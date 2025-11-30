import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd

from utils.rand_augment import RandAugment


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank, use_grid_sampler=False, aug_params=None, landmark_csv=None):
        super(MXFaceDataset, self).__init__()
        self.use_grid_sampler = use_grid_sampler
        self.landmark_csv = landmark_csv
        self.ldmk_info = None
        
        if self.landmark_csv and os.path.exists(self.landmark_csv):
            self.ldmk_info = pd.read_csv(self.landmark_csv, sep=',', index_col=0)

        if self.use_grid_sampler and aug_params is not None:
            from utils.data_aug_grid_sampler import GridSampleAugmenter
            self.grid_augmenter = GridSampleAugmenter(aug_params, input_size=112)
            
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        
        landmark = None
        if self.ldmk_info is not None:
             try:
                 # Use 'index' (dataset index) to lookup in ldmk_info
                 # Note: predict_landmark.py uses dataset index as 'idx' in CSV.
                 ldmk_vals = self.ldmk_info.loc[index].values 
                 landmark = torch.from_numpy(ldmk_vals.reshape(-1, 2)).float()
             except KeyError:
                 pass

        if self.use_grid_sampler:
            sample_pil = Image.fromarray(sample)
            sample_aug, theta = self.grid_augmenter.augment(sample_pil)
            sample = self.transform(sample_aug)
            
            if landmark is not None:
                 landmark = self.transform_ldmk(landmark, theta)
            
            if landmark is not None:
                return sample, label, theta, landmark
            return sample, label, theta
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            
            if landmark is not None:
                return sample, label, torch.zeros(2,3), landmark
            return sample, label

    def __len__(self):
        return len(self.imgidx)
        
    def transform_ldmk(self, ldmk, theta):
        inv_theta = self.inv_matrix(theta.unsqueeze(0)).squeeze(0)
        # ldmk is Nx2. We need to append 1 for affine transform
        # ldmk range is [0, 1] (normalized) or pixel? 
        # DFA output is usually normalized [-1, 1] or [0, 1].
        # CVLface predict_landmark.py output seems to be normalized [0, 1] based on `visualize` usage.
        # But let's check `repeated_dataset_with_ldmk_theta.py` again.
        # It does: (((ldmk) * 2 - 1) @ inv_theta.T) / 2 + 0.5
        # This implies ldmk is in [0, 1].
        
        ldmk = torch.cat([ldmk, torch.ones(ldmk.shape[0], 1)], dim=1).float()
        transformed_ldmk = (((ldmk) * 2 - 1) @ inv_theta.T) / 2 + 0.5
        
        # Handle mirror if needed (determinant < 0)
        if inv_theta[0, 0] < 0:
            transformed_ldmk = self.mirror_ldmk(transformed_ldmk)
        return transformed_ldmk

    def mirror_ldmk(self, ldmk):
        # Assuming 5 landmarks
        new_ldmk = ldmk.clone()
        # Swap left/right eye (0, 1)
        tmp = new_ldmk[1, :].clone()
        new_ldmk[1, :] = new_ldmk[0, :]
        new_ldmk[0, :] = tmp
        # Swap left/right mouth corner (3, 4)
        tmp1 = new_ldmk[4, :].clone()
        new_ldmk[4, :] = new_ldmk[3, :]
        new_ldmk[3, :] = tmp1
        return new_ldmk

    def inv_matrix(self, theta):
        # torch batched version
        assert theta.ndim == 3
        a, b, t1 = theta[:, 0,0], theta[:, 0,1], theta[:, 0,2]
        c, d, t2 = theta[:, 1,0], theta[:, 1,1], theta[:, 1,2]
        det = a * d - b * c
        inv_det = 1.0 / det
        inv_mat = torch.stack([
            torch.stack([d * inv_det, -b * inv_det, (b * t2 - d * t1) * inv_det], dim=1),
            torch.stack([-c * inv_det, a * inv_det, (c * t1 - a * t2) * inv_det], dim=1)
        ], dim=1)
        return inv_mat


class FaceDatasetFolder(Dataset):
    def __init__(self, root_dir, local_rank, number_sample=10):
        super(FaceDatasetFolder, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((112,112)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.number_sample=number_sample

        self.imgidx, self.labels=self.scan(root_dir)
    def scan(self,root):
        imgidex=[]
        labels=[]
        lb=-1
        list_dir=os.listdir(root)
        list_dir.sort()
        for l in list_dir:
            images=os.listdir(os.path.join(root,l))
            lb += 1
            if (len(images)>=self.number_sample):
                ln=self.number_sample
            else:
                ln=len(images)
            for i in range(ln):
                img=images[i]
                imgidex.append(os.path.join(l,img))
                labels.append(lb)
        return imgidex,labels
    def readImage(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img=self.readImage(path)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)