import torch
import numpy as np
import csv
import os
from .lmdb_dataset import LmdbDataset
import torchvision.transforms.functional as TF
from PIL import Image
import io

class RepeatedLmdbDataset(LmdbDataset):
    def __init__(self,
                 lmdb_file,
                 landmark_path,
                 transforms=None,
                 lmdb_handler=None,
                 aug_params=None,
                 repeated_augment_prob=0.0,
                 use_same_image=False,
                 disable_repeat=True,
                 skip_aug_prob_in_disable_repeat=0.0,
                 second_img_augment=False):
        
        # Initialize base LmdbDataset
        # We force use_grid_sampler=True internally for this class as it relies on it for theta
        # But we handle the augmentation call manually in get_one_sample
        super(RepeatedLmdbDataset, self).__init__(lmdb_file, 
                                                  transforms=transforms, 
                                                  lmdb_handler=lmdb_handler, 
                                                  use_grid_sampler=True, 
                                                  aug_params=aug_params)
        
        self.grid_augmenter = getattr(self, 'grid_augmenter', None)

        self.landmark_path = landmark_path
        self.repeated_augment_prob = repeated_augment_prob
        self.use_same_image = use_same_image
        self.disable_repeat = disable_repeat
        self.skip_aug_prob_in_disable_repeat = skip_aug_prob_in_disable_repeat
        self.second_img_augment = second_img_augment

        # Load landmarks
        # Assuming CSV format: index, ldmk1_x, ldmk1_y, ...
        # We load it into a dictionary for O(1) access
        self.ldmk_info = {}
        with open(landmark_path, 'r') as f:
            reader = csv.reader(f)
            # Check if header exists. If first row has non-numeric, skip it.
            # We'll try to parse the first row.
            rows = list(reader)
            start_row = 0
            try:
                float(rows[0][0])
            except ValueError:
                start_row = 1
            
            for row in rows[start_row:]:
                if not row: continue
                idx = int(float(row[0])) # First column is index
                vals = [float(x) for x in row[1:]]
                self.ldmk_info[idx] = np.array(vals)
        
        self.identity_theta = torch.zeros(2, 3)
        self.identity_theta[0, 0] = 1
        self.identity_theta[1, 1] = 1

        self.do_augment = True
        self.prev_index = None
        self.prev_label = None
        self.repeated = False
        
        # Build label index map for sampling different images of same identity
        self._build_label_index_map()

    def _build_label_index_map(self):
        """
        Builds a mapping from label to a list of indices for that label.
        This is required for sampling a different image of the same identity.
        """
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def set_augmentation(self, value):
        self.do_augment = value

    def get_one_sample(self, index, augment=True):
        # Read image and label from LMDB
        image_bytes = self.handler.get(self.keys[index])
        image_bytes = io.BytesIO(image_bytes)
        image_pil = Image.open(image_bytes)
        
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)

        theta = None
        sample = image_pil

        if augment and self.grid_augmenter is not None:
            # grid_augmenter.augment returns (aug_sample_pil, theta)
            sample, theta = self.grid_augmenter.augment(sample)
        
        # Convert to tensor
        sample = TF.pil_to_tensor(sample)
        
        if self._transforms is not None:
            sample = self._transforms(sample)

        # Load landmark
        if index not in self.ldmk_info:
             raise KeyError(f"Index {index} not found in landmark file {self.landmark_path}")
        
        ldmk = self.ldmk_info[index]

        if len(ldmk) == 10:
            ldmk = ldmk.reshape(-1, 2)
        else:
            # Assuming 3D points or other format, take first 2 coords
            ldmk = ldmk.reshape(-1, 3)[:, :2]

        ldmk = torch.from_numpy(ldmk)
        
        if theta is not None:
            ldmk = self.transform_ldmk(ldmk, theta)
        else:
            theta = self.identity_theta.clone()

        ldmk = ldmk.float()
        return sample, label, ldmk, theta

    def __getitem__(self, index):
        placeholder = 0
        augment = self.do_augment

        # Repeated sampling logic
        if self.prev_index is not None and augment:
            if self.repeated_augment_prob > 0:
                if np.random.rand() < self.repeated_augment_prob and not self.repeated:
                    self.repeated = True
                    if self.use_same_image:
                        index = self.prev_index
                    else:
                        # Sample another image from the previous label
                        indices = self.label_to_indices[self.prev_label]
                        if len(indices) > 0:
                            import random
                            index = random.choice(indices)
                        else:
                            index = self.prev_index
                else:
                    self.repeated = False

        if self.disable_repeat:
            if np.random.rand() < self.skip_aug_prob_in_disable_repeat:
                augment = False
        
        sample1, target, ldmk1, theta1 = self.get_one_sample(index, augment=augment)

        if self.repeated:
            if self.prev_label is not None:
                if augment and self.prev_label != target.item():
                    print(f'Warning repeated label different {target.item()} {self.prev_label}')

        self.prev_index = index
        self.prev_label = target.item()

        if self.disable_repeat:
             return sample1, target, ldmk1, theta1, placeholder, placeholder, placeholder

        # Get extra image index
        if self.use_same_image:
            extra_index = index
        else:
            indices = self.label_to_indices[target.item()]
            if len(indices) > 0:
                import random
                extra_index = random.choice(indices)
            else:
                extra_index = index

        extra_augment = augment and self.second_img_augment
        sample2, target2, ldmk2, theta2 = self.get_one_sample(extra_index, augment=extra_augment)
        
        # Ensure targets match (sanity check)
        # Note: In rare cases if label map is wrong, this might fail, but it should be correct.
        assert target.item() == target2.item() 

        return sample1, target, ldmk1, theta1, sample2, ldmk2, theta2

    def transform_ldmk(self, ldmk, theta):
        inv_theta = inv_matrix(theta.unsqueeze(0)).squeeze(0)
        ldmk = torch.cat([ldmk, torch.ones(ldmk.shape[0], 1)], dim=1).float()
        
        transformed_ldmk = (((ldmk) * 2 - 1) @ inv_theta.T) / 2 + 0.5
        if inv_theta[0, 0] < 0:
            transformed_ldmk = self.mirror_ldmk(transformed_ldmk)
        return transformed_ldmk

    def mirror_ldmk(self, ldmk):
        if len(ldmk) == 5:
            return self.mirror_ldmk_5(ldmk)
        else:
            # For robustness, just return as is or implement 68 points if needed
            return ldmk

    def mirror_ldmk_5(self, ldmk):
        new_ldmk = ldmk.clone()
        # Swap left/right eye (indices 0 and 1)
        tmp = new_ldmk[1, :].clone()
        new_ldmk[1, :] = new_ldmk[0, :]
        new_ldmk[0, :] = tmp
        # Swap left/right mouth corner (indices 3 and 4)
        tmp1 = new_ldmk[4, :].clone()
        new_ldmk[4, :] = new_ldmk[3, :]
        new_ldmk[3, :] = tmp1
        return new_ldmk

def inv_matrix(theta):
    # torch batched version
    # theta: (B, 2, 3)
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
