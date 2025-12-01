import torch
import numpy as np
import io
from PIL import Image
import torchvision.transforms.functional as TF
from .lmdb_dataset import LmdbDataset
import random

class RepeatedLmdbDataset(LmdbDataset):
    def __init__(self,
                 lmdb_file,
                 transforms=None,
                 lmdb_handler=None,
                 aug_params=None,
                 repeated_augment_prob=0.0, # Kept for compatibility, ignored
                 use_same_image=False,
                 disable_repeat=True, # Kept for compatibility, ignored (Always False behavior)
                 skip_aug_prob_in_disable_repeat=0.0, # Kept for compatibility, ignored
                 second_img_augment=False,
                 landmark_path=None): # Kept for compatibility
        
        # Initialize base LmdbDataset
        use_grid_sampler = aug_params is not None
        
        super(RepeatedLmdbDataset, self).__init__(lmdb_file, 
                                                  transforms=transforms, 
                                                  lmdb_handler=lmdb_handler, 
                                                  use_grid_sampler=use_grid_sampler, 
                                                  aug_params=aug_params)
        
        self.grid_augmenter = getattr(self, 'grid_augmenter', None)

        self.use_same_image = use_same_image
        self.second_img_augment = second_img_augment
        
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

    def get_one_sample(self, index, augment=True):
        # Read image and label from LMDB
        image_bytes = self.handler.get(self.keys[index])
        image_bytes = io.BytesIO(image_bytes)
        image_pil = Image.open(image_bytes)
        
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)

        sample = image_pil
        theta = None

        if augment and self.use_grid_sampler and self.grid_augmenter is not None:
            # grid_augmenter.augment returns (aug_sample_pil, theta)
            sample, theta = self.grid_augmenter.augment(sample)
        
        # Convert to tensor
        sample = TF.pil_to_tensor(sample)
        
        if self._transforms is not None:
            sample = self._transforms(sample)

        return sample, label, theta

    def __getitem__(self, index):
        # Always return pairs, ignoring disable_repeat and burst logic
        
        # 1. Get first sample
        sample1, target, theta1 = self.get_one_sample(index, augment=True)

        # 2. Determine index for second sample
        if self.use_same_image:
            extra_index = index
        else:
            # Sample another image from the same label
            indices = self.label_to_indices[target.item()]
            if len(indices) > 0:
                extra_index = random.choice(indices)
            else:
                extra_index = index

        # 3. Get second sample
        # CVLface logic: augment second image if second_img_augment is True
        # Note: CVLface repeated_dataset.py lines 45-46: if not skip_augment and self.repeated_sampling_cfg.second_img_augment:
        # We assume skip_augment is False (standard training)
        
        augment_second = self.second_img_augment
        sample2, target2, theta2 = self.get_one_sample(extra_index, augment=augment_second)
        
        # Ensure targets match
        assert target.item() == target2.item()

        # 4. Return
        # Match CVLface return signature: sample1, sample2, target, [theta1, theta2]
        
        if theta1 is not None:
            # If theta1 exists, we assume theta2 might also exist (or be None if not augmented)
            # CVLface returns theta1, theta2. If theta2 is None (no augment), we might need a dummy?
            # CVLface line 27: theta2 = torch.tensor([0])
            # If sample2 not augmented, theta2 remains dummy.
            
            if theta2 is None:
                theta2 = torch.tensor([0]) # Dummy
                
            return sample1, sample2, target, theta1, theta2
        else:
            return sample1, sample2, target
