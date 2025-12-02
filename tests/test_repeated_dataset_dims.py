#!/usr/bin/env python3
"""Test script to verify RepeatedLmdbDataset output dimensions."""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, '/Users/sumanthrc/Documents/Antigravity_projects/kd_fr')

from utils.repeated_dataset import RepeatedLmdbDataset
from omegaconf import OmegaConf

# Minimal config for testing
aug_params = {
    'scale_min': 0.8,
    'scale_max': 1.2,
    'rot_prob': 0.2,
    'max_rot': 30,
    'hflip_prob': 0.5,
    'extra_offset': 0.15,
    'photometric_num_ops': 2,
    'photometric_magnitude': 14,
    'photometric_magnitude_offset': 9,
    'photometric_num_magnitude_bins': 31,
    'blur_magnitude': 1.0,
    'blur_prob': 0.3,
    'cutout_prob': 0.2,
}

print("Testing RepeatedLmdbDataset...")
print("=" * 80)

# You'll need to provide a valid lmdb path here
lmdb_path = "/workspace/webface4m_subset.lmdb"  # Adjust this path

if not os.path.exists(lmdb_path):
    print(f"ERROR: LMDB file not found: {lmdb_path}")
    print("Please update the lmdb_path in this test script.")
    sys.exit(1)

# Create dataset with grid sampler enabled
dataset = RepeatedLmdbDataset(
    lmdb_file=lmdb_path,
    aug_params=aug_params,
    repeated_augment_prob=0.15,
    use_same_image=False,
    disable_repeat=False,
    skip_aug_prob_in_disable_repeat=0.0,
    second_img_augment=False  # This should create dummy theta2
)

print(f"Dataset length: {len(dataset)}")
print()

# Test single sample
print("Testing single sample...")
sample = dataset[0]
print(f"Number of returned values: {len(sample)}")

if len(sample) == 5:
    img1, img2, label, theta1, theta2 = sample
    print(f"img1 shape: {img1.shape}, type: {type(img1)}")
    print(f"img2 shape: {img2.shape}, type: {type(img2)}")
    print(f"label: {label}, type: {type(label)}")
    print(f"theta1 shape: {theta1.shape}, type: {type(theta1)}, dtype: {theta1.dtype}")
    print(f"theta2 shape: {theta2.shape}, type: {type(theta2)}, dtype: {theta2.dtype}")
    print()
    
    # Test batching
    print("Testing batching...")
    batch_size = 4
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Use 0 to avoid multiprocessing issues
    )
    
    batch = next(iter(dataloader))
    img1_batch, img2_batch, label_batch, theta1_batch, theta2_batch = batch
    
    print(f"Batched img1 shape: {img1_batch.shape}")
    print(f"Batched img2 shape: {img2_batch.shape}")
    print(f"Batched label shape: {label_batch.shape}")
    print(f"Batched theta1 shape: {theta1_batch.shape}, dtype: {theta1_batch.dtype}")
    print(f"Batched theta2 shape: {theta2_batch.shape}, dtype: {theta2_batch.dtype}")
    print()
    
    # Test concatenation (as done in train_AdaDistill_data_aug.py)
    print("Testing concatenation (line 258 in train script)...")
    try:
        theta_concat = torch.cat([theta1_batch, theta2_batch], dim=0)
        print(f"✓ Concatenation successful! Shape: {theta_concat.shape}")
    except RuntimeError as e:
        print(f"✗ Concatenation failed with error:")
        print(f"  {e}")
        print(f"  theta1_batch.ndim = {theta1_batch.ndim}, theta2_batch.ndim = {theta2_batch.ndim}")
else:
    print(f"Unexpected return: {sample}")

print("=" * 80)
print("Test complete!")
