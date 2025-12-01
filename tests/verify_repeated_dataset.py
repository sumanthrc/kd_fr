import os
import sys
import shutil
import numpy as np
import torch
from PIL import Image
import csv

# Add root to path
sys.path.append(os.getcwd())

from utils.lmdb_dataset import LmdbDataset
from utils.repeated_dataset import RepeatedLmdbDataset

def create_dummy_data(root_dir):
    img_dir = os.path.join(root_dir, 'imgs')
    lmdb_dir = os.path.join(root_dir, 'lmdb')
    os.makedirs(img_dir, exist_ok=True)
    
    # Create 2 classes, 2 images each
    classes = ['c1', 'c2']
    img_paths = []
    labels = []
    
    idx = 0
    for i, c in enumerate(classes):
        c_dir = os.path.join(img_dir, c)
        os.makedirs(c_dir, exist_ok=True)
        for j in range(2):
            img_path = os.path.join(c_dir, f'img_{j}.jpg')
            # Create random image
            img = Image.fromarray(np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
            img.save(img_path)
            img_paths.append(img_path)
            labels.append(i)
            idx += 1
            
    # Create LMDB
    print("Creating dummy LMDB...")
    LmdbDataset._create_database_from_image_folder(img_dir, lmdb_dir)
    
    # Create dummy landmarks CSV
    # 4 images, 5 landmarks each (10 values)
    # Format: index, x1, y1, x2, y2, ...
    # Normalized 0-1
    ldmks = np.random.rand(4, 10)
    
    csv_path = os.path.join(root_dir, 'landmarks.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['index'] + [f'v{i}' for i in range(10)])
        for i in range(4):
            row = [i] + ldmks[i].tolist()
            writer.writerow(row)
    
    return lmdb_dir, csv_path

def test_repeated_dataset():
    temp_dir = 'tests/temp_data'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        lmdb_path, ldmk_path = create_dummy_data(temp_dir)
        
        aug_params = {
            'scale_min': 0.7,
            'scale_max': 2.0,
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
            'cutout_prob': 0.2
        }
        
        print("Initializing RepeatedLmdbDataset...")
        dataset = RepeatedLmdbDataset(
            lmdb_file=lmdb_path,
            landmark_path=ldmk_path,
            aug_params=aug_params,
            repeated_augment_prob=0.5,
            use_same_image=False,
            disable_repeat=False,
            second_img_augment=True
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Test __getitem__
        print("Fetching a sample...")
        sample1, target, ldmk1, theta1, sample2, ldmk2, theta2 = dataset[0]
        
        print("Sample 1 shape:", sample1.shape)
        print("Target:", target)
        print("Ldmk 1 shape:", ldmk1.shape)
        print("Theta 1 shape:", theta1.shape)
        print("Sample 2 shape:", sample2.shape)
        print("Ldmk 2 shape:", ldmk2.shape)
        print("Theta 2 shape:", theta2.shape)
        
        assert sample1.shape == (3, 112, 112)
        assert theta1.shape == (2, 3)
        assert ldmk1.shape == (5, 2)
        
        # Test with disable_repeat=True
        print("\nTesting disable_repeat=True...")
        dataset.disable_repeat = True
        out = dataset[0]
        # Should return placeholders for second part
        # sample1, target, ldmk1, theta1, placeholder, placeholder, placeholder
        assert len(out) == 7
        assert out[4] == 0
        
        print("\nVerification Successful!")
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_repeated_dataset()
