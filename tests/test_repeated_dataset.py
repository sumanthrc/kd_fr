import unittest
import os
import shutil
import torch
import numpy as np
from PIL import Image
import sys

# Force correct project root
project_root = '/Users/sumanthrc/Documents/Antigravity_projects/kd_fr'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
else:
    # Move to front
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)

# Force correct project root
project_root = '/Users/sumanthrc/Documents/Antigravity_projects/kd_fr'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
else:
    # Move to front
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)

from utils.lmdb_dataset import LmdbDataset
from utils.repeated_dataset import RepeatedLmdbDataset
# We need to mock or create a dummy landmark file for the other dataset
from utils.repeated_dataset_with_ldmk_theta import RepeatedLmdbDataset as RepeatedLdmkDataset

class TestRepeatedDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for test data
        cls.test_dir = 'test_data_repeated'
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Create dummy images
        cls.img_dir = os.path.join(cls.test_dir, 'images')
        os.makedirs(cls.img_dir, exist_ok=True)
        
        # Create 2 identities, 2 images each
        cls.identities = ['id1', 'id2']
        for identity in cls.identities:
            id_dir = os.path.join(cls.img_dir, identity)
            os.makedirs(id_dir, exist_ok=True)
            for i in range(2):
                img = Image.new('RGB', (112, 112), color=tuple(np.random.randint(0, 255, (3,)).tolist()))
                img.save(os.path.join(id_dir, f'{i}.jpg'))
        
        # Create LMDB
        cls.lmdb_path = os.path.join(cls.test_dir, 'test.lmdb')
        LmdbDataset._create_database_from_image_folder(cls.img_dir, cls.lmdb_path)
        
        # Create dummy landmark file
        cls.ldmk_path = os.path.join(cls.test_dir, 'landmarks.csv')
        with open(cls.ldmk_path, 'w') as f:
            # Header
            f.write('index,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5\n')
            # We have 4 images total (0, 1, 2, 3)
            for i in range(4):
                # Dummy 5 landmarks
                f.write(f'{i},10,10,20,20,30,30,40,40,50,50\n')

    @classmethod
    def tearDownClass(cls):
        # Cleanup
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_repeated_dataset(self):
        dataset = RepeatedLmdbDataset(
            lmdb_file=self.lmdb_path,
            repeated_augment_prob=1.0, # Force repeat
            use_same_image=False,
            disable_repeat=False,
            second_img_augment=False
        )
        
        # Test getting an item
        # Since we have few images, we might get the same index if random choice picks it, 
        # but let's just check structure.
        sample1, sample2, target = dataset[0]
        
        self.assertIsInstance(sample1, torch.Tensor)
        self.assertIsInstance(sample2, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
        
        # Check shapes
        self.assertEqual(sample1.shape, (3, 112, 112)) # Assuming TF.pil_to_tensor keeps size
        
        # Check target consistency (dataset logic ensures this)
        # However, since we return target from sample1, we implicitly trust sample2 is same identity.
        # In our dummy dataset, index 0 is id1. 
        # id1 has indices 0 and 1.
        # So sample2 should be index 0 or 1.
        # Both have label 0 (since id1 is first folder).
        
        # Let's verify label
        self.assertEqual(target.item(), 0)

    def test_repeated_dataset_same_image(self):
        dataset = RepeatedLmdbDataset(
            lmdb_file=self.lmdb_path,
            repeated_augment_prob=1.0,
            use_same_image=True,
            disable_repeat=False
        )
        
        sample1, sample2, target = dataset[0]
        self.assertEqual(target.item(), 0)
        # With same image and no augmentation (default), samples might be identical
        # But we didn't pass transforms, so they are just tensors of the image.
        self.assertTrue(torch.allclose(sample1, sample2))

    def test_repeated_ldmk_dataset(self):
        dataset = RepeatedLdmkDataset(
            lmdb_file=self.lmdb_path,
            landmark_path=self.ldmk_path,
            repeated_augment_prob=1.0,
            use_same_image=False,
            disable_repeat=False,
            second_img_augment=False
        )
        
        # Returns: sample1, target, ldmk1, theta1, sample2, ldmk2, theta2
        item = dataset[0]
        self.assertEqual(len(item), 7)
        
        sample1, target, ldmk1, theta1, sample2, ldmk2, theta2 = item
        
        self.assertEqual(target.item(), 0)
        self.assertEqual(ldmk1.shape, (5, 2))
        self.assertEqual(theta1.shape, (2, 3))

if __name__ == '__main__':
    unittest.main()
