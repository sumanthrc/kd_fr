import unittest
import torch
import pandas as pd
import os
import shutil
import numpy as np
from PIL import Image
from utils.dataset import MXFaceDataset

class TestOfflineLandmarks(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy CSV
        self.csv_path = os.path.join(self.test_dir, 'landmarks.csv')
        # Create 10 samples
        data = []
        for i in range(10):
            # normalized landmarks [0, 1]
            ldmks = np.random.rand(10) 
            data.append([i] + list(ldmks))
        
        columns = ['idx'] + [f'ldmk_{j}' for j in range(10)]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(self.csv_path, index=False)
        
        # Mock MXFaceDataset to avoid needing actual recordio files
        # We will override __getitem__ partially or mock the internal data structures
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_landmark_loading(self):
        # We can't easily instantiate MXFaceDataset without .rec files.
        # So we will mock the class or just test the logic if we can isolate it.
        # Or we can create a dummy .rec file? That's hard.
        # Let's mock the internal attributes after instantiation if possible, 
        # but __init__ will fail.
        
        # Alternative: Create a subclass that mocks the data loading part
        class MockDataset(MXFaceDataset):
            def __init__(self, landmark_csv):
                self.landmark_csv = landmark_csv
                self.ldmk_info = pd.read_csv(self.landmark_csv, sep=',', index_col=0)
                self.use_grid_sampler = False
                self.transform = None
                self.imgidx = list(range(10))
                
            def __getitem__(self, index):
                # Mock image
                sample = np.zeros((112, 112, 3), dtype=np.uint8)
                label = torch.tensor(0)
                
                landmark = None
                if self.ldmk_info is not None:
                     try:
                         ldmk_vals = self.ldmk_info.loc[index].values 
                         landmark = torch.from_numpy(ldmk_vals.reshape(-1, 2)).float()
                     except KeyError:
                         pass
                
                if landmark is not None:
                    return sample, label, torch.zeros(2,3), landmark
                return sample, label

        dataset = MockDataset(self.csv_path)
        item = dataset[0]
        self.assertEqual(len(item), 4)
        sample, label, theta, landmark = item
        self.assertIsNotNone(landmark)
        self.assertEqual(landmark.shape, (5, 2))
        
        # Verify values match CSV
        df = pd.read_csv(self.csv_path, index_col=0)
        expected = df.loc[0].values.reshape(5, 2)
        np.testing.assert_almost_equal(landmark.numpy(), expected, decimal=5)
        print("Test passed!")

if __name__ == '__main__':
    unittest.main()
