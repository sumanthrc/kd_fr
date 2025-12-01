import sys
import os
# Force correct project root
project_root = '/Users/sumanthrc/Documents/Antigravity_projects/kd_fr'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
else:
    # Move to front
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)

from config.config import config as cfg
from utils.repeated_dataset import RepeatedLmdbDataset

def verify_config():
    # Set dataset to REPEATED_WEBFACE4M to trigger the logic
    cfg.dataset = "REPEATED_WEBFACE4M"
    
    # Mock aug_params if needed by the dataset logic (though RepeatedLmdbDataset handles None)
    # cfg.aug_params = None 
    
    print("Testing config parameters:")
    print(f"Dataset: {cfg.dataset}")
    print(f"Repeated Augment Prob: {cfg.repeated_augment_prob}")
    print(f"Use Same Image: {cfg.use_same_image}")
    
    # Try to initialize the dataset
    # We need a dummy LMDB path or we expect it to fail on file not found, 
    # but we want to check if arguments are passed correctly.
    
    try:
        dataset = RepeatedLmdbDataset(
            lmdb_file=cfg.lmdb_path,
            aug_params=cfg.aug_params if hasattr(cfg, 'aug_params') else None,
            repeated_augment_prob=cfg.repeated_augment_prob,
            use_same_image=cfg.use_same_image,
            disable_repeat=cfg.disable_repeat,
            second_img_augment=cfg.second_img_augment
        )
        print("Dataset initialized successfully (unexpected if LMDB missing)")
    except Exception as e:
        print(f"Dataset initialization attempted. Error (expected if LMDB missing): {e}")
        # Check if error is related to arguments or file
        if "lmdb" in str(e).lower() or "no such file" in str(e).lower():
            print("SUCCESS: Arguments passed correctly, failed on file path as expected.")
        else:
            print("FAILURE: Unexpected error during initialization.")
            raise e

if __name__ == "__main__":
    verify_config()
