import argparse
import os
import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
from config.config import config as default_cfg
from utils.lmdb_dataset import LmdbDataset
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Landmarks and Augmentation')
    parser.add_argument('--config', type=str, default=None, help='path to config file')
    parser.add_argument('--landmark_csv', type=str, required=True, help='path to landmark csv')
    parser.add_argument('--output_dir', type=str, default='vis_output', help='output directory')
    parser.add_argument('--num_samples', type=int, default=10, help='number of samples to visualize')
    return parser.parse_args()

def merge_config(args):
    cfg = default_cfg
    if args.config:
        yaml_cfg = OmegaConf.load(args.config)
        for k, v in yaml_cfg.items():
            setattr(cfg, k, v)
    return cfg

def draw_landmarks(img, landmarks, color=(0, 255, 0)):
    """
    Draw landmarks on an image.
    img: numpy array (H, W, 3) in BGR or RGB
    landmarks: numpy array (5, 2) normalized [0, 1] or pixel coordinates?
               Based on dataset logic, they are normalized [0, 1].
    """
    h, w = img.shape[:2]
    img_copy = img.copy()
    
    for i, (x, y) in enumerate(landmarks):
        # Denormalize if values are small (likely normalized)
        if x <= 1.0 and y <= 1.0:
            cx, cy = int(x * w), int(y * h)
        else:
            cx, cy = int(x), int(y)
            
        cv2.circle(img_copy, (cx, cy), 2, color, -1)
        # cv2.putText(img_copy, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
    return img_copy

def main():
    args = parse_args()
    cfg = merge_config(args)
    
    # Force grid sampler ON for visualization
    cfg.use_grid_sampler = True
    
    # Setup Dataset
    print(f"Loading Dataset: {cfg.dataset}")
    if cfg.dataset != "WEBFACE4M":
        print("This script is optimized for WEBFACE4M as requested, but trying generic load.")

    # Transform to tensor is needed for LmdbDataset
    tfm = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        # We don't want to normalize for visualization, or we need to denormalize.
        # LmdbDataset applies transform AFTER augmentation.
        # But Augmenter expects PIL. LmdbDataset handles conversion.
        # Let's use a simple ToTensor so we get the augmented tensor.
    ])
    
    # We need to manually instantiate LmdbDataset to pass landmark_csv
    # Note: LmdbDataset expects 'aug_params' if use_grid_sampler is True
    dataset = LmdbDataset(
        lmdb_file=cfg.lmdb_path, 
        transforms=tfm, 
        use_grid_sampler=True, 
        aug_params=cfg.grid_sampler_aug_params,
        landmark_csv=args.landmark_csv
    )
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Visualizing {args.num_samples} samples to {args.output_dir}")
    
    for i in range(args.num_samples):
        # Get item
        # Returns: image_tensor, label, theta, landmark
        try:
            item = dataset[i]
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
            
        if len(item) == 4:
            img_tensor, label, theta, landmark = item
        else:
            print(f"Sample {i} did not return 4 items (len={len(item)}). Check if landmarks are loaded.")
            continue
            
        # Convert Tensor to Numpy Image (H, W, C)
        # Tensor is (C, H, W)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        # If we didn't normalize, it's [0, 1]. Scale to [0, 255]
        img_np = (img_np * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Draw Augmented Landmarks
        # landmark is a Tensor (5, 2)
        ldmk_np = landmark.numpy()
        
        vis_img = draw_landmarks(img_bgr, ldmk_np, color=(0, 0, 255)) # Red for augmented
        
        # Save
        save_path = os.path.join(args.output_dir, f"sample_{i}_aug.jpg")
        cv2.imwrite(save_path, vis_img)
        print(f"Saved {save_path}")
        
        # Optional: Visualize Original (Non-augmented)
        # We can't easily get the original from the same call because augmentation is random inside __getitem__.
        # But we can manually load it from dataset.handler if we wanted, but that's complex.
        # Just showing the augmented result is sufficient to prove theta was applied.

if __name__ == "__main__":
    main()
