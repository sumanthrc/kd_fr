import sys
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
from PIL import Image

# Add kd_fr to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'kd_fr'))

from utils.lmdb_dataset import LmdbDataset
from aligners import get_aligner
from omegaconf import OmegaConf

def visualize(img_tensor, landmarks):
    """
    Visualizes landmarks on an image tensor.
    img_tensor: (C, H, W), normalized [-1, 1]
    landmarks: (5, 2)
    """
    # Denormalize and convert to numpy (H, W, C) BGR
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5 + 0.5) * 255
    img = np.clip(img, 0, 255).astype(np.uint8).copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    h, w, _ = img.shape
    
    # Draw landmarks
    for idx, point in enumerate(landmarks):
        x, y = int(point[0]), int(point[1])
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        
    return img

def main():
    parser = argparse.ArgumentParser(description='Predict landmarks using DFA and save to CSV')
    parser.add_argument('--lmdb_path', type=str, required=True, help='Path to LMDB dataset')
    parser.add_argument('--save_name', type=str, default='landmarks.csv', help='Output CSV filename')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--padding_ratio', type=float, default=0.215, help='Padding ratio for DFA')
    args = parser.parse_args()

    # Load Aligner Config
    config_path = os.path.join(os.path.dirname(__file__), 'kd_fr/aligners/configs/dfa.yaml')
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    print(f"Loading aligner from {config_path}")
    cfg = OmegaConf.load(config_path)
    aligner = get_aligner(cfg).to('cuda').eval()

    # Load Dataset
    # Standard transform: ToTensor and Normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print(f"Loading dataset from {args.lmdb_path}")
    dataset = LmdbDataset(lmdb_file=args.lmdb_path, transforms=transform)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False
    )

    # Determine save path
    # If save_name is just a filename, save it in the same directory as the LMDB
    if os.path.dirname(args.save_name):
        save_path = args.save_name
    else:
        save_path = os.path.join(os.path.dirname(args.lmdb_path), args.save_name)
        
    print(f"Saving landmarks to {save_path}")
    
    f = open(save_path, 'w')
    # Write Header
    # Assuming 5 points (10 values)
    header = "idx," + ",".join([f"ldmk_{i}" for i in range(10)]) + "\n"
    f.write(header)

    vis_dir = os.path.join(os.path.dirname(save_path), 'vis_landmarks')
    
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Predicting'):
        # LmdbDataset returns (img, label) or (img, label, theta)
        if len(batch) == 2:
            imgs, labels = batch
        else:
            imgs, labels, _ = batch
            
        imgs = imgs.to('cuda')
        
        with torch.no_grad():
            # aligner forward pass
            # returns: aligned_x, orig_pred_ldmks, aligned_ldmks, score, thetas, bbox
            # We want orig_pred_ldmks which are coordinates on the input image
            _, orig_pred_ldmks, _, _, _, _ = aligner(imgs, padding_ratio_override=args.padding_ratio)
            
        # orig_pred_ldmks shape: (B, 5, 2)
        
        for i, ldmk in enumerate(orig_pred_ldmks):
            # Flatten: x1, y1, x2, y2...
            ldmk_flat = ldmk.cpu().numpy().reshape(-1)
            
            global_idx = idx * args.batch_size + i
            
            line = f"{global_idx}," + ",".join([f"{x:.5f}" for x in ldmk_flat]) + "\n"
            f.write(line)
            
        # Visualization (every 100 batches)
        if idx % 100 == 0:
            os.makedirs(vis_dir, exist_ok=True)
            # Visualize first image in batch
            vis_img = visualize(imgs[0], orig_pred_ldmks[0])
            cv2.imwrite(os.path.join(vis_dir, f'{idx * args.batch_size}.jpg'), vis_img)

    f.close()
    print("Done.")

if __name__ == '__main__':
    main()