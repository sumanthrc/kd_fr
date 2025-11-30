import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from core.models import build_aligner
from utils.dataset import MXFaceDataset, FaceDatasetFolder
from utils.lmdb_dataset import LmdbDataset
from omegaconf import OmegaConf
from config.config import config as default_cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Predict Landmarks using DFA')
    parser.add_argument('--config', type=str, default=None, help='path to config file')
    parser.add_argument('--save_name', type=str, default='landmarks.csv', help='output csv filename')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    return parser.parse_args()

def merge_config(args):
    cfg = default_cfg
    if args.config:
        yaml_cfg = OmegaConf.load(args.config)
        for k, v in yaml_cfg.items():
            setattr(cfg, k, v)
    return cfg

def main():
    args = parse_args()
    cfg = merge_config(args)
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build Aligner
    print("Building Aligner...")
    aligner = build_aligner(cfg, local_rank=0)
    if aligner is None:
        raise ValueError("Aligner configuration not found or failed to build.")
    aligner = aligner.to(device)
    aligner.eval()
    
    # Setup Dataset
    print(f"Loading Dataset: {cfg.dataset}")
    if cfg.dataset == "WEBFACE4M":
        # Ensure we have a transform that returns a tensor
        from torchvision import transforms
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = LmdbDataset(lmdb_file=cfg.lmdb_path, transforms=tfm, use_grid_sampler=False)
        
    elif getattr(cfg, "db_file_format", "") == "rec":
        dataset = MXFaceDataset(root_dir=cfg.rec, local_rank=0, use_grid_sampler=False)
    else:
        dataset = FaceDatasetFolder(root_dir=cfg.data_path, local_rank=0, number_sample=cfg.sample)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False
    )
    
    # Open CSV for writing
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
    save_path = os.path.join(cfg.output, args.save_name)
        
    print(f"Saving landmarks to {save_path}")
    
    with open(save_path, 'w') as f:
        # Write Header
        # Assuming 5 landmarks (10 values)
        header = ['idx'] + [f'ldmk_{i}' for i in range(10)]
        f.write(','.join(header) + '\n')
        
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                if len(batch) == 2:
                    imgs, labels = batch
                elif len(batch) == 3:
                    imgs, labels, _ = batch # ignore theta if present
                
                imgs = imgs.to(device)
                
                # Predict
                # DFA forward returns: aligned_x, orig_pred_ldmks, aligned_ldmks, score, thetas, bbox
                _, orig_pred_ldmks, _, _, _, _ = aligner(imgs)
                
                # Flatten landmarks
                # orig_pred_ldmks shape: (B, 5, 2)
                orig_pred_ldmks = orig_pred_ldmks.view(orig_pred_ldmks.size(0), -1).cpu().numpy()
                
                # Calculate global indices
                start_idx = batch_idx * args.batch_size
                
                for i, ldmk in enumerate(orig_pred_ldmks):
                    global_idx = start_idx + i
                    row = [str(global_idx)] + [f'{x:.5f}' for x in ldmk]
                    f.write(','.join(row) + '\n')
                    
    print("Done!")

if __name__ == "__main__":
    main()
