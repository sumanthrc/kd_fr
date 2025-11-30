import argparse
import os
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from config.config import config as default_cfg
from core.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch KD_FR Training')
    parser.add_argument('--config', type=str, default=None, help='path to config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    # Add other overrides as needed
    return parser.parse_args()

def merge_config(args):
    # Start with default config
    cfg = default_cfg
    
    # Override with yaml config if provided
    if args.config:
        yaml_cfg = OmegaConf.load(args.config)

        for k, v in yaml_cfg.items():
            setattr(cfg, k, v)
            
    # Override with args
    if args.local_rank:
        cfg.local_rank = args.local_rank
    if args.resume:
        cfg.resume = args.resume
        
    return cfg

def main():
    args = parse_args()
    
    # Init distributed
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Load config
    cfg = merge_config(args)
    
    # Initialize Trainer
    trainer = Trainer(cfg, local_rank, world_size)
    
    # Start Training
    trainer.train()

if __name__ == "__main__":
    main()
