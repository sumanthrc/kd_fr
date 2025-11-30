import torch
from torchvision import transforms
from utils.dataset import MXFaceDataset, DataLoaderX, FaceDatasetFolder
from utils.lmdb_dataset import LmdbDataset
from utils.adaface_data_aug.record_dataset import AugmentRecordDataset

def get_dataloader(cfg, local_rank, world_size):
    """
    Factory function to create a distributed data loader based on configuration.
    """
    
    # Define transforms
    if getattr(cfg, "use_adaface_aug", False):
        # AdaFace augmentation pipeline
        post_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
        
        trainset = AugmentRecordDataset(
            root_dir=cfg.rec_root,
            transform=post_transform,
            crop_augmentation_prob=cfg.crop_aug_p,
            photometric_augmentation_prob=cfg.photo_aug_p,
            low_res_augmentation_prob=cfg.lowres_aug_p,
            swap_color_channel=False,
        )
        
    elif cfg.dataset == "WEBFACE4M":
        # WebFace4M LMDB dataset
        tfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
        # Handle grid sampler if enabled
        use_grid_sampler = getattr(cfg, "use_grid_sampler", False)
        aug_params = getattr(cfg, "grid_sampler_aug_params", None) if use_grid_sampler else None
        
        trainset = LmdbDataset(
            lmdb_file=cfg.lmdb_path, 
            transforms=tfm, 
            use_grid_sampler=use_grid_sampler, 
            aug_params=aug_params,
            landmark_csv=getattr(cfg, "landmark_csv", None)
        )
        
    elif getattr(cfg, "db_file_format", "") == "rec" or cfg.dataset == "emoreIresNet":
        # MXNet RecordIO dataset
        use_grid_sampler = getattr(cfg, "use_grid_sampler", False)
        aug_params = getattr(cfg, "grid_sampler_aug_params", None) if use_grid_sampler else None
        
        trainset = MXFaceDataset(
            root_dir=cfg.rec, 
            local_rank=local_rank, 
            use_grid_sampler=use_grid_sampler, 
            aug_params=aug_params,
            landmark_csv=getattr(cfg, "landmark_csv", None)
        )
        
    else:
        # Image Folder dataset
        trainset = FaceDatasetFolder(
            root_dir=cfg.data_path, 
            local_rank=local_rank, 
            number_sample=cfg.sample
        )

    # Distributed Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True
    )

    # DataLoader
    train_loader = DataLoaderX(
        local_rank=local_rank, 
        dataset=trainset, 
        batch_size=cfg.batch_size,
        sampler=train_sampler, 
        num_workers=16, 
        pin_memory=True, 
        drop_last=True
    )
    
    return train_loader, train_sampler
