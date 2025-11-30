import torch
import torch.nn as nn
import os
import logging
from torch.nn.parallel.distributed import DistributedDataParallel

from backbones.mobilefacenet import MobileFaceNet
from backbones.iresnet import iresnet100, iresnet50, iresnet18
from backbones import net as adaface_net
from backbones.kprpe_models.vit import load_model as load_vit_model
from backbones.kprpe_models.vit_kprpe import load_model as load_vit_kprpe_model
from aligners import get_aligner
from omegaconf import OmegaConf

def load_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg

def build_aligner(cfg, local_rank):
    if hasattr(cfg, 'aligner_config_path') and cfg.aligner_config_path:
        aligner_cfg = load_config(cfg.aligner_config_path)
        aligner = get_aligner(aligner_cfg)
        aligner = aligner.to(local_rank).eval()
        return aligner
    return None

def build_backbone(name, embedding_size, use_se=False, input_size=(112, 112)):
    """
    Factory function to create a backbone model.
    """
    if name == "mobilefacenet":
        return MobileFaceNet(input_size=input_size)
    elif name == "iresnet50":
        return iresnet50(num_features=embedding_size, use_se=use_se)
    elif name == "iresnet100":
        return iresnet100(num_features=embedding_size, use_se=use_se)
    elif name == "iresnet18":
        return iresnet18(num_features=embedding_size, use_se=use_se)
    else:
        raise ValueError(f"Unknown backbone: {name}")

def build_teacher(cfg, local_rank):
    """
    Factory function to create and load a teacher model.
    """
    if cfg.teacher == "iresnet100":
        backbone_t = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE)
    elif cfg.teacher == "iresnet50":
        backbone_t = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE)
    elif cfg.teacher == "iresnet18":
        backbone_t = iresnet18(num_features=cfg.embedding_size, use_se=cfg.SE)
    elif cfg.teacher == "adaface_res50":
        # Special case for AdaFace teacher which uses a different build process
        backbone_t = adaface_net.build_model('ir_50')
    elif cfg.teacher == "Vit_b":
        vit_config = load_config(cfg.vit_config_path)
        backbone_t = load_vit_model(vit_config)
    elif cfg.teacher == "Vit_b_kprpe":

        backbone_t = load_vit_kprpe_model(load_config(cfg.vit_config_path))
        
    else:
        raise ValueError(f"Unknown teacher: {cfg.teacher}")

    backbone_t = backbone_t.to(local_rank)

    # Load weights
    if cfg.teacher == "adaface_res50":
        if os.path.exists(cfg.pretrained_teacher_path):
            checkpoint = torch.load(cfg.pretrained_teacher_path, map_location=torch.device(local_rank))
            if 'state_dict' in checkpoint:
                state_dict = {key.replace('model.', ''): val for key, val in checkpoint['state_dict'].items() if 'model.' in key}
                backbone_t.load_state_dict(state_dict)
            else:
                 backbone_t.load_state_dict(checkpoint)
    else:
        if hasattr(cfg, 'pretrained_teacher_path') and cfg.pretrained_teacher_path and os.path.exists(cfg.pretrained_teacher_path):
            try:
                checkpoint = torch.load(cfg.pretrained_teacher_path, map_location=torch.device(local_rank))
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                     state_dict = {key.replace('module.', ''): val for key, val in checkpoint['state_dict'].items()}
                     backbone_t.load_state_dict(state_dict, strict=False)
                else:
                    backbone_t.load_state_dict(checkpoint, strict=False)
                logging.info(f"Teacher loaded from {cfg.pretrained_teacher_path}")
            except Exception as e:
                logging.warning(f"Failed to load teacher from {cfg.pretrained_teacher_path}: {e}")
                raise e
        else:
            logging.warning("No pretrained teacher path provided or file not found.")

    return backbone_t

def wrap_distributed(model, local_rank, broadcast_buffers=False):
    return DistributedDataParallel(
        module=model, broadcast_buffers=broadcast_buffers, device_ids=[local_rank]
    )
