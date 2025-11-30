import torch
from utils import losses

def build_header(cfg, local_rank):
    """
    Factory function to create the loss header.
    """
    if cfg.loss == "ArcFace":
        if hasattr(cfg, 'adaptive_alpha'):
             header = losses.AdaptiveAArcFace(
                 in_features=cfg.embedding_size, 
                 out_features=cfg.num_classes, 
                 s=cfg.s, 
                 m=cfg.m,  
                 adaptive_weighted_alpha=cfg.adaptive_alpha
             )
        else:
             header = losses.ArcFace(
                 in_features=cfg.embedding_size, 
                 out_features=cfg.num_classes, 
                 s=cfg.s, 
                 m=cfg.m
             )
             
    elif cfg.loss == "CosFace":
        if hasattr(cfg, 'adaptive_alpha'):
            header = losses.AdaptiveACosFace(
                in_features=cfg.embedding_size, 
                out_features=cfg.num_classes, 
                s=cfg.s, 
                m=cfg.m, 
                adaptive_weighted_alpha=cfg.adaptive_alpha
            )
        else:
            header = losses.CosFace(
                in_features=cfg.embedding_size, 
                out_features=cfg.num_classes, 
                s=cfg.s, 
                m=cfg.m
            )
            
    elif cfg.loss == "AdaFace":
        if hasattr(cfg, 'adaptive_alpha'):
             header = losses.AdaptiveAdaFace(
                 in_features=cfg.embedding_size, 
                 out_features=cfg.num_classes, 
                 s=cfg.s, 
                 m=cfg.m, 
                 h=cfg.h, 
                 t_alpha=cfg.t_alpha, 
                 adaptive_weighted_alpha=cfg.adaptive_alpha
             )
        else:
             header = losses.AdaFace(
                 in_features=cfg.embedding_size, 
                 out_features=cfg.num_classes, 
                 s=cfg.s, 
                 m=cfg.m,
                 h=cfg.h, 
                 t_alpha=cfg.t_alpha
             )
             
    elif cfg.loss == "MSE":
        # MSE loss doesn't use a header, but we return a dummy identity or None
        return None
        
    else:
        raise ValueError(f"Unknown loss header: {cfg.loss}")

    header = header.to(local_rank)
    return header