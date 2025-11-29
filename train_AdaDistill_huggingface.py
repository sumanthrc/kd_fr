import argparse
import logging
import os
import time
import sys
import shutil
from typing import Optional, Union, List
from pathlib import Path 

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

# --- your imports ---
from backbones.mobilefacenet import MobileFaceNet
from utils import losses
from config.config import config as cfg
from utils.dataset import MXFaceDataset, DataLoaderX, FaceDatasetFolder
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from omegaconf import OmegaConf

from backbones.iresnet import iresnet100, iresnet50, iresnet18
from backbones.vit import VisionTransformer
from backbones import net
from backbones.kprpe_models.vit_kprpe import load_model

# --- HF imports (minimal) ---
from transformers import AutoModel
from huggingface_hub import hf_hub_download

torch.backends.cudnn.benchmark = True

def load_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg

# ---------- utilities for HF/local detection ----------
def _is_hf_spec(p: Optional[str]) -> bool:
    return isinstance(p, str) and p.startswith("hf:")

def _hf_repo_id(p: str) -> str:
    # e.g. "hf:facebook/deit-base-patch16-224" -> "facebook/deit-base-patch16-224"
    return p[len("hf:"):]

# ---------- helper functions copied from HF-style loading ----------
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)

def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model

def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)


def _build_adaface_res50(local_rank: int):
    model = net.build_model('ir_50')
    checkpoint = torch.load(
        "teacher/adaface_ir50_ms1mv2.pth",
        map_location=torch.device(local_rank)
    )
    statedict = checkpoint["state_dict"]
    model_statedict = {k[6:]: v for k, v in statedict.items() if k.startswith("model.")}
    model.load_state_dict(model_statedict)
    return model


def _build_teacher(local_rank: int, rank: int):
    """
    Builds teacher by:
      1) special-case 'adaface_res50'
      2) if cfg.pretrained_teacher_path is a LOCAL path: build from TEACHER_BUILDERS + load_state_dict
      3) if cfg.pretrained_teacher_path starts with 'hf:': download to cache dir and load from local path
      4) else: build from TEACHER_BUILDERS without weights
    """
    TEACHER_BUILDERS = {
        "iresnet100": lambda: iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE),
        "iresnet50":  lambda: iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE),
        "iresnet18":  lambda: iresnet18(num_features=cfg.embedding_size, use_se=cfg.SE),
        "TransFace-B": lambda: VisionTransformer(
            img_size=112, patch_size=9, num_classes=512, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05,
            using_checkpoint=True),
        "Vit-B-KPRPE": lambda: load_model(load_config(cfg.config_path)),
        "adaface_res50": lambda: _build_adaface_res50(local_rank)
    }
    teacher_key = cfg.teacher
    pretrained_spec = getattr(cfg, "pretrained_teacher_path", None)
    hf_token = getattr(cfg, "hf_token", None) or os.environ.get("HF_TOKEN")

    # 1) Adaface special-case
    if teacher_key == "adaface_res50":
        if rank == 0:
            logging.info("Teacher: using special-case 'adaface_res50' (local checkpoint loader).")
        return TEACHER_BUILDERS[teacher_key]().to(local_rank), None

    # 2) LOCAL checkpoint -> build from registry + load state_dict
    if pretrained_spec and os.path.exists(pretrained_spec):
        if teacher_key not in TEACHER_BUILDERS:
            raise ValueError(f"Unknown cfg.teacher={teacher_key} (options: {list(TEACHER_BUILDERS)})")
        model = TEACHER_BUILDERS[teacher_key]().to(local_rank)
        try:
            state = torch.load(pretrained_spec, map_location=torch.device(local_rank))
            sd = state.get("state_dict", state)
            model.load_state_dict(sd, strict=True)
            if rank == 0:
                logging.info(f"Teacher loaded successfully from local: {pretrained_spec}")
        except Exception as e:
            logging.info(f"Teacher local load failed ({pretrained_spec}), continuing uninitialized. Error: {e}")
        return model, None

    # 3) HF id -> download to cache and load locally with trust_remote_code
    if _is_hf_spec(pretrained_spec):
        repo = _hf_repo_id(pretrained_spec)  # strip "hf:"
        cache_dir = os.path.expanduser(f"~/.cvlface_cache/{repo}")
        if rank == 0:
            logging.info(f"Teacher: downloading HF repo '{repo}' to cache: {cache_dir}")
            download(repo, cache_dir, HF_TOKEN=hf_token)
        if dist.is_initialized():
            dist.barrier()
        model = load_model_from_local_path(cache_dir, HF_TOKEN=hf_token).to(local_rank)
        return model, None

    # 4) default builder without weights
    if teacher_key not in TEACHER_BUILDERS:
        raise ValueError(f"Unknown cfg.teacher={teacher_key} (options: {list(TEACHER_BUILDERS)})")
    if rank == 0:
        logging.info("Teacher: building from registry without pretrained weights.")
    return TEACHER_BUILDERS[teacher_key]().to(local_rank), None


def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if not os.path.exists(cfg.output) and rank == 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    os.chdir("/workspace/kd_fr")
    logging.info(f"CWD={os.getcwd()}")
    if (cfg.db_file_format != "rec"):
        trainset = FaceDatasetFolder(root_dir=cfg.data_path, local_rank=local_rank, number_sample=cfg.sample)
    else:
        trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=16, pin_memory=True, drop_last=True
    )
    
    # --- Teacher (now supports HF or local) ---
    backbone_t, _ = _build_teacher(local_rank, rank)

    # --- Student (unchanged) ---
    if cfg.network == "mobilefacenet":
        backbone = MobileFaceNet(input_size=(112, 112)).to(local_rank)
    elif cfg.network == "iresnet50":
        backbone = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet18":
        backbone = iresnet18(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    else:
        backbone = None
        logging.info("create backbone failed!")
        exit()

    # --- Resume student weights if requested (unchanged) ---
    if cfg.global_step:
        try:
            backbone_pth = os.path.join(cfg.output, str(cfg.global_step) + "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank == 0:
                logging.info("backbone resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError) as e:
            logging.info(f"load backbone resume init failed: {e}")

    # broadcast student params (unchanged)
    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    # DDP wrap
    backbone_t = DistributedDataParallel(module=backbone_t, broadcast_buffers=False, device_ids=[local_rank])
    backbone_t.eval()
    backbone = DistributedDataParallel(module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    # header (unchanged)
    if cfg.loss == "ArcFace":
        header = losses.AdaptiveAArcFace(
            in_features=cfg.embedding_size, out_features=cfg.num_classes,
            s=cfg.s, m=cfg.m, adaptive_weighted_alpha=cfg.adaptive_alpha
        ).to(local_rank)
    elif cfg.loss == "CosFace":
        header = losses.AdaptiveACosFace(
            in_features=cfg.embedding_size, out_features=cfg.num_classes,
            s=cfg.s, m=cfg.m, adaptive_weighted_alpha=cfg.adaptive_alpha
        ).to(local_rank)
    elif cfg.loss == "AdaFace":
        header = losses.AdaptiveAAdaFace(
            in_features=cfg.embedding_size, out_features=cfg.num_classes,
            s=cfg.s, m=cfg.m, h=cfg.h, t_alpha=cfg.adaface_t_alpha
        ).to(local_rank)
    else:
        raise ValueError("Header not implemented")

    header = DistributedDataParallel(module=header, broadcast_buffers=False, device_ids=[local_rank])
    header.eval()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay
    )
    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    criterion = CrossEntropyLoss()

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0:
        logging.info("Total Step is: %d" % total_step)

    if cfg.global_step:
        rem_steps = (total_step - cfg.global_step)
        cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
        logging.info("resume from estimated epoch {}".format(cur_epoch))
        logging.info("remaining steps {}".format(rem_steps))
        start_epoch = cur_epoch
        scheduler_backbone.last_epoch = cur_epoch
        opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_last_lr()[0]

    #callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.benchmarks)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = cfg.global_step

    CANON_5PTS = torch.tensor([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
    ], dtype=torch.float32)

    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for _, (img, label) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)
            B,_,H,W = img.shape
            kp = CANON_5PTS.to(img.device).unsqueeze(0).repeat(B,1,1)
            features = backbone(img)
            with torch.no_grad():
                features_t = backbone_t(img, keypoints=kp)
            if cfg.loss == "AdaFace":
                norms = features.norm(p=2, dim=1, keepdim=True)
                thetas, target_logit_mean, lma, cos_theta_tmp = header(features, features_t, label, norms=norms)
            else:
                thetas, target_logit_mean, lma, cos_theta_tmp = header(
                    F.normalize(features), F.normalize(features_t), label
                )
            loss_v = criterion(thetas, label)
            loss_v.backward()
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
            opt_backbone.step()
            opt_backbone.zero_grad()
            loss.update(loss_v.item(), 1)

            callback_logging(global_step, loss, epoch, target_logit_mean, lma, cos_theta_tmp)
            #callback_verification(global_step, backbone)

        scheduler_backbone.step()
        callback_checkpoint(global_step, backbone, header)

    #callback_verification(-1, backbone)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch AdaDistill training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    args_ = parser.parse_args()
    main(args_)