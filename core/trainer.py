import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
import logging
import os
import time

from core.data import get_dataloader
from core.models import build_backbone, build_teacher, wrap_distributed, build_aligner
from core.losses import build_header
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter

class Trainer:
    def __init__(self, cfg, local_rank, world_size):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        self.rank = dist.get_rank()
        
        self.setup_logging()
        self.setup_data()
        self.setup_models()
        self.setup_optimizer()
        self.setup_callbacks()
        
    def setup_logging(self):
        if self.rank == 0:
            if not os.path.exists(self.cfg.output):
                os.makedirs(self.cfg.output)
            
            # Log configuration details
            self.log_configuration()
        else:
            time.sleep(2) 

    def log_configuration(self):
        logging.info("=" * 60)
        logging.info("Training Configuration Summary")
        logging.info("=" * 60)
        
        # Dataset
        logging.info(f"Dataset: {getattr(self.cfg, 'dataset', 'Unknown')}")
        #logging.info(f"Data Path: {getattr(self.cfg, 'data_path', getattr(self.cfg, 'rec', 'Unknown'))}")
        
        # Model & Teacher
        logging.info(f"Student Network: {self.cfg.network}")
        if hasattr(self.cfg, 'teacher') and self.cfg.teacher:
            logging.info(f"Teacher Network: {self.cfg.teacher}")
            logging.info(f"Pretrained Teacher Path: {getattr(self.cfg, 'pretrained_teacher_path', 'None')}")
        else:
            logging.info("Teacher Network: None (Standalone Training)")

        # Loss Function
        logging.info(f"Loss Function: {self.cfg.loss}")
        logging.info(f"Margin (m): {getattr(self.cfg, 'm', 'N/A')}")
        logging.info(f"Scale (s): {getattr(self.cfg, 's', 'N/A')}")
        
        # Data Augmentation
        augs = []
        if getattr(self.cfg, "use_adaface_aug", False):
            augs.append("AdaFace Augmentation")
        if getattr(self.cfg, "use_grid_sampler", False):
            augs.append("Grid Sampler")
        if getattr(self.cfg, "use_aroface_aug", False):
            augs.append("AroFace Augmentation")
        
        if not augs:
            augs.append("Standard/None")
            
        logging.info(f"Data Augmentations: {', '.join(augs)}")
        logging.info("=" * 60)
            
    def setup_data(self):
        self.train_loader, self.train_sampler = get_dataloader(self.cfg, self.local_rank, self.world_size)
        self.total_step = int(len(self.train_loader.dataset) / self.cfg.batch_size / self.world_size * self.cfg.num_epoch)
        if self.rank == 0:
            logging.info(f"Total Step is: {self.total_step}")

    def setup_models(self):
        # Student
        self.backbone = build_backbone(self.cfg.network, self.cfg.embedding_size, self.cfg.SE).to(self.local_rank)
        
        # Teacher (only if needed)
        self.backbone_t = None
        if hasattr(self.cfg, 'teacher') and self.cfg.teacher:
            self.backbone_t = build_teacher(self.cfg, self.local_rank)
            self.backbone_t = wrap_distributed(self.backbone_t, self.local_rank)
            self.backbone_t.eval()
        
        # Header
        self.header = build_header(self.cfg, self.local_rank)
        
        # Aligner (optional)
        self.aligner = build_aligner(self.cfg, self.local_rank)
        
        # Resume logic
        if self.cfg.global_step:
            self.load_checkpoint()
            
        # Wrap Distributed
        for ps in self.backbone.parameters():
            dist.broadcast(ps, 0)
        self.backbone = wrap_distributed(self.backbone, self.local_rank)
        self.backbone.train()
        
        self.header = wrap_distributed(self.header, self.local_rank)
        if hasattr(self.cfg, 'teacher') and self.cfg.teacher:
             self.header.eval()
        else:
             self.header.train()

    def setup_optimizer(self):
        self.opt_backbone = torch.optim.SGD(
            params=[{'params': self.backbone.parameters()}],
            lr=self.cfg.lr / 512 * self.cfg.batch_size * self.world_size,
            momentum=0.9, weight_decay=self.cfg.weight_decay)
            
        # Header optimizer (only if trainable)
        self.opt_header = None
        if self.header.training:
            self.opt_header = torch.optim.SGD(
                params=[{'params': self.header.parameters()}],
                lr=self.cfg.lr / 512 * self.cfg.batch_size * self.world_size,
                momentum=0.9, weight_decay=self.cfg.weight_decay)

        self.scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.opt_backbone, lr_lambda=self.cfg.lr_func)
            
        if self.opt_header:
            self.scheduler_header = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.opt_header, lr_lambda=self.cfg.lr_func)

        self.criterion = CrossEntropyLoss()

    def setup_callbacks(self):
        self.callback_verification = CallBackVerification(self.cfg.eval_step, self.rank, self.cfg.val_targets, self.cfg.benchmarks)
        self.callback_logging = CallBackLogging(getattr(self.cfg, 'log_interval', 50), self.rank, self.total_step, self.cfg.batch_size, self.world_size, writer=None)
        self.callback_checkpoint = CallBackModelCheckpoint(self.rank, self.cfg.output)
        self.loss_meter = AverageMeter()

    def load_checkpoint(self):
        try:
            backbone_pth = os.path.join(self.cfg.output, str(self.cfg.global_step) + "backbone.pth")
            self.backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(self.local_rank)))
            if self.rank == 0:
                logging.info("backbone resume loaded successfully!")
        except Exception as e:
            logging.info(f"load backbone resume init, failed! {e}")
            
        if not (hasattr(self.cfg, 'teacher') and self.cfg.teacher):
             try:
                header_pth = os.path.join(self.cfg.output, str(self.cfg.global_step) + "header.pth")
                self.header.load_state_dict(torch.load(header_pth, map_location=torch.device(self.local_rank)))
                if self.rank == 0:
                    logging.info("header resume loaded successfully!")
             except Exception as e:
                logging.info(f"header resume init, failed! {e}")

    def train(self):
        start_epoch = 0
        global_step = self.cfg.global_step
        
        if global_step:
            rem_steps = (self.total_step - global_step)
            cur_epoch = self.cfg.num_epoch - int(self.cfg.num_epoch / self.total_step * rem_steps)
            start_epoch = cur_epoch
            self.scheduler_backbone.last_epoch = cur_epoch
            if self.scheduler_header:
                self.scheduler_header.last_epoch = cur_epoch
                
        for epoch in range(start_epoch, self.cfg.num_epoch):
            self.train_sampler.set_epoch(epoch)
            
            for _, batch in enumerate(self.train_loader):
                global_step += 1
                
                landmark = None
                if getattr(self.cfg, 'use_grid_sampler', False):
                    if len(batch) == 4:
                        img, label, theta, landmark = batch
                    else:
                        img, label, theta = batch
                else:
                    if len(batch) == 4:
                        img, label, _, landmark = batch
                    else:
                        img, label = batch
                    
                img = img.cuda(self.local_rank, non_blocking=True)
                label = label.cuda(self.local_rank, non_blocking=True)
                
                # Aligner processing
                keypoints = None
                if landmark is not None:
                    keypoints = landmark.cuda(self.local_rank, non_blocking=True)
                elif hasattr(self, 'aligner') and self.aligner:
                    with torch.no_grad():
                        aligned_x, orig_ldmks, aligned_ldmks, score, theta, bbox = self.aligner(img)
                        #img = aligned_x.cuda(self.local_rank, non_blocking=True)
                        #keypoints = aligned_ldmks.cuda(self.local_rank, non_blocking=True)
                        keypoints = orig_ldmks.cuda(self.local_rank, non_blocking=True)
                features = self.backbone(img)
                
                if self.backbone_t:
                    with torch.no_grad():
                
                        if keypoints is not None:
                             features_t = self.backbone_t(img, keypoints=keypoints)
                        else:
                             features_t = self.backbone_t(img)
                        
                    if self.header is None:
                        # MSE Loss case (no header)
                        # features and features_t are comparable (same dim)
                        loss_v = F.mse_loss(features, features_t)
                        thetas = None # No logits
                        target_logit_mean, lma, cos_theta_tmp = 0.0, 0.0, 0.0
                    
                    elif self.cfg.loss == "AdaFace":
                        norms = torch.norm(features, p=2, dim=1, keepdim=True)
                        thetas, target_logit_mean, lma, cos_theta_tmp = self.header(
                            F.normalize(features), F.normalize(features_t), norms, label
                        )
                        loss_v = self.criterion(thetas, label)
                    else:
                        thetas, target_logit_mean, lma, cos_theta_tmp = self.header(
                            F.normalize(features), F.normalize(features_t), label
                        )
                        loss_v = self.criterion(thetas, label)
                else:
                    features = F.normalize(features)
                    thetas = self.header(features, label)
                    target_logit_mean, lma, cos_theta_tmp = 0.0, 0.0, 0.0
                    loss_v = self.criterion(thetas, label)
                loss_v.backward()
                
                clip_grad_norm_(self.backbone.parameters(), max_norm=5, norm_type=2)
                
                self.opt_backbone.step()
                self.opt_backbone.zero_grad()
                
                if self.opt_header:
                    self.opt_header.step()
                    self.opt_header.zero_grad()
                    
                self.loss_meter.update(loss_v.item(), 1)
                
                self.callback_logging(global_step, self.loss_meter, epoch, target_logit_mean, lma, cos_theta_tmp)
                self.callback_verification(global_step, self.backbone)
                
            self.scheduler_backbone.step()
            if self.scheduler_header:
                self.scheduler_header.step()
                
            self.callback_checkpoint(global_step, self.backbone, self.header)
            
        self.callback_verification(-1, self.backbone)
        dist.destroy_process_group()
