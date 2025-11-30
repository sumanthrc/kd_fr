from easydict import EasyDict as edict

config = edict()

# -----------------------------------------------------------------------------
# General Configuration
# -----------------------------------------------------------------------------
config.output = "output/"               # train model output folder
config.embedding_size = 512             # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128                 # batch size per GPU (128*4GPU = 512)
config.lr = 0.1
config.global_step = 0                  # step to resume
config.log_interval = 50                # log every N steps

# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------
config.dataset = "emoreIresNet"         # Default dataset

# Dictionary of dataset-specific parameters
DATASETS = {
    "emoreIresNet": {
        "rec": "./datasets/train_datasets/faces_emore",
        "db_file_format": "rec",
        "num_classes": 85742,
        "num_image": 5822653,
        "num_epoch": 26,
        "warmup_epoch": -1,
        "val_targets": ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"],
        "eval_step": 5686,
        "milestones": [8, 14, 20, 25],
    },
    "Idifface": {
        "data_path": "./datasets/train_datasets/Idifface",
        "db_file_format": "folder",
        "num_classes": 10049,
        "num_image": 502450,
        "num_epoch": 60,
        "warmup_epoch": -1,
        "val_targets": ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"],
        "eval_step": 982 * 4,
        "milestones": [40, 48, 52],
        "sample": 50,
    },
    "CASIA_WebFace": {
        "rec": "./datasets/train_datasets/faces_webface_112x112",
        "db_file_format": "rec",
        "num_classes": 10575,
        "num_image": 494414,
        "num_epoch": 60,
        "warmup_epoch": -1,
        "val_targets": ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"],
        "eval_step": 3916,
        "milestones": [40, 48, 52],
    },
    "WEBFACE4M": {
        "lmdb_path": "./datasets/train_datasets/webface4m_112x112.lmdb_dataset",
        "num_classes": 205990,
        "num_image": 4235242,
        "num_epoch": 26,
        "warmup_epoch": -1,
        "val_targets": ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"],
        "eval_step": 5686,
        "milestones": [8, 14, 20, 25],
    },
}

# Apply dataset config
if config.dataset in DATASETS:
    for k, v in DATASETS[config.dataset].items():
        config[k] = v
else:
    raise ValueError(f"Unknown dataset: {config.dataset}")

# -----------------------------------------------------------------------------
# Learning Rate Scheduler
# -----------------------------------------------------------------------------
def get_lr_func(milestones, warmup_epoch=-1):
    def lr_step_func(epoch):
        if epoch < warmup_epoch:
            return ((epoch + 1) / (4 + 1)) ** 2 
        else:
            return 0.1 ** len([m for m in milestones if m - 1 <= epoch])
    return lr_step_func

config.lr_func = get_lr_func(config.milestones, config.warmup_epoch)


# -----------------------------------------------------------------------------
# Model & Loss Configuration
# -----------------------------------------------------------------------------
config.network = "mobilefacenet"        # [iresnet100 | iresnet50 | iresnet18 | mobilefacenet]
config.SE = False                       # SEModule

config.loss = "ArcFace"                 # [ArcFace | CosFace | AdaFace | MSE]
config.s = 64.0
config.m = 0.45
config.adaptive_weighted_alpha = True

# AdaFace specific
config.h = 0.333
config.t_alpha = 0.01

# -----------------------------------------------------------------------------
# Teacher Configuration (for KD)
# -----------------------------------------------------------------------------
config.teacher = "iresnet50"            # [iresnet100 | iresnet50 | mobilefacenet | adaface_res50 | Vit_b]
config.pretrained_teacher_path = "teacher/resnet50_arcfaceloss_ms1mv2_data_aug.pth"
config.vit_config_path = None           # Path to ViT config (if using ViT teacher/vit_b_kprpe)
config.aligner_config_path = None       # Path to Aligner config (if using DFA)
config.landmark_csv = None              # Path to offline landmarks CSV (if using offline landmarks)

# -----------------------------------------------------------------------------
# Augmentation Configuration
# -----------------------------------------------------------------------------
# AdaFace Augmentation
config.use_adaface_aug = True          
config.rec_root = "./datasets/train_datasets/faces_emore"
config.train_rec = "faces_emore"
config.crop_aug_p = 0.2
config.photo_aug_p = 0.2
config.lowres_aug_p = 0.2

# Grid Sampler (Default Off)
config.use_grid_sampler = False
config.grid_sampler_aug_params = { 
    'scale_min': 0.8,
    'scale_max': 1.2,
    'rot_prob': 0.2,
    'max_rot': 20,
    'hflip_prob': 0.5,
    'extra_offset': 0.1,
    'photometric_num_ops': 2,
    'photometric_magnitude': 14,
    'photometric_magnitude_offset': 9,
    'photometric_num_magnitude_bins': 31,
    'blur_magnitude': 1.0,
    'blur_prob': 0.2,
    'cutout_prob': 0.2,
}

if __name__ == "__main__":
    import pprint
    pprint.pprint(config)