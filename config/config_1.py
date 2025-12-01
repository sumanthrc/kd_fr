from easydict import EasyDict as edict

config = edict()
config.dataset = "REPEATED_WEBFACE4M" # training dataset
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128 # batch size per GPU (128*4GPU = 512)
config.lr = 0.1

# Saving path
config.output = "output/" # train model output folder
config.benchmarks = "datasets/test_datasets"

# teacher path
config.pretrained_teacher_path = "teacher/vit_b_webface4m.pt" # teacher folder
#config.pretrained_teacher_header_path = "teacher/resnet50_ms1mv2_aug_2_12_08/295672header.pth" # teacher folder

config.global_step=0# step to resume

# Margin-penalty loss configurations
config.s=64.0
config.m=0.45

# AdaFace specific parameters
#config.h=0.333
#config.adaface_t_alpha=0.01

                        # ----------------Vit_B_KPRPE backbone configuration ------------------

#config.config_path = 'backbones/kprpe_models/vit_kprpe/configs/v1_base_kprpe_splithead_unshared.yaml'

                        #----------------Vit_B backbone configuration ------------------

config.vit_config_path = 'backbones/kprpe_models/vit/configs/v1_base.yaml'

#AdaDistill configuration
config.adaptive_alpha=True

config.loss="ArcFace"  #  Option : ArcFace, CosFace, MLLoss

# type of network to train [iresnet100 | iresnet50 | iresnet18 | mobilefacenet | Transface_B]
config.network = "mobilefacenet" # iresnet100, iresnet50, iresnet18, mobilefacenet
config.teacher = "Vit_b" # iresnet100, iresnet50, mobilefacenet, adaface_res50

config.SE=False # SEModule

#-------------------- Grid sampler augmentation parameters ------------------
config.use_grid_sampler = True
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
              'cutout_prob': 0.2,}

if config.dataset == "emoreIresNet":
    config.rec = "./datasets/train_datasets/faces_emore"
    config.db_file_format="rec"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch =  26
    config.warmup_epoch = -1
    config.val_targets =  ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]
    config.eval_step=5686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

if config.dataset == "Idifface":
    #config.rec = "./train_datasets/faces_emore"
    config.data_path="./datasets/train_datasets/Idifface"
    config.db_file_format="folder"

    config.num_classes = 10049
    config.num_image = 502450
    config.num_epoch = 60
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step= 982 * 4
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [40, 48, 52] if m - 1 <= epoch])
    config.lr_func = lr_step_func
    config.sample = 50

if config.dataset == "CASIA_WebFace":
    config.rec = "./datasets/train_datasets/faces_webface_112x112"
    config.db_file_format="rec"
    config.num_classes = 10575
    config.num_image = 494414
    config.num_epoch = 60
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]
    config.eval_step= 3916
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [40, 48, 52] if m - 1 <= epoch])
    config.lr_func = lr_step_func

if config.dataset == "WEBFACE4M":
    config.lmdb_path = "./datasets/train_datasets/webface4m_112x112.lmdb_dataset"
    config.num_classes = 205990
    config.num_image = 4235242
    config.num_epoch =  26
    config.warmup_epoch = -1
    config.val_targets =  ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step=5686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

if config.dataset == "REPEATED_WEBFACE4M":
    config.lmdb_path = "./datasets/train_datasets/webface4m_112x112.lmdb_dataset"
    config.landmark_path = "./datasets/train_datasets/webface4m_112x112_landmarks.csv"
    config.num_classes = 205990
    config.num_image = 4235242
    config.num_epoch =  26
    config.warmup_epoch = -1
    config.val_targets =  ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step=5686
    
    # Repeated dataset specific
    config.repeated_augment_prob = 0.1
    config.use_same_image = False
    config.second_img_augment = False
    config.disable_repeat = True
    config.skip_aug_prob_in_disable_repeat = 0.0
    
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func