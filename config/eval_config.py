from easydict import EasyDict as edict

cfg = edict()

# -------------------------------------------------------------------- #
#  Basics
cfg.model_path   = "output/resnet50_aug2_12_08/295672backbone.pth"   # <- checkpoint to test
cfg.backbone     = "iresnet50"                    # iresnet18 / 50 / 100 / mobilefacenet
cfg.image_size   = 112                            # input resolution
cfg.rank         = 0                              # gpu index (0 for single-GPU)

# -------------------------------------------------------------------- #
#  IJB settings (set eval_target later in validation.py loop)
cfg.ijb_root     = "datasets/eval_datasets"                # folder containing IJBB/ & IJBC/
cfg.batch_size_ijb   = 128
cfg.use_flip_ijb     = True
cfg.use_norm_score   = True          # feature-norm weighting
cfg.use_detector_score = False       # requires detector conf scores

# -------------------------------------------------------------------- #
#  TinyFace settings
cfg.tinyface_root    = "datasets/eval_datasets"  # folder containing TinyFace dataset
cfg.batch_size_tf    = 128
cfg.use_flip_tf      = True
cfg.fusion_method    = "norm_weighted_avg"

#  Where to store results
cfg.output_dir       = "results/eval_results"
