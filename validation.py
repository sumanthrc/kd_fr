import os, sys, torch, importlib
from pathlib import Path
from config.eval_config import cfg
from backbones.iresnet import iresnet18, iresnet50, iresnet100
from backbones.mobilefacenet import MobileFaceNet
from evaluation_code.validate_ijb import ijb_eval
from evaluation_code.validate_tinyface import tinyface_eval

# --------- 1.  Build backbone -------------------------------------------------
if cfg.backbone.startswith("iresnet"):
    builder = dict(iresnet18=iresnet18, iresnet50=iresnet50,
                   iresnet100=iresnet100)[cfg.backbone]
    net = builder(num_features=512, use_se=False).to(f"cuda:{cfg.rank}")
elif cfg.backbone == "mobilefacenet":
    net = MobileFaceNet(input_size=(cfg.image_size, cfg.image_size)
                       ).to(f"cuda:{cfg.rank}")
else:
    sys.exit(f"Unsupported backbone: {cfg.backbone}")

state = torch.load(cfg.model_path, map_location=f"cuda:{cfg.rank}")
net.load_state_dict(state, strict=True)
net.eval()

# --------- 2.  IJB-B / IJB-C evaluation --------------------------------------

for target in ["IJBB", "IJBC"]:
    ijb_kwargs = dict(
        model_path=cfg.model_path,
        eval_target=target,
        eval_desc=f"{Path(cfg.model_path).stem}_{target.lower()}",
        batch_size_eval=cfg.batch_size_ijb,
        image_size=cfg.image_size,
        eval_path=cfg.ijb_root,
        use_flip_test=cfg.use_flip_ijb,
        use_detector_score=cfg.use_detector_score,
        use_norm_score=cfg.use_norm_score,
        output=os.path.join(cfg.output_dir, target.lower())
    )
    print(f"\n=== Evaluating {target} ===")
    ijb_eval(rank=cfg.rank, model=net, **ijb_kwargs)

# --------- 3.  TinyFace evaluation -------------------------------------------

tf_kwargs = dict(
    eval_path   = cfg.tinyface_root,
    fusion_method = cfg.fusion_method,
    eval_desc   = Path(cfg.model_path).stem,
    use_flip_test = cfg.use_flip_tf,
    batch_size_eval = cfg.batch_size_tf,
    output      = os.path.join(cfg.output_dir, "tinyface")
)
print("\n=== Evaluating TinyFace ===")
tinyface_eval(rank=cfg.rank, model=net, **tf_kwargs)
