#!/usr/bin/env bash
export OMP_NUM_THREADS=16

# Set visible devices (optional, adjust as needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Number of GPUs to use
NGPUS=4

# Run training using torchrun
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NGPUS \
  train.py \
  --config config/config.py

# Optional: Cleanup
pkill -f train.py || true
