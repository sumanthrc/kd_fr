#!/usr/bin/env bash
export OMP_NUM_THREADS=4

# if you really have two cards, mask them here; otherwise drop this line
export CUDA_VISIBLE_DEVICES=0

# number of GPUs you actually want to use
NGPUS=1

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NGPUS \
  train_standalone.py

# optional cleanup
pkill -f train_standalone.py || true