#!/usr/bin/env bash
export OMP_NUM_THREADS=16

# if you really have two cards, mask them here; otherwise drop this line
export CUDA_VISIBLE_DEVICES=0,1,2,3

# number of GPUs you actually want to use
NGPUS=4

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NGPUS \
  train_Adadistill_KPRPE.py

# optional cleanup
pkill -f train_Adadistill_KPRPE.py || true