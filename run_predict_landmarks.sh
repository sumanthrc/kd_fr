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
  predict_landmarks.py \
  --lmdb_path "./datasets/train_datasets/webface4m_112x112.lmdb_dataset" \
  --save_name "./output/webface4m_112x112_landmarks.csv" \
  --batch_size 128 \
  --num_workers 4

# optional cleanup
pkill -f predict_landmarks.py.py || true