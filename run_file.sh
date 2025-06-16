#!/bin/bash

# === Setup ===
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# === First Script: cpo_atlas_ndp.py ===
echo "Starting cpo_atlas_ndp.py on GPUs 1,2,3,4..."
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --main_process_port=29515 --num_processes=3 cpo_atlas_ndp.py \
    > "$LOG_DIR/cpo_ndp_out.log" 2> "$LOG_DIR/cpo_ndp_err.log"

# === Second Script: cpo_atlas_rbf.py ===
echo "Starting cpo_atlas_rbf.py on GPUs 1,2,3,4..."
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --main_process_port=29516 --num_processes=3 cpo_atlas_rbf.py \
    > "$LOG_DIR/cpo_rbf_out.log" 2> "$LOG_DIR/cpo_rbf_err.log"

echo "✅ Both scripts completed."
