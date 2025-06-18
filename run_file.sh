#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4

# === Setup ===
LOG_DIR="/localssd/chouaib/geo_ai/logs/"
mkdir -p "$LOG_DIR"
source /localssd/chouaib/anaconda3/etc/profile.d/conda.sh
conda activate geo_ai

echo "Starting test of xalma_atlas_ndp.py on one gpu"
 
/localssd/chouaib/anaconda3/envs/geo_ai/bin/python test_xalma_ndp.py > "$LOG_DIR/test_xalma_ndp_out.log" 2> "$LOG_DIR/test_xalma_ndp_err.log"


echo "Starting test of xalma_nllb_ndp.py on one gpu"

/localssd/chouaib/anaconda3/envs/geo_ai/bin/python NLLB/test_xalma_ndp.py > "$LOG_DIR/test_xalma_NLLB_ndp_out.log" 2> "$LOG_DIR/test_xalma_NLLB_ndp_err.log"

echo "✅ Both scripts completed."
