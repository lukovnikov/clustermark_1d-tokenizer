#!/bin/bash

# Set PYTHONPATH to the project root (one level up from the script's directory)
export PYTHONPATH="$(cd "$(dirname "$0")/.."; pwd):$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python wavelet_WM_apply.py --indir experiments_v1/gen_clean_v1_50000samples_rar_xl --method dwtDct --message_bits_len 64
CUDA_VISIBLE_DEVICES=0 python wavelet_WM_apply.py --indir experiments_v1/gen_clean_v1_50000samples_rar_xl --method dwtDctSvd --message_bits_len 64
CUDA_VISIBLE_DEVICES=3 python wavelet_WM_apply.py --indir experiments_v1/gen_clean_v1_50000samples_rar_xl --method rivaGan --message_bits_len 32

