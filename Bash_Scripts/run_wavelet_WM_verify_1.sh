#!/bin/bash

# Set PYTHONPATH to the project root (one level up from the script's directory)
export PYTHONPATH="$(cd "$(dirname "$0")/.."; pwd):$PYTHONPATH"

CUDA_VISIBLE_DEVICES=1 python wavelet_WM_verify.py --dir experiments_v1/gen_clean_v1_50000samples_rar_xl_dwtDctSvd_50000posthocsamples_messagelen64  --evalfirstk 2000 --perturbationset full                                                                 &&         
CUDA_VISIBLE_DEVICES=1 python wavelet_WM_verify.py --dir experiments_v1/gen_clean_v1_50000samples_rar_xl_dwtDctSvd_50000posthocsamples_messagelen64  --evalfirstk 2000  --perturbationset ae_small                                                            &&              
CUDA_VISIBLE_DEVICES=1 python wavelet_WM_verify.py --dir experiments_v1/gen_clean_v1_50000samples_rar_xl_dwtDctSvd_50000posthocsamples_messagelen64  --evalfirstk 2000  --perturbationset negative   --imgdir experiments_v1/gen_clean_v1_50000samples_rar_xl &&
#CUDA_VISIBLE_DEVICES=1 python eval_cleanfid.py --verbose --generated_path     experiments_v1/gen_clean_v1_50000samples_rar_xl_dwtDctSvd_50000posthocsamples_messagelen64                                                                                      &&
CUDA_VISIBLE_DEVICES=1 python wavelet_WM_verify.py --dir experiments_v1/gen_clean_v1_50000samples_rar_xl_dwtDctSvd_50000posthocsamples_messagelen64  --evalfirstk 2000  --perturbationset ctrl_small                                                                          