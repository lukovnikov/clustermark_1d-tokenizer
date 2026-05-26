#!/bin/bash

# Set PYTHONPATH to the project root (one level up from the script's directory)
export PYTHONPATH="$(cd "$(dirname "$0")/.."; pwd):$PYTHONPATH"

CUDA_VISIBLE_DEVICES=1 python wavelet_WM_verify.py --evalfirstk 2000 --dir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-B_c2i_wm_clusters0_penalty0_greenfrac=0.5_dwtDct_50000posthocsamples_messagelen64    --perturbationset negative   --imgdir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-B_c2i_wm_clusters0_penalty0_greenfrac=0.5 --overwrite &&
CUDA_VISIBLE_DEVICES=1 python wavelet_WM_verify.py --evalfirstk 2000 --dir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-L_c2i_wm_clusters0_penalty0_greenfrac=0.5_dwtDct_50000posthocsamples_messagelen64    --perturbationset negative   --imgdir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-L_c2i_wm_clusters0_penalty0_greenfrac=0.5 --overwrite

CUDA_VISIBLE_DEVICES=2 python wavelet_WM_verify.py --evalfirstk 2000 --dir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-B_c2i_wm_clusters0_penalty0_greenfrac=0.5_dwtDctSvd_50000posthocsamples_messagelen64     --perturbationset negative   --imgdir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-B_c2i_wm_clusters0_penalty0_greenfrac=0.5 --overwrite &&
CUDA_VISIBLE_DEVICES=2 python wavelet_WM_verify.py --evalfirstk 2000 --dir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-L_c2i_wm_clusters0_penalty0_greenfrac=0.5_dwtDctSvd_50000posthocsamples_messagelen64     --perturbationset negative   --imgdir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-L_c2i_wm_clusters0_penalty0_greenfrac=0.5 --overwrite

CUDA_VISIBLE_DEVICES=3 python wavelet_WM_verify.py --evalfirstk 2000 --dir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-B_c2i_wm_clusters0_penalty0_greenfrac=0.5_rivaGan_50000posthocsamples_messagelen32     --perturbationset negative   --imgdir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-B_c2i_wm_clusters0_penalty0_greenfrac=0.5 --overwrite
CUDA_VISIBLE_DEVICES=0 python wavelet_WM_verify.py --evalfirstk 2000 --dir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-L_c2i_wm_clusters0_penalty0_greenfrac=0.5_rivaGan_50000posthocsamples_messagelen32     --perturbationset negative   --imgdir experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-L_c2i_wm_clusters0_penalty0_greenfrac=0.5 --overwrite