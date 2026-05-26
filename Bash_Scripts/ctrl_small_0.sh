#!/bin/bash

# Set PYTHONPATH to the project root (one level up from the script's directory)
export PYTHONPATH="$(cd "$(dirname "$0")/.."; pwd):$PYTHONPATH"

python eval_wm.py --dir experiments_v1/gen_wm_v1_50000samples_rar_xl_0clusters_greenfrac0.5_penalty2     --evalfirstk 2000 --perturbationset ctrl_small       --numworkers 0  --batsize 32   --device 0  &&              
python eval_wm.py --dir experiments_v1/gen_wm_v1_50000samples_rar_xl_0clusters_greenfrac0.5_penalty5     --evalfirstk 2000 --perturbationset ctrl_small       --numworkers 0  --batsize 32   --device 0  &&           
python eval_wm.py --dir experiments_v1/gen_wm_v1_50000samples_rar_xl_0clusters_greenfrac0.25_penalty2    --evalfirstk 2000 --perturbationset ctrl_small       --numworkers 0  --batsize 32   --device 0  &&            
python eval_wm.py --dir experiments_v1/gen_wm_v1_50000samples_rar_xl_0clusters_greenfrac0.25_penalty5    --evalfirstk 2000 --perturbationset ctrl_small       --numworkers 0  --batsize 32   --device 0  &&            
python eval_wm.py --dir experiments_v1/gen_wm_v1_50000samples_rar_xl_8clusters_greenfrac0.5_penalty2     --evalfirstk 2000 --perturbationset ctrl_small       --numworkers 0  --batsize 32   --device 0  &&           
python eval_wm.py --dir experiments_v1/gen_wm_v1_50000samples_rar_xl_8clusters_greenfrac0.5_penalty5     --evalfirstk 2000 --perturbationset ctrl_small       --numworkers 0  --batsize 32   --device 0         