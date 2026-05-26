#!/bin/bash

# Set PYTHONPATH to the project root (one level up from the script's directory)
export PYTHONPATH="$(cd "$(dirname "$0")/.."; pwd):$PYTHONPATH"

# Default sweep values
num_samples_value=2000
num_clusters_list=(0)
wm_red_penalty_list=(0)
wm_green_fraction_list=(0.5)
model_list=('rar_xl')
perturbationset_list=('full' 'ae_small' 'ctrl_small')
methods_list=('dwtDct' 'dwtDctSvd' 'rivaGan')

device_value=1

# Parse optional arguments
for arg in "$@"; do
  case $arg in
    --num_clusters=*)
      num_clusters_list=("${arg#*=}")
      ;;
    --wm_red_penalty=*)
      wm_red_penalty_list=("${arg#*=}")
      ;;
    --wm_green_fraction=*)
      wm_green_fraction_list=("${arg#*=}")
      ;;
    --num_samples=*)
      num_samples_value="${arg#*=}"
      ;;
    --device=*)
      device_value="${arg#*=}"
      ;;
    --model=*)
      model_value="${arg#*=}"
      ;;
    --perturbationset=*)
      perturbationset_value="${arg#*=}"
      ;;
    --method=*)
      method_value="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done


# Main loop (print full command for each config)
for green_fraction in "${wm_green_fraction_list[@]}"; do
  for model in "${model_list[@]}"; do
    for num_clusters in "${num_clusters_list[@]}"; do
      for red_penalty in "${wm_red_penalty_list[@]}"; do
        for perturbationset in "${perturbationset_list[@]}"; do
          for method in "${methods_list[@]}"; do
            # we need to set a variable here dynamically depending on the method
            # if it is rivaGan, set it to 32
            # else set it to 64
            if [ "$method" == "rivaGan" ]; then
              messagelen=32
            else
              messagelen=64
            fi

            echo "python eval_ROC_old.py --dir experiments_v1/gen_clean_v1_50000samples_${model}_${method}_50000posthocsamples_messagelen${messagelen}  --perturbationset ${perturbationset}"
          done
        done
      done
    done
  done
done

# read -p "Press Enter to proceed..."

# Main loop (print full command for each config)
for green_fraction in "${wm_green_fraction_list[@]}"; do
  for model in "${model_list[@]}"; do
    for num_clusters in "${num_clusters_list[@]}"; do
      for red_penalty in "${wm_red_penalty_list[@]}"; do
        for perturbationset in "${perturbationset_list[@]}"; do
          for method in "${methods_list[@]}"; do
            # we need to set a variable here dynamically depending on the method
            # if it is rivaGan, set it to 32
            # else set it to 64
            if [ "$method" == "rivaGan" ]; then
              messagelen=32
            else
              messagelen=64
            fi

            echo "python eval_ROC_old.py --dir experiments_v1/gen_clean_v1_50000samples_${model}_${method}_50000posthocsamples_messagelen${messagelen}  --perturbationset ${perturbationset}"
            python eval_ROC_old.py --dir experiments_v1/gen_clean_v1_50000samples_${model}_${method}_50000posthocsamples_messagelen${messagelen}  --perturbationset ${perturbationset}
          done
        done
      done
    done
  done
done
