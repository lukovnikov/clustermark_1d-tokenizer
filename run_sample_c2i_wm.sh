#!/bin/bash

# Default sweep values
num_samples_value=2000
num_clusters_list=(0)  # (8 16 32 64 128)
wm_red_penalty_list=(5 2)
wm_green_fraction_list=(0.5 0.25)

# for 50k experiments with XL model
# num_samples_value=50000
# num_clusters_list=(8 16 32 128)
# wm_red_penalty_list=(1 2 5)
# wm_green_fraction_list=(0.5 0.25)

# Default single-value arguments
batch_size_value=100
# model_value="GPT-B"

model_value="rar_xl"

# num_samples_value=50000
# num_clusters_list=(8 128)
# wm_red_penalty_list=(5 2)
# wm_green_fraction_list=(0.5)

device_value=0
prefix_value=0

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
    --batsize=*)
      batch_size_value="${arg#*=}"
      ;;
    --num_samples=*)
      num_samples_value="${arg#*=}"
      ;;
    --device=*)
      device_value="${arg#*=}"
      ;;
    --prefix=*)
      prefix_value="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

# Main loop (print full command for each config)
for num_clusters in "${num_clusters_list[@]}"; do
  for red_penalty in "${wm_red_penalty_list[@]}"; do
    for green_fraction in "${wm_green_fraction_list[@]}"; do
      echo "python sample_c2i_wm.py --modelsize $model_value --num_clusters $num_clusters --wm_red_penalty $red_penalty --wm_green_fraction $green_fraction --num_samples $num_samples_value --device $device_value --wm_seed_prefix $prefix_value"
    done
  done
done

# read -p "Press Enter to proceed..."
sleep 10s

# Main loop (print full command for each config)
for num_clusters in "${num_clusters_list[@]}"; do
  for red_penalty in "${wm_red_penalty_list[@]}"; do
    for green_fraction in "${wm_green_fraction_list[@]}"; do
      echo "python sample_c2i_wm.py --modelsize $model_value --num_clusters $num_clusters --wm_red_penalty $red_penalty --wm_green_fraction $green_fraction --num_samples $num_samples_value --device $device_value --wm_seed_prefix $prefix_value"
      python sample_c2i_wm.py --modelsize $model_value --num_clusters $num_clusters --wm_red_penalty $red_penalty --wm_green_fraction $green_fraction --batsize $batch_size_value --num_samples $num_samples_value --device $device_value --wm_seed_prefix $prefix_value
    done
  done
done
