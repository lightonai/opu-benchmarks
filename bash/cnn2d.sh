#!/bin/bash

# Script for compression to iterate through multiple models.

# ============= Common parameters =============

#Set the respective flag to True to run the OPU/backprop simulation
OPU_simulation=true
backprop_simulation=true

main_path=../scripts/images/OPU_training.py
main_bp_path=../scripts/images/backprop_training.py

save_path=../data/
dataset_path=~/datasets/

batch_size=32
OPU="Saturn"

# To iterate on multiple models uncomment the line below
#declare -a models=("resnet18" "resnet34" "resnet50" "resnet101" "resnet152")

declare -a models=("densenet169")
#we advide using norelu for VGG models, noavgpool for ResNets and full for DenseNets
model_options="full"

# ============= OPU parameters =============

n_components=2
dtype="float32"

encode_type='positive'

alpha_exp_min=6
alpha_exp_max=8
alpha_space=5

# ============= Backprop parameters =============

n_epochs=5

for model in "${models[@]}"; do
  if [[ "$OPU_simulation" == true ]]
  then
    python3 $main_path $model $dataset $OPU -model_options=$model_options -model_dtype=$dtype \
            -n_components=$n_components -encode_type=$encode_type\
            -alpha_exp_min=$alpha_exp_min -alpha_exp_max=$alpha_exp_max -alpha_space=$alpha_space \
            -dataset_path=$dataset_path -save_path=$save_path
    echo -e '\n'
  fi

  if [[ "$backprop_simulation" == true ]]
  then
    python3 $main_bp_path $model $dataset Adam $OPU $n_epochs -dataset_path=$dataset_path -save_path=$save_path
    echo -e '\n'
  fi
done
