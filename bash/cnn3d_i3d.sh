#!/bin/bash

# Script for training the 3D CNN I3D on video frames and flow with OPU.

OPU_simulation=true
backprop_simulation=false

OPU="Saturn"
n_components=5

batch_size=2
num_workers=12
crop_size=224 #height and width of the frames

RP_device="opu"
device="cuda:0"

opu_script=../scripts/videos/i3d_opu.py
bp_script=../scripts/videos/i3d_backprop.py

combine_script=../scripts/videos/combine_streams.py

rgb_path=../pretrained_weights/i3d_rgb_imagenet_kin.pt
flow_path=../pretrained_weights/i3d_flow_imagenet_kin.pt
save_path=../data/"$OPU"_"$n_components"/

dataset="hmdb51"
dataset_path=/data/home/luca/datasets_video/HMDB51/

fold=1 #the dataset split: 1, 2, 3

#Note: I use the same number of frames for the train and test set
frames_min=64
frames_max=64
frames_step=1

step_train=100
step_test=100

dtype="float32"
encode_type='positive'
decode_type="mixing"

alpha_exp_min=5
alpha_exp_max=7
alpha_space=5

#=========== backprop parameters ============

n_epochs=100

for ((frames = $frames_min; frames <= $frames_max; frames += $frames_step)); do

  if [[ "$OPU_simulation" == true ]]; then

    python3 $opu_script "rgb" $dataset  \
      -frames_train=$frames -frames_test=$frames -step_train=$step_train -step_test=$step_test \
      -batch_size=$batch_size -num_workers=$num_workers -crop_size=$crop_size -fold=$fold -device=$device \
      -RP_device=$RP_device -model_dtype=$dtype -encode_type=$encode_type -decode_type=$decode_type -n_components=$n_components \
      -alpha_exp_min=$alpha_exp_min -alpha_exp_max=$alpha_exp_max -alpha_space=$alpha_space \
      -pretrained_path_rgb=$rgb_path -dataset_path=$dataset_path -save_path=$save_path
    echo -e '\n'

    python3 $opu_script "flow" $dataset $OPU \
      -frames_train=$frames -frames_test=$frames -step_train=$step_train -step_test=$step_test \
      -batch_size=$batch_size -num_workers=$num_workers -crop_size=$crop_size -fold=$fold -device=$device \
      -RP_device=$RP_device -model_dtype=$dtype -encode_type=$encode_type -decode_type=$decode_type -n_components=$n_components \
      -alpha_exp_min=$alpha_exp_min -alpha_exp_max=$alpha_exp_max -alpha_space=$alpha_space \
      -pretrained_path_flow=$flow_path -dataset_path=$dataset_path -save_path=$save_path
    echo -e '\n'

    python3 $combine_script $save_path"$dataset"_"$frames"_"$step_train"/class_proba/ \
      -save_path=$save_path"$dataset"_"$frames"_"$step_train"/

  fi

  if [[ "$backprop_simulation" == true ]]; then
    python3 $bp_script $n_epochs "rgb" $dataset \
      -frames_train=$frames -frames_test=$frames -step_train=$step_train -step_test=$step_test \
      -batch_size=$batch_size -num_workers=$num_workers -crop_size=$crop_size -fold=$fold -device=$device \
      -pretrained_path_rgb=$rgb_path -dataset_path=$dataset_path -save_path=$save_path

    python3 $bp_script $n_epochs "flow" $dataset \
      -frames_train=$frames -frames_test=$frames -step_train=$step_train -step_test=$step_test \
      -batch_size=$batch_size -num_workers=$num_workers -crop_size=$crop_size -fold=$fold -device=$device \
      -pretrained_path_flow=$flow_path -dataset_path=$dataset_path -save_path=$save_path

  fi
done
