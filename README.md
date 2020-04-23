# OPU Benchmarks

## Installation

We advise creating a `virtualenv` before running these commands. You can create one with `python3 -m venv <venv_name>`. 
Activate it with source `<path_to_venv>/bin/activate`  before proceeding. We used `python 3.5` and `pytorch 1.2` 
for all the simulations.

- Clone the repository and then do `pip install <path_to_repo>`. 

## Transfer learning on images


Download the dataset from the [Kaggle page](https://www.kaggle.com/alessiocorrado99/animals10). The dataset should be in the root folder of this repository, but all scripts have an option to change the path with `-dataset_path`.

Use the script `OPU_training.py` in the `scripts/images` folder. An example call is the following one:

```
python3 OPU_training.py resnet50 Saturn -model_options noavgpool -model_dtype float32 
-n_components 2 -device cuda:0 -dataset_path ~/datasets/ -save_path data/ 
```

The script implements the following pipeline:

- Extract the convolutional features of the dataset with a ResNet50 model (without the avgpool at the end); The features 
are extracted in the dtype specified by `model_dtype` (either `float32` or `float16`) with Pytorch on the specified 
`device`: `cuda:n` selects the GPU #n, whereas `cpu` uses the CPU.
- Encode the data, project it to a space that is half the original size with the OPU (`n_components=2`) and decode 
the outcome.
- Fit a Ridge Classifier to the random features matrix of the training dataset. 

The steps are the same for the inference phase, except that the ridge fit is replaced by the evaluation on the 
matrix of random features of the test set.

There are two arguments, `dataset_path` and `save_path`, that allow to specify the path to the dataset and
save folder respectively.

For the backpropagation training, you can use the `backprop_training.py` script in the same folder.
 
```
python3 backprop_training.py $model $dataset Adam $OPU $n_epochs -dataset_path=$dataset_path -save_path=$save_path
```

Should you want to iterate on multiple models, you can use the `cnn2d.sh` script in the `bash` folder. 
Call `chmod +x cnn2d.sh` and execute with `./cnn2d.sh`. There are instructions inside to pick the models for the simulations

## Graph simulation

In the `bash` folder there is a script called `graphs.sh`. Just launch that and it will run the same simulation with 
both the OPU and GPU for graphs of different sizes.
 
## Transfer Learning on videos

#### Datasets and model

The most popular datasets in the action recognition task are:

- [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) 
- [UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/)

The videos are in `.avi` format. To use the flow/frames of the videos you can either extract them yourself with `ffmpeg`, or 
download them from [here](https://github.com/feichtenhofer/twostreamfusion). We opted for the download of the pre-extracted flow/frames for better reproducibility of the results.
 
Obtain the archives and extract them, then rename the folder `frames`/`flow` depending on the stream you picked. 

Download the three splits [for the HMDB51](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar) 
and [for the UCF101](https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip) dataset to get the annotation files.
Rename the resulting folder `annotations`.

The final structure for both datasets should be something like this:
```
dataset_folder
|----frames
|----flow
|----annotations
```
Finally, download the pretrained rgb/flow models from this [git repository](https://github.com/AlxDel/i3d_crf/tree/master/models).   
   
#### Simulations

We recommend using the bash script `cnn3d_i3d.sh` in the `bash` folder. 
 - Go to the `bash` folder and open the `cnn3d_i3d.sh` script;
 - Set the `OPU`/`backprop` flag to `true` depending on which simulation you want to launch 
 - Edit the parameters at the top to match the path to the dataset, script and save folder along with other things you might want to change;
 - Make it executable with `chmod +x cnn3d_i3d.sh` and run with `./cnn3d_i3d.sh`.
 
## Finetuning with RAY

```
python3 i3d_backprop_tune.py rgb hmdb51 -pretrained_path_rgb /home/ubuntu/opu-benchmarks/pretrained_weights/i3d_rgb_imagenet_kin.pt -dataset_
path /home/ubuntu/datasets_video/HMDB51/ -save_path /home/ubuntu/opu-benchmarks/data/
```

## Hardware specifics

All the simulations have been run on a Tesla P100 GPU with 16GB memory and a Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz with 12 cores. 
For the int8 simulations we use an RTX 2080 with 12GB memory.
