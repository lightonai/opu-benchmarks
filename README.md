# OPU Benchmarks

This repository contains three different benchmarks to compare the performance of CPU and GPU with LightOn's Optical Processing Unit (OPU).

## Installation

We advise creating a `virtualenv` before running these commands. You can create one with `python3 -m venv <venv_name>`. 
Activate it with source `<path_to_venv>/bin/activate`  before proceeding. We used `python 3.5` and `pytorch 1.2` 
for all the simulations.

- Clone the repository and then do `pip install <path_to_repo>`. 

## Transfer learning on images

The standard transfer learning procedure with a CNN usually involves choosing a neural network architecture, pre-trained on ImageNet (e.g. a VGG, ResNet or DenseNet), and then fine-tune its weights on the dataset of choice.

The fine-tuning is typically carried out with the backpropagation algorithm on one or more GPUs, whereas the inference might be carried out on either a CPU or GPU, since the inference phase can be optimized in many ways for faster execution.

The pipeline involving the OPU is the following:

- Compute the convolutional features by processing the training set through a CNN.
- Encode the convolutional features in a binary format. We use an encoding scheme based on the sign: +1 if the sign of an element is positive, 0 otherwise.
- Project the encoded convolutional features in a lower dimensional space with the OPU.
- Fit a linear classifier to the random features of the train set. We chose the Ridge Classifier due to the fast implementation in Scikit-Learn.

In the inference phase we repeat these steps, except of course for the fitting of the linear classifier, which is instead used to get the class predictions.

The advantage of this algorithm is that it does not require the computation of gradients in the training phase, and the training itself requires just one pass over the training set. The bottleneck is represented by the fit of the classifier, but its running time can be improved by projecting to a lower dimensional space.

### Run it

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

Our data consists in a time-evolving graph, and we want to detect changes in its structure, such as an abnormal increase or decrease in the number of connections between nodes or the formation of one or more tightly connected communities *cliques*. 

We propose to use NEWMA, presented in [this paper](https://ieeexplore.ieee.org/document/9078835). It involves computing two exponentially weighted moving averages (EWMA), with different forgetting factors. These track a certain function of the adjacency matrix, and flag a possible change point whenever the difference between the two EWMAs cross a threshold.

With this method, we can detect the formation of a clique in the graph and discriminate it from a simple increase in the number of edges. Once the clique has been detected, we can diagonalize the matrix to recover the eigenvector of the second largest eigenvalue, and recover the members of the clique.

### Run it

In the `bash` folder there is a script called `graphs.sh`. Just launch that and it will run the same simulation with 
both the OPU and GPU for graphs of different sizes.
 
## Transfer Learning on videos

We propose to use the pipeline employed for images on videos. 
Training on videos can be performed in many different ways. In this document we focus on the method proposed in [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750) that allows to reach state-of-the-art results in the video action recognition task.

### Run it

#### Datasets and model
The training pipeline was developed starting from this paper:  
[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).

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

The method used with the OPU has far less hyperparameters and it is a quick, easy way to get good performance. To have an idea of how much time it can take to fine-tune the hyperparameters of gradient-based optimization, this is a script that performs a hyperparameter search using `ray[tune]`.

```
python3 i3d_backprop_tune.py rgb hmdb51 -pretrained_path_rgb /home/ubuntu/opu-benchmarks/pretrained_weights/i3d_rgb_imagenet_kin.pt -dataset_
path /home/ubuntu/datasets_video/HMDB51/ -save_path /home/ubuntu/opu-benchmarks/data/
```

## Hardware specifics

All the simulations have been run on a Tesla P100 GPU with 16GB memory and a Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz with 12 cores. 
For the int8 simulations we use an RTX 2080 with 12GB memory.
