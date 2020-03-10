# OPU Benchmarks

## Installation

We advise creating a `virtualenv` before running these commands. You can create one with `python3 -m venv <venv_name>`. 
Activate it with source `<path_to_venv>/bin/activate`  before proceeding. We used `python 3.5` and `pytorch 1.2` 
for all the simulations.

- Clone the repository and then do `pip install <path_to_repo>`.

- (optional) If you would like to replicate the results with `TensorRT` in `int8` you need to download the appropriate version from the
 [official NVIDIA website](https://developer.nvidia.com/tensorrt). We tested the code with `TensorRT 6.0.1.5` with `CUDA 10.1`.

- Download the dataset from the [Kaggle page](https://www.kaggle.com/alessiocorrado99/animals10). The dataset should be in the root folder of this repository, but all scripts have an option to change the path with `-dataset_path`. 

**NOTE**: there are problems with the `Pillow` package because this combination of Pytorch and TensorRT requires version
 `Pillow 6.1` in the `onnx` conversion of the model. If you have the same problems, uninstall `Pillow` and then retry with
 `pip install Pillow==6.1`. 

## Transfer learning on images

A general description of the method used is available in [this blog post](https://medium.com/@LightOnIO/au-revoir-backprop-bonjour-optical-transfer-learning-5f5ae18e4719) on Lighton's Medium.

Use the script `OPU_training.py` in the `scripts` folder. An example call is the following one:

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

#### Replicate the TensorRT results  

Navigate to the `script` folder and then launch the following command: 

```
python3 tensorrt_training.py densenet169 Saturn -dtype_train int8 -dtype_inf int8 -block 10 -layer 12 
-n_components 2 -encode_type plain_th -encode_thr 0 -alpha_exp_min 6 -alpha_exp_max 8 
-save_path ~/dummy/int8/ -features_path ~/datasets_conv_features/int8_features/
``` 

Substitute the `save_path` with your desired destination folder. In the above example I had pre-extracted the features 
on a GPU which supported `int8` (RTX 2080) and then moved them to the machine connected to the OPU. If the GPU on your machine already supports `int8` just drop the `-features_path` argument.

If you want to just extract the dataset features you can use the `tensorrt_extract_features.py`.

## Graph simulation

Coming soon!

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
 
## Hardware specifics

All the simulations have been run on a Tesla P100 GPU with 16GB memory and a Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz with 12 cores. 
For the int8 simulations we use an RTX 2080 with 12GB memory.
