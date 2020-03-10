import os
import pathlib
import json

import torch
import numpy as np

def get_video_acc_3d(data, labels, loader, epoch=None, clf=None, save_path=None):
    """
    Computes the accuracy of each video clip by majority rule on the label assigned to each frame and by averaging the
    class probability scores.
    Expects the test data to be ordered like this:

    video1_frame1
    video1_frame2
    ...
    video2_frame1
    video2_frame2

    Do NOT shuffle the loader.

    Parameters
    ----------
    clf: scikitlearn classifier:
        classifier for the accuracy evaluation
    loader: Pytorch dataloader,
        dataloader used to generate the features.
    data: numpy array or torch tensor,
        array of features FOR EACH CLIP in the shape (n_frames, n_features).
    labels: numpy array,
        labels associated to EACH CLIP.


    Returns
    -------
    acc_per_clip: float,
        test accuracy per clip.
    acc_video_maj: float,
        test accuracy per video, computed with a majority rule over the clips of each video.
    acc_video_mean: float,
        test accuracy per video, computed by averaging the class probabilities for the frames of each video.
    save_path: str or None,
        savepath for the class probabilities per video. If None, they will not be saved
    """

    n_videos = loader.dataset.n_videos
    video_preds = {k: [] for k in range(n_videos)}

    if clf is not None:
        acc_per_clip = clf.score(data, labels) * 100
        preds_labels = clf.predict(data)
        probabilities = clf._predict_proba_lr(data)
        filename = "proba_{}_{}_{}.json".format(loader.dataset.frames_per_clip, loader.dataset.mode, int(clf.alpha))

    else:
        _, preds_labels = torch.max(data, 1)
        acc_per_clip = torch.sum(labels == preds_labels).item() / len(loader.dataset) * 100
        preds_labels = preds_labels.cpu().numpy()
        probabilities = data.cpu().numpy()
        filename = "proba_{}_{}_{}.json".format(loader.dataset.frames_per_clip, loader.dataset.mode, epoch)

    for i, pred in enumerate(preds_labels):
        clip_idx = loader.dataset.clips_idx[i]
        video_preds[clip_idx].append(pred)

    video_preds = {k: np.argmax(np.bincount(v)) for k, v in video_preds.items()}
    acc_maj = sum([video_preds[i] == loader.dataset.video_targets[i] for i in range(n_videos)]) / n_videos * 100

    video_preds = {k: [] for k in range(n_videos)}

    for i, pred in enumerate(probabilities):
        clip_idx = loader.dataset.clips_idx[i]
        video_preds[clip_idx].append(pred)

    # Use .tolist() to serialize dict as .json
    video_preds = {k: np.mean(v, 0).tolist() for k, v in video_preds.items()}

    if save_path is not None:
        save_path = os.path.join(save_path, "class_proba")
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_path, filename), "w") as file:
            json.dump(video_preds, file)

        filename = "ground_truth.json"
        with open(os.path.join(save_path, filename), "w") as file:
            json.dump(loader.dataset.video_targets, file)

    video_preds = {k: np.argmax(v) for k, v in video_preds.items()}
    acc_softmax = sum([video_preds[i] == loader.dataset.video_targets[i] for i in range(n_videos)]) / n_videos * 100

    return acc_per_clip, acc_maj, acc_softmax


def get_model_size(model):
    """
    Returns the model size and the number of parameters of the convolutional, batchnorm and linear layers.

    Parameters
    ----------

    model: torch model,
        pytorch model.

    Returns
    -------
    model_size = float,
        size of the model in MB.
    total_weights = int,
        number of weights in the model.

    NOTE: the number of bits is obtained from the last two characters of the dtype (ex: 'float32' -> 32 bits).
        As such, it will not work for bits lower than 10 (ex: 'int8' -> t8)
    """

    tot_conv_weights, tot_batchnorm_weights, tot_linear_weights = 0, 0, 0
    tot_conv_size, tot_batchnorm_size, tot_linear_size = 0., 0., 0.

    for index, layer in model.named_modules():

        if isinstance(layer, torch.nn.Conv2d):
            layer_weights = np.prod(layer.weight.data.shape[:])
            tot_conv_weights += layer_weights
            tot_conv_size += layer_weights * int(str(layer.weight.data.dtype)[-2:]) / (8 * 2 ** 10 * 2 ** 10)

        elif isinstance(layer, torch.nn.BatchNorm2d):
            layer_weights = np.prod(layer.weight.data.shape[:])
            tot_batchnorm_weights += layer_weights
            tot_batchnorm_size += layer_weights * int(str(layer.weight.data.dtype)[-2:]) / (8 * 2 ** 10 * 2 ** 10)

        elif isinstance(layer, torch.nn.Linear):
            layer_weights = np.prod(layer.weight.data.shape[:])
            tot_linear_weights += layer_weights
            tot_linear_size += layer_weights * int(str(layer.weight.data.dtype)[-2:]) / (8 * 2 ** 10 * 2 ** 10)

    total_weights = tot_conv_weights + tot_batchnorm_weights + tot_linear_weights
    model_size = tot_conv_size + tot_batchnorm_size + tot_linear_size

    return model_size, total_weights, tot_linear_size

def get_output_size(model, input_shape=(1, 3, 224, 224), device="cpu", dtype='float32'):
    """
    Returns the shape of the convolutional features in output to the model.

    Parameters
    ----------
    model pytorch model,
        neural network model.
    input_shape: tuple of int,
        shape of the images in input to the model in the form (batch_size, channels, height, width).
        Defaults to (1, 3, 224, 224).

    Return
    ------
    output_size : int,
        shape of the flattened convolutional features in output to the model.

    Note: It is not possible to do model(x) on CPU with f16. To avoid problems, the model is cast to float32 for this
    computation and then it is converted back to float16.
    """

    if dtype == "float16":
        model.float()

    dummy_input = torch.ones(input_shape).to(device)
    output_size = model(dummy_input).shape[1:].numel()

    if dtype == "float16":
        model.half()

    return output_size
