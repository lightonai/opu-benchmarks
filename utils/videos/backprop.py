from time import time

import torch
from utils.videos.statistics import get_video_acc_3d

def train_model(model, train_loader, criterion, optimizer, device='cpu'):
    """
    Trains the given model for one epoch on the given dataset

    model: Pytorch model,
        neural net model.
    train_loader: torch Dataloader,
        contains the training images.
    criterion: torch.nn.modules.loss,
        criterion for the determination of the loss.
    acc_toll: float,
        tollerance on the train accuracy. If the difference between two consecutive epochs goes below this,
        stop the training.
    device: string,
        device to use for the computation. Choose between 'cpu' and 'gpu:x', where
        x is the GPU number. Defaults to 'cpu'.

    Returns
    -------
    model: torch model,
        model trained on the given dataset.
    epoch_loss: float,
        loss on the train set.
    epoch_acc: float,
        accuracy on the train set [%].
    """

    tot_train_images = len(train_loader.dataset)

    model.train()

    running_loss = 0.0
    running_corrects = 0

    for batch_id, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(torch.device(device))
        labels = labels.to(torch.device(device))

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs.data, 1)
        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data)
        torch.cuda.synchronize()

        del images, outputs

    epoch_loss = running_loss.item() / tot_train_images
    epoch_acc = running_corrects.item() / tot_train_images * 100

    return model, epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, dtype='f32', device='cpu', epoch=None, save_path=None):
    """
    Evaluates a 3D model on a video dataset. Tested with HMDB51 and UCF101 on I3D.

    model: Pytorch model,
        neural net model.
    test_loader: torch Dataloader,
        contains the test images.
    criterion: torch.nn.modules.loss,
        criterion for the determination of the loss. Defaults to CrossEntropyLoss.
    dtype: str,
        dtype to use for the evaluation. Choose among f32 and f16.
    device: string,
        device to use for the computation. Choose between 'cpu' and 'gpu:x', where
        x is the GPU number. Defaults to 'cpu'.
    epoch: int,
        training epoch number. Used to discriminate between the different class probability files.
    save_path: str,
        save path for the class probability files.

    Returns
    -------

    test_loss: float,
        loss on the test set.
    acc_per_clip: float,
        test accuracy per clip.
    acc_video_maj: float,
        test accuracy per video, computed with a majority rule over the clips of each video.
    acc_video_mean: float,
        test accuracy per video, computed by averaging the class probabilities for the frames of each video.
    inference_full: float,
        inference time, including the data loading.
    inference_conv_time: float,
        inference time for the convolutional part only.

    NOTE: Please pass the test_loader with shuffle=False, since they are used to determine
    which frame belong to which video.
    """

    tot_test_images = len(test_loader.dataset)

    inference_conv_time = 0

    if dtype == 'f16':
        model.half()

    model.to(torch.device(device)).eval()

    torch.cuda.synchronize()
    full_start = time()

    with torch.no_grad():
        clip_labels = torch.zeros(len(test_loader.dataset), dtype=torch.int64).to(device)
        logits = torch.zeros((len(test_loader.dataset), len(test_loader.dataset.classes))).to(device)

        for batch_id, (images, labels) in enumerate(test_loader):
            batch_size = images.shape[0]  # this is basically for the last batch

            images = images.to(torch.device(device))
            labels = labels.to(torch.device(device))

            if dtype == 'f16':
                images = images.half()

            torch.cuda.synchronize()
            t0 = time()

            logits[batch_id * batch_size: (batch_id + 1) * batch_size, :] = model(images)
            clip_labels[batch_id * batch_size: (batch_id + 1) * batch_size] = labels
            torch.cuda.synchronize()
            inference_conv_time += time() - t0

        test_loss = criterion(logits, clip_labels).item() / tot_test_images
        test_acc_clip, test_acc_maj, test_acc_softmax = get_video_acc_3d(logits, clip_labels, test_loader, epoch=epoch,
                                                                         clf=None, save_path=save_path)

    torch.cuda.synchronize()
    inference_full = time() - full_start

    return test_loss, test_acc_clip, test_acc_maj, test_acc_softmax, inference_full, inference_conv_time
