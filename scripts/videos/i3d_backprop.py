import os
import pathlib
import json
import argparse
from argparse import RawTextHelpFormatter
from time import time

import torch
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils.videos.models.i3d as i3d
from utils.videos.datasets.HMDB51 import HMDB51Frames3D, HMDB51Flow3D
from utils.videos.datasets.UCF101 import UCF101Frames3D, UCF101Flow3D
from utils.videos.backprop import train_model, evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="TL with the OPU on videos - I3D.",
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument("n_epochs", help="Number of epochs.", type=int)
    parser.add_argument("mode", help="Mode for the 3D CNN. Choose between rgb and flow", type=str)
    parser.add_argument("dataset_name", help='Base model for TL.', type=str, choices=["hmdb51", "ucf101"])

    parser.add_argument("-frames_train", help="Frames per clip for the train set. Default=3", type=int, default=3)
    parser.add_argument("-frames_test", help="Frames per clip for the test set. Default=3", type=int, default=3)
    parser.add_argument("-step_train", help="Distance between clips for the train set. Default=100", type=int,
                        default=100)
    parser.add_argument("-step_test", help="Distance between clips for the test set. Default=100", type=int,
                        default=100)

    parser.add_argument("-batch_size", help='Batch size for the training and testing phases. Default=32.',
                        type=int, default=32)
    parser.add_argument("-num_workers", help='Number of workers. Default=12.', type=int, default=12)
    parser.add_argument("-lr", help='learning rate. Default=1e-3.', type=float, default=0.001)

    parser.add_argument("-crop_size", help='Size of the center crop area.', type=int, default=224)
    parser.add_argument("-fold", help='Dataset split. Default=1.', type=int, default=1, choices=[1, 2, 3])

    parser.add_argument("-device",
                        help="Device for the GPU computation, specified as 'cuda:x', where x is the GPU number."
                             "Choose 'cpu' to use the CPU for all computations. Default='cuda:0'", type=str,
                        default='cuda:0')

    parser.add_argument('-model_dtype', help="dtype for the network weights. Defaults to 'float32'.",
                        choices=['float32', 'float16'], type=str, default="float32")

    parser.add_argument("-pretrained_path_rgb", help='Path to the pretrained weights for rgb model.', type=str,
                        default="/data/home/luca/mlvideos/models_3d_pretrained/pretrained_weights/i3d_rgb_imagenet_kin.pt")
    parser.add_argument("-pretrained_path_flow", help='Path to the pretrained weights for flow model.', type=str,
                        default="/data/home/luca/mlvideos/models_3d_pretrained/pretrained_weights/i3d_flow_imagenet_kin.pt")

    parser.add_argument("-dataset_path",
                        help='Path to the dataset folder (excluded). Defaults to /data/home/luca/datasets_video/HMDB51/.',
                        type=str, default="/data/home/luca/datasets_video/HMDB51/")
    parser.add_argument("-save_path", help='Path to the save folder. If None, results will not be saved. Default=None.',
                        type=str, default=None)

    args = parser.parse_args()

    return args


def get_loader(dataset_name, dataset_path, mode, n_frames, step_between_clips, batch_size, crop_size,
               fold, num_workers, set_type, shuffle):
    """
    Helper function to load the train/test loader. Here you can check all the transforms applied to the data.
    """
    normalize_rgb = transforms.Normalize(mean=(0.3697, 0.3648, 0.3201), std=(0.2617, 0.2575, 0.2581))

    data_transform_rgb = transforms.Compose(
        [transforms.Resize((crop_size, crop_size)), transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), normalize_rgb])

    data_transform_flow = transforms.Compose([transforms.Grayscale(), transforms.Resize((crop_size, crop_size)),
                                              transforms.ToTensor()])

    frames_path = os.path.join(dataset_path, "frames")
    flow_path = os.path.join(dataset_path, "flow")
    annotations_path = os.path.join(dataset_path, "annotations")

    if dataset_name == "ucf101":
        if mode == "rgb":
            dataset = UCF101Frames3D(frames_path, annotations_path, n_frames, step_between_clips,
                                     fold=fold, set_type=set_type, transform=data_transform_rgb)


        else:
            dataset = UCF101Flow3D(flow_path, annotations_path, n_frames, step_between_clips,
                                   fold=fold, set_type=set_type, transform=data_transform_flow)

    else:
        if mode == "rgb":
            dataset = HMDB51Frames3D(frames_path, annotations_path, n_frames, step_between_clips,
                                     fold=fold, set_type=set_type, transform=data_transform_rgb)
        else:
            dataset = HMDB51Flow3D(flow_path, annotations_path, n_frames, step_between_clips,
                                   fold=fold, set_type=set_type, transform=data_transform_flow)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader

def main(args):
    if args.save_path is not None:
        save_path = os.path.join(args.save_path, "{}_{}_{}".format(args.dataset_name, args.frames_train, args.step_train))
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)

    train_loader = get_loader(args.dataset_name, args.dataset_path, args.mode, args.frames_train, args.step_train,
                              args.batch_size, args.crop_size, args.fold, args.num_workers, "train", shuffle=True)

    # Technically, creating the test_loader here is worthless, but I want to show the info along with the train set.
    test_loader = get_loader(args.dataset_name, args.dataset_path, args.mode, args.frames_test, args.step_test,
                             args.batch_size, args.crop_size, args.fold, args.num_workers, "test", shuffle=False)

    train_loader.dataset.__info__()
    test_loader.dataset.__info__()

    if args.mode == "rgb":
        n_channels = 3
        pretrained_path = args.pretrained_path_rgb
    else:
        n_channels = 2
        pretrained_path = args.pretrained_path_flow

    model = i3d.InceptionI3d(400, in_channels=n_channels, only_feat=False, spatial_squeeze=True, final_endpoint="Logits")
    model.name = "inception_i3d"
    model.load_state_dict(torch.load(pretrained_path))
    model.replace_logits(len(train_loader.dataset.classes))
    model.to(args.device)

    criterion_backprop = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    milestones = [args.n_epochs * i//100 for i in range(20, 100, 20)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.25)

    final_data = []

    for epoch in range(args.n_epochs):
        epoch_data = []
        torch.cuda.synchronize()
        t0 = time()

        model, epoch_train_loss, epoch_train_acc = train_model(model, train_loader, criterion_backprop, optimizer,
                                                               device=args.device)

        torch.cuda.synchronize()
        epoch_train_time = time() - t0
        print('Epoch = {} \nTrain Loss = {:.4f} \t Train acc = {:.2f}%\t Epoch time = {:.2f} s'
              .format(epoch, epoch_train_loss, epoch_train_acc, epoch_train_time))

        epoch_data.extend((epoch, epoch_train_loss, epoch_train_acc, epoch_train_time))

        # Test phase
        torch.cuda.synchronize()
        t0 = time()

        test_loss, test_acc_clip, test_acc_maj, test_acc_softmax, inf_full_time, inf_conv_time = evaluate_model(
            model,
            test_loader,
            criterion_backprop,
            device=args.device,
            epoch=epoch,
            save_path=save_path)

        print('Test Loss = {0:.4f} \t Test acc: frames = {1:2.2f}\tvideos (m) = {2:2.2f}\tvideos (s) = {3:2.2f}'
              .format(test_loss, test_acc_clip, test_acc_maj, test_acc_softmax))

        epoch_data.extend(
            (test_loss, test_acc_clip, test_acc_maj, test_acc_softmax, inf_full_time, inf_conv_time))

        final_data.append(epoch_data)
        scheduler.step()

        if args.save_path is not None:

            savedata = pd.DataFrame(final_data, columns=["epoch", "train_loss", "train_acc", "train_time",
                                                         "test_loss", "test_acc_frames", "test_acc_maj",
                                                         "test_acc_softmax", "inf_full_time", "inf_conv_time"])
            savedata.to_csv(os.path.join(save_path, "data_{}.csv".format(args.mode)), sep='\t', index=False)

            with open(os.path.join(save_path, "parameters_{}.json".format(args.mode)), "w") as file:
                json.dump(args.__dict__, file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
