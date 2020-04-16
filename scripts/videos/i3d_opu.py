import os
import pathlib
import json
import argparse
from argparse import RawTextHelpFormatter
from time import time
from datetime import datetime

import torch
import numpy as np
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from sklearn.linear_model import RidgeClassifier

import utils.videos.models.i3d as i3d
from utils.videos.features import fast_conv_features, decoding, get_random_features, dummy_predict_GPU, generate_RM
from utils.videos.datasets.HMDB51 import HMDB51Frames3D, HMDB51Flow3D
from utils.videos.datasets.UCF101 import UCF101Frames3D, UCF101Flow3D
from utils.videos.statistics import get_video_acc_3d, get_model_size, get_output_size


def parse_args():
    parser = argparse.ArgumentParser(description="TL with the OPU on videos - I3D.",
                                     formatter_class=RawTextHelpFormatter)

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

    parser.add_argument("-crop_size", help='Size of the center crop area.', type=int, default=224)
    parser.add_argument("-fold", help='Dataset split. Default=1.', type=int, default=1, choices=[1, 2, 3])

    parser.add_argument("-RP_device", help='Device for the Random projection.', type=str, choices=["gpu", "opu"],
                        default="opu")
    parser.add_argument("-device",
                        help="Device for the GPU computation, specified as 'cuda:x', where x is the GPU number."
                             "Choose 'cpu' to use the CPU for all computations. Default='cuda:0'", type=str,
                        default='cuda:0')

    parser.add_argument("-encode_type",
                        help="Type of encoding. 'float32'=standard float32 | 'positive'= sign encoding",
                        type=str, default='positive')
    parser.add_argument("-decode_type", help='Type of decoding. Default=mixing', type=str,
                        choices=['none', 'mixing'], default='mixing')

    parser.add_argument("-n_components", help='Number of random features as a fraction of the original size.'
                                              '0 --> no projection. Default=3', type=int, default=3)

    parser.add_argument("-alpha_exp_min",
                        help='Minimum order of magnitude for the regularization coefficient. Default=3.', type=int,
                        default=2)
    parser.add_argument("-alpha_exp_max",
                        help='Maximum order of magnitude for the regularization coefficient. Default=5.', type=int,
                        default=5)
    parser.add_argument("-alpha_space",
                        help='Spacing between the mantissa of the regularization coefficients. Default=5.',
                        type=int, default=5)

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

    data_transform_rgb = transforms.Compose([transforms.Resize((crop_size, crop_size)), transforms.RandomHorizontalFlip(),
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


def save_results(args, train_data, test_data, save_path):
    """
    Helper function to save the results of the training
    """

    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    train_columns = ['RP', 'batch size', "frames_train", "step_train", 'conv f', 'conv features shape',
                     "encoding", "generation time", 'rand f', 'decoding', 'alpha', 'fit time', 'total time', 'model size [MB]',
                     'ridge size [MB]', 'date']

    savedata = pd.DataFrame(train_data, columns=train_columns)
    savedata.to_csv(os.path.join(save_path, "train_{}.csv".format(args.mode)), sep='\t', index=False)

    test_columns = ['RP', 'batch size', "frames_test", "step_test", 'conv f', "encoding", 'rand f',
                    'decoding', 'predict time', 'inference time', 'acc_test_frames', "acc_test_video_m",
                    "acc_test_video_s", 'model size [MB]', 'ridge size [MB]']

    savedata = pd.DataFrame(test_data[args.frames_test], columns=test_columns)
    savedata.to_csv(os.path.join(save_path, "inference_{}_{}.csv".format(args.mode, str(args.frames_test).zfill(3))),
                    sep='\t', index=False)

    with open(os.path.join(save_path, "parameters_{}.json".format(args.mode)), "w") as file:
        json.dump(args.__dict__, file)

    return


def main(args):
    if args.save_path is not None:
        save_path = os.path.join(args.save_path, "{}_{}".format(args.RP_device, args.n_components),
                                 "{}_{}_{}".format(args.dataset_name, args.frames_train, args.step_train))

    torch.manual_seed(0)

    train_loader = get_loader(args.dataset_name, args.dataset_path, args.mode, args.frames_train, args.step_train,
                              args.batch_size, args.crop_size, args.fold, args.num_workers, "train", shuffle=False)

    # Technically, creating the test_loader here is worthless, but I want to show the info along with the train set.
    test_loader = get_loader(args.dataset_name, args.dataset_path, args.mode, args.frames_test, args.step_test,
                             args.batch_size, args.crop_size, args.fold, args.num_workers, "test", shuffle=False)

    train_loader.dataset.__info__()
    test_loader.dataset.__info__()

    alpha_mant = np.linspace(1, 9, args.alpha_space)
    alphas = np.concatenate([alpha_mant * 10 ** i for i in range(args.alpha_exp_min, args.alpha_exp_max + 1)])

    if args.mode == "rgb":
        n_channels = 3
        pretrained_path = args.pretrained_path_rgb
    else:
        n_channels = 2
        pretrained_path = args.pretrained_path_flow

    model = i3d.InceptionI3d(400, in_channels=n_channels)
    model.name = "inception_i3d"
    model.load_state_dict(torch.load(pretrained_path))
    model.replace_logits(len(train_loader.dataset.classes))
    output_size = get_output_size(model, input_shape=(1, n_channels, args.frames_train, args.crop_size, args.crop_size))

    if args.n_components != 0:
        args.n_components = output_size // args.n_components
        print("Random Projection from {} to {}".format(output_size, args.n_components))

    model_size, total_weights, tot_linear_size = get_model_size(model)

    model.to(args.device)

    enc_train_features, train_labels, train_conv_time, train_enc_time = fast_conv_features(train_loader, model,
                                                                                           output_size,
                                                                                           device=args.device,
                                                                                           encode_type=args.encode_type)

    print("Train conv features time = {0:3.2f} s\tencoding = {1:1.5f} s - shape {2}"
          .format(train_conv_time, train_enc_time, enc_train_features.shape))

    enc_test_features, test_labels, test_conv_time, test_enc_time = fast_conv_features(test_loader, model, output_size,
                                                                                       device=args.device,
                                                                                       encode_type=args.encode_type)
    print("Test conv features time = {0:3.2f} s\tencoding = {1:1.5f} s - shape {2}"
          .format(test_conv_time, test_enc_time, enc_test_features.shape))

    if args.RP_device == "gpu":
        R, generation_time = generate_RM(args.n_components, output_size)
        print("Generation time = {0:3.2f} s".format(generation_time))

    else:
        R = None
        generation_time = 0.

    # Encode, get the random features and decode
    train_proj_time, train_random_features = get_random_features(enc_train_features, args.n_components, matrix=R)
    test_proj_time, test_random_features = get_random_features(enc_test_features, args.n_components, matrix=R)
    print('Train Projection time = {0:3.2f} s\nTest Projection time = {1:3.2f} s'.format(train_proj_time, test_proj_time))

    del enc_train_features, enc_test_features

    train_decode_time, dec_train_random_features = decoding(train_random_features, decode_type=args.decode_type)
    # I ignore the test decode time because after the Ridge I will simulate it on GPU
    _, dec_test_random_features = decoding(test_random_features, decode_type=args.decode_type)

    torch.cuda.empty_cache()

    current_date = str(datetime.now())
    final_train_data = []
    test_data = {k: [] for k in range(1, args.frames_test + 1)}

    # Run the ridge classifier
    for alpha in alphas:
        clf = RidgeClassifier(alpha=alpha)
        since = time()
        clf.fit(dec_train_random_features, train_labels)
        fit_time = time() - since

        train_acc_clip, train_acc_maj, train_acc_softmax = get_video_acc_3d(dec_train_random_features, train_labels,
                                                                            train_loader, clf=clf)

        total_train_time = train_conv_time + train_enc_time + train_proj_time + train_decode_time + fit_time

        ridge_size = np.prod(clf.coef_.shape) * 32 / (8 * 2 ** 10 * 2 ** 10)

        train_data = [args.n_components, args.batch_size, args.frames_train, args.step_train,
                      train_conv_time, output_size, train_enc_time, generation_time, train_proj_time, train_decode_time, alpha, fit_time,
                      total_train_time, model_size, ridge_size, current_date]

        final_train_data.append(train_data)

        print('\nalpha = {0:.1e}\tTrain acc: frames = {1:2.2f}\tvideos (m) = {2:2.2f}\tvideos (s) = {3:2.2f}'
              .format(alpha, train_acc_clip, train_acc_maj, train_acc_softmax))



        test_acc_clip, test_acc_maj, test_acc_softmax = get_video_acc_3d(dec_test_random_features, test_labels,
                                                                         test_loader, clf=clf, save_path=save_path)

        test_decode_time, predict_time = dummy_predict_GPU(clf, dec_test_random_features, device=args.device)

        total_inference_time = test_conv_time + test_enc_time + test_proj_time + test_decode_time + predict_time

        # In order, test_times contains the convolutional, encoding and projection time.
        inference_data = [args.n_components, args.batch_size, args.frames_test, args.step_test,
                          test_conv_time, test_enc_time, test_proj_time, test_decode_time, predict_time,
                          total_inference_time, test_acc_clip, test_acc_maj, test_acc_softmax, model_size,
                          ridge_size]

        test_data[args.frames_test].append(inference_data)

        print('\tTest frames = {0}\tTest acc: frames = {1:2.2f}\tvideos (m) = {2:2.2f}\tvideos (s) = {3:2.2f}'
              .format(args.frames_test, test_acc_clip, test_acc_maj, test_acc_softmax))

        if args.save_path is not None:
            save_results(args, final_train_data, test_data)

    print("Results saved in {}".format(save_path))

    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
