import os
import re
import pathlib
import argparse
from argparse import RawTextHelpFormatter
from time import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier

from utils.images.model_utils import pick_model, get_model_size
from utils.images.features import fast_conv_features, decoding, get_random_features, dummy_predict_GPU
from utils.images.dataset import Animals10


def parse_args():
    parser = argparse.ArgumentParser(description="Transfer Learning with the OPU", formatter_class=RawTextHelpFormatter)

    parser.add_argument("model_name", help='Base model for TL.', type=str)
    parser.add_argument("OPU", help='OPU model. For file naming.', type=str, choices=['Zeus', 'Vulcain', "Saturn"])

    parser.add_argument("-device",
                        help="Device for the GPU computation, specified as 'cuda:x', where x is the GPU number."
                             "Choose 'cpu' to use the CPU for all computations. Defaults to 'cuda:0'", type=str,
                        default='cuda:0')
    parser.add_argument("-num_workers", help="Number of workers. Defaults to 12", type=int, default=12)
    parser.add_argument("-batch_size", help="Batch size. Defaults to 32", type=int, default=32)

    parser.add_argument('-model_options', help='Options for the removal of specific layers in the architecture.'
                                               'Defaults to full.',
                        choices=['full', 'noavgpool', 'norelu', 'norelu_maxpool'], type=str, default="full")
    parser.add_argument('-model_dtype', help="dtype for the network weights. Defaults to 'float32'.",
                        choices=['float32', 'float16'], type=str, default="float32")
    parser.add_argument("-encode_type",
                        help='Type of encoding, done on GPU. Defaults to positive. The float32 is done with 1 bit.',
                        type=str, choices=['float32', 'positive'], default='positive')
    parser.add_argument("-decode_type", help='Type of decoding. Defaults to mixing', type=str,
                        choices=['none', 'mixing'],
                        default='mixing')
    parser.add_argument("-exp_bits", help='Number of bits for encoding and decoding. Defaults to 1', type=int,
                        default=1)

    parser.add_argument("-n_components", help='Sets the RP size to original_size/n_components. If 0, no RP is applied.'
                                              'Defaults to 2.', type=int, default=2)

    parser.add_argument("-alpha_exp_min",
                        help='Minimum order of magnitude for the regularization coefficient. Defaults to 3.', type=int,
                        default=3)
    parser.add_argument("-alpha_exp_max",
                        help='Maximum order of magnitude for the regularization coefficient.Defaults to 5.', type=int,
                        default=5)
    parser.add_argument("-alpha_space",
                        help='Spacing between the mantissa of the regularization coefficients. Defaults to 5.',
                        type=int, default=5)

    parser.add_argument("-dataset_path", help='Path to the dataset folder (excluded).', type=str,
                        default='../datasets/')
    parser.add_argument("-save_path",
                        help='Path to the save folder. If None, results will not be saved. Defaults to None.',
                        type=str, default=None)

    args = parser.parse_args()

    return args


def get_loaders(dataset_path, batch_size=32, num_workers=12, mean=None, std=None):
    """
    Function to load the train/test loaders.

    Parameters
    ----------
    dataset_path: str, dataset path.

    batch_size: int, batch size.
    num_workers: int, number of workers.
    mean:None or torch.Tensor, mean per channel
    std:None or torch.Tensor, std per channel

    Returns
    -------
    train_loader: Pytorch dataloader, dataloader for the train set.
    test_loader: Pytorch dataloader, dataloader for the test set.
    """

    transform_list = [transforms.Resize((224, 224)), transforms.ToTensor()]
    if mean is not None:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    data_transform = transforms.Compose(transform_list)

    dataset_path = os.path.join(dataset_path, "animals10/raw-img/")

    train_dataset = Animals10(dataset_path, test_ratio=20, mode="train", transform=data_transform)
    test_dataset = Animals10(dataset_path, test_ratio=20, mode="test", transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


def get_mean_std(train_loader):
    """
    Computes the mean and std per channel on the train dataset.

    Parameters
    ----------
    train_loader: Pytorch dataloader, dataloader for the train set

    Returns
    -------
    mean: torch.Tensor, mean per channel
    std: torch.Tensor, std per channel
    """

    mean, std = torch.zeros(3), torch.zeros(3)

    for batch_id, (image, target) in enumerate(train_loader):
        mean += torch.mean(image, dim=(0, 2, 3))
        std += torch.std(image, dim=(0, 2, 3))

    mean = mean / len(train_loader)
    std = std / len(train_loader)

    return mean, std

def save_results(args, final_train_data, final_inference_data):
    base_path = os.path.join(args.save_path, '{}_{}'.format(args.model_name, args.OPU),
                             'OPU_{}_{}_{}_{}'.format(args.device, args.n_components, args.model_options, args.model_dtype))

    pathlib.Path(os.path.join(base_path, 'train')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(base_path, 'inference')).mkdir(parents=True, exist_ok=True)

    train_data_columns = ['RP', 'model dtype', 'conv f', 'conv features shape', args.encode_type,
                          'rand f', 'to float32', 'alpha', 'fit time', 'total time',
                          'acc_test', 'model size [MB]', 'total weights', 'ridge size [MB]', 'date']

    inference_data_columns = ['RP', 'model dtype', 'conv f', args.encode_type, 'rand f',
                              'to float32', 'predict time', 'inference time', 'model size [MB]',
                              'ridge size [MB]']

    csv_name_train = os.path.join("train", "{}_{}_train.csv".format(args.model_name, args.model_options))
    csv_name_inference = os.path.join("inference", "{}_{}_inference.csv".format(args.model_name, args.model_options))

    pd.DataFrame(final_train_data, columns=train_data_columns).to_csv(os.path.join(base_path, csv_name_train),
                                                                      sep='\t', index=False)

    pd.DataFrame(final_inference_data, columns=inference_data_columns).to_csv(os.path.join(base_path, csv_name_inference),
                                                                              sep='\t', index=False)
    print("Results saved in ", base_path)
    return

def main(args):

    print('Model = {}\tModel options = {}\tDevice = {}\n'.format(args.model_name, args.model_options, args.device))

    train_loader, test_loader = get_loaders(args.dataset_path, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Computing dataset mean...")
    mean, std = get_mean_std(train_loader)
    train_loader, test_loader = get_loaders(args.dataset_path, batch_size=args.batch_size, num_workers=args.num_workers,
                                            mean=mean, std=std)

    print("Train images = {}\tTest images = {}".format(len(train_loader.dataset), len(test_loader.dataset)))

    alpha_mant = np.linspace(1, 9, args.alpha_space)
    alphas = np.concatenate([alpha_mant * 10 ** i for i in range(args.alpha_exp_min, args.alpha_exp_max + 1)])

    # Load the model.
    model, output_size = pick_model(model_name=args.model_name, model_options=args.model_options, pretrained=True,
                                    dtype=args.model_dtype)

    model_size, total_weights, model_size_linear = get_model_size(model)
    print("model size = {0:3.2f} MB".format(model_size))

    # Get the encoded convolutional features.

    model.to(torch.device(args.device))
    if args.n_components != 0:
        n_components = output_size // args.n_components
        print("Random projection from {} to {}".format(output_size, n_components))
    else:
        n_components = 0

    enc_train_features, train_labels, train_conv_time, train_encode_time = fast_conv_features(train_loader, model,
                                                                                              output_size,
                                                                                              encode_type=args.encode_type,
                                                                                              dtype=args.model_dtype,
                                                                                              device=args.device)

    print("{0} - train conv features time = {1:3.2f} s\tencoding = {2:1.5f} s"
          .format(args.model_name, train_conv_time, train_encode_time))

    enc_test_features, test_labels, test_conv_time, test_encode_time = fast_conv_features(test_loader, model,
                                                                                          output_size,
                                                                                          encode_type=args.encode_type,
                                                                                          dtype=args.model_dtype,
                                                                                          device=args.device)

    print("{0} - test conv features time  = {1:3.2f} s\tencoding = {2:1.5f} s"
          .format(args.model_name, test_conv_time, test_encode_time))

    # Get the random features and decode

    train_proj_time, train_random_features = get_random_features(enc_train_features, n_components)
    test_proj_time, test_random_features = get_random_features(enc_test_features, n_components)

    del model, enc_train_features, enc_test_features

    train_decode_time, dec_train_random_features = decoding(train_random_features, decode_type=args.decode_type)
    test_decode_time, dec_test_random_features = decoding(test_random_features, decode_type=args.decode_type)

    print("Train projection time = {0:3.2f} s\tTrain decode time = {1:3.2f} s".format(train_proj_time, train_decode_time))
    print("Test projection time = {0:3.2f} s\tTest decode time = {1:3.2f} s".format(test_proj_time, test_decode_time))
    torch.cuda.empty_cache()

    current_date = str(datetime.now())
    final_train_data = []
    final_inference_data = []

    # Run the ridge classifier

    for alpha in alphas:
        clf = RidgeClassifier(alpha=alpha)
        since = time()
        clf.fit(dec_train_random_features, train_labels)
        fit_time = time() - since

        train_accuracy = clf.score(dec_train_random_features, train_labels) * 100
        test_accuracy = clf.score(dec_test_random_features, test_labels) * 100

        test_decode_time, predict_time = dummy_predict_GPU(clf, dec_test_random_features, device=args.device)

        total_train_time = train_conv_time + train_encode_time + train_proj_time + train_decode_time + fit_time

        total_inference_time = test_conv_time + test_encode_time + test_proj_time + test_decode_time + predict_time

        n_bits = int(re.findall(r"\d+", args.model_dtype)[0])
        ridge_size = np.prod(clf.coef_.shape) * n_bits / (8 * 2 ** 10 * 2 ** 10)

        train_data = [n_components, args.model_dtype, train_conv_time, output_size,
                      train_encode_time, train_proj_time, train_decode_time, alpha, fit_time, total_train_time,
                      test_accuracy, model_size, total_weights, ridge_size, current_date]

        inference_data = [n_components, args.model_dtype, test_conv_time, test_encode_time,
                          test_proj_time, test_decode_time, predict_time, total_inference_time, model_size, ridge_size]

        final_train_data.append(train_data)
        final_inference_data.append(inference_data)

        print('Alpha = {0:.2e}\tTrain acc = {1:3.2f}\tTest acc = {2:2.2f}\tInference time = {3:3.2f} s'
              .format(alpha, train_accuracy, test_accuracy, total_inference_time))

    if args.save_path is not None:
        save_results(args, final_train_data, final_inference_data)

    return


if __name__ == '__main__':
    args = parse_args()
    main(args)
