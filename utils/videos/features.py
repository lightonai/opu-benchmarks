from time import time

import torch
import numpy as np

try:
    from lightonml.encoding.base import Float32Encoder, MixingBitPlanDecoder
except ModuleNotFoundError:
    print("Missing lightonml module. Check that it is correctly installed.")


def encode_GPU(X, encode_type="positive"):
    """
    Encodes a batch on GPU.

    Parameters
    ----------
    X: torch tensor, the tensor to encode.
    encode_type: type of encoding. Choose between "positive"-->sign encoding | "float32"-->1st exponent bit of float32
        Anything else results in no encoding

    Returns
    -------
    X: torch tensor, binarized version of X.
    encode_time: float, encoding time
    """

    if encode_type == 'positive':
        torch.cuda.synchronize()
        start = time()
        X = (X > 0)
        torch.cuda.synchronize()
        encode_time = time() - start

    elif encode_type == 'float32':
        torch.cuda.synchronize()
        start = time()
        X = (torch.abs(X) > 2)
        torch.cuda.synchronize()
        encode_time = time() - start
    else:
        encode_time = 0.

    return X, encode_time


def fast_conv_features(loader, model, out_shape, encode_type='positive', device='cpu'):
    """
    Computes the convolutional features of the images in the loader, and optionally encodes them on GPU
    based on their sign.

    Parameters
    ----------

    loader: torch Dataloader,
        contains the images to extract the training features from.
    model: torchvision.models,
        architecture to use to get the convolutional features.
    out_shape: int,
        output size of the last layer.
    encode_type: type of encoding. Choose between "positive"-->sign encoding | "float32"-->1st exponent bit of float32
        Anything else results in no encoding
    device: string,
        device to use for the computation. Choose between 'cpu' and 'gpu:x', where
        x is the GPU number. Defaults to 'cpu'.

    Returns
    -------

    conv_features: numpy array,
        array containing the convolutional features. format is (# of samples * # of features).
        They are already moved to CPU.
    labels: list of int,
        labels associated to each image.
    conv_time: float,
        time required to compute the convolutional features. It includes the data loading.
    encode_time: float,
        encoding time, done on the GPU.
    """

    n_images = len(loader.dataset)

    model.eval()
    batch_size = loader.batch_size

    encode_time = 0

    with torch.no_grad():

        if encode_type is not None:
            conv_features = torch.ByteTensor(n_images, out_shape)
        else:
            conv_features = torch.FloatTensor(n_images, out_shape)

        labels = np.empty(n_images, dtype='uint8')

        torch.cuda.synchronize()
        t0 = time()

        for i, (images, targets) in enumerate(loader):

            images = images.to(torch.device(device))

            outputs = model(images)

            if encode_type is not None:
                outputs, batch_encode_time = encode_GPU(outputs, encode_type=encode_type)
                encode_time += batch_encode_time

            conv_features[i * batch_size: (i + 1) * batch_size, :] = outputs.data.view(images.size(0), -1).to(
                torch.device("cpu"))
            labels[i * batch_size: (i + 1) * batch_size] = targets.numpy()

        torch.cuda.synchronize()
        conv_time = time() - t0 - encode_time

    return conv_features, labels, conv_time, encode_time


def decoding(random_features, exp_bits=1, sign_bit=False, mantissa_bits=0, decode_type='mixing'):
    """
    Decodes the random features.

    Parameters
    ----------

    random_features: numpy 2d array,
        random features.
    exp_bits: int,
        number of bits for the exponent. Defaults to 1.
    sign_bit: boolean,
        if True, a bit will be used for the sign of the encoded numbers. Defaults to False.
    mantissa_bits: int,
        number of bits for the mantissa. Defaults to 0.
    encode type: string,
        type of decoding to perform. If not 'mixing', it just converts the dtype to float32 for better compatibility
        with scikit-learn.

    Returns
    -------
    decoding_time: float,
        encoding time.
    dec_train_random_features: numpy 2d array,
        decoded random features.
    """

    if decode_type == 'mixing':
        if sign_bit is True:
            decode_bits = exp_bits + mantissa_bits + 1
        else:
            decode_bits = exp_bits + mantissa_bits
        decoder = MixingBitPlanDecoder(n_bits=decode_bits)
        since = time()
        dec_random_features = decoder.transform(random_features)
        train_decode_time = time() - since

    else:
        since = time()
        dec_random_features = random_features.astype('float32')
        train_decode_time = time() - since

    return train_decode_time, dec_random_features


def dummy_predict(clf, dec_test_random_features):
    """
    Computes the predicted labels of the given input features.

    Parameters
    ----------
    clf: Ridge classifier,
        Ridge classifier object from scikit-learn library.
    dec_test_random_features: numpy array,
        random features of the test set.

    Returns
    -------
    predict_time: float,
        prediction time.
    """

    since = time()
    clf.predict(dec_test_random_features)
    predict_time = time() - since

    return predict_time


def dummy_predict_GPU(clf, test_random_features, device='cpu'):
    """
    Performs a dummy decoding + predict on the random features on CPU or GPU.

    Parameters
    ----------
    clf: Ridge classifier,
        Ridge classifier object from scikit-learn library.
    dec_test_random_features: numpy array,
        random features of the test set.
    device: string,
        device to use for the computation. Choose between 'cpu' and 'gpu:x', where
        x is GPU number. Defaults to 'cpu'.

    Returns
    -------
    predict_time: float,
        prediction time.
    """

    x = torch.ByteTensor(test_random_features).to(device)

    torch.cuda.synchronize()
    start = time()
    x = x.float()
    torch.cuda.synchronize()
    decode_time = time() - start

    ridge_coefficients = torch.FloatTensor(clf.coef_.T).to(device)
    start = time()
    torch.cuda.synchronize()
    torch.mm(x, ridge_coefficients)

    torch.cuda.synchronize()
    predict_time = time() - start

    del ridge_coefficients, x
    torch.cuda.empty_cache()

    return decode_time, predict_time
