from time import time

import torch
import numpy as np

try:
    from lightonml.encoding.base import Float32Encoder, MixingBitPlanDecoder
    from lightonml.projections.sklearn import OPUMap
except ModuleNotFoundError:
    print("Missing lightonml module. Check that it is correctly installed.")


def encode_GPU(X, encode_type="positive"):
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
    n_images: int,
        number of images.
    encode: str,
        encodes the convolutional features on GPU. Choices: positive -> sign encoding / float32 -> 1 bit float32.
        Eventual other options result in no encoding.
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


def generate_RM(n_components, n_features, n_ram=10, normalize=True):
    """
    Generates the splits for the random matrix, ready to be moved to GPU.

    Parameters
    ----------

    n_components: int,
        number of random projections.
    n_features: int,
        number of convolutional features of the input matrix.
    n_ram: int,
        number of splits for the random matrix.
    normalize: boolean,
        if True, normalizes the matrix by dividing each entry by np.sqrt(n_features). defaults to True

    Returns
    -------

    R: list of torch tensor,
        random projection matrix.
    generation_time: float,
        time to generate the matrix.

    """

    matrix_shape = (n_features, n_components // n_ram)
    R = []
    since = time()

    for i in range(n_ram):
        print('Generating random matrix # ', i + 1)
        # allocate the right amount of memory
        R_tmp_real = torch.randn(matrix_shape, dtype=torch.float, requires_grad=False)
        R_tmp_im = torch.randn(matrix_shape, dtype=torch.float, requires_grad=False)
        if normalize is True:
            R_tmp_real /= np.sqrt(n_components)
            R_tmp_im /= np.sqrt(n_components)

        R.append((R_tmp_real, R_tmp_im))

    generation_time = time() - since

    return R, generation_time


def get_rand_features_GPU(R, X, device="cuda:0"):
    """
    Computes the random projection on GPU.

    Parameters
    ----------

    R: list of torch tensor,
        random projection matrix.
    X: numpy array,
        matrix to project. Expects a list of tuples in the form [Real part, Imaginary part] of the different blocks of
        the matrix.
    device: str,
        device for the random projection.

    Returns
    -------
    random_features: torch tensor,
        array of random features.
    proj_time: float,
        projection time.

    """
    random_features = []


    with torch.no_grad():
        X = X.to(device)

        torch.cuda.synchronize()
        t0 = time()

        for real_part, im_part in R:
            matrix = real_part.to(device)
            real_dot = torch.mm(X, matrix)

            matrix = im_part.to(device)
            im_dot = torch.mm(X, matrix)

            random_features.append(real_dot**2 + im_dot**2)

        random_features = torch.cat(random_features, axis=1).to("cpu")

        torch.cuda.synchronize()
        proj_time = time() - t0

    return random_features, proj_time


def get_random_features(X, n_components, matrix=None):
    """
    Performs the random projection of the encoded random features X using the OPU.

    Parameters
    ----------
    X: numpy 2d array,
        encoded convolutional training features. Make sure that the dtype is int8 if n_components!=0.
    n_components: int,
        number of random projections.
    matrix: None or numpy array,
        Matrix to use for the random projection on GPU. If None, the OPU will be used.
    Returns
    -------

    proj_time: float,
        projection time for the features.
    random_features: numpy 2d array,
        random features of the training set. If n_components=0, the original train random features are returned.

    """

    if n_components == 0:
        train_time = 0.

        # The conversion to numpy is needed for compatibility with the MixingBitPlanDecoder.
        if type(X) is not np.ndarray:
            X = X.numpy()

        return train_time, X

    if matrix is not None:
        random_features, train_time = get_rand_features_GPU(matrix, X)

    else:
        opu = OPUMap(n_components=n_components)

        since = time()
        random_features = opu.transform(X)
        train_time = time() - since

    return train_time, random_features



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
