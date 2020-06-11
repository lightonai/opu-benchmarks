from collections import namedtuple
from time import time

import torch
import numpy as np

try:
    from lightonml.encoding.base import Float32Encoder, MixingBitPlanDecoder
    from lightonml.projections.sklearn import OPUMap
except ModuleNotFoundError:
    print("Missing lightonml module. Check that it is correctly installed.")


def get_random_features(X, n_components, opu_map=None, matrix=None, conv_blocks=1, device="cuda:0"):
    """
    Performs the random projection of the encoded random features X using the OPU.

    Parameters
    ----------
    X: numpy 2d array or torch.tensor,
        encoded convolutional training features. Make sure that the dtype is int8 if n_components!=0.
    n_components: int,
        number of random projections.
    opu_map: OPUMap object or None,
        OPUMap object for performing the projection with the OPU. If None, it is generated automatically.
        You should pass it when you plan to do multiple projections in the same script.
    matrix: None or torch.tensor,
        Matrix to use for the random projection on GPU. If None, the OPU will be used.
    conv_blocks: int,
        number of splits for the input matrix if GPU is used.
    device: str,
        device for the random projection if the OPU is not used. Examples are "cpu" or "cuda:0".

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
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X_blocks = X.split(conv_blocks)
        random_features, train_time = get_rand_features_GPU(X_blocks, matrix, device=device)

    else:
        if opu_map is None:
            opu_map = OPUMap(n_components=n_components)

        since = time()
        random_features = opu_map.transform(X)
        train_time = time() - since

    return train_time, random_features


def get_rand_features_GPU(X_blocks, R_blocks, device="cuda:0"):
    """
    Computes the random projection on GPU, using splits for both the random matrix and the input matrix.

    Parameters
    ----------
    X_blocks: tuple of torch.Tensor,
        splits for the input matrix. It must be on CPU.
    R_blocks: tuple of torch.Tensor,
        splits for the random matrix. It must be on CPU.
    device: str,
        device for the random projection if the OPU is not used. Examples are "cpu" or "cuda:0".

    Returns
    -------
    proj_time: float,
        projection time for the features.
    random_features: numpy 2d array,
        random features of the training set. If n_components=0, the original train random features are returned.

    """

    random_features = []
    n_blocks = len(R_blocks.real)

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time()
        for Xi in X_blocks:
            product_row = []
            Xi = Xi.to(device)

            for idx in range(n_blocks):
                R = R_blocks.real[idx].to(device)
                prod = torch.mm(Xi, R) ** 2

                R = R_blocks.im[idx].to(device)
                prod += torch.mm(Xi, R) ** 2

                product_row.append(prod.to("cpu"))

            random_features.append(torch.cat(product_row, axis=1))
        random_features = torch.cat(random_features, axis=0)

        torch.cuda.synchronize()
        proj_time = time() - t0

    return random_features, proj_time


class GPU_matrix:
    """
    Class that generates a random matrix and computes the optimal split for both the random matrix and the input
    matrix so that everything will fit in the GPU memory.

    NOTE: It works well if the random projection is the only thing you do. If there are other things in the script,
    account for the additional memory cost.

    Attributes
    ----------
    n_samples: int, number of samples of the input matrix (i.e. the rows)
    n_features: int, number of features of the input matrix (i.e. the columns)
    n_components: int, number of random projections.
    GPU_memory: int, memory available for the Random projection.
    normalize: boolean, whether to normalize or not the random matrix.

    R_blocks_size: int, optimal size for the blocks of the random matrix (along the columns)
    conv_blocks_size: int, optimal size for the blocks of the input matrix (along the rows)

    Methods
    -------
    optimize_split: returns the optimal size for the input and random matrix split.
    generate_RM: generates the split random matrix as a tuple of torch tensors.

    """
    def __init__(self, n_samples, n_features, n_components, GPU_memory=16, normalize=True):
        """

        Parameters
        ----------
        n_samples: int, number of samples of the input matrix (i.e. the rows)
        n_features: int, number of features of the input matrix (i.e. the columns)
        n_components: int, number of random projections.
        GPU_memory: int, memory available for the Random projection.
        normalize: boolean, whether to normalize or not the random matrix.
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_components = n_components

        self.GPU_memory = GPU_memory
        self.normalize = normalize

        self.R_blocks_size, self.conv_blocks_size = self.optimize_split()

    def optimize_split(self):
        """
        Computes the optimal size for the splits of the convolutional and random matrix. Memory is always in GB.
        The random matrix is assumed to be complex, the conv matrix real.

        The number of blocks used to split the matrices is:

        Random matrix = n_components//R_blocks_size
        Conv matrix = n_samples//conv_blocks_size
        """

        GB_scaling = 32 / (8 * 1024 * 1024 * 1024)

        R_memory = self.n_features * self.n_components * GB_scaling * 2
        conv_memory = self.n_samples * self.n_features * GB_scaling
        prod_memory = self.n_samples * self.n_components * GB_scaling

        total_memory = R_memory + conv_memory + prod_memory

        R_blocks_size, conv_blocks_size = 1, 1
        while total_memory > self.GPU_memory:

            if R_memory > conv_memory:
                R_blocks_size += 1
            else:
                conv_blocks_size += 1

            R_memory = self.n_features * self.n_components * GB_scaling * 2 / R_blocks_size
            conv_memory = self.n_samples * self.n_features * GB_scaling / conv_blocks_size
            prod_memory = self.n_samples * self.n_components * GB_scaling / (conv_blocks_size * R_blocks_size)

            total_memory = R_memory + conv_memory + prod_memory

        return R_blocks_size, conv_blocks_size

    def generate_RM(self):
        """
        Generates the splits for the random matrix, real and imaginary part.

        Returns
        -------
        R: named_tuple,
            Random matrix. R.real and R.im return the real and imaginary part. They are tuples of tensors, where element
            i contains the split i of the real/imaginary part.
        generation_time: float,
            time to generate and split the random matrix.
        """

        since = time()
        matrix = namedtuple("R", ["real", "im"])

        R_real = torch.randn((self.n_features, self.n_components), requires_grad=False)
        if self.normalize:
            R_real /= np.sqrt(self.n_components)

        R_real = R_real.split(self.n_components//self.R_blocks_size, dim=1)

        R_im = torch.randn((self.n_features, self.n_components), requires_grad=False)
        if self.normalize:
            R_im /= np.sqrt(self.n_components)

        R_im = R_im.split(self.n_components//self.R_blocks_size, dim=1)

        R = matrix(real=R_real, im=R_im)

        generation_time = time() - since

        return R, generation_time
