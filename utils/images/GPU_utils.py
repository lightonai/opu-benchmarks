import cupy as cp
import numpy as np
from time import time


def compute_dot_split(x, random_matrix):
    """
    Computes the random projection dot product on GPU.

    Parameters
    ----------
    x = numpy array,
        contains the data to project.
    random_matrix = ????,
        random projection matrix.

    Returns
    -------

    output: cupy array ???,
        contains the random projected matrix.
    """

    rm = cp.asarray(random_matrix)
    output = cp.asnumpy(cp.dot(x, rm))
    return output


def generate_RM(n_ram, n_components, n_features, normalize=True):
    """
    Generates the splits for the random matrix, ready to be moved to GPU.

    Parameters
    ----------

    n_ram: int,
        number of splits for the random matrix.
    n_components: int,
        number of random projections.
    n_features: int,
        number of convolutional features of the input matrix.
    normalize: boolean,
        if True, normalizes the matrix by dividing each entry by np.sqrt(n_features). defaults to True
    Returns
    -------

    R: list of numpy array,
        blocks of the random projection matrix.
    generation_time: float,
        time to generate the matrix.

    """

    matrix_shape = (n_features, n_components // n_ram)
    R = []
    since = time()

    for i in range(n_ram):
        print('Generating random matrix # ', i + 1)
        # allocate the right amount of memory
        R_tmp = np.zeros(shape=matrix_shape, dtype='float32')
        # fill that amount of memory and no more
        R_tmp[:] = np.random.randn(*matrix_shape)
        if normalize is True:
            R_tmp /= np.sqrt(n_components)
        R.append(R_tmp)

    generation_time = time() - since

    return R, generation_time


def get_rand_features_GPU(R, X):
    """
    Computes the random projection on GPU.

    Parameters
    ----------

    R: numpy array,
        random projection matrix.
    X: numpy array,
        matrix to project.

    Returns
    -------
    togpu_time: float,
        time to move the features to GPU.
    proj_time: float,
        projection time.
    tocpu_time: float,
        time to move the featrues back to CPU.
    random_features: numpy array,
        matrix of random features
    """

    random_features = []

    t0 = time()
    # Export the features to GPU
    X = cp.asarray(X)
    togpu_time = time() - t0

    # Do the RP

    t0 = time()
    for matrix in R:
        random_features.append(compute_dot_split(X, matrix))
    proj_time = time() - t0

    # Turn the features back to numpy arrays.
    t0 = time()
    random_features = cp.asnumpy(random_features)
    random_features = np.concatenate(random_features, axis=1)
    tocpu_time = time() - t0

    return togpu_time, proj_time, tocpu_time, random_features
