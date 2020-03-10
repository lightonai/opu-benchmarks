from time import time
import numpy as np

def compute_dot_split(x, random_matrix):
    """
    Computes the random projection dot product on CPU.

    Parameters
    ----------
    x = numpy array,
        contains the data to project.
    random_matrix = numpy array,
        random projection matrix.

    Returns
    -------
    output: numpy array,
        contains the matrix of random features.
    """
    t0 = time()
    output = np.dot(x, random_matrix)
    proj_time = time() - t0
    return proj_time, output

def generate_RM(n_components, n_features, normalize=True):
    """
    Generates a random matrix whose entries are drawn according to a normal distribution.

    Parameters
    ----------
    n_components: int,
        number of random projections.
    n_features: int,
        number of convolutional features of the input matrix.
    normalize: boolean,
        if True, normalizes the matrix by dividing each entry by np.sqrt(n_features). Defaults to True.

    Returns
    -------
    R:numpy array,
        Random matrix.
    generation_time: float,
        time to generate the matrix.

    """
    since = time()

    R = np.random.randn(n_features, n_components)
    if normalize is True:
        R /= np.sqrt(n_components)

    generation_time = time() - since
    return R, generation_time
