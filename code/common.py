
import numpy as np


def polynomialize(n_levels):
    mat = np.zeros([n_levels, n_levels])
    vec = np.array([(i+1) - 0.5 * (n_levels+1) for i in range(n_levels)])
    for i in range(n_levels):
        mat[:, i] = vec ** i
    return mat


def orthogonalize(mat):
    q, r = np.linalg.qr(mat)
    eigen = np.diag(np.diag(r))
    return np.matmul(q, eigen)


def normalize(mat):
    n_rows, n_cols = mat.shape
    for i in range(1, n_rows):
        mat[:, i] = mat[:, i] / np.sqrt(np.sum(mat[:, i] ** 2))
    return mat


def create_poly(n_levels):
    mp = polynomialize(n_levels)
    mo = orthogonalize(mp)
    mn = normalize(mo)
    # drop first column
    mat = mn[:, 1:]
    # replace new first column with a different scale
    mat[:, 0] = [(i - 0.5 * n_levels - 0.5) / 10 for i in range(1, n_levels+1)]
    return mat
