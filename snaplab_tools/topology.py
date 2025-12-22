import numpy as np

def threshold_adj(A, q=0.8, abs=True, fill_diag=True, binarize=True):
    if abs:
        A = np.abs(A)

    if fill_diag:
        np.fill_diagonal(A, 0)

    thresh = np.quantile(A, q=q)
    mask = A >= thresh

    A_out = A.copy()
    A_out[~mask] = 0
    if binarize:
        A_out[mask] = 1

    return A_out

