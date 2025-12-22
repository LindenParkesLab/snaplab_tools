import numpy as np
import bct

from snaplab_tools.utils import normalize_x

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


def volume_normalize_adjacency(adjacency, region_size):

    region_size = normalize_x(region_size) + 1e-5
    size_matrix = np.add.outer(region_size, region_size) / 2
    adjacency_norm = np.divide(adjacency, size_matrix)
    adjacency_mask = adjacency_norm > 0
    adjacency_norm[adjacency_mask] += 1
    adjacency_norm = np.log(adjacency_norm, out=np.zeros_like(adjacency_norm), where=(adjacency_norm != 0))
    
    return adjacency_norm


def threshold_consistency(A, thr=0.60):
    """
    Thresholds edges from a group averaged adjacency matrix by retaining edges that are non-zero in some proportion of
    subjects (defined by thr). NaN values are treated as missing data.
    Args:
        A: n x n x m structural adjacency matrix with subjects along m
        thr: proportion of subjects that an edge needs to exist in order to be retained
    Returns:
        Am: thresholded mean adjacency matrix
    """
    # get group averaged A matrix (ignoring NaNs)
    Am = np.nanmean(A, axis=2)
    
    # find non-zero elements in A
    Ab = A > 0
    
    # count non-zero elements along subject dimension
    Ab_count = np.sum(Ab, axis=2)
    
    # count valid (non-NaN) subjects for each edge
    valid_count = np.sum(~np.isnan(A), axis=2)
    
    # compute fraction of non-zero elements over valid subjects
    # where valid_count is 0, set proportion to 0 (will be masked anyway)
    Ab_prop = np.divide(Ab_count, valid_count, 
                        out=np.zeros_like(Ab_count, dtype=float),
                        where=valid_count > 0)
    
    # define mask using threshold
    mask = Ab_prop < thr
    
    # set elements within mask in mean A matrix to zero
    Am[mask] = 0
    
    return Am


def get_norm_rc(A, n_perms=1000, weighted=True, directed=False):
    
    if directed == True:
        def deg_func(A):
            _, _, degree = bct.degrees_dir(A)
            return degree
        randmio_func = bct.randmio_dir
    elif directed == False:
        deg_func = bct.degrees_und
        randmio_func = bct.randmio_und

    degree = deg_func(A)
    kmax = int(np.max(degree))
    print(kmax)

    if weighted == True and directed == True:
        def rc_func(A, kmax):
            R = bct.rich_club_wd(A, klevel=kmax)
            return R
    elif weighted == True and directed == False:
        def rc_func(A, kmax):
            R = bct.rich_club_wu(A, klevel=kmax)
            return R
    elif weighted == False and directed == True:
        def rc_func(A, kmax):
            R, _, _ = bct.rich_club_bd(A, klevel=kmax)
            return R
    elif weighted == False and directed == False:
        def rc_func(A, kmax):
            R, _, _ = bct.rich_club_bu(A, klevel=kmax)
            return R
    
    R = rc_func(A, kmax)

    if n_perms > 0:
        print('Running permutation...')
        R_perm = np.zeros((n_perms, kmax))
        for i in np.arange(n_perms):
            np.random.seed(i)
            A_rand, _ = randmio_func(A, itr=5)
            R_perm[i, :] = rc_func(A_rand, kmax)

    # compute normalized rich club coefficient
    if n_perms > 0:
        R_norm = np.divide(R, np.nanmean(R_perm, axis=0))

        # compute p values
        p_val = np.zeros(kmax)
        for i in np.arange(kmax):
            p_val[i] = np.nanmean(R[i] <= R_perm[:, i])

    if n_perms > 0:
        return degree, R, R_perm, R_norm, p_val
    else:
        return degree, R
