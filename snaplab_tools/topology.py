"""
Topology analysis functions for network neuroscience.

This module provides functions for analyzing network topology including
thresholding, normalization, and rich club analysis.
"""

import numpy as np
import bct
from typing import Tuple, Optional, Union

from snaplab_tools.utils import normalize_x

__all__ = [
    'threshold_adjacency',
    'volume_normalize_adjacency',
    'threshold_adjacency_consistency',
    'get_norm_rc',
]


def threshold_adjacency(
    A: np.ndarray,
    q: float = 0.8,
    use_abs: bool = True,
    fill_diag: bool = True,
    binarize: bool = True
) -> np.ndarray:
    """
    Threshold an adjacency matrix based on quantile values.
    
    Parameters
    ----------
    A : np.ndarray, shape (n_nodes, n_nodes)
        Input adjacency matrix
    q : float, default=0.8
        Quantile threshold (0-1). Edges below this quantile will be set to zero.
    use_abs : bool, default=True
        Whether to take absolute value of adjacency matrix before thresholding
    fill_diag : bool, default=True
        Whether to set diagonal elements to zero
    binarize : bool, default=True
        Whether to binarize the output (1 for edges above threshold, 0 otherwise)
        
    Returns
    -------
    A_out : np.ndarray, shape (n_nodes, n_nodes)
        Thresholded adjacency matrix
        
    Examples
    --------
    >>> A = np.random.randn(10, 10)
    >>> A_thresh = threshold_adj(A, q=0.9, binarize=True)
    """
    # Apply absolute value if requested
    A_proc = np.abs(A) if use_abs else A.copy()
    
    # Zero out diagonal
    if fill_diag:
        np.fill_diagonal(A_proc, 0)
    
    # Calculate threshold and create mask
    thresh = np.quantile(A_proc, q=q)
    mask = A_proc >= thresh
    
    # Apply threshold
    A_out = A_proc.copy()
    A_out[~mask] = 0
    
    # Binarize if requested
    if binarize:
        A_out[mask] = 1
    
    return A_out


def volume_normalize_adjacency(
    adjacency: np.ndarray,
    region_size: np.ndarray
) -> np.ndarray:
    """
    Normalize adjacency matrix by regional volume/size.
    
    This function normalizes edge weights by the average size of the connected
    regions, applies log transformation, and handles zero values appropriately.
    
    Parameters
    ----------
    adjacency : np.ndarray, shape (n_nodes, n_nodes)
        Input adjacency matrix with edge weights
    region_size : np.ndarray, shape (n_nodes,)
        Size/volume of each region
        
    Returns
    -------
    adjacency_norm : np.ndarray, shape (n_nodes, n_nodes)
        Volume-normalized adjacency matrix (log-transformed)
        
    Notes
    -----
    The normalization follows these steps:
    1. Normalize region sizes to [0, 1] and add small epsilon
    2. Create size matrix as average of connected region sizes
    3. Divide adjacency by size matrix
    4. Add 1 to non-zero elements
    5. Apply log transformation
    
    Examples
    --------
    >>> adjacency = np.random.rand(10, 10)
    >>> region_size = np.random.rand(10) * 1000  # region volumes
    >>> adj_norm = volume_normalize_adjacency(adjacency, region_size)
    """
    # Normalize region sizes to [0, 1] and add small constant to avoid division by zero
    region_size_norm = normalize_x(region_size) + 1e-5
    
    # Create matrix of average region sizes for each edge
    size_matrix = np.add.outer(region_size_norm, region_size_norm) / 2
    
    # Normalize adjacency by region size
    adjacency_norm = np.divide(adjacency, size_matrix)
    
    # Identify non-zero edges
    adjacency_mask = adjacency_norm > 0
    
    # Add 1 to non-zero elements before log transform
    adjacency_norm[adjacency_mask] += 1
    
    # Apply log transformation (only to non-zero elements)
    adjacency_norm = np.log(
        adjacency_norm,
        out=np.zeros_like(adjacency_norm),
        where=(adjacency_norm != 0)
    )
    
    return adjacency_norm


def threshold_adjacency_consistency(
    A: np.ndarray,
    thr: float = 0.60
) -> np.ndarray:
    """
    Threshold edges based on consistency across subjects.
    
    Retains edges that are non-zero in at least a specified proportion of
    subjects. Handles missing data (NaN values) appropriately.
    
    Parameters
    ----------
    A : np.ndarray, shape (n_nodes, n_nodes, n_subjects)
        Structural adjacency matrix with subjects along the third dimension.
        Can contain NaN values for missing data.
    thr : float, default=0.60
        Proportion threshold (0-1). Edges must be non-zero in at least this
        fraction of subjects to be retained in the output.
        
    Returns
    -------
    Am : np.ndarray, shape (n_nodes, n_nodes)
        Thresholded mean adjacency matrix. Edges present in fewer than `thr`
        proportion of subjects are set to zero.
        
    Notes
    -----
    - NaN values are treated as missing data and excluded from proportion calculations
    - The function computes the group average only for edges meeting the threshold
    - Edges with no valid (non-NaN) data across subjects will be set to zero
    
    Examples
    --------
    >>> A = np.random.rand(10, 10, 20)  # 10 nodes, 20 subjects
    >>> # Set some edges to zero in some subjects
    >>> A[A < 0.3] = 0
    >>> # Introduce some missing data
    >>> A[0, 1, :5] = np.nan
    >>> Am = threshold_consistency(A, thr=0.6)
    """
    # Compute group averaged adjacency matrix (ignoring NaNs)
    Am = np.nanmean(A, axis=2)
    
    # Create binary matrix indicating non-zero elements
    Ab = A > 0
    
    # Count non-zero elements across subjects for each edge
    Ab_count = np.sum(Ab, axis=2)
    
    # Count valid (non-NaN) subjects for each edge
    valid_count = np.sum(~np.isnan(A), axis=2)
    
    # Compute proportion of non-zero elements over valid subjects
    # Set proportion to 0 where valid_count is 0 (will be masked anyway)
    Ab_prop = np.divide(
        Ab_count,
        valid_count,
        out=np.zeros_like(Ab_count, dtype=float),
        where=valid_count > 0
    )
    
    # Create mask for edges below threshold
    mask = Ab_prop < thr
    
    # Set masked elements to zero in the mean adjacency matrix
    Am[mask] = 0
    
    return Am


def get_norm_rc(
    A: np.ndarray,
    n_perms: int = 1000,
    weighted: bool = True,
    directed: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Compute normalized rich club coefficients with permutation testing.
    
    Calculates rich club coefficients and normalizes them against a null
    distribution generated from randomized networks that preserve the degree
    sequence.
    
    Parameters
    ----------
    A : np.ndarray, shape (n_nodes, n_nodes)
        Adjacency matrix (weighted or binary, directed or undirected)
    n_perms : int, default=1000
        Number of permutations for null distribution. Set to 0 to skip
        permutation testing and return only unnormalized coefficients.
    weighted : bool, default=True
        Whether the network is weighted. If False, treated as binary.
    directed : bool, default=False
        Whether the network is directed.
        
    Returns
    -------
    If n_perms > 0:
        degree : np.ndarray, shape (n_nodes,)
            Degree of each node
        R : np.ndarray, shape (kmax,)
            Rich club coefficients for each k-level
        R_perm : np.ndarray, shape (n_perms, kmax)
            Rich club coefficients from permuted networks
        R_norm : np.ndarray, shape (kmax,)
            Normalized rich club coefficients (R / mean(R_perm))
        p_val : np.ndarray, shape (kmax,)
            P-values for each k-level (proportion of permutations with R >= observed)
            
    If n_perms == 0:
        degree : np.ndarray, shape (n_nodes,)
            Degree of each node
        R : np.ndarray, shape (kmax,)
            Rich club coefficients for each k-level
    
    Notes
    -----
    - Uses Brain Connectivity Toolbox (BCT) functions for calculations
    - Permuted networks preserve the degree sequence using the Maslov-Sneppen algorithm
    - kmax is determined as the maximum degree in the network
    - P-values are computed as the proportion of permutations with coefficients
      greater than or equal to the observed value
    
    Examples
    --------
    >>> A = np.random.rand(50, 50)
    >>> A = (A + A.T) / 2  # Make symmetric
    >>> degree, R, R_perm, R_norm, p_val = get_norm_rc(A, n_perms=100)
    >>> # Find significant rich club levels (p < 0.05)
    >>> sig_levels = np.where(p_val < 0.05)[0]
    """
    # Select appropriate functions based on network properties
    if directed:
        def deg_func(A):
            _, _, degree = bct.degrees_dir(A)
            return degree
        randmio_func = bct.randmio_dir
    else:
        deg_func = bct.degrees_und
        randmio_func = bct.randmio_und
    
    # Calculate degree and maximum degree
    degree = deg_func(A)
    kmax = int(np.max(degree))
    print(f"Maximum degree (kmax): {kmax}")
    
    # Select rich club function based on network properties
    if weighted and directed:
        def rc_func(A, kmax):
            return bct.rich_club_wd(A, klevel=kmax)
    elif weighted and not directed:
        def rc_func(A, kmax):
            return bct.rich_club_wu(A, klevel=kmax)
    elif not weighted and directed:
        def rc_func(A, kmax):
            R, _, _ = bct.rich_club_bd(A, klevel=kmax)
            return R
    else:  # binary undirected
        def rc_func(A, kmax):
            R, _, _ = bct.rich_club_bu(A, klevel=kmax)
            return R
    
    # Compute rich club coefficients for observed network
    R = rc_func(A, kmax)
    
    # Perform permutation testing if requested
    if n_perms > 0:
        print(f'Running {n_perms} permutations...')
        R_perm = np.zeros((n_perms, kmax))
        
        for i in range(n_perms):
            np.random.seed(i)
            # Generate randomized network preserving degree sequence
            A_rand, _ = randmio_func(A, itr=5)
            R_perm[i, :] = rc_func(A_rand, kmax)
        
        # Compute normalized rich club coefficients
        R_norm = np.divide(R, np.nanmean(R_perm, axis=0))
        
        # Compute p-values (proportion of permutations >= observed)
        p_val = np.zeros(kmax)
        for k in range(kmax):
            p_val[k] = np.nanmean(R[k] <= R_perm[:, k])
        
        return degree, R, R_perm, R_norm, p_val
    else:
        return degree, R
