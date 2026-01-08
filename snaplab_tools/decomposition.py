import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy import stats


def pca_with_nan_handling(data, n_components=None, standardize=False, 
                          return_full_shape=False, impute=False):
    """
    Perform PCA with NaN handling.
    
    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Input data where samples are in rows and features in columns.
        For your case: (n_regions, n_measures)
    n_components : int, float, or None
        Number of components to keep. If None, keeps all components.
        If float between 0 and 1, selects number of components such that
        the explained variance is greater than the percentage specified.
    standardize : bool, default=False
        If True, z-score the data (per feature) before PCA.
    return_full_shape : bool, default=False
        If True, returns scores in original shape with NaNs reinserted.
        If False, returns scores only for valid samples.
        Only relevant when impute=False.
    impute : bool, default=False
        If True, impute NaN values using median of each column.
        If False, remove rows containing any NaNs.
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'scores': PC scores, shape (n_valid_samples, n_components) or 
                    (n_samples, n_components) if return_full_shape=True or impute=True
        - 'loadings': PC loadings (weights), shape (n_features, n_components)
        - 'explained_variance': Variance explained by each component
        - 'explained_variance_ratio': Proportion of variance explained
        - 'cumulative_variance_ratio': Cumulative proportion of variance
        - 'pca_object': Fitted PCA object
        - 'valid_mask': Boolean mask of valid (non-NaN) samples (None if impute=True)
        - 'mean': Mean of each feature (after NaN handling, before PCA)
        - 'std': Std of each feature if standardized, else None
        - 'imputer': SimpleImputer object if impute=True, else None
    """
    if impute:
        # Impute NaNs using median
        imputer = SimpleImputer(strategy='median')
        data_clean = imputer.fit_transform(data)
        valid_mask = None
    else:
        # Identify rows without NaNs
        valid_mask = ~np.isnan(data).any(axis=1)
        n_valid = valid_mask.sum()
        
        if n_valid == 0:
            raise ValueError("All samples contain NaNs")
        
        # Extract valid data
        data_clean = data[valid_mask]
        imputer = None
    
    # Store mean for centering information
    mean = np.mean(data_clean, axis=0)
    std = None
    
    # Optional standardization
    if standardize:
        data_clean = stats.zscore(data_clean, axis=0, ddof=1)
        std = np.std(data_clean, axis=0, ddof=1)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data_clean)
    
    # Get loadings (components_ is transposed)
    loadings = pca.components_.T
    
    # Optionally return full shape with NaNs (only when not imputing)
    if not impute and return_full_shape:
        scores_full = np.full((data.shape[0], scores.shape[1]), np.nan)
        scores_full[valid_mask] = scores
        scores = scores_full
    
    # Calculate cumulative variance
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    results = {
        'scores': scores,
        'loadings': loadings,
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': cumulative_variance_ratio,
        'pca_object': pca,
        'valid_mask': valid_mask,
        'mean': mean,
        'std': std,
        'imputer': imputer
    }
    
    return results
