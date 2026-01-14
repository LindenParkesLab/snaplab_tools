import os, wget
import numpy as np
import pandas as pd
import nibabel as nib
from statsmodels.stats import multitest
from sklearn.linear_model import LinearRegression

def normalize_x(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def get_schaefer_system_mask(roi_names, system='Vis'):
    n_parcels = len(roi_names)
    system_mask = np.zeros((n_parcels,)).astype(bool)
    for roi in np.arange(n_parcels):
        if system in roi_names[roi]:
            system_mask[roi] = True

    return system_mask


# The exponential decay function
def exp_decay(x, tau, init):
    return init*np.e**(-x/tau)


def get_parcelwise_average_nifti(data_file, parc_file):
    # load parcellation
    parc = nib.load(parc_file).get_fdata().squeeze()
    unique_labels = np.unique(parc)

    file_name, file_extension = os.path.splitext(data_file)

    # load gifti file
    if file_extension == '.nii' or file_extension == '.gz':
        data = nib.load(data_file)
        data = data.get_fdata().squeeze()

    # mean over labels
    data_mean = []
    for i in unique_labels:
        data_mean.append(np.mean(data[parc == i]))

    return np.asarray(data_mean)


def get_parcelwise_average_surface(data_file, annot_file):
    # load parcellation
    labels, ctab, surf_names = nib.freesurfer.read_annot(annot_file)
    unique_labels = np.unique(labels)

    file_name, file_extension = os.path.splitext(data_file)

    # load gifti file
    if file_extension == '.gii':
        data = nib.load(data_file)
        data = data.darrays[0].data
    elif file_extension == '.mgh':
        data = nib.load(data_file)
        data = data.get_fdata().squeeze()
    elif file_extension == '.curv':
        data = nib.freesurfer.io.read_morph_data(data_file)
    elif file_extension == '.txt':
        data = np.loadtxt(data_file)

    # mean over labels
    data_mean = []
    for i in unique_labels:
        data_mean.append(np.mean(data[labels == i]))

    return np.asarray(data_mean)


def load_schaefer_parc(n_parcels=200, order=17, annot='fsaverage', out_dir='~/schaefer_parc'):
    # output dir
    out_dir = os.path.expanduser(out_dir)
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    # github link
    remote_path = 'https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal'

    # nifti image in mni
    file = 'Schaefer2018_{0}Parcels_{1}Networks_order_FSLMNI152_1mm.nii.gz'.format(n_parcels, order)
    interim_dir = 'Parcellations/MNI'
    if os.path.exists(os.path.join(out_dir, file)) == False:
        wget.download(os.path.join(remote_path, interim_dir, file), out_dir)
    nifti_file = os.path.join(out_dir, file)

    # roi names and coords in MNI
    file = 'Schaefer2018_{0}Parcels_{1}Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'.format(n_parcels, order)
    interim_dir = 'Parcellations/MNI/Centroid_coordinates'
    if os.path.exists(os.path.join(out_dir, file)) == False:
        wget.download(os.path.join(remote_path, interim_dir, file), out_dir)
    centroids = pd.read_csv(os.path.join(out_dir, file), index_col=0)

    # annotation files for fsaverage
    if os.path.exists(os.path.join(out_dir, annot)) == False:
        os.makedirs(os.path.join(out_dir, annot))
    interim_dir = 'Parcellations/FreeSurfer5.3/{0}/label'.format(annot)
    files = ['lh.Schaefer2018_{0}Parcels_{1}Networks_order.annot'.format(n_parcels, order),
             'rh.Schaefer2018_{0}Parcels_{1}Networks_order.annot'.format(n_parcels, order)]
    for file in files:
        if os.path.exists(os.path.join(out_dir, annot, file)) == False:
            wget.download(os.path.join(remote_path, interim_dir, file), os.path.join(out_dir, annot))
    lh_annot_file = os.path.join(out_dir, annot, files[0])
    rh_annot_file = os.path.join(out_dir, annot, files[1])

    # cifti file for HCP space (32k fs_LR)
    # file = 'Schaefer2018_{0}Parcels_{1}Networks_order.dscalar.nii'.format(n_parcels, order)
    file = 'Schaefer2018_{0}Parcels_{1}Networks_order.dlabel.nii'.format(n_parcels, order)
    interim_dir = 'Parcellations/HCP/fslr32k/cifti'
    if os.path.exists(os.path.join(out_dir, file)) == False:
        wget.download(os.path.join(remote_path, interim_dir, file), out_dir)
    hcp_file = os.path.join(out_dir, file)

    return nifti_file, centroids, lh_annot_file, rh_annot_file, hcp_file


def schaefer_ordering_mapper(out_dir='~/schaefer_ordering_mapper',
                             n_parcels=400, input_order=17, output_order=7):

    # output dir
    out_dir = os.path.expanduser(out_dir)
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    # download data from Yeo lab github
    # github link
    remote_path = 'https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI'

    # download schaefer files
    for order in [input_order, output_order]:
        # centroids and labels
        file = 'Schaefer2018_{0}Parcels_{1}Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'.format(n_parcels, order)
        if os.path.exists(os.path.join(out_dir, file)) == False:
            wget.download(os.path.join(remote_path, 'Centroid_coordinates', file), out_dir)

    # load centroids
    file = 'Schaefer2018_{0}Parcels_{1}Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'.format(n_parcels, input_order)
    centroids_inorder = pd.read_csv(os.path.join(out_dir, file), index_col=0)

    file = 'Schaefer2018_{0}Parcels_{1}Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'.format(n_parcels, output_order)
    centroids_outorder = pd.read_csv(os.path.join(out_dir, file), index_col=0)

    # remap data
    # mapped = pd.DataFrame(index=centroids_inorder.index, columns=centroids_inorder.columns)
    mapped = pd.DataFrame(index=centroids_inorder.index)
    for i, data in centroids_outorder.iterrows():
        # get coords to be matched from output order
        coords = [data['R'], data['A'], data['S']]

        # find index of matching node from input order
        idx = np.where(np.all(coords == centroids_inorder[['R', 'A', 'S']].values, axis=1))[0][0]

        # store
        mapped.loc[i, 'input_roi'] = centroids_inorder.loc[idx + 1, 'ROI Name']
        mapped.loc[i, 'output_roi'] = centroids_outorder.loc[i, 'ROI Name']
        mapped.loc[i, 'mapped_indices'] = idx

    mapped['mapped_indices'] = mapped['mapped_indices'].astype(int)

    # save out
    mapped.to_csv(os.path.join(out_dir, 'Schaefer_{0}-{1}_mappedto_{0}-{2}.csv'.format(
        n_parcels, input_order, output_order)))

    return mapped


def get_fdr_p(p_vals, alpha=0.05):
    if p_vals.ndim == 2:
        do_reshape = True
        dims = p_vals.shape
        p_vals = p_vals.flatten()
    else:
        do_reshape = False

    out = multitest.multipletests(p_vals, alpha=alpha, method='fdr_bh')
    p_fdr = out[1]

    if do_reshape:
        p_fdr = p_fdr.reshape(dims)

    return p_fdr


def winsorize(data, lower_percentile=1, upper_percentile=100):
    """
    Winsorize data by clipping values at specified percentiles.
    
    Parameters
    ----------
    data : np.ndarray or pd.Series
        Data to winsorize
    lower_percentile : float
        Lower percentile threshold (default: 1)
    upper_percentile : float
        Upper percentile threshold (default: 99)
    
    Returns
    -------
    np.ndarray or pd.Series
        Winsorized copy of the data
    """
    if isinstance(data, pd.Series):
        values = data.values.copy()
        lower_bound = np.nanpercentile(values, lower_percentile)
        upper_bound = np.nanpercentile(values, upper_percentile)
        winsorized = np.clip(values, lower_bound, upper_bound)
        return pd.Series(winsorized, index=data.index, name=data.name)
    else:
        values = np.array(data).copy()
        lower_bound = np.nanpercentile(values, lower_percentile)
        upper_bound = np.nanpercentile(values, upper_percentile)
        return np.clip(values, lower_bound, upper_bound)


def winsorize_iqr(vector, k=1.5, inplace=False):
    """
    Winsorizes a vector using the IQR method to handle outliers.
    
    Parameters:
    -----------
    vector : array-like
        Input data to be winsorized (list, numpy array, or pandas Series)
    k : float, optional (default=1.5)
        Multiplier for IQR to determine outlier thresholds
    inplace : bool, optional (default=False)
        If True, modifies the input vector in place (only works with mutable input)
        
    Returns:
    --------
    winsorized_vector : numpy array
        Winsorized version of the input vector
    """
    # Convert input to numpy array if it isn't already
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)
    
    # Calculate quartiles and IQR
    q1 = np.percentile(vector, 25)
    q3 = np.percentile(vector, 75)
    iqr = q3 - q1
    
    # Calculate lower and upper bounds
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    # Create a copy unless inplace is True and input is mutable
    if inplace and isinstance(vector, np.ndarray):
        winsorized_vector = vector
    else:
        winsorized_vector = vector.copy()
    
    # Winsorize the values
    winsorized_vector[winsorized_vector < lower_bound] = lower_bound
    winsorized_vector[winsorized_vector > upper_bound] = upper_bound
    
    return winsorized_vector


def nuis_reg(X, y, use_sklearn=False):
    if use_sklearn:
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]

        regr = LinearRegression()
        regr.fit(X, y)
        predicted = regr.predict(X)
        residuals = y - predicted
    else:
        beta = np.dot(np.linalg.pinv(X), y)
        predicted = np.dot(X, beta)
        residuals = y - predicted
    
    return residuals
