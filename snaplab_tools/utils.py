"""Parcellation fetching, parcel-wise averaging, and small numerics.

Parcellations
    :func:`load_schaefer_parc` and :func:`schaefer_ordering_mapper` download Schaefer2018
    parcellation files from the Yeo lab's CBIG repository and cache them on disk;
    :func:`get_schaefer_system_mask` selects a Yeo system by name.

Parcel-wise averaging
    :func:`get_parcelwise_average_nifti` and :func:`get_parcelwise_average_surface` reduce
    volumetric or surface data to one value per parcel.

Numerics
    :func:`normalize_x` rescales to [0, 1]; :func:`exp_decay` is the model function to fit when
    estimating a decay timescale.

Note that the parcellation fetchers hit the network on first call. If you only need Schaefer
centroids or geodesic distances, :mod:`snaplab_tools.nulls` ships those offline.

Statistics that used to live here -- FDR correction, winsorizing, nuisance regression -- are now
in :mod:`snaplab_tools.stats`, alongside the rest of them.
"""
import os, wget
import numpy as np
import pandas as pd
import nibabel as nib

__all__ = [
    'normalize_x',
    'get_schaefer_system_mask',
    'exp_decay',
    'get_parcelwise_average_nifti',
    'get_parcelwise_average_surface',
    'load_schaefer_parc',
    'schaefer_ordering_mapper',
]


def normalize_x(x):
    """Rescale an array to the unit interval via min-max normalization.

    Parameters
    ----------
    x : array-like
        Values to rescale. A constant array yields a divide-by-zero.

    Returns
    -------
    ndarray
        ``(x - min(x)) / (max(x) - min(x))``, so the output spans exactly [0, 1].
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def get_schaefer_system_mask(roi_names, system='Vis'):
    """Boolean mask selecting the Schaefer parcels belonging to one Yeo system.

    Matching is a plain substring test against each ROI name, so `system` must appear verbatim
    in the Schaefer naming scheme (e.g. 'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic',
    'Cont', 'Default' for the 7-network order).

    Parameters
    ----------
    roi_names : sequence of str
        Parcel names, in parcellation order (e.g. '7Networks_LH_Vis_1').
    system : str
        Substring identifying the system.

    Returns
    -------
    (n_parcels,) ndarray of bool
        True where the parcel name contains `system`.
    """
    n_parcels = len(roi_names)
    system_mask = np.zeros((n_parcels,)).astype(bool)
    for roi in np.arange(n_parcels):
        if system in roi_names[roi]:
            system_mask[roi] = True

    return system_mask


# The exponential decay function
def exp_decay(x, tau, init):
    """Exponential decay ``init * exp(-x / tau)``.

    Intended as the model function passed to a curve fitter (e.g. ``scipy.optimize.curve_fit``)
    when estimating a decay timescale -- fitting it to an autocorrelation function recovers tau
    as the intrinsic timescale.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate (e.g. autocorrelation lags).
    tau : float
        Decay constant, in the same units as `x`. Larger tau decays more slowly.
    init : float
        Value at ``x = 0``.

    Returns
    -------
    ndarray
        The decay evaluated at `x`.
    """
    return init*np.e**(-x/tau)


def get_parcelwise_average_nifti(data_file, parc_file):
    """Average volumetric data within each parcel of a NIfTI parcellation.

    Parameters
    ----------
    data_file : str
        Path to the data volume, '.nii' or '.nii.gz'.
    parc_file : str
        Path to the parcellation volume, in the same space and resolution as `data_file`.

    Returns
    -------
    (n_labels,) ndarray
        Mean of the data over each unique label in the parcellation, ordered by sorted label
        value. Label 0 (background) is included, so you will usually want to drop the first
        element.

    Raises
    ------
    ValueError
        If `data_file` does not have a recognised extension.
    """
    # Validated up front, before any file is read: an unrecognised extension used to fall through
    # and leave `data` undefined, so the failure surfaced as an UnboundLocalError several lines
    # later -- after the parcellation had already been loaded.
    file_name, file_extension = os.path.splitext(data_file)
    if file_extension not in ('.nii', '.gz'):
        raise ValueError(
            f"Unsupported data file extension {file_extension!r} for {data_file!r}; "
            f"expected '.nii' or '.nii.gz'."
        )

    # load parcellation
    parc = nib.load(parc_file).get_fdata().squeeze()
    unique_labels = np.unique(parc)

    data = nib.load(data_file)
    data = data.get_fdata().squeeze()

    # mean over labels
    data_mean = []
    for i in unique_labels:
        data_mean.append(np.mean(data[parc == i]))

    return np.asarray(data_mean)


def get_parcelwise_average_surface(data_file, annot_file):
    """Average surface data within each parcel of a FreeSurfer annotation.

    Parameters
    ----------
    data_file : str
        Path to per-vertex data. Accepted extensions: '.gii' (first data array), '.mgh',
        '.curv' (FreeSurfer morphometry), '.txt' (whitespace-delimited).
    annot_file : str
        FreeSurfer '.annot' file for the same hemisphere and surface as `data_file`.

    Returns
    -------
    (n_labels,) ndarray
        Mean of the data over each unique label, ordered by sorted label value. The medial
        wall (label 0) is included, so you will usually want to drop the first element.

    Raises
    ------
    ValueError
        If `data_file` does not have a recognised extension.
    """
    # Validated up front, before any file is read: an unrecognised extension used to fall through
    # and leave `data` undefined, so the failure surfaced as an UnboundLocalError several lines
    # later -- after the annotation had already been loaded.
    file_name, file_extension = os.path.splitext(data_file)
    if file_extension not in ('.gii', '.mgh', '.curv', '.txt'):
        raise ValueError(
            f"Unsupported data file extension {file_extension!r} for {data_file!r}; "
            f"expected one of '.gii', '.mgh', '.curv', '.txt'."
        )

    # load parcellation
    labels, ctab, surf_names = nib.freesurfer.read_annot(annot_file)
    unique_labels = np.unique(labels)

    if file_extension == '.gii':
        data = nib.load(data_file)
        data = data.darrays[0].data
    elif file_extension == '.mgh':
        data = nib.load(data_file)
        data = data.get_fdata().squeeze()
    elif file_extension == '.curv':
        data = nib.freesurfer.io.read_morph_data(data_file)
    else:  # '.txt', the only remaining option after the check above
        data = np.loadtxt(data_file)

    # mean over labels
    data_mean = []
    for i in unique_labels:
        data_mean.append(np.mean(data[labels == i]))

    return np.asarray(data_mean)


def load_schaefer_parc(n_parcels=200, order=17, annot='fsaverage', out_dir='~/schaefer_parc'):
    """Download (and cache) a Schaefer2018 parcellation in volume, surface, and CIFTI form.

    Files are fetched from the Yeo lab's CBIG GitHub repository into `out_dir` on first call and
    reused thereafter, so only the first call needs network access.

    Parameters
    ----------
    n_parcels : int
        Parcellation resolution (100, 200, ..., 1000).
    order : int
        Yeo network order, 7 or 17.
    annot : str
        FreeSurfer surface for the annotation files, 'fsaverage' or 'fsaverage5'.
    out_dir : str
        Cache directory; '~' is expanded and the directory is created if absent.

    Returns
    -------
    nifti_file : str
        Path to the parcellation in MNI152 1mm volumetric space.
    centroids : pandas.DataFrame
        Parcel names and R/A/S centroid coordinates.
    lh_annot_file, rh_annot_file : str
        Paths to the left and right FreeSurfer annotation files.
    hcp_file : str
        Path to the '.dlabel.nii' CIFTI parcellation in fs_LR 32k space.
    """
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
    """Build an index map between the 7- and 17-network orderings of a Schaefer parcellation.

    The two Yeo orderings cover the same parcels in a different sequence. Matching is done on
    exact R/A/S centroid coordinates, which are identical across orderings at a given
    resolution. Use ``mapped['mapped_indices']`` to reindex a data vector from `input_order`
    into `output_order`::

        mapping = schaefer_ordering_mapper(n_parcels=400, input_order=17, output_order=7)
        data_7 = data_17[mapping['mapped_indices'].values]

    Centroid files are downloaded to `out_dir` on first call, and the resulting map is written
    there as a CSV.

    Parameters
    ----------
    out_dir : str
        Cache/output directory; '~' is expanded and the directory is created if absent.
    n_parcels : int
        Parcellation resolution, the same for both orderings.
    input_order, output_order : int
        Yeo network orders to map from and to (7 or 17).

    Returns
    -------
    pandas.DataFrame
        Indexed by output-order parcel, with columns 'input_roi', 'output_roi', and
        'mapped_indices' (0-based positions into the input-order vector).
    """
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

