import os, wget
import numpy as np
import pandas as pd
import nibabel as nib

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
