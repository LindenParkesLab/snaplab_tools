## combine_glasser_msa.py 
## Appends the MSA subcortical nifti atlas to the Glasser cortical atlas. Voxels that have labels in both atlases (e.g. the hippocampus) will only have the MSA subcortical label.  
## FSL was used to first flip the x axis of the subcortical nifti file to ensure 
##
## 241107 Created by Amber Howell (Rutgers University)


import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np

data_loc = "/Users/ah2252/Downloads/"

## - Load the two NIfTI files
nifti_cortical = nib.load(data_loc + "atlas-Glasser_space-MNI152NLin6Asym_res-01_dseg.nii.gz")
nifti_subcortical = nib.load(data_loc + "Tian_Subcortex_S4_3T_1mm_flipped.nii.gz") 
## - Note: The MSA nifti file has the x axis flipped so we need to flip it using FSL before running this code 
## - fslswapdim Tian_Subcortex_S4_3T_1mm.nii -x y z Tian_Subcortex_S4_3T_1mm_flipped.nii 
## - to check if the nifti files are aligned use fslhd on both nifti files and make sure the xyz matrices match between the niftis. 


## - Load in the cortical and flipped subcortical nifti files
data_cortical = nifti_cortical.get_fdata()
data_subcortical = nifti_subcortical.get_fdata()

## -
data_subcortical_relabel = np.zeros_like(data_subcortical)
data_subcortical_relabel[data_subcortical>0] =data_subcortical[data_subcortical>0]+np.unique(data_cortical)[-1]

## - Output nifti file will have all of the subcortical labels and cortical labels (subcortical labels take precedence over the cortical labels)
combined_data = np.copy(data_cortical)
combined_data[data_subcortical_relabel>0] =data_subcortical_relabel[data_subcortical_relabel>0]

## - Create and save a new NIfTI image with the combined data
combined_nifti = nib.Nifti1Image(combined_data, affine=nifti_cortical.affine, header=nifti_cortical.header)
nib.save(combined_nifti, data_loc + "atlas-Glasser_space-MNI152NLin6Asym_res-01_dseg_Tian_Subcortex_S4_3T_1mm_flipped_drop-duplicate-glasser-labels.nii")


