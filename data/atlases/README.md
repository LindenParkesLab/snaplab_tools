# Atlases Directory

This directory contains brain atlas data and scripts related to the Glasser and MSA (Multi-Scale Atlas) atlases. The contents include specific atlases and a Python script for combining data from these atlases.

## Folder Contents

- `Glasser/`: Contains files related to the Glasser atlas, which provides a parcellation of the cerebral cortex based on multimodal MRI data. Files were taken directly from https://github.com/PennLINC/xcp_d/tree/main/xcp_d/data/atlases/atlas-Glasser. 
- `MSA/`: Contains data for the Melbourne Subcortical Atlas (MSA), which provides parcellations at multiple spatial scales of the subcortex. The files were taken directly from https://github.com/yetianmed/subcortex/tree/master/Group-Parcellation/3T/Subcortex-Only and the 1mm nifti file was used in the combined Glasser/MSA atlas.
- `GlasserMSA/`: Includes files combining the Glasser atlas with the MSA, created using the combine_glasser_msa.py script. For voxels that had labels in Glasser and MSA (duplicate ROIs; e.g. hippocampus) the Glasser label was dropped. In other words, in ROIs like the hippocampus, voxels that were present in both atlases will only be labeled with MSAl labels. This also means that the Glasser hippocampus labels (e.g. id 120) will only be the partial Glasser ROI.  
- `combine_glasser_msa.py`: A Python script for combining data from the Glasser and MSA atlases. It can be used to generate a whole-brain atlas file or other combined outputs.

