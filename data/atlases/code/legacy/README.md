# Legacy atlas code

These notebooks and scripts produced the atlases in the old layout (one folder per atlas,
`atlas-<name>_space-<template>_res-<r>_dseg.nii.gz`). They are kept as a record of where the
component atlases came from. They write the old layout and use hard-coded paths, so they are
not meant to be rerun as they are.

| File | Made | Status |
| --- | --- | --- |
| `schaefer_1_rename.ipynb` | Schaefer volumes (TemplateFlow) and fsLR dlabels (CBIG) | Outputs kept: `tpl-*/*_atlas-Schaefer<N>7_*`, `sourcedata/Schaefer2018/` |
| `msa_rename_relabel.ipynb` | MSA1–4, cut out of Tian's Schaefer100+subcortex volumes | Outputs kept: `tpl-*/*_atlas-MSA<S>_*` |
| `cerebellum.sh` | MDTB10, resampled with ANTs into TemplateFlow's grids | Outputs kept: `tpl-*/*_atlas-MDTB10_*` |
| `download_templateflow.sh` | Local TemplateFlow copy used by the above | — |
| `schaefer_2_tian_relabel.ipynb` | SchaeferMSA (Tian's combined volumes, renumbered cortex-first) | **Replaced** by `../build_atlases.py` |
| `schaefer_3_msa_mdtb.ipynb` | SchaeferMSA + MDTB10 | **Replaced**: see below |
| `glasser_msa_mdtb.ipynb` | GlasserMSA, GlasserMSA + MDTB10 (now `Glasser360MSA*`) | **Replaced**: see below |
| `combine_glasser_msa.py` | An earlier Glasser + MSA combination | **Replaced** |

The combining notebooks added the voxel arrays of their inputs without checking the images'
orientation. In MNI152NLin6Asym the MSA volumes are stored LAS and Glasser and MDTB10 RAS, so
in the MNI152NLin6Asym versions of GlasserMSA* the subcortex was mirrored left–right, and in
SchaeferMSA*MDTB10 the cerebellum was. (MNI152NLin2009cAsym was unaffected: all its inputs are
RAS.) `../build_atlases.py` combines in world space and checks every label's hemisphere.
