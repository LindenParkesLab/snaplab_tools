#!/bin/bash

# Download and transform the MDTB atlas from
# King, M., Hernandez-Castillo, C.R., Poldrack, R.R., Ivry, R., and Diedrichsen, J. (2019).
# Functional Boundaries in the Human Cerebellum revealed by a Multi-Domain Task Battery. Nat. Neurosci.

DOWNLOAD_DIR=/home/lindenmp/atlases/MDTB10
rm -rf ${DOWNLOAD_DIR}
mkdir -p ${DOWNLOAD_DIR}
cd ${DOWNLOAD_DIR}

# Get the atlas description
wget https://github.com/DiedrichsenLab/cerebellar_atlases/raw/master/King_2019/atlas_description.json
mv atlas_description.json dataset_description.json

# Get the labels (already in TSV!!)
wget https://github.com/DiedrichsenLab/cerebellar_atlases/raw/master/King_2019/atl-MDTB10.tsv
mv atl-MDTB10.tsv atlas-MDTB10_dseg.tsv
sed -i -e 's/name/label/' atlas-MDTB10_dseg.tsv

# Get the actual atlas
wget https://github.com/DiedrichsenLab/cerebellar_atlases/raw/master/King_2019/atl-MDTB10_space-MNI_dseg.nii

# This is in a strange volume, but in coordinate space it's
# "..._space-MNI.nii: volume file aligned to FNIRT MNI space", which is NLin6Asym.

TEMPLATEFLOW_HOME=/home/lindenmp/templateflow
# Resample it into the official TemplateFlow volume
for RES in 1 2; do
    # To NLin6Asym (no transform)
    antsApplyTransforms \
        -d 3 \
        -i atl-MDTB10_space-MNI_dseg.nii \
        -o atlas-MDTB10_space-MNI152NLin6Asym_res-0${RES}_dseg.nii.gz \
        -r ${TEMPLATEFLOW_HOME}/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-0${RES}_desc-brain_mask.nii.gz \
        --interpolation GenericLabel \
        -v 1

    # To 2009cAsym
    antsApplyTransforms \
        -d 3 \
        -i atl-MDTB10_space-MNI_dseg.nii \
        -o atlas-MDTB10_space-MNI152NLin2009cAsym_res-0${RES}_dseg.nii.gz \
        -t ${TEMPLATEFLOW_HOME}/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5 \
        -r ${TEMPLATEFLOW_HOME}/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-0${RES}_desc-brain_mask.nii.gz \
        --interpolation GenericLabel \
        -v 1
        # -o tpl-MNI152NLin2009cAsym_atlas-MDTB10_res-01_dseg.nii.gz  \
done

rm atl-MDTB10_space-MNI_dseg.nii