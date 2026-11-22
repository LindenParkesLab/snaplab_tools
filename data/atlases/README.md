# SNaP Lab atlases

The lab's parcellations, as one [BIDS atlas dataset](https://bids-specification.readthedocs.io/)
(BIDS 1.11, `DatasetType: derivative`). Pipelines should take their parcellations from here.
XCP-D 26+ reads it directly (`--datasets snaplab=<this folder>`).

## Layout

```text
data/atlases/
├── dataset_description.json
├── atlas-<name>_description.json      # per atlas: description, license, references, sources
├── tpl-MNI152NLin6Asym/               # tpl-MNI152NLin6Asym_atlas-<name>_res-0{1,2}_dseg.{nii.gz,tsv,json}
├── tpl-MNI152NLin2009cAsym/           # same
├── tpl-fsLR/                          # tpl-fsLR_atlas-<name>_den-{32k,91k}_dseg.{dlabel.nii,tsv,json}
├── sourcedata/                        # inputs that aren't atlases here (see below)
└── code/
    ├── build_atlases.py               # builds the combined atlases and all fsLR files, and checks them
    ├── import_yan2023.py              # makes the Yan<N>7 components from CBIG's release
    ├── fetch_atlaspack.sh             # gets the 4S atlases (AtlasPack), for pipelines other than XCP-D
    └── legacy/                        # the notebooks that made the component atlases (old layout)
```

Every image has a TSV (`index`, `name`, `hemisphere`, `cortex`, `subcortex`, `cerebellum`,
plus source-specific columns) and a JSON sidecar with the same entities. Volumes are 3D int16
label images with qform = sform.

**TSV columns XCP-D reads.** XCP-D matches CIFTI parcels to the TSV by `name`, unless the TSV
has a `cifti_label` or `label_7network` column, which it then matches through instead. None of
the TSVs here has either (Glasser's HCP names, e.g. `R_V1_ROI`, are in `hcp_label`).

**fsLR densities.** Cortex-only atlases are `den-32k`: 59,412 surface vertices (32k fsLR,
medial wall excluded). Atlases with subcortex or cerebellum are `den-91k`: the standard 91,282
grayordinates, the same as fMRIPrep's `--cifti-output 91k`, with subcortical and cerebellar
labels sampled from the MNI152NLin6Asym 2 mm volume.

## Atlases

| Atlas | Parcels | What |
| --- | --- | --- |
| `Schaefer<N>7`, N = 100, 200, 400, 600, 800, 1000 | N | Schaefer 2018 cortex, 7-network order |
| `Yan<N>7`, N = 100, 200, 400, 600, 800, 1000 | N | Yan 2023 homotopic cortex, Yeo 7-network labels: left parcel i and right parcel i + N/2 are homotopic |
| `Glasser360` | 360 | HCP-MMP1.0 cortex (Glasser 2016). Not `Glasser`: XCP-D has a built-in atlas of that name and stops when two atlas sets share one |
| `MSA<S>`, S = 1–4 | 16, 32, 50, 54 | Melbourne Subcortex Atlas (Tian 2020), 3T, scales S1–S4 |
| `MDTB10` | 10 | MDTB cerebellum (King 2019) |
| `Schaefer<N>7MSA<S>`, N = 100, 200, 400, 800 | N + MSA | cortex + subcortex |
| `Schaefer<N>7MSA<S>MDTB10` | N + MSA + 10 | cortex + subcortex + cerebellum |
| `Yan<N>7MSA<S>`, N = 100, 200, 400, 800 | N + MSA | cortex + subcortex |
| `Yan<N>7MSA<S>MDTB10` | N + MSA + 10 | cortex + subcortex + cerebellum |
| `Glasser360MSA<S>` | 360 + MSA | cortex + subcortex |
| `Glasser360MSA<S>MDTB10` | 360 + MSA + 10 | cortex + subcortex + cerebellum |

In the combined atlases, labels are numbered cortex first, then MSA, then MDTB10 (whose names
get a `cerebellum_` prefix). Where the components overlap, MSA overrides cortex and MDTB10
overrides both.

**Schaefer names.** `name` holds CBIG's current parcel names (as in AtlasPack and Tian's
releases). The TemplateFlow volumes came with older names for some parcels (e.g. `FrOper` for
`FrOperIns`, `PCC` for `pCunPCC`); those are in `templateflow_name`. Same parcels, same
indices.

**Yan atlases.** CBIG's release has Yeo 7, Yeo 17 and Kong 17 versions with the same parcels,
numbered differently. Only the Yeo 7 version is an atlas here; the TSVs give each parcel's
`yeo17_index`/`yeo17_name` and `kong17_index`/`kong17_name`, and its `homotopic_index`. CBIG
releases no MNI152NLin2009cAsym volumes: `code/import_yan2023.py` resamples the
MNI152NLin6Asym ones with TemplateFlow's transform (the one fMRIPrep uses). Resampled the same
way, Schaefer agrees with TemplateFlow's own MNI152NLin2009cAsym Schaefer on 92-97% of voxels
(1 mm) and 96-99% (2 mm), depending on scale. Direction checked: the transform raises the
correlation of TemplateFlow's MNI152NLin6Asym T1w with its MNI152NLin2009cAsym T1w (within the
brain) from 0.80 to 0.90. (TemplateFlow's file named for the opposite direction is not its
inverse: it holds nearly the same mapping.)

**4S atlases** (Schaefer + CIT168 subcortex, thalamus, hippocampus, amygdala, cerebellum) are
not kept here. XCP-D and QSIRecon include them; for other pipelines, run
`code/fetch_atlaspack.sh`, which puts PennLINC's AtlasPack (checksum-verified) in
`data/AtlasPack/` (gitignored). Don't pass both AtlasPack and XCP-D's own copy to XCP-D: it
stops when two atlas datasets contain the same atlas.

## Rebuilding

```bash
python code/build_atlases.py     # nibabel, numpy, pandas; about 2-10 minutes
```

The `Yan<N>7` components come from `sourcedata/Yan2023/` and only need remaking if that changes
(before `build_atlases.py`; also needs scipy and nitransforms, and downloads TemplateFlow's
205 MB MNI152NLin6Asym -> MNI152NLin2009cAsym transform unless given `--xfm`; about 10 minutes):

```bash
python code/import_yan2023.py [--xfm tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5]
```

It rebuilds the combined atlases and all fsLR files from the components and `sourcedata/`,
rewrites the TSVs, sidecars and descriptions, and checks the result. It stops with an error if
any check fails:

- the inputs' voxel grids correspond voxel for voxel (combining happens in world space, never on
  raw arrays);
- every label in a TSV is present in its image, and nothing else is;
- labels named for a hemisphere lie in it: volumes ≥ 90% of voxels on that side (one voxel of
  tolerance at the midline); cortical surfaces entirely on that hemisphere;
- combinations equal their components recombined by sampling each one in world coordinates
  (independent of the combining code, and the only check that sees a mirrored MDTB10, whose
  labels have no hemisphere);
- component volumes are 3D integer label images with qform = sform;
- fsLR label tables match the TSVs, every TSV label has grayordinates, the intent is
  ConnDenseLabel, and XCP-D's parcel-name matching would succeed.

Rerunning produces byte-identical files. To add an atlas, add its component files (volumes and
TSV) and, for cortex, its fsLR source in `sourcedata/`, then extend `build_atlases.py`.

### sourcedata

| File | From |
| --- | --- |
| `Schaefer2018/Schaefer2018_<N>Parcels_7Networks_order.dlabel.nii` | CBIG Schaefer2018_LocalGlobal, HCP/fslr32k/cifti (64,984 vertices, medial wall included) |
| `Yan2023/MNI/yeo7/*_FSLMNI152_{1,2}mm.nii.gz`, `Yan2023/HCP/fsLR32k/{yeo7,yeo17,kong17}/*.dlabel.nii` | CBIG `stable_projects/brain_parcellation/Yan2023_homotopic/parcellations`, commit `80cf681d25ef8a0d259c5773e92f0d39537aaca1` (N = 100, 200, 400, 600, 800, 1000) |
| `xcpd/tpl-fsLR_atlas-Glasser_den-32k_dseg.*` | XCP-D 26.3.0's built-in atlases (`XCPDAtlases`); the fsLR source for `Glasser360` |
| `hcp/91282_Greyordinates.dscalar.nii` | HCPpipelines `global/templates/91282_Greyordinates/`, commit `e2d5bbad6d48452deb15c4293556b535c1973a75` (sha256 `46a61fff…6de9552`) |
| `MDTB10/atlas_description.json` | DiedrichsenLab cerebellar_atlases, King_2019 |

## Notes

- Schaefer1000: two parcels (`7Networks_RH_Vis_33`, `7Networks_RH_Cont_Cing_1`) lie entirely in
  the medial wall in CBIG's fsLR file, so the fsLR version has 998 parcels; its TSV leaves the
  two out (indices of the rest unchanged) and its JSON lists them. In general an fsLR file and
  its TSV hold only labels that have grayordinates: Workbench's `-cifti-parcellate` drops empty
  labels, and XCP-D stops when the TSV has labels the CIFTI lacks.
- In the 91k fsLR files, subcortical and cerebellar parcels keep the voxels that fall inside the
  91k grayordinates: for MSA2, 76–100% of each parcel; for MDTB10, 74–81%.
- The MSA volumes are those in Tian's combined Schaefer + subcortex releases (cut out by
  `code/legacy/msa_rename_relabel.ipynb`). They equal Tian's Subcortex-Only files (MNI152NLin6Asym
  1 and 2 mm, MNI152NLin2009cAsym 2 mm) voxel for voxel; Tian releases MNI152NLin2009cAsym 1 mm
  only in the combined files, which they also equal. In MSA2-4, the left ventral anterior
  thalamus parcels reach one voxel past the midline (about 5% of their voxels), as in Tian's
  files.

## History

**2026-11-21.** Moved to the BIDS 1.11 atlas layout (what XCP-D 26+ requires), with these changes:

- Fixed: in the MNI152NLin6Asym versions of `GlasserMSA*` and `GlasserMSA*MDTB10` (now
  `Glasser360MSA*`) the subcortex
  was mirrored left–right, and in `Schaefer*MSA*MDTB10` the cerebellum was (the old notebooks
  added voxel arrays of LAS- and RAS-stored images). MNI152NLin2009cAsym versions were correct
  and are unchanged.
- All combinations are now built by `code/build_atlases.py` from the components. For
  `Schaefer*MSA*` this means the cortex is TemplateFlow's Schaefer (as in `Schaefer<N>7`); in
  MNI152NLin2009cAsym that differs slightly from the cortex in Tian's combined files, which
  were used before (85–89% of cortical voxels agree). MNI152NLin6Asym is unchanged.
- Added: Schaefer 800 combinations; fsLR versions of every atlas.
- Schaefer `name`s are now CBIG's current names everywhere (before, standalone Schaefer used
  TemplateFlow's older names and the combinations CBIG's).
- Removed `4SParcels/` (see above).
- Glasser is now `Glasser360` everywhere: `Glasser360`, `Glasser360MSA<S>`,
  `Glasser360MSA<S>MDTB10` (were `Glasser`, `GlasserMSA<S>`, `GlasserMSA<S>MDTB10`). XCP-D has
  its own `Glasser` and stops when two atlas sets share a name.
- TSV `label` column renamed `name` (required by XCP-D); `res-1`/`res-2` are now `res-01`/`res-02`.
- Added the Yan 2023 homotopic parcellation (`Yan<N>7`) and its combinations.
- Glasser TSV column `cifti_label` renamed `hcp_label`: with it, XCP-D failed on every Glasser
  atlas in CIFTI mode (it matches parcels through `cifti_label`, and the label tables use `name`).
- fsLR files and TSVs leave out labels without grayordinates (Schaefer1000: 2), which XCP-D
  could not handle; dlabel intent set to ConnDenseLabel.
- Component volumes stored as 3D int16 with qform = sform (MSA were float32 with qform_code 0,
  MDTB10 and Glasser360 float64, Glasser360 4D); voxel values and affines unchanged.
