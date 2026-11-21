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
    ├── fetch_atlaspack.sh             # gets the 4S atlases (AtlasPack), for pipelines other than XCP-D
    └── legacy/                        # the notebooks that made the component atlases (old layout)
```

Every image has a TSV (`index`, `name`, `hemisphere`, `cortex`, `subcortex`, `cerebellum`,
plus source-specific columns) and a JSON sidecar with the same entities.

**fsLR densities.** Cortex-only atlases are `den-32k`: 59,412 surface vertices (32k fsLR,
medial wall excluded). Atlases with subcortex or cerebellum are `den-91k`: the standard 91,282
grayordinates, the same as fMRIPrep's `--cifti-output 91k`, with subcortical and cerebellar
labels sampled from the MNI152NLin6Asym 2 mm volume.

## Atlases

| Atlas | Parcels | What |
| --- | --- | --- |
| `Schaefer<N>7`, N = 100, 200, 400, 600, 800, 1000 | N | Schaefer 2018 cortex, 7-network order |
| `Glasser360` | 360 | HCP-MMP1.0 cortex (Glasser 2016). Not `Glasser`: XCP-D has a built-in atlas of that name and stops when two atlas sets share one |
| `MSA<S>`, S = 1–4 | 16, 32, 50, 54 | Melbourne Subcortex Atlas (Tian 2020), 3T, scales S1–S4 |
| `MDTB10` | 10 | MDTB cerebellum (King 2019) |
| `Schaefer<N>7MSA<S>`, N = 100, 200, 400, 800 | N + MSA | cortex + subcortex |
| `Schaefer<N>7MSA<S>MDTB10` | N + MSA + 10 | cortex + subcortex + cerebellum |
| `Glasser360MSA<S>` | 360 + MSA | cortex + subcortex |
| `Glasser360MSA<S>MDTB10` | 360 + MSA + 10 | cortex + subcortex + cerebellum |

In the combined atlases, labels are numbered cortex first, then MSA, then MDTB10 (whose names
get a `cerebellum_` prefix). Where the components overlap, MSA overrides cortex and MDTB10
overrides both.

**Schaefer names.** `name` holds CBIG's current parcel names (as in AtlasPack and Tian's
releases). The TemplateFlow volumes came with older names for some parcels (e.g. `FrOper` for
`FrOperIns`, `PCC` for `pCunPCC`); those are in `templateflow_name`. Same parcels, same
indices.

**4S atlases** (Schaefer + CIT168 subcortex, thalamus, hippocampus, amygdala, cerebellum) are
not kept here. XCP-D and QSIRecon include them; for other pipelines, run
`code/fetch_atlaspack.sh`, which puts PennLINC's AtlasPack (checksum-verified) in
`data/AtlasPack/` (gitignored). Don't pass both AtlasPack and XCP-D's own copy to XCP-D: it
stops when two atlas datasets contain the same atlas.

## Rebuilding

```bash
python code/build_atlases.py     # nibabel, numpy, pandas; about 2-10 minutes
```

It rebuilds the combined atlases and all fsLR files from the components and `sourcedata/`,
rewrites the TSVs, sidecars and descriptions, and checks the result. It stops with an error if
any check fails:

- the inputs' voxel grids correspond voxel for voxel (combining happens in world space, never on
  raw arrays);
- every label in a TSV is present in its image, and nothing else is;
- labels named for a hemisphere lie in it: volumes ≥ 90% of voxels on that side (one voxel of
  tolerance at the midline); cortical surfaces entirely on that hemisphere;
- fsLR label tables match the TSVs.

Rerunning produces byte-identical files. To add an atlas, add its component files (volumes and
TSV) and, for cortex, its fsLR source in `sourcedata/`, then extend `build_atlases.py`.

### sourcedata

| File | From |
| --- | --- |
| `Schaefer2018/Schaefer2018_<N>Parcels_7Networks_order.dlabel.nii` | CBIG Schaefer2018_LocalGlobal, HCP/fslr32k/cifti (64,984 vertices, medial wall included) |
| `xcpd/tpl-fsLR_atlas-Glasser_den-32k_dseg.*` | XCP-D 26.3.0's built-in atlases (`XCPDAtlases`); the fsLR source for `Glasser360` |
| `hcp/91282_Greyordinates.dscalar.nii` | HCPpipelines `global/templates/91282_Greyordinates/`, commit `e2d5bbad6d48452deb15c4293556b535c1973a75` (sha256 `46a61fff…6de9552`) |
| `MDTB10/atlas_description.json` | DiedrichsenLab cerebellar_atlases, King_2019 |

## Notes

- Schaefer1000: two parcels (`7Networks_RH_Vis_33`, `7Networks_RH_Cont_Cing_1`) lie entirely in
  the medial wall in CBIG's fsLR file, so the fsLR version has 998 parcels.
- In the 91k fsLR files, subcortical and cerebellar parcels keep the voxels that fall inside the
  91k grayordinates: for MSA2, 76–100% of each parcel; for MDTB10, 74–81%.
- The MSA volumes are those in Tian's combined Schaefer + subcortex releases (cut out by
  `code/legacy/msa_rename_relabel.ipynb`).

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
