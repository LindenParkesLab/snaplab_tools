# Third-party notices

This repository redistributes brain atlases and surface geometry produced by others. The `LICENSE`
at the root (BSD 3-Clause) covers the original source code here; each file listed below remains
under the terms of its own licence, and those terms travel with the file.

All of them are permissive and compatible with BSD 3-Clause, so using this package raises no
licence conflict. What they do carry is attribution requirements, which is why they are recorded
here.

If you use any of this data, cite the source paper. That is a condition of several of these
licences and, regardless, it is the right thing to do.

---

## Bundled inside the installed package

These ship inside the wheel (`snaplab_tools/nulls/resources/`) so `snaplab_tools.nulls` works
offline.

### Schaefer2018 parcellations

`parcellations/Schaefer2018_*Parcels_7Networks_order*.{dlabel.nii,csv}`

From the CBIG repository, **MIT licence**, which permits redistribution provided the copyright
notice travels with it:

> Copyright (c) 2016–2019 Thomas Yeo Lab, CBIG, National University of Singapore.
> Licensed under the MIT License. See
> <https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md>

Schaefer, A., Kong, R., Gordon, E.M., Laumann, T.O., Zuo, X.-N., Holmes, A.J., Eickhoff, S.B., &
Yeo, B.T.T. (2018). Local-global parcellation of the human cerebral cortex from intrinsic
functional connectivity MRI. *Cerebral Cortex*, 28(9), 3095–3114.
<https://doi.org/10.1093/cercor/bhx179>

### fsLR-32k midthickness surfaces

`surfaces/tpl-fsLR_den-32k_hemi-{L,R}_midthickness.surf.gii`

Standard fs_LR-32k surface geometry, as redistributed by TemplateFlow (`tpl-fsLR`). Derived from
the Human Connectome Project's surface pipelines.

Van Essen, D.C., Smith, S.M., Barch, D.M., Behrens, T.E.J., Yacoub, E., & Ugurbil, K. (2013). The
WU-Minn Human Connectome Project: an overview. *NeuroImage*, 80, 62–79.
<https://doi.org/10.1016/j.neuroimage.2013.05.041>

### Derived distance matrices

`distances/schaefer*-7_geodesic_{distance,hemi}.npy`

Computed in this repository from the two items above using Connectome Workbench
(`wb_command -surface-geodesic-distance`). As derived works they inherit the terms of their
sources.

---

## In `data/atlases/` (not installed with the package)

### Glasser (HCP-MMP1.0)

Taken from [PennLINC/xcp_d](https://github.com/PennLINC/xcp_d/tree/main/xcp_d/data/atlases),
which redistributes the parcellation in NIfTI form.

Glasser, M.F., Coalson, T.S., Robinson, E.C., Hacker, C.D., Harwell, J., Yacoub, E., Ugurbil, K.,
Andersson, J., Beckmann, C.F., Jenkinson, M., Smith, S.M., & Van Essen, D.C. (2016). A multi-modal
parcellation of human cerebral cortex. *Nature*, 536, 171–178.
<https://doi.org/10.1038/nature18933>

> **Note.** The original release of this parcellation is distributed via BALSA under the HCP Data
> Use Terms, which restrict redistribution. The copy here came from xcp_d rather than from BALSA
> directly. If you intend to redistribute it further, check the terms that apply to your source.

### Melbourne Subcortical Atlas (MSA / Tian)

Taken from
[yetianmed/subcortex](https://github.com/yetianmed/subcortex/tree/master/Group-Parcellation/3T/Subcortex-Only).

Tian, Y., Margulies, D.S., Breakspear, M., & Zalesky, A. (2020). Topographic organization of the
human subcortex unveiled with functional connectivity gradients. *Nature Neuroscience*, 23,
1421–1432. <https://doi.org/10.1038/s41593-020-00711-6>

### MDTB

King, M., Hernandez-Castillo, C.R., Poldrack, R.A., Ivry, R.B., & Diedrichsen, J. (2019).
Functional boundaries in the human cerebellum revealed by a multi-domain task battery. *Nature
Neuroscience*, 22, 1371–1378. <https://doi.org/10.1038/s41593-019-0436-x>

### Combined atlases

`GlasserMSA/`, `SchaeferMSA/`, `*MDTB10/` are combinations produced in this repository from the
above. They inherit the terms of their constituent atlases.

---

## Method dependencies

Not redistributed here — pip installs them separately — but required at runtime and worth citing
when used. Both are GPL-3.0, which is worth knowing if you redistribute a work built on this one:

- **BrainSMASH** (GPL-3.0) — used by `snaplab_tools.nulls`. Burt, J.B., Helmer, M., Shinn, M.,
  Anticevic, A., & Murray, J.D. (2020). Generative modeling of brain maps with spatial
  autocorrelation. *NeuroImage*, 220, 117038.
  <https://doi.org/10.1016/j.neuroimage.2020.117038>
- **Brain Connectivity Toolbox** (`bctpy`, GPL-3.0) — used by `snaplab_tools.topology`. Rubinov,
  M., & Sporns, O. (2010). Complex network measures of brain connectivity: uses and
  interpretations. *NeuroImage*, 52(3), 1059–1069.
  <https://doi.org/10.1016/j.neuroimage.2009.10.003>
