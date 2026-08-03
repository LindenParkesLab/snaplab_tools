# Installation

## Requirements

Python 3.9 or newer. Runtime dependencies are declared in `setup.py` and installed automatically —
there is no separate list to keep in sync.

## Install

```bash
git clone https://github.com/LindenParkesLab/snaplab_tools.git
cd snaplab_tools
pip install -e .
```

The `-e` (editable) install is the one to use in a lab setting: `git pull` then picks up new
functions without reinstalling.

Installing the package ships about 39 MB of bundled resources with it — Schaefer parcellations at
every published resolution (100 to 1000 in steps of 100), fsLR-32k midthickness surfaces, and a
precomputed geodesic distance matrix for each, plus the Glasser (HCP-MMP1.0) distance matrix and
centroids. {mod}`snaplab_tools.nulls` works offline because of them. Most of that size is the
distance matrices, which grow with the square of the resolution: the one for 1000 parcels is 8 MB
on its own.

## Optional extras

```bash
pip install -e ".[surface]"      # cortical surface rendering
pip install -e ".[changepoint]"  # multi change-point detection
pip install -e ".[docs]"         # build this documentation
```

`surface`
: Installs `surfplot` and `brainspace`, which pull in VTK. Needed only for
  {func}`~snaplab_tools.plotting.plotting.plot_brain_surface_data` and
  {func}`~snaplab_tools.plotting.plotting.plot_brain_surface_data_single`. Both import it lazily,
  so everything else in the package works without it.

`changepoint`
: Installs `ruptures`. Needed only for more than one change point, or a non-L2 cost model, in
  {func}`~snaplab_tools.gams.detect_change_point`. The single-boundary L2 detector is exact and
  needs no extra dependency.

## Extra setup for surface plotting

The surface plotting functions read parcel values onto FreeSurfer surfaces, and need Schaefer
`.annot` annotation files that are not bundled (they are large and come in many variants). Download
them once:

```python
from snaplab_tools.utils import load_schaefer_parc

load_schaefer_parc(n_parcels=400, order=7, annot='fsaverage5',
                   out_dir='~/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5')
```

By default the plotting module looks under
`~/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3`. Point it elsewhere with the
`SCHAEFER_ANNOT_DIR` environment variable:

```bash
export SCHAEFER_ANNOT_DIR=/path/to/Parcellations/FreeSurfer5.3
```

:::{note}
`SCHAEFER_ANNOT_DIR` is read on every call, so setting it with `os.environ` part-way through a
session works — including after the package has been imported. It used to be captured at import
time, which meant a late `os.environ` assignment silently did nothing.
:::

## Building geodesic distance matrices yourself

Distance matrices for all ten Schaefer 7-network resolutions (100 to 1000 in steps of 100) and for
Glasser (HCP-MMP1.0, 360 areas) ship with the package, so
{func}`~snaplab_tools.nulls.load_distance_matrix` works out of the box for any of them:

```python
D, hemi = load_distance_matrix(n_regions=400)                      # Schaefer, the default
D, hemi = load_distance_matrix(n_regions=360, atlas='glasser')     # HCP-MMP1.0
```

Any *other* parcellation needs {func}`~snaplab_tools.nulls.build_geodesic_distance_matrix`, which
shells out to Connectome Workbench — install `wb_command` and put it on your `PATH`, or point at it
with the `WB_COMMAND` environment variable.

:::{warning}
Parcel order differs between the two atlases. Schaefer runs **left hemisphere first**; HCP-MMP1.0
runs **right first** (areas 1–180 right, 181–360 left). Both are kept in the published order, so a
map parcellated with the atlas as distributed needs no reordering — but if you assemble a map by
concatenating hemispheres yourself, check which convention you built it in.
:::

:::{note}
The Glasser *parcellation* is not bundled — it is distributed via BALSA under the HCP Data Use
Terms, which restrict redistribution, so only the derived distance matrix and centroids ship (see
`THIRD_PARTY_NOTICES.md`). That is enough to load and use the basis offline. Rebuilding it needs
your own BALSA copy, passed to `scripts/nulls/build_distance_matrices.py` via `--dlabel`.
:::

:::{note}
At 1000 parcels the upstream CBIG fsLR-32k parcellation assigns no vertices to two of its parcels
(533 `7Networks_RH_Vis_33` and 903 `7Networks_RH_Cont_Cing_1`), so the geodesic basis covers 998
of the 1000. Those two carry an all-NaN row;
{func}`~snaplab_tools.nulls.generate_surrogates` drops them with a warning and returns them as
NaN. They do have centroids, so `kind='euclidean'` covers all 1000.
:::

## Checking the install

```python
import snaplab_tools
print(snaplab_tools.__version__)

from snaplab_tools.nulls import load_distance_matrix
distance_matrix, hemi = load_distance_matrix(n_regions=400, kind='geodesic')
print(distance_matrix.shape)   # (400, 400)
```

If that runs, the bundled resources resolved correctly. See the [API reference](api/index.md) for
what is available.
