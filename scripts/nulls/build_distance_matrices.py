"""Rebuild the parcel distance matrices bundled in ``snaplab_tools/nulls/resources``.

Provenance for the files that ship inside the wheel: this is how they were made.

**Schaefer** (all ten published 7-network resolutions). Downloads the CBIG parcellations (fsLR-32k
dlabel + FSLMNI152 1mm centroid CSV), then builds each geodesic matrix with Connectome Workbench.
Everything needed is fetched automatically.

**Glasser** (HCP-MMP1.0, 360 areas). The parcellation is distributed via BALSA under the HCP Data
Use Terms, which restrict redistribution, so it is *not* bundled and cannot be downloaded here --
supply your own copy with ``--dlabel``. The centroid CSV is derived from the volumetric atlas in
``data/atlases`` (``--centroids``); only these derived products ship.

Nothing at run time needs this -- the outputs are committed. Run it to add a resolution, to pick up
an upstream correction, or to check that what is committed is what this pipeline produces.

Usage::

    export WB_COMMAND=/Applications/ConnectomeWorkbench.app/Contents/usr/bin/wb_command

    python scripts/nulls/build_distance_matrices.py                  # Schaefer, only what is missing
    python scripts/nulls/build_distance_matrices.py --check          # diff against what is committed
    python scripts/nulls/build_distance_matrices.py --atlas glasser --centroids \\
        --dlabel /path/to/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors_with_Atlas_ROIs2.32k_fs_LR.dlabel.nii

Cost is one wb_command call per parcel: roughly seven minutes for the full Schaefer set on a
laptop, ninety seconds for Glasser.
"""
import argparse
import urllib.request

import nibabel as nib
import numpy as np
import pandas as pd

from snaplab_tools.nulls.maps import (
    build_geodesic_distance_matrix,
    _ATLASES,
    _centroid_csv_path,
    _distance_cache_path,
    _dlabel_path,
    _hemi_cache_path,
)

CBIG = (
    "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation"
    "/Schaefer2018_LocalGlobal/Parcellations"
)

# Sources for the Glasser centroid CSV, both already redistributed in this repository.
GLASSER_VOLUME = (
    "data/atlases/Glasser/atlas-Glasser/atlas-Glasser_space-MNI152NLin6Asym_res-1_dseg.nii.gz"
)
GLASSER_TSV = "data/atlases/Glasser/atlas-Glasser/atlas-Glasser_dseg.tsv"


def fetch_schaefer(n_regions):
    """Download the CBIG dlabel and centroid CSV for one resolution, if not already present."""
    wanted = [
        (_dlabel_path(n_regions),
         f"{CBIG}/HCP/fslr32k/cifti/Schaefer2018_{n_regions}Parcels_7Networks_order.dlabel.nii"),
        (_centroid_csv_path(n_regions),
         f"{CBIG}/MNI/Centroid_coordinates/Schaefer2018_{n_regions}Parcels_7Networks_order"
         f"_FSLMNI152_1mm.Centroid_RAS.csv"),
    ]
    for path, url in wanted:
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  fetching {path.name}")
        urllib.request.urlretrieve(url, path)


def build_glasser_centroids():
    """Derive the Glasser centroid CSV from the volumetric atlas in ``data/atlases``.

    MNI152NLin6Asym 1mm is the space the Schaefer CSVs call FSLMNI152 1mm, so the Euclidean basis
    means the same thing across both atlases. Names come from the TSV's ``cifti_label`` column,
    which is what the BALSA dlabel's label table uses -- the two agree exactly, which is what makes
    a matrix built from the surface safe to pair with centroids derived from the volume.
    """
    img = nib.load(GLASSER_VOLUME)
    data = np.asarray(img.dataobj).astype(int).squeeze()
    tsv = pd.read_csv(GLASSER_TSV, sep="\t")

    rows = []
    for _, entry in tsv.iterrows():
        k = int(entry["index"])
        voxels = np.argwhere(data == k)
        if voxels.size == 0:
            raise ValueError(f"label {k} ({entry['cifti_label']}) has no voxels in {GLASSER_VOLUME}")
        ras = nib.affines.apply_affine(img.affine, voxels.mean(axis=0))
        rows.append({"ROI Label": k, "ROI Name": entry["cifti_label"],
                     "R": int(round(ras[0])), "A": int(round(ras[1])), "S": int(round(ras[2]))})

    path = _centroid_csv_path(360, "glasser")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  wrote {path.name} ({len(rows)} parcels)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", default="schaefer", choices=sorted(_ATLASES))
    parser.add_argument("--dlabel", help="parcellation to build from, for a non-bundled atlas")
    parser.add_argument("--centroids", action="store_true",
                        help="also rebuild the derived centroid CSV (Glasser only)")
    parser.add_argument("--rebuild", action="store_true", help="rebuild matrices that already exist")
    parser.add_argument("--check", action="store_true",
                        help="rebuild in memory and diff against the committed files")
    parser.add_argument("--resolutions", type=int, nargs="+")
    args = parser.parse_args()

    resolutions = args.resolutions or list(_ATLASES[args.atlas]["resolutions"])
    if args.atlas == "glasser" and args.centroids:
        build_glasser_centroids()

    for n in resolutions:
        print(f"{args.atlas}{n}")
        if args.atlas == "schaefer":
            fetch_schaefer(n)
        elif args.dlabel is None:
            raise SystemExit(
                f"--dlabel is required for {args.atlas}: its parcellation is not bundled, because "
                f"its licence does not permit redistribution. See THIRD_PARTY_NOTICES.md."
            )

        dpath = _distance_cache_path(n, "geodesic", args.atlas)
        hpath = _hemi_cache_path(n, "geodesic", args.atlas)
        if dpath.exists() and not (args.rebuild or args.check):
            print("  distance matrix present, skipping")
            continue

        D, hemi = build_geodesic_distance_matrix(n, args.atlas, dlabel=args.dlabel)

        if args.check:
            # wb_command writes float32 GIFTI, so rebuilds agree to float32 precision rather than
            # exactly; anything larger than that is a real difference worth looking at.
            committed = np.load(dpath)
            finite = np.isfinite(committed)
            assert np.array_equal(finite, np.isfinite(D)), "NaN pattern differs from what is committed"
            print(f"  max |diff| vs committed: {np.abs(D[finite] - committed[finite]).max():.2e} mm")
            continue

        np.save(dpath, D)
        np.save(hpath, hemi)
        placed = np.isfinite(D).any(axis=1)
        print(f"  built: {placed.sum()}/{n} parcels placed, "
              f"max {np.nanmax(D):.1f} mm, {dpath.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
