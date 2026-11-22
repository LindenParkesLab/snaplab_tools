#!/usr/bin/env python
"""Import the Yan 2023 homotopic parcellations (Yeo 7-network labels) as component atlases.

    python code/import_yan2023.py [--xfm PATH]      # needs nibabel, numpy, pandas, scipy, nitransforms

Then run code/build_atlases.py, which makes the combinations and the fsLR files.

Input: sourcedata/Yan2023/, the CBIG release (ThomasYeoLab/CBIG at commit 80cf681d,
stable_projects/brain_parcellation/Yan2023_homotopic/parcellations):
  - MNI/yeo7/<N>Parcels_Yeo2011_7Networks_FSLMNI152_{1,2}mm.nii.gz   (FSL MNI152 = MNI152NLin6Asym)
  - HCP/fsLR32k/{yeo7,yeo17,kong17}/<N>Parcels_*.dlabel.nii
The three network versions have the same parcels, numbered differently; the Yeo 17 and Kong 17
indices and names of each parcel go into extra TSV columns.

Output, for N = 100, 200, 400, 600, 800, 1000: tpl-*/tpl-*_atlas-Yan<N>7_res-0{1,2}_dseg.{nii.gz,tsv}
  - MNI152NLin6Asym: CBIG's volumes, stored on the same grid (and orientation) as the other atlases.
  - MNI152NLin2009cAsym: CBIG releases none, so the 1 mm MNI152NLin6Asym volume is resampled with
    TemplateFlow's MNI152NLin6Asym -> MNI152NLin2009cAsym transform (the one fMRIPrep uses).
    res-01: nearest label at each voxel centre. res-02: most common label (background included)
    of 27 points spread through each voxel.

Checks (the script stops if one fails): each volume has labels 1..N; the volume and the fsLR
file give the same label to the same cortex (sampled at the fsLR midthickness vertices; >= 90%
of parcels, as for Schaefer, whose files are known to match); the
three network versions have identical boundaries. It also resamples Schaefer<N>7 the same way
and reports its agreement with TemplateFlow's MNI152NLin2009cAsym Schaefer, as a check of the
transform.
"""
import argparse
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nitransforms.manip import TransformChain
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'sourcedata' / 'Yan2023'
SCALES = (100, 200, 400, 600, 800, 1000)
NETWORKS = {'yeo7': 'Yeo2011_7Networks', 'yeo17': 'Yeo2011_17Networks', 'kong17': 'Kong2022_17Networks'}
SURFACES = ROOT.parents[1] / 'snaplab_tools' / 'nulls' / 'resources' / 'surfaces'
XFM_URL = ('https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/'
           'tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5')
XFM_SHA256 = '2e3869a07b96aec406e0419ca2e434afc54882d37cc212b933b139d1b63a4dfe'


def vol_path(space, name, res, ext='nii.gz'):
    return ROOT / f'tpl-{space}' / f'tpl-{space}_atlas-{name}_res-0{res}_dseg.{ext}'


def labels(img):
    return np.asarray(img.dataobj).squeeze().astype(np.int32)


def save(data, ref, path):
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.int16)
    hdr.set_slope_inter(1, 0)
    img = nib.Nifti1Image(data.astype(np.int16), ref.affine, hdr)
    img.set_qform(ref.affine, code=int(ref.header['qform_code']) or 1)
    img.set_sform(ref.affine, code=int(ref.header['sform_code']) or 1)
    nib.save(img, path)


def on_grid(img, ref):
    """img's labels on ref's grid; the grids must correspond voxel for voxel."""
    m = np.linalg.inv(img.affine) @ ref.affine
    if not np.allclose(m, np.round(m), atol=1e-4):
        raise SystemExit(f'grids do not correspond voxel for voxel:\n{img.affine}\n{ref.affine}')
    m = np.round(m).astype(int)
    ijk = np.indices(ref.shape[:3]).reshape(3, -1)
    src = m[:3, :3] @ ijk + m[:3, 3:4]
    if (src < 0).any() or (src >= np.array(img.shape[:3])[:, None]).any():
        raise SystemExit('grids do not cover the same field of view')
    return labels(img)[tuple(src)].reshape(ref.shape[:3])


def sample(data, affine, world):
    """Nearest-voxel labels at world coordinates (n, 3); 0 outside the image."""
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(affine), world)).astype(int)
    ok = np.all((ijk >= 0) & (ijk < data.shape), axis=1)
    out = np.zeros(len(world), np.int32)
    out[ok] = data[tuple(ijk[ok].T)]
    return out


def resample(src, xfm, ref, points_per_axis, chunk=500_000):
    """src (1 mm MNI152NLin6Asym image) on ref's grid in MNI152NLin2009cAsym."""
    data = labels(src)
    ijk = np.argwhere(np.ones(ref.shape[:3], bool))
    if points_per_axis == 1:
        offsets = np.zeros((1, 3))
    else:
        o = (np.arange(points_per_axis) + 0.5) / points_per_axis - 0.5   # within the voxel
        offsets = np.stack(np.meshgrid(o, o, o, indexing='ij'), -1).reshape(-1, 3)
    out = np.zeros(len(ijk), np.int32)
    for start in range(0, len(ijk), chunk):
        block = ijk[start:start + chunk]
        pts = (block[:, None, :] + offsets[None]).reshape(-1, 3)
        world = nib.affines.apply_affine(ref.affine, pts)
        lab = sample(data, src.affine, xfm.map(world)).reshape(len(block), len(offsets))
        out[start:start + chunk] = lab[:, 0] if len(offsets) == 1 else stats.mode(lab, axis=1, keepdims=False).mode
    return out.reshape(ref.shape[:3])


def dlabel(n, net):
    img = nib.load(SRC / 'HCP' / 'fsLR32k' / net / f'{n}Parcels_{NETWORKS[net]}.dlabel.nii')
    table = img.header.get_axis(0).label[0]
    return np.asarray(img.dataobj)[0].astype(int), {k: v[0] for k, v in table.items() if k}, img.header.get_axis(1)


def get_xfm(path):
    if path is None:
        path = Path(tempfile.mkdtemp()) / Path(XFM_URL).name
        print(f'downloading {XFM_URL} (205 MB)')
        urllib.request.urlretrieve(XFM_URL, path)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != XFM_SHA256:
        raise SystemExit(f'sha256 mismatch for {path}')
    return TransformChain.from_filename(str(path), fmt='itk')   # maps 2009c points to MNI152NLin6Asym


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--xfm', type=Path, help=f'local copy of {Path(XFM_URL).name} (default: download it)')
    xfm = get_xfm(ap.parse_args().xfm)
    surf = {h: nib.load(SURFACES / f'tpl-fsLR_den-32k_hemi-{h}_midthickness.surf.gii').darrays[0].data
            for h in ('L', 'R')}
    failures = []
    for n in SCALES:
        name = f'Yan{n}7'
        ref6 = {r: nib.load(vol_path('MNI152NLin6Asym', f'Schaefer{n}7', r)) for r in (1, 2)}
        ref9 = {r: nib.load(vol_path('MNI152NLin2009cAsym', f'Schaefer{n}7', r)) for r in (1, 2)}

        # Table: Yeo 7 names, plus each parcel's Yeo 17 / Kong 17 index and name (same vertices).
        d7, t7, bm = dlabel(n, 'yeo7')
        if sorted(t7) != list(range(1, n + 1)):
            failures.append(f'{name}: fsLR keys are not 1..{n}')
        df = pd.DataFrame({'index': list(range(1, n + 1)), 'name': [t7[i] for i in range(1, n + 1)]})
        df['homotopic_index'] = np.where(df['index'] <= n // 2, df['index'] + n // 2, df['index'] - n // 2)
        for net in ('yeo17', 'kong17'):
            d, t, _ = dlabel(n, net)
            pairs = dict(zip(d7[d7 > 0], d[d7 > 0]))
            if not (np.array_equal(d7 > 0, d > 0) and len(set(zip(d7[d7 > 0], d[d7 > 0]))) == n == len(set(pairs.values()))):
                failures.append(f'{name}: {net} boundaries differ from yeo7')
            df[f'{net}_index'] = df['index'].map(pairs)
            df[f'{net}_name'] = df[f'{net}_index'].map(t)

        # MNI152NLin6Asym: CBIG's volumes on our grid.
        mni6 = {}
        for r in (1, 2):
            img = nib.load(SRC / 'MNI' / 'yeo7' / f'{n}Parcels_Yeo2011_7Networks_FSLMNI152_{r}mm.nii.gz')
            mni6[r] = on_grid(img, ref6[r])
            if sorted(set(np.unique(mni6[r])) - {0}) != list(range(1, n + 1)):
                failures.append(f'{name} MNI152NLin6Asym res-0{r}: labels are not 1..{n}')
            save(mni6[r], ref6[r], vol_path('MNI152NLin6Asym', name, r))

        # The volume and the fsLR file label the same cortex: sample the 1 mm volume at the
        # midthickness vertices (nearest labelled voxel within 3 mm).
        agree = []
        for struct, sl, b in bm.iter_structures():
            h = 'L' if struct.endswith('LEFT') else 'R'
            xyz = surf[h][b.vertex]
            best = np.zeros(len(xyz), np.int32)
            for d in sorted({(i, j, k) for i in range(-3, 4) for j in range(-3, 4) for k in range(-3, 4)},
                            key=lambda v: np.dot(v, v)):
                if np.dot(d, d) > 9:
                    continue
                s = sample(mni6[1], ref6[1].affine, xyz + d)
                best = np.where(best == 0, s, best)
            surf_lab = d7[sl]
            for k in np.unique(surf_lab[surf_lab > 0]):
                got = best[(surf_lab == k) & (best > 0)]
                agree.append(len(got) > 0 and np.bincount(got).argmax() == k)
        frac = float(np.mean(agree))
        print(f'{name}: volume vs fsLR, parcels whose vertices mostly sample their own label: {frac:.1%}')
        if frac < 0.90:   # the same test on Schaefer, whose volumes and fsLR are known to match:
            #               99.2% at 400 parcels, 91.7% at 1000 (small parcels sample less cleanly)
            failures.append(f'{name}: volume and fsLR labels disagree ({frac:.1%} of parcels match)')

        # MNI152NLin2009cAsym: resample the 1 mm volume through TemplateFlow's transform.
        src = nib.Nifti1Image(mni6[1].astype(np.int16), ref6[1].affine)
        for r, ppa in ((1, 1), (2, 3)):
            out = resample(src, xfm, ref9[r], ppa)
            missing = set(range(1, n + 1)) - set(np.unique(out))
            if missing:
                failures.append(f'{name} MNI152NLin2009cAsym res-0{r}: labels lost in resampling: {sorted(missing)[:8]}')
            save(out, ref9[r], vol_path('MNI152NLin2009cAsym', name, r))

            # Same resampling of Schaefer<N>7, compared with TemplateFlow's own 2009c Schaefer.
            sch = resample(nib.load(vol_path('MNI152NLin6Asym', f'Schaefer{n}7', 1)), xfm, ref9[r], ppa)
            tf = labels(ref9[r])
            both = (sch > 0) & (tf > 0)
            print(f'  res-0{r}: Schaefer{n}7 resampled this way vs TemplateFlow: {np.mean(sch[both] == tf[both]):.1%} '
                  f'of voxels agree; labelled voxels {np.sum(sch > 0) / np.sum(tf > 0):.2f}x TemplateFlow\'s')

        for space in ('MNI152NLin6Asym', 'MNI152NLin2009cAsym'):
            for r in (1, 2):
                df.to_csv(vol_path(space, name, r, 'tsv'), sep='\t', index=False, na_rep='n/a')

    if failures:
        print(f'\n{len(failures)} CHECK(S) FAILED:')
        for f in failures:
            print('  ', f)
        sys.exit(1)
    print('\nall checks passed; now run code/build_atlases.py')


if __name__ == '__main__':
    main()
