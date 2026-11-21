#!/usr/bin/env python
"""Build the combined atlases and every fsLR (CIFTI) atlas in this BIDS atlas dataset.

Run from anywhere; rerunning rewrites the same outputs:

    python code/build_atlases.py            # needs nibabel, numpy, pandas

Inputs (not changed by this script):
  - component atlases, volumetric: tpl-MNI152NLin{6,2009c}Asym/tpl-*_atlas-{Schaefer<N>7,
    Glasser360,MSA<S>,MDTB10}_res-0{1,2}_dseg.nii.gz, and their TSVs (index, name, ...)
  - cortical surface labels: sourcedata/Schaefer2018/*.dlabel.nii (CBIG, 32k fsLR) and
    sourcedata/xcpd/tpl-fsLR_atlas-Glasser_den-32k_dseg.dlabel.nii
  - the HCP 91k grayordinate definition: sourcedata/hcp/91282_Greyordinates.dscalar.nii

Outputs:
  - combined volumetric atlases: Schaefer<N>7MSA<S>, Schaefer<N>7MSA<S>MDTB10, Glasser360MSA<S>,
    Glasser360MSA<S>MDTB10. Labels are numbered cortex first, then MSA, then MDTB10. Where
    components overlap, MSA overrides cortex and MDTB10 overrides both.
  - fsLR atlases for every atlas: cortex-only atlases as den-32k (59412 surface vertices,
    medial wall excluded), atlases with subcortex/cerebellum as den-91k (91282 grayordinates).
    Cortex comes from the surface labels; subcortex and cerebellum are sampled from the
    MNI152NLin6Asym 2 mm volume at the 91k voxels.
  - TSV (index, name, hemisphere, cortex, subcortex, cerebellum, ...) and JSON sidecars next to
    every image, an atlas-<name>_description.json per atlas, and dataset_description.json.

Components are combined in world space: every image is mapped onto the reference grid through
its affine, and the build stops if the grids don't correspond voxel for voxel. (The MSA
volumes are stored LAS and the others RAS; combining raw arrays mirrors one of them.)

Checks, all of which must pass: every label in each TSV is present in its image; labels named
for a hemisphere lie in that hemisphere (volumes: >= 90% of voxels on that side of x = 0;
surfaces: all vertices on that hemisphere); fsLR label tables match the TSVs.
"""
import colorsys
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPACES = ('MNI152NLin6Asym', 'MNI152NLin2009cAsym')
RESOLUTIONS = (1, 2)
SCHAEFER_COMBO_SCALES = (100, 200, 400, 800)
MSA_SCALES = (1, 2, 3, 4)
FLAGS = ('cortex', 'subcortex', 'cerebellum')

REFS = {
    'Schaefer': ['Schaefer, A., Kong, R., Gordon, E. M., Laumann, T. O., Zuo, X. N., Holmes, A. J., '
                 'Eickhoff, S. B., & Yeo, B. T. (2018). Local-global parcellation of the human cerebral '
                 'cortex from intrinsic functional connectivity MRI. Cerebral Cortex, 28(9), 3095-3114. '
                 'https://doi.org/10.1093/cercor/bhx179',
                 'https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal'],
    'Glasser': ['Glasser, M. F., Coalson, T. S., Robinson, E. C., et al. (2016). A multi-modal '
                'parcellation of human cerebral cortex. Nature, 536, 171-178. https://doi.org/10.1038/nature18933',
                'https://github.com/PennLINC/xcp_d'],
    'MSA': ['Tian, Y., Margulies, D. S., Breakspear, M., & Zalesky, A. (2020). Topographic organization '
            'of the human subcortex unveiled with functional connectivity gradients. Nature Neuroscience, '
            '23, 1421-1432. https://doi.org/10.1038/s41593-020-00711-6',
            'https://github.com/yetianmed/subcortex'],
    'MDTB10': ['King, M., Hernandez-Castillo, C. R., Poldrack, R. A., Ivry, R. B., & Diedrichsen, J. (2019). '
               'Functional boundaries in the human cerebellum revealed by a multi-domain task battery. '
               'Nature Neuroscience, 22, 1371-1378. https://doi.org/10.1038/s41593-019-0436-x',
               'https://github.com/DiedrichsenLab/cerebellar_atlases'],
}
LICENSES = {
    'Schaefer': 'MIT (https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md)',
    'Glasser': 'HCP Data Use Terms apply to the original release; see THIRD_PARTY_NOTICES.md',
    'MSA': 'See https://github.com/yetianmed/subcortex',
    'MDTB10': 'See https://github.com/DiedrichsenLab/cerebellar_atlases',
}
MSA_PARCELS = {1: 16, 2: 32, 3: 50, 4: 54}

COLUMN_DOCS = {
    'index': {'Description': 'Label value in the image.'},
    'name': {'Description': 'Region name.'},
    'hemisphere': {'Description': 'Hemisphere the region belongs to, from its name.',
                   'Levels': {'L': 'left', 'R': 'right', 'bilateral': 'spans both hemispheres'}},
    'cortex': {'Description': 'Region comes from a cortical atlas (Schaefer or Glasser).'},
    'subcortex': {'Description': 'Region comes from the Melbourne Subcortex Atlas (MSA).'},
    'cerebellum': {'Description': 'Region comes from the MDTB10 cerebellar atlas.'},
    'templateflow_name': {'Description': "The parcel's name in TemplateFlow's Schaefer2018 release, which "
                                         'predates CBIG renaming some parcels. Same parcel, same index; '
                                         '`name` holds the current CBIG name.'},
}


# --------------------------------------------------------------------------- helpers
def family(name):
    for fam in ('Schaefer', 'Glasser', 'MSA', 'MDTB10'):
        if name.startswith(fam):
            return fam
    raise ValueError(name)


def hemisphere(label):
    if re.search(r'_LH_|^Left_|-lh$', label):
        return 'L'
    if re.search(r'_RH_|^Right_|-rh$', label):
        return 'R'
    return 'bilateral'


def vol_path(space, name, res, ext='nii.gz'):
    return ROOT / f'tpl-{space}' / f'tpl-{space}_atlas-{name}_res-0{res}_dseg.{ext}'


def fslr_path(name, den, ext):
    return ROOT / 'tpl-fsLR' / f'tpl-fsLR_atlas-{name}_den-{den}_dseg.{ext}'


def read_tsv(path):
    df = pd.read_csv(path, sep='\t', na_values='n/a', keep_default_na=False)
    return df.drop(columns=[c for c in ('hemisphere', *FLAGS) if c in df.columns])


def write_tsv(df, path):
    df.to_csv(path, sep='\t', index=False, na_rep='n/a')


def write_json(obj, path):
    path.write_text(json.dumps(obj, indent=4) + '\n')


def on_grid(img, ref):
    """Return img's labels on ref's voxel grid. The grids must correspond voxel for voxel."""
    data = np.asarray(img.dataobj).squeeze().astype(np.int32)
    m = np.linalg.inv(img.affine) @ ref.affine      # ref voxel -> img voxel
    if not np.allclose(m, np.round(m), atol=1e-4):
        raise ValueError(f'grids do not correspond voxel for voxel:\n{img.affine}\n{ref.affine}')
    m = np.round(m).astype(int)
    ijk = np.indices(ref.shape[:3]).reshape(3, -1)
    src = m[:3, :3] @ ijk + m[:3, 3:4]
    if (src < 0).any() or (src >= np.array(data.shape)[:, None]).any():
        raise ValueError('grids do not cover the same field of view')
    return data[tuple(src)].reshape(ref.shape[:3])


def save_label_volume(data, ref, path):
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.int16)
    hdr.set_slope_inter(1, 0)
    img = nib.Nifti1Image(data.astype(np.int16), ref.affine, hdr)
    img.set_qform(ref.affine, code=int(ref.header['qform_code']) or 1)
    img.set_sform(ref.affine, code=int(ref.header['sform_code']) or 1)
    nib.save(img, path)


def voxel_size(affine):
    return float(np.abs(affine[0, 0]))


def on_side(x, hemi, tol):
    """Fraction of x coordinates in the named hemisphere, allowing one voxel past the midline
    (medial parcels touch x = 0; a left-right mirroring puts most voxels on the wrong side)."""
    return float(np.mean(x >= -tol)) if hemi == 'R' else float(np.mean(x <= tol))


def with_flags(df, fam):
    df = df.copy()
    df.insert(2, 'hemisphere', [hemisphere(n) for n in df['name']])
    for i, flag in enumerate(FLAGS):
        df.insert(3 + i, flag, {'cortex': fam in ('Schaefer', 'Glasser'),
                                'subcortex': fam == 'MSA', 'cerebellum': fam == 'MDTB10'}[flag])
    return df


def colors_for(df, cortex_table):
    """Label colors: from the cortical source table, the TSV's color column, or generated."""
    out = {}
    names = cortex_table or {}
    for k, (idx, name) in enumerate(zip(df['index'], df['name'])):
        if name in names:
            out[idx] = names[name]
        elif 'color' in df.columns and isinstance(df['color'].iloc[k], str):
            h = df['color'].iloc[k].lstrip('#')
            out[idx] = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,)
        else:
            out[idx] = colorsys.hsv_to_rgb((k * 0.618034) % 1, 0.65, 0.9) + (1.0,)
    return out


# --------------------------------------------------------------------------- build
class Builder:
    def __init__(self):
        self.report = []
        self.failures = []
        self.atlases = {}   # name -> dict(fam, sources, tsv DataFrame, kind)
        self.grey = nib.load(ROOT / 'sourcedata' / 'hcp' / '91282_Greyordinates.dscalar.nii').header.get_axis(1)

    def fail(self, msg):
        self.failures.append(msg)

    # -- components --------------------------------------------------------------
    def register_components(self):
        for f in sorted((ROOT / 'tpl-MNI152NLin6Asym').glob('*_res-02_dseg.tsv')):
            name = re.search(r'atlas-([^_]+)_', f.name).group(1)
            if not re.fullmatch(r'Schaefer\d+7|Glasser360|MSA\d|MDTB10', name):
                continue   # a combination: rebuilt below
            fam = family(name)
            df = with_flags(read_tsv(f), fam)
            if fam == 'Schaefer':
                # `name`: CBIG's current parcel names (also used by AtlasPack and by Tian's
                # Schaefer+subcortex release). The TemplateFlow volumes came with older names
                # for some parcels; those are kept in `templateflow_name`.
                if 'templateflow_name' not in df.columns:
                    df['templateflow_name'] = df['name']
                df['name'] = df['index'].map(self.cbig_names(name))
                assert df['name'].notna().all(), name
            self.atlases[name] = {'fam': fam, 'sources': [name], 'tsv': df, 'component': True}

    # -- combinations ------------------------------------------------------------
    def combos(self):
        # A combination is named after its components, e.g. Glasser360MSA2MDTB10.
        cortices = [f'Schaefer{n}7' for n in SCHAEFER_COMBO_SCALES] + ['Glasser360']
        for ctx in cortices:
            for s in MSA_SCALES:
                yield f'{ctx}MSA{s}', [ctx, f'MSA{s}']
                yield f'{ctx}MSA{s}MDTB10', [ctx, f'MSA{s}', 'MDTB10']

    def build_combo_tables(self):
        for name, parts in self.combos():
            dfs, offset = [], 0
            for p in parts:
                df = self.atlases[p]['tsv'].copy()
                df['index'] = df['index'] + offset
                if p == 'MDTB10':
                    df['name'] = 'cerebellum_' + df['name']
                offset = int(df['index'].max())
                dfs.append(df)
            df = pd.concat(dfs, ignore_index=True)
            assert df['index'].tolist() == list(range(1, len(df) + 1)), name
            self.atlases[name] = {'fam': 'combo', 'sources': parts, 'tsv': df, 'component': False}

    def build_combo_volumes(self):
        for name, parts in self.combos():
            for space in SPACES:
                for res in RESOLUTIONS:
                    ref = nib.load(vol_path(space, parts[0], res))
                    out = on_grid(ref, ref)
                    offset = len(self.atlases[parts[0]]['tsv'])
                    for p in parts[1:]:
                        lab = on_grid(nib.load(vol_path(space, p, res)), ref)
                        out = np.where(lab > 0, lab + offset, out)
                        offset += len(self.atlases[p]['tsv'])
                    save_label_volume(out, ref, vol_path(space, name, res))

    # -- fsLR ----------------------------------------------------------------------
    def cortex_vertex_labels(self, ctx):
        """Per-hemisphere vertex arrays (32492) of the cortical atlas's indices, and its colors."""
        if ctx.startswith('Schaefer'):
            n = re.fullmatch(r'Schaefer(\d+)7', ctx).group(1)
            src = ROOT / 'sourcedata' / 'Schaefer2018' / f'Schaefer2018_{n}Parcels_7Networks_order.dlabel.nii'
            key_col = 'name'
        else:
            src = ROOT / 'sourcedata' / 'xcpd' / 'tpl-fsLR_atlas-Glasser_den-32k_dseg.dlabel.nii'
            key_col = 'cifti_label'
        img = nib.load(src)
        lab_axis, bm = img.header.get_axis(0), img.header.get_axis(1)
        table = lab_axis.label[0]
        data = np.asarray(img.dataobj)[0].astype(int)
        df = self.atlases[ctx]['tsv']
        if key_col == 'name':
            # Schaefer: the TSV index is the CBIG key (same parcels, checked spatially against
            # the TemplateFlow volumes), so map by key.
            key_to_index = {k: k for k in table if k != 0}
            if set(key_to_index) != set(df['index']):
                self.fail(f'{ctx}: keys in {src.name} differ from the TSV indices')
        else:
            to_index = dict(zip(df[key_col], df['index']))
            key_to_index = {k: to_index[v[0]] for k, v in table.items() if k != 0}
            missing = set(df[key_col]) - {table[k][0] for k in table}
            if missing:
                self.fail(f'{ctx}: labels missing from {src.name}: {sorted(missing)[:5]}')
        verts = {}
        for struct, sl, b in bm.iter_structures():
            arr = np.zeros(b.nvertices[struct], int)
            arr[b.vertex] = [key_to_index.get(k, 0) for k in data[sl]]
            verts[struct] = arr
        name_of = dict(zip(df['index'], df['name']))
        colors = {name_of[i]: table[k][1] for k, i in key_to_index.items() if i in name_of}
        return verts, colors

    def cbig_names(self, ctx):
        n = re.fullmatch(r'Schaefer(\d+)7', ctx).group(1)
        src = ROOT / 'sourcedata' / 'Schaefer2018' / f'Schaefer2018_{n}Parcels_7Networks_order.dlabel.nii'
        table = nib.load(src).header.get_axis(0).label[0]
        return {k: v[0] for k, v in table.items() if k != 0}

    def build_fslr(self, name):
        info = self.atlases[name]
        df = info['tsv']
        parts = info['sources']
        ctx = next((p for p in parts if family(p) in ('Schaefer', 'Glasser')), None)
        has_vol = any(family(p) in ('MSA', 'MDTB10') for p in parts)
        bm = self.grey if has_vol else self.grey[self.grey.surface_mask]
        den = '91k' if has_vol else '32k'
        values = np.zeros(len(bm), int)
        colors = {}
        if ctx:
            verts, colors = self.cortex_vertex_labels(ctx)
            for struct, sl, b in bm.iter_structures():
                if struct in verts:
                    values[sl] = verts[struct][b.vertex]
        if has_vol:
            ref = nib.load(vol_path('MNI152NLin6Asym', name, 2))
            vol = np.asarray(ref.dataobj).squeeze().astype(int)
            vox = bm.voxel[bm.volume_mask]
            xyz = nib.affines.apply_affine(bm.affine, vox)
            ijk = nib.affines.apply_affine(np.linalg.inv(ref.affine), xyz)
            if not np.allclose(ijk, np.round(ijk), atol=1e-4):
                raise ValueError(f'{name}: MNI152NLin6Asym 2 mm grid does not match the 91k volume')
            sampled = vol[tuple(np.round(ijk).astype(int).T)]
            noncortex = set(df.loc[~df['cortex'], 'index'])
            values[bm.volume_mask] = np.where(np.isin(sampled, list(noncortex)), sampled, 0)
        rgba = colors_for(df, colors)
        table = {0: ('???', (0.0, 0.0, 0.0, 0.0))}
        table.update({int(i): (n, tuple(rgba[i])) for i, n in zip(df['index'], df['name'])})
        lab_axis = nib.cifti2.LabelAxis([name], [table])
        out = fslr_path(name, den, 'dlabel.nii')
        nib.save(nib.Cifti2Image(values[None, :].astype(np.float32), header=(lab_axis, bm)), out)
        self.check_fslr(name, out, df)
        return den

    # -- checks ----------------------------------------------------------------------
    def check_volume(self, name, space, res):
        img = nib.load(vol_path(space, name, res))
        data = np.asarray(img.dataobj).squeeze().astype(int)
        df = self.atlases[name]['tsv']
        present = set(np.unique(data)) - {0}
        missing = set(df['index']) - present
        extra = present - set(df['index'])
        if missing or extra:
            self.fail(f'{name} {space} res-0{res}: missing {sorted(missing)[:8]} extra {sorted(extra)[:8]}')
        x = nib.affines.apply_affine(img.affine, np.argwhere(data > 0))[:, 0]
        lab = data[data > 0]
        for idx, hemi in zip(df['index'], df['hemisphere']):
            if hemi == 'bilateral' or idx not in present:
                continue
            xs = x[lab == idx]
            frac = on_side(xs, hemi, voxel_size(img.affine))
            if frac < 0.9:
                self.fail(f'{name} {space} res-0{res}: label {idx} ({hemi}) has only {frac:.0%} of voxels in that hemisphere')

    def check_fslr(self, name, path, df):
        img = nib.load(path)
        lab_axis, bm = img.header.get_axis(0), img.header.get_axis(1)
        table = lab_axis.label[0]
        if {k: v[0] for k, v in table.items() if k} != dict(zip(df['index'], df['name'])):
            self.fail(f'{name} fsLR: label table does not match the TSV')
        data = np.asarray(img.dataobj)[0].astype(int)
        missing = set(df['index']) - set(np.unique(data))
        if missing:
            names = df.set_index('index').loc[sorted(missing), 'name'].tolist()
            self.report.append(f'{name} fsLR: {len(missing)} label(s) have no grayordinates: {names}')
        hemi_of = dict(zip(df['index'], df['hemisphere']))
        for struct, sl, b in bm.iter_structures():
            vals = data[sl]
            if struct in ('CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT'):
                want = 'L' if struct.endswith('LEFT') else 'R'
                wrong = {v for v in np.unique(vals) if v and hemi_of.get(v) not in (want, 'bilateral')}
                if wrong:
                    self.fail(f'{name} fsLR {struct}: labels from the other hemisphere: {sorted(wrong)[:5]}')
        if bm.volume_mask.any():
            x = nib.affines.apply_affine(bm.affine, bm.voxel[bm.volume_mask])[:, 0]
            vals = data[bm.volume_mask]
            for idx in np.unique(vals[vals > 0]):
                h = hemi_of[idx]
                if h != 'bilateral':
                    frac = on_side(x[vals == idx], h, voxel_size(bm.affine))
                    if frac < 0.9:
                        self.fail(f'{name} fsLR volume: label {idx} ({h}) has only {frac:.0%} of voxels in that hemisphere')

    # -- metadata ----------------------------------------------------------------------
    def describe(self, name):
        info = self.atlases[name]
        if info['component']:
            fam = info['fam']
            if fam == 'Schaefer':
                n = re.fullmatch(r'Schaefer(\d+)7', name).group(1)
                desc = (f'Schaefer 2018 cortical parcellation, {n} parcels, 7-network order. Volumes from '
                        'TemplateFlow; fsLR from CBIG (medial wall removed).')
            elif fam == 'Glasser':
                desc = ('HCP-MMP1.0 cortical parcellation (Glasser et al. 2016), 360 parcels, from XCP-D. '
                        'Named Glasser360 here because XCP-D has a built-in atlas called "Glasser" '
                        '(the same parcellation) and stops when two atlas datasets share a name.')
            elif fam == 'MSA':
                s = int(name[-1])
                desc = f'Melbourne Subcortex Atlas (Tian et al. 2020), scale S{s} ({MSA_PARCELS[s]} parcels), 3T group parcellation.'
            else:
                desc = 'MDTB 10-region functional parcellation of the cerebellum (King et al. 2019).'
            obj = {'Name': name, 'Description': desc, 'License': LICENSES[fam], 'ReferencesAndLinks': REFS[fam]}
        else:
            parts = info['sources']
            desc = (' + '.join(parts) + ': built by code/build_atlases.py. Labels are numbered '
                    + ', then '.join(parts) + '. Where they overlap, '
                    + ('MSA overrides cortex and MDTB10 overrides both.' if 'MDTB10' in parts else 'MSA overrides cortex.'))
            obj = {'Name': name, 'Description': desc,
                   'License': 'Inherits the terms of its source atlases.',
                   'ReferencesAndLinks': [r for p in parts for r in REFS[family(p)]],
                   'SourceAtlases': {p: {'License': LICENSES[family(p)]} for p in parts}}
        write_json(obj, ROOT / f'atlas-{name}_description.json')

    def sidecars(self, name, den):
        df = self.atlases[name]['tsv']
        cols = {c: COLUMN_DOCS[c] for c in df.columns if c in COLUMN_DOCS}
        generated = {'GeneratedBy': 'code/build_atlases.py'} if not self.atlases[name]['component'] else {}
        for space in SPACES:
            for res in RESOLUTIONS:
                write_tsv(df, vol_path(space, name, res, 'tsv'))
                write_json({'Name': name, 'Resolution': f'{res} mm isotropic', **generated, **cols},
                           vol_path(space, name, res, 'json'))
        write_tsv(df, fslr_path(name, den, 'tsv'))
        dens = {'32k': '59412 cortical surface vertices (32k fsLR, medial wall excluded). No subcortical data.',
                '91k': '91282 grayordinates: 59412 cortical surface vertices (32k fsLR) and 31870 subcortical '
                       'and cerebellar voxels (MNI152NLin6Asym, 2 mm).'}
        write_json({'Name': name, 'Density': {den: dens[den]}, 'GeneratedBy': 'code/build_atlases.py', **cols},
                   fslr_path(name, den, 'json'))

    # -- main ------------------------------------------------------------------------
    def run(self):
        (ROOT / 'tpl-fsLR').mkdir(exist_ok=True)
        self.register_components()
        self.build_combo_tables()
        print(f'{sum(a["component"] for a in self.atlases.values())} components, '
              f'{sum(not a["component"] for a in self.atlases.values())} combinations')
        self.build_combo_volumes()
        expected = set()
        for name in sorted(self.atlases):
            for space in SPACES:
                for res in RESOLUTIONS:
                    self.check_volume(name, space, res)
            den = self.build_fslr(name)
            self.sidecars(name, den)
            self.describe(name)
            expected |= {f'atlas-{name}_description.json', *(p.name for p in [fslr_path(name, den, e) for e in ('dlabel.nii', 'tsv', 'json')])}
            expected |= {vol_path(s, name, r, e).name for s in SPACES for r in RESOLUTIONS for e in ('nii.gz', 'tsv', 'json')}
            print(f'  {name}: {len(self.atlases[name]["tsv"])} labels, fsLR den-{den}')
        # Stale files from atlases no longer built.
        found = {p.name for p in ROOT.glob('atlas-*_description.json')} | {p.name for p in ROOT.glob('tpl-*/*')}
        if found - expected:
            self.fail(f'files not produced by this build: {sorted(found - expected)[:10]}')
        write_json({
            'Name': 'SNaP Lab atlases',
            'BIDSVersion': '1.11.0',
            'DatasetType': 'derivative',
            'GeneratedBy': [{'Name': 'snaplab_tools build_atlases.py', 'CodeURL': 'code/build_atlases.py'}],
            'License': 'Each atlas carries the terms of its sources; see atlas-<name>_description.json '
                       'and THIRD_PARTY_NOTICES.md in the snaplab_tools repository.',
        }, ROOT / 'dataset_description.json')
        for line in self.report:
            print('note:', line)
        if self.failures:
            print(f'\n{len(self.failures)} CHECK(S) FAILED:')
            for f in self.failures:
                print('  ', f)
            sys.exit(1)
        print('\nall checks passed')


if __name__ == '__main__':
    Builder().run()
