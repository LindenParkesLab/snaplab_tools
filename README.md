# SNaP Lab Tools

[![Documentation Status](https://readthedocs.org/projects/snaplab-tools/badge/?version=latest)](https://snaplab-tools.readthedocs.io/en/latest/?badge=latest)

Common analysis functions and tools used by the SNaP Lab: spatial null models, GAM fitting and
change-point detection, correlation statistics, network topology, cross-validated prediction, and
publication-figure plotting for human neuroimaging data.

**Documentation: https://snaplab-tools.readthedocs.io**

## Installation

```bash
git clone https://github.com/LindenParkesLab/snaplab_tools.git
cd snaplab_tools
pip install -e .
```

Optional extras:

```bash
pip install -e ".[surface]"      # VTK-backed cortical surface rendering (surfplot, brainspace)
pip install -e ".[changepoint]"  # multi change-point / non-L2 cost models (ruptures)
pip install -e ".[docs]"         # build the documentation locally
```

Dependencies are declared in [`setup.py`](setup.py) and installed automatically -- there is no
separate list to keep in sync.

## Quick start

Testing a correlation between two brain maps against a spatial null, using the Schaefer geodesic
distance matrices bundled with the package. This runs as written — swap the two synthetic maps for
your own `(400,)` vectors in Schaefer 400 7-network order:

```python
import matplotlib.pyplot as plt
from snaplab_tools.datasets import make_spatial_map, make_correlated_map
from snaplab_tools.nulls import generate_surrogates, corr_with_null
from snaplab_tools.plotting.plotting import plot_correlation

x = make_spatial_map(n_regions=400, seed=0)
y = make_correlated_map(x, rho=0.35, seed=1)

surrogates = generate_surrogates(y, n_perms=5000)
result = corr_with_null(x, y, surrogates, method="spearman")
print(f"rho = {result['r']:.2f}, p_smash = {result['p_smash']:.3f}")

fig, ax = plt.subplots(figsize=(3, 3))
plot_correlation(x, y, ax, x_label="Map X", y_label="Map Y", method="spearman")
plt.show()
```

See the [API reference](https://snaplab-tools.readthedocs.io/en/latest/api/index.html) for
everything else.

## Building the docs

```bash
pip install -e ".[docs,surface,changepoint]"
cd docs && make html
```

Executable pages are run as part of the build, so a broken API fails the build rather than leaving
stale output behind. Open `docs/_build/html/index.html` when it finishes.

## Contributing

These tools are developed around the lab's own needs. Pull requests are welcome — bug fixes,
missing dependencies, and improvements to what is already here especially. We are not taking
feature requests.

See the [contributing guide](https://snaplab-tools.readthedocs.io/en/latest/contributing.html) for
docstring conventions and how to add a tutorial.

## Licence

The source code in this repository is released under the
[BSD 3-Clause License](LICENSE), the usual choice in this corner of the field — the same licence
as nilearn, MNE-Python and BrainSpace.

The brain atlases and surface files redistributed here carry their own terms (all permissive, and
compatible with the above). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full list
and the papers to cite if you use them.
