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
distance matrices bundled with the package:

```python
import matplotlib.pyplot as plt
from snaplab_tools.nulls import generate_surrogates, corr_with_null
from snaplab_tools.plotting.plotting import plot_correlation

# x and y are (400,) parcellated maps in Schaefer 400 7-network order
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

Found a missing dependency, a bug, or something undocumented? Open an issue or a pull request. See
the [contributing guide](https://snaplab-tools.readthedocs.io/en/latest/contributing.html) for
docstring conventions and how to add a tutorial.
