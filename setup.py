import re
from pathlib import Path

from setuptools import find_packages, setup

# Read the version straight out of the package rather than importing it -- importing would
# require every runtime dependency to already be installed, which is not true during a build.
_init = Path(__file__).parent / "snaplab_tools" / "__init__.py"
version = re.search(r'^__version__ = ["\'](.*)["\']', _init.read_text(), re.M).group(1)

setup(
    name="snaplab_tools",
    version=version,
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "nibabel",
        "nilearn",
        "brainsmash",       # BrainSMASH variogram nulls (snaplab_tools.nulls)
        "bctpy",
        "pygam",            # penalized-spline GAM fitting (snaplab_tools.gams)
        "joblib",           # parallel bootstrap engine (snaplab_tools.gams)
        "tqdm",
        "matplotlib",
        "seaborn>=0.11",   # set_theme() (snaplab_tools.plotting.utils) landed in 0.11
        "Pillow",
        "GitPython",
        "wget",             # remote parcellation/brain-map fetching (utils, brainmaps)
    ],
    extras_require={
        # Multi change-point / non-L2 cost models in snaplab_tools.gams. The single-boundary
        # L2 detector is exact and dependency-free; ruptures is only needed beyond that.
        "changepoint": ["ruptures"],
        # VTK-backed cortical surface rendering (plot_brain_surface_data and friends). Heavy
        # (pulls VTK) and only imported lazily inside the functions that need it.
        "surface": [
            "surfplot",
            "brainspace",
        ],
        # Building the Sphinx documentation site. ipykernel/jupyter-cache are what execute
        # notebook-backed pages during the build; nothing uses them yet, but the tutorials being
        # rewritten will, and myst-nb is configured for it.
        "docs": [
            "sphinx>=7.2",
            "myst-nb>=1.1",
            "pydata-sphinx-theme>=0.15",
            "sphinx-copybutton>=0.5",
            "sphinx-design>=0.6",
            "ipykernel",
            "jupyter-cache",
        ],
    },
    # Ship the bundled null-model resources (surfaces, parcellations, prebuilt distance matrices)
    # inside the wheel so `snaplab_tools.nulls` is self-contained on any machine, not just an
    # editable dev checkout.
    package_data={
        "snaplab_tools.nulls": [
            "resources/surfaces/*.surf.gii",
            "resources/parcellations/*.dlabel.nii",
            "resources/parcellations/*.csv",
            "resources/distances/*.npy",
        ],
    },
)
