"""Sphinx configuration for the snaplab_tools documentation.

Every notebook under ``tutorials/`` is executed during the build (see the myst-nb settings below),
so a change that breaks the API breaks the build rather than leaving stale output behind. That is
the point of the arrangement -- keep it that way.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# -- Rendering environment ---------------------------------------------------------------------
# Where the Schaefer FreeSurfer .annot files live, for the cortical-surface plotting functions.
# snaplab_tools reads this on each call, so it does not matter whether the package has already
# been imported by the time we get here.
os.environ.setdefault(
    "SCHAEFER_ANNOT_DIR",
    os.path.expanduser("~/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3"),
)
# Deliberately NOT setting MPLBACKEND here. Notebook kernels inherit this environment, and forcing
# 'Agg' makes plt.show() a no-op that discards the figure instead of emitting it as cell output --
# every tutorial would render with no images at all. ipykernel's default (matplotlib_inline) is
# already headless and is what captures figures. Brain-surface rendering goes through VTK, which
# does need a display; that is handled by the Xvfb block below.

if os.environ.get("READTHEDOCS") == "True":
    # surfplot/brainspace render through VTK, which needs an X display even when writing to a file.
    # Start a virtual one and point DISPLAY at it; notebook kernels are spawned as child processes
    # and inherit both.
    try:
        subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1024x768x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = ":99"
    except FileNotFoundError:
        # No Xvfb (it comes from build.apt_packages in .readthedocs.yaml). Surface rendering will
        # fail loudly during notebook execution rather than silently producing blank images.
        print("WARNING: Xvfb not found; VTK surface rendering will fail.", file=sys.stderr)

# -- Project information -----------------------------------------------------------------------
project = "snaplab_tools"
author = "SNaP Lab"
copyright = "%Y, SNaP Lab"

# Read the version from the package source rather than importing it, so the docs build does not
# depend on the package being importable at config time.
_init = (_ROOT / "snaplab_tools" / "__init__.py").read_text()
release = re.search(r'^__version__ = ["\'](.*)["\']', _init, re.M).group(1)
version = release

# -- General configuration ---------------------------------------------------------------------
extensions = [
    "myst_nb",                    # MyST Markdown + executable notebooks (pulls in myst_parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # NumPy-style docstrings, as used throughout the package
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# -- Autodoc / autosummary ---------------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
# Keep signatures readable: show argument names as written rather than resolving type aliases.
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Intersphinx -------------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "nilearn": ("https://nilearn.github.io/stable/", None),
}
# Missing objects in third-party packages should not fail a -W build.
nitpicky = False

# -- MyST / notebook execution -----------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
]
myst_heading_anchors = 3

# Notebooks are NOT executed here -- they ship with their outputs committed, and this build only
# renders them.
#
# Executing during the docs build meant every build installed the full scientific stack (VTK and
# brainspace alone are ~200 MB) and depended on several third-party downloads succeeding, so a
# transient network failure took the documentation offline for reasons unrelated to the docs.
#
# The check that notebooks still run has moved to CI (.github/workflows/tutorials.yml), which
# executes every notebook on push and fails if any of them break. That keeps the guarantee while
# making the published build cheap and reliable. See docs/contributing.md for the workflow.
nb_execution_mode = "off"
nb_merge_streams = True

# -- HTML output -------------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"{project} {release}"

html_theme_options = {
    "github_url": "https://github.com/LindenParkesLab/snaplab_tools",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/LindenParkesLab/snaplab_tools",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_prev_next": True,
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

html_sidebars = {
    "index": [],
}

html_context = {
    "github_user": "LindenParkesLab",
    "github_repo": "snaplab_tools",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}
