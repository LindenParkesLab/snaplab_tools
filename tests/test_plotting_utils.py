"""Tests for plotting configuration and the parcel-averaging input checks.

These pin down behaviour that used to be surprising: import-time side effects, an environment
variable that could only be set before import, and file-extension handling that failed with an
UnboundLocalError instead of a message.
"""
import os

import matplotlib.pyplot as plt
import pytest

from snaplab_tools.plotting.utils import register_custom_colormaps, set_plotting_params
from snaplab_tools.utils import get_parcelwise_average_nifti, get_parcelwise_average_surface


def test_importing_plotting_does_not_enable_interactive_mode():
    """A library import must not flip matplotlib into interactive mode process-wide.

    Checked in a fresh interpreter: importlib.reload() would not tell us what a first import
    does in a clean process, and reloading pollutes this one's warning state.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, '-c',
         'import matplotlib; matplotlib.use("Agg");'
         'import matplotlib.pyplot as plt;'
         'import snaplab_tools.plotting.plotting;'
         'print(plt.isinteractive())'],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith('False'), (
        f'importing the plotting module enabled interactive mode: {result.stdout!r}'
    )


def test_schaefer_annot_dir_is_read_at_call_time(monkeypatch):
    """SCHAEFER_ANNOT_DIR must take effect even when set after the module was imported.

    It used to be captured into a module-level constant at import, so setting it later did
    nothing at all -- silently, and only visible as a confusing 'file not found'.
    """
    from snaplab_tools.plotting.plotting import _schaefer_annot_dir

    monkeypatch.setenv('SCHAEFER_ANNOT_DIR', '/tmp/annots-one')
    assert _schaefer_annot_dir() == '/tmp/annots-one'

    monkeypatch.setenv('SCHAEFER_ANNOT_DIR', '/tmp/annots-two')
    assert _schaefer_annot_dir() == '/tmp/annots-two'


def test_schaefer_annot_dir_falls_back_to_default(monkeypatch):
    from snaplab_tools.plotting.plotting import _schaefer_annot_dir

    monkeypatch.delenv('SCHAEFER_ANNOT_DIR', raising=False)
    assert _schaefer_annot_dir() == os.path.expanduser(
        '~/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3'
    )


def test_set_plotting_params_applies_the_documented_values():
    """The 8pt size is documented and explicitly requested, so it must survive.

    seaborn's set_theme() rewrites rcParams wholesale; calling it after the explicit assignments
    silently replaced 8pt with seaborn's 'paper' default of 9.6pt.
    """
    set_plotting_params(format='pdf')

    assert plt.rcParams['font.size'] == 8
    assert plt.rcParams['pdf.fonttype'] == 42     # Type-42 keeps text editable in Illustrator
    assert plt.rcParams['ps.fonttype'] == 42
    assert plt.rcParams['svg.fonttype'] == 'none'
    assert plt.rcParams['savefig.format'] == 'pdf'


def test_set_plotting_params_leaves_the_font_cache_alone():
    """It used to run `rm -rf ~/.cache/matplotlib` on macOS on every call."""
    import matplotlib

    cache_dir = matplotlib.get_cachedir()
    existed = os.path.isdir(cache_dir)
    set_plotting_params()
    assert os.path.isdir(cache_dir) == existed


def test_registered_presets_are_ours_and_not_another_library_s():
    """`cmap='<preset>'` must resolve to the vetted preset, not a same-named map from elsewhere.

    Matplotlib's colormap registry is process-global and first registration wins, silently. The
    presets used to be called 'blue_orange' and 'blue_red' among others, and importing nilearn --
    which snaplab_tools.plotting.utils does at import time -- registers colormaps under both of
    those names, so two presets resolved to nilearn's maps instead. Hence the 'snap_' prefix.
    """
    import matplotlib as mpl

    presets = register_custom_colormaps()
    assert presets, 'no colormaps registered'

    for name, cmap in presets.items():
        assert name.startswith('snap_'), f'{name} is unprefixed and can be claimed by any library'
        registered = mpl.colormaps[name]
        assert registered(0.0) == cmap(0.0) and registered(1.0) == cmap(1.0), (
            f"cmap='{name}' resolves to a different colormap than the preset of that name"
        )
        assert mpl.colormaps[name + '_r'](0.0) == cmap(1.0), f'{name}_r is not the reverse of {name}'


def test_register_custom_colormaps_is_idempotent():
    """Documented as safe to call at the top of every notebook, so a second call must not raise."""
    register_custom_colormaps()
    register_custom_colormaps()


@pytest.mark.parametrize('func,bad_file,expected', [
    (get_parcelwise_average_nifti, 'volume.mgz', '.nii'),
    (get_parcelwise_average_surface, 'surface.foo', '.gii'),
])
def test_unsupported_extension_raises_before_reading_anything(func, bad_file, expected):
    """A bad extension is a clear ValueError, raised before any file is opened.

    Previously it fell through the if/elif chain leaving `data` unbound, so the error surfaced
    as an UnboundLocalError several lines later -- and only after the parcellation had been read.
    """
    with pytest.raises(ValueError) as excinfo:
        func(bad_file, '/definitely/does/not/exist.annot')

    message = str(excinfo.value)
    assert 'Unsupported data file extension' in message
    assert expected in message
