"""Tests for figure export and task-block shading helpers.

`save_panel` exists so a project's panels cannot drift in resolution or background, which only
holds if its defaults are predictable: the content-aware DPI choice and the transparent/opaque
switch are the parts worth pinning.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from snaplab_tools.plotting.utils import DPI_LINE, DPI_RASTER, save_panel


@pytest.fixture
def line_fig():
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    yield fig
    plt.close(fig)


@pytest.fixture
def raster_fig():
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(np.arange(400).reshape(20, 20))
    yield fig
    plt.close(fig)


class TestSavePanel:
    def test_writes_every_requested_format_and_returns_the_shared_stem(self, line_fig, tmp_path):
        stem = save_panel(line_fig, str(tmp_path / 'panels'), 'fig1')
        assert stem == str(tmp_path / 'panels' / 'fig1')
        for ext in ('png', 'svg'):
            assert (tmp_path / 'panels' / f'fig1.{ext}').exists()

    def test_creates_the_output_directory(self, line_fig, tmp_path):
        save_panel(line_fig, str(tmp_path / 'a' / 'b'), 'fig1', formats=('png',))
        assert (tmp_path / 'a' / 'b' / 'fig1.png').exists()

    # Sizes are compared within one figure, never across two: bbox_inches='tight' crops to the
    # drawn content, so two different figures at identical DPI need not share a pixel size.
    def test_a_raster_panel_defaults_to_DPI_RASTER(self, raster_fig, tmp_path):
        auto = Image.open(save_panel(raster_fig, str(tmp_path), 'auto', formats=('png',)) + '.png')
        pinned = Image.open(save_panel(raster_fig, str(tmp_path), 'pin',
                                       dpi=DPI_RASTER, formats=('png',)) + '.png')
        assert auto.size == pinned.size

    def test_a_line_panel_defaults_to_DPI_LINE(self, line_fig, tmp_path):
        auto = Image.open(save_panel(line_fig, str(tmp_path), 'auto', formats=('png',)) + '.png')
        pinned = Image.open(save_panel(line_fig, str(tmp_path), 'pin',
                                       dpi=DPI_LINE, formats=('png',)) + '.png')
        assert auto.size == pinned.size

    def test_the_two_defaults_actually_differ(self, line_fig, tmp_path):
        low = Image.open(save_panel(line_fig, str(tmp_path), 'lo',
                                    dpi=DPI_LINE, formats=('png',)) + '.png')
        high = Image.open(save_panel(line_fig, str(tmp_path), 'hi',
                                     dpi=DPI_RASTER, formats=('png',)) + '.png')
        assert high.width > low.width

    def test_explicit_dpi_overrides_the_content_choice(self, raster_fig, tmp_path):
        auto = Image.open(save_panel(raster_fig, str(tmp_path), 'auto', formats=('png',)) + '.png')
        forced = Image.open(save_panel(raster_fig, str(tmp_path), 'forced',
                                       dpi=100, formats=('png',)) + '.png')
        assert forced.width < auto.width

    def test_transparent_leaves_the_background_clear(self, line_fig, tmp_path):
        p = save_panel(line_fig, str(tmp_path), 'clear', formats=('png',), transparent=True)
        assert Image.open(p + '.png').convert('RGBA').getpixel((0, 0))[3] == 0

    def test_default_background_is_opaque(self, line_fig, tmp_path):
        p = save_panel(line_fig, str(tmp_path), 'solid', formats=('png',))
        assert Image.open(p + '.png').convert('RGBA').getpixel((0, 0))[3] == 255
