"""Building blocks for the figures in :mod:`snaplab_tools.plotting.plotting`.

Useful on their own, not just as internals.

Style and colour
    :func:`set_plotting_params` sets the lab's matplotlib defaults (8pt fonts, editable Type-42
    PDF text, seaborn paper style) -- call it once at the top of a notebook.
    :func:`get_my_colors` returns the named lab palette.

Colormaps
    :func:`make_diverging_cmap` and :func:`make_sequential_cmap` build and register matplotlib
    colormaps (each with an automatic ``_r`` reverse); :func:`register_custom_colormaps` installs
    the presets and :func:`show_colormaps` previews them. :func:`cvd_min_delta_e` measures the
    smallest perceptual distance between two colours across simulated colour-vision deficiencies
    -- use it to check a categorical palette stays distinguishable.

Statistics annotation
    :func:`create_correlation_text`, :func:`format_pvalue`, and :func:`add_stats_annotation` turn a pair of vectors into the annotated text block on a
    correlation plot; :func:`create_null_inset` draws the embedded null distribution. The
    statistics themselves come from :mod:`snaplab_tools.stats` -- this module only formats them.

Axis helpers
    :func:`style_correlation_axis`, :func:`add_module_lines` (system boundaries on a matrix
    plot), :func:`process_input_data` (shared input validation), and :func:`roi_to_vtx` (project
    parcel values onto surface vertices).
"""
import os
import numpy as np
import pandas as pd
import scipy as sp
import nibabel as nib

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from nilearn import plotting

from snaplab_tools.stats import significance_stars

__all__ = [
    'set_plotting_params',
    'get_my_colors',
    'make_diverging_cmap',
    'make_sequential_cmap',
    'register_custom_colormaps',
    'show_colormaps',
    'cvd_min_delta_e',
    'create_correlation_text',
    'format_pvalue',
    'add_stats_annotation',
    'create_null_inset',
    'style_correlation_axis',
    'add_module_lines',
    'process_input_data',
    'roi_to_vtx',
]


def set_plotting_params(format='png'):
    """Set global matplotlib/seaborn parameters for publication figures.

    Sets Type-42 (editable) PDF/PS fonts, an 8pt base font size, the savefig format, and a seaborn
    white/paper style (no background grid).

    Type-42 fonts and ``svg.fonttype='none'`` are the settings that matter at submission time:
    they keep text as text rather than outlines, so figures stay editable in Illustrator and
    journals stop complaining.

    Parameters
    ----------
    format : str
        Default savefig format (e.g. 'png', 'pdf', 'svg').

    Notes
    -----
    This used to delete the matplotlib font cache (``rm -rf ~/.cache/matplotlib``) on macOS every
    time it was called. That is not this function's business, it made every call slow, and it
    forced a full font re-scan on the next plot. If you genuinely need to rebuild the cache after
    installing a font, do it once yourself::

        import matplotlib, shutil
        shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)
    """
    # seaborn first: set_theme() rewrites rcParams wholesale, including font.size, so the explicit
    # values below have to come after it or they are silently discarded.
    sns.set_theme(style='white', context='paper', font_scale=1)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['savefig.format'] = format
    plt.rcParams['font.size'] = 8
    plt.rcParams['svg.fonttype'] = 'none'


def get_my_colors(normalize=True, as_list=False, cat_trio=False):
    """Return the lab's named colour palette.

    Parameters
    ----------
    normalize : bool
        Scale RGB values from 0-255 to 0-1.
    as_list : bool
        Return a list of colours instead of a name->colour dict.
    cat_trio : bool
        Return only the three categorical colours (raspberry, blue, green).

    Returns
    -------
    dict or list
        Named colours, or a list of them if as_list.
    """
    # color palette (RGB / HEX):
    # raspberry blush: rgba(234,86,81,255) / #ea5651
    # conch shell: rgba(238,186,169,255) / #eebaa9
    # cinnamon: rgba(165,74,54,255) / #a54a36
    # wenge: rgba(63,44,41,255) / #3f2c29
    # savannah green: rgba(194,158,62,255) / #c29e3e
    # new age: rgba(217,206,209,255) / #d9ced1
    # starry night blue: rgba(48,65,121,255) / #304179
    # north sea green: rgba(0,111,116,255) / #006f74
    my_colors = dict()
    my_colors['raspberry_blush'] = [234, 86, 81]
    my_colors['starry_night_blue'] = [48, 65, 121]
    my_colors['north_sea_green'] = [0, 111, 116]
    if not cat_trio:
        my_colors['conch_shell'] = [238, 186, 169]
        my_colors['cinnamon'] = [165, 74, 54]
        my_colors['wenge'] = [63, 44, 41]
        my_colors['savannah_green'] = [194, 158, 62]
        my_colors['new_age'] = [217, 206, 209]

    if normalize:
        for key in my_colors.keys():
            my_colors[key] = [color / 255 for color in my_colors[key]]

    if as_list:
        my_colors = list(my_colors.values())

    return my_colors


def roi_to_vtx(roi_data, annot_file):
    """Map per-ROI values onto surface vertices using a FreeSurfer annotation.

    Parameters
    ----------
    roi_data : (n_rois,) array-like
        Per-ROI values (ROI label i maps to roi_data[i - 1]; label 0 is medial wall).
    annot_file : str
        Path to the FreeSurfer .annot file.

    Returns
    -------
    vtx_data : ndarray
        Per-vertex values.
    vtx_data_min, vtx_data_max : float
        Min/max of the mapped values (both 0 if fewer than 2 unique values).
    """
    labels, ctab, surf_names = nib.freesurfer.read_annot(annot_file)
    vtx_data = np.zeros(labels.shape)

    unique_labels = np.unique(labels)
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    for i in unique_labels:
        vtx_data[labels == i] = roi_data[i - 1]

    # get min/max for plottin
    x = np.sort(np.unique(vtx_data))

    if x.shape[0] > 1:
        vtx_data_min = x[0]
        vtx_data_max = x[-1]
    else:
        vtx_data_min = 0
        vtx_data_max = 0

    return vtx_data, vtx_data_min, vtx_data_max


def add_module_lines(modules, ax):
    """Draw white boundary boxes around contiguous module blocks on a matrix axis.

    Parameters
    ----------
    modules : pd.Series
        Per-node module labels, in matrix order.
    ax : matplotlib Axes
        Axis showing the matrix to annotate.
    """

    # get unqiue modules
    unique_modules = modules.unique()
    print(unique_modules)

    previous = -1
    for i in np.arange(len(unique_modules)):

        # get box boundaries using first and last occurence of module name
        bool_array = np.asarray(modules == unique_modules[i])
        n = len(bool_array)
        first = -1
        last = -1
        for i in range(0, n):
            if (bool_array[i] != True):
                continue
            if (first == -1):
                first = i
            last = i

        # draw box
        ax.hlines(last + 1, previous + 1, last + 1, colors='w')
        ax.vlines(last + 1, previous + 1, last + 1, colors='w')
        ax.hlines(first, previous + 1, last + 1, colors='w')
        ax.vlines(first, previous + 1, last + 1, colors='w')

        # update previous
        previous = last
        

def process_input_data(x, y, data_group=None):
    """
    Process and validate input data for correlation plotting.
    
    Parameters
    ----------
    x, y : array-like or pd.Series
        Input variables to correlate
    data_group : array-like, optional
        Group labels for data points
        
    Returns
    -------
    dict
        Dictionary containing:
        - x_clean : cleaned x data
        - y_clean : cleaned y data
        - x_label : extracted x label (or None)
        - y_label : extracted y label (or None)
        - valid_mask : boolean mask of valid data
        - data_group_clean : cleaned data_group (if provided)
    """
    # Extract Series names if present
    x_label = x.name if isinstance(x, pd.Series) else None
    y_label = y.name if isinstance(y, pd.Series) else None
    
    # Convert to numpy arrays
    x = np.array(x, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    
    # Validate dimensions
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length. Got x={len(x)}, y={len(y)}")
    
    # Handle data_group if provided
    if data_group is not None:
        data_group = np.array(data_group)
        if len(data_group) != len(x):
            raise ValueError(
                f"data_group length ({len(data_group)}) must match x and y length ({len(x)})"
            )
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if len(x_clean) < 3:
        raise ValueError("Need at least 3 valid data points for correlation")
    
    result = {
        'x_clean': x_clean,
        'y_clean': y_clean,
        'x_label': x_label,
        'y_label': y_label,
        'valid_mask': valid_mask,
        'n_valid': len(x_clean),
        'n_invalid': len(x) - len(x_clean)
    }
    
    if data_group is not None:
        result['data_group_clean'] = data_group[valid_mask]
    
    return result


def format_pvalue(p_value):
    """
    Format p-value for display.
    
    Parameters
    ----------
    p_value : float
        P-value to format
        
    Returns
    -------
    str
        Formatted p-value string
    """
    if np.isnan(p_value):
        return 'p=NaN'
    elif p_value < 0.01:
        return f'p={p_value:.2e}'
    else:
        return f'p={p_value:.2f}'


def create_correlation_text(corr_coef, p_value, method='pearson', 
                            n_outliers=0, model_info=None, 
                            data_group_info=None, ttest_stats=None):
    """
    Create formatted text for correlation statistics.
    
    Parameters
    ----------
    corr_coef : float
        Correlation coefficient
    p_value : float
        P-value for correlation (can be computed or custom)
    method : str
        Correlation method ('pearson' or 'spearman')
    n_outliers : int, optional
        Number of outliers detected
    model_info : dict, optional
        Information about polynomial model fit (for plot_correlation)
        Should contain: 'degree', 'r_squared', 'rmse', 'mae', 'metric_used'
    data_group_info : dict, optional
        Information about data groups (for plot_correlation)
        Should contain: 'unique_groups', 'group_type'
    ttest_stats : dict, optional
        T-test statistics (for plot_correlation_unity)
        Should contain: 't_stat', 'p_value'
        
    Returns
    -------
    str
        Formatted statistics text
    """
    # Get significance and formatted p-value
    sig_stars = significance_stars(p_value)
    p_str = format_pvalue(p_value)
    
    # Create correlation symbol
    corr_symbol = 'ρ' if method.lower() == 'spearman' else 'r'
    
    # Build statistics text
    stats_text = f"{corr_symbol}={corr_coef:.2f}{sig_stars}\n{p_str}"
    
    # Add data group information if provided
    if data_group_info is not None:
        unique_groups = data_group_info.get('unique_groups', [])
        if len(unique_groups) <= 5:
            if data_group_info.get('group_type') == 'string':
                group_labels = ", ".join([str(g) for g in unique_groups])
                if len(group_labels) <= 20:
                    stats_text += f"\n({group_labels})"
    
    # Add model information if provided (for auto_polynomial)
    if model_info is not None:
        degree_names = {1: 'Linear', 2: 'Quadratic', 3: 'Cubic', 
                       4: 'Quartic', 5: 'Quintic', 6: 'Sextic'}
        degree = model_info.get('degree', 1)
        metric_used = model_info.get('metric_used', 'variance_explained')
        
        if metric_used != 'linear_only':
            model_name = degree_names.get(degree, f"Degree {degree}")
            
            metric_values = {
                'variance_explained': model_info.get('r_squared', 0),
                'rmse': model_info.get('rmse', 0),
                'mae': model_info.get('mae', 0)
            }
            metric_labels = {
                'variance_explained': 'R²',
                'rmse': 'RMSE',
                'mae': 'MAE'
            }
            
            metric_value = metric_values.get(metric_used, 0)
            metric_label = metric_labels.get(metric_used, metric_used)
            stats_text += f"\n{metric_label}={metric_value:.2f}"
    
    # Add outlier count if present
    if n_outliers > 0:
        stats_text += f"\nOutliers: {n_outliers}"
    
    # Add t-test information if provided (for plot_correlation_unity)
    if ttest_stats is not None:
        t_stat = ttest_stats.get('t_stat')
        t_p_value = ttest_stats.get('p_value')
        if not np.isnan(t_stat) and not np.isnan(t_p_value):
            t_sig_stars = significance_stars(t_p_value)
            t_p_str = format_pvalue(t_p_value)
            stats_text += f"\nt = {t_stat:.2f}{t_sig_stars}\n{t_p_str}"
    
    return stats_text


def add_stats_annotation(ax, stats_text, position='upper left', fontsize=6):
    """
    Add statistics text annotation to axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add annotation to
    stats_text : str
        Text to display
    position : str
        Position of text box ('upper left', 'upper right', 'lower left', 'lower right')
    fontsize : float
        Font size for text
        
    Returns
    -------
    None
        Modifies ax in place
    """
    position_dict = {
        'upper left': (0.05, 0.95),
        'upper right': (0.95, 0.95),
        'lower left': (0.05, 0.05),
        'lower right': (0.95, 0.05)
    }
    
    text_x, text_y = position_dict.get(position, (0.05, 0.95))
    ha = 'left' if text_x < 0.5 else 'right'
    va = 'top' if text_y > 0.5 else 'bottom'
    
    ax.text(text_x, text_y, stats_text, transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    edgecolor='gray', alpha=0.8),
           fontsize=fontsize-2, ha=ha, va=va, family='monospace')


def create_null_inset(ax, custom_null, corr_coef, stats_position, 
                     line_color, fontsize):
    """
    Create inset axes showing null distribution with observed correlation.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Main axis to add inset to
    custom_null : array-like
        Array of null distribution values
    corr_coef : float
        Observed correlation coefficient
    stats_position : str
        Position of stats box (used to determine inset placement)
    line_color : str or color
        Color for marking observed value
    fontsize : float
        Base font size
        
    Returns
    -------
    matplotlib.axes.Axes or None
        Inset axes object, or None if insufficient data
    """
    custom_null = np.array(custom_null)
    custom_null = custom_null[~np.isnan(custom_null)]
    
    if len(custom_null) < 10:
        return None
    
    # Determine inset position to avoid overlap with stats box
    inset_position_map = {
        'upper left': (0.05, 0.1, 0.255, 0.2125),     # lower left
        'upper right': (0.70, 0.1, 0.255, 0.2125),    # lower right
        'lower left': (0.05, 0.75, 0.255, 0.2125),    # upper left
        'lower right': (0.70, 0.75, 0.255, 0.2125)    # upper right
    }
    
    inset_bounds = inset_position_map.get(stats_position, (0.05, 0.1, 0.255, 0.2125))
    
    # Create inset axes
    axins = ax.inset_axes(inset_bounds)
    
    # Make background transparent
    axins.patch.set_alpha(0)
    
    # Compute KDE
    kde = sp.stats.gaussian_kde(custom_null)
    x_kde = np.linspace(custom_null.min(), custom_null.max(), 200)
    y_kde = kde(x_kde)
    
    # Plot KDE
    axins.plot(x_kde, y_kde, color='gray', linewidth=1, alpha=0.8)
    axins.fill_between(x_kde, 0, y_kde, color='gray', alpha=0.3)
    
    # Mark observed correlation
    axins.axvline(corr_coef, color=line_color, linewidth=1, 
                 linestyle='--', alpha=0.8)
    
    # Shade tail beyond observed value
    if corr_coef > 0:
        # Shade right tail for positive correlations
        tail_mask = x_kde >= corr_coef
        axins.fill_between(x_kde[tail_mask], 0, y_kde[tail_mask], 
                          color=line_color, alpha=0.2)
    else:
        # Shade left tail for negative correlations
        tail_mask = x_kde <= corr_coef
        axins.fill_between(x_kde[tail_mask], 0, y_kde[tail_mask], 
                          color=line_color, alpha=0.2)
    
    # Style inset
    axins.tick_params(labelsize=fontsize-2, pad=1)
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)
    axins.set_yticks([])
    
    # Set x-axis to show observed value
    x_margin = (custom_null.max() - custom_null.min()) * 0.1
    axins.set_xlim(custom_null.min() - x_margin, 
                  custom_null.max() + x_margin)
    
    # Format x-tick labels to 1 decimal place
    axins.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    return axins


def style_correlation_axis(ax, x_label, y_label, title=None, 
                           fontsize=6, grid=True, grid_alpha=0.3):
    """
    Apply consistent styling to correlation plot axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    title : str, optional
        Plot title
    fontsize : float
        Base font size
    grid : bool
        Whether to show grid
    grid_alpha : float
        Grid transparency
        
    Returns
    -------
    None
        Modifies ax in place
    """
    # Set labels
    ax.set_xlabel(x_label or 'X Variable', fontsize=fontsize)
    ax.set_ylabel(y_label or 'Y Variable', fontsize=fontsize)
    
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight='bold', pad=10)
    
    # Grid
    if grid:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=fontsize,
                  length=6, width=1.2, colors='black')


# =====================================================================================================
# Custom colormaps (two hue poles + a neutral midpoint) + factories and a preview helper
# =====================================================================================================
# Every map is generated the same way, from a (low, mid, high) hex triple. The two hue poles keep an
# OKLab dE >= 8 under deuteranope / protanope / tritanope simulation, so a diverging quantity's sign
# stays readable for colorblind viewers; the midpoint is a neutral grey (a diverging map must never
# place a hue at the midpoint). 'sydnor_sa' is the fixed S-A axis identity map (orange -> purple).
_DIVERGING_PRESETS = {
    # Ordered most colorblind-safe first (min CVD OKLab dE, ~x100; check any pair with cvd_min_delta_e).
    'navy_gold':       ('#1A3A6B', '#EDEDED', '#C99700'),   # ~38 (dark-bg friendly)
    'indigo_gold':     ('#3B4CC0', '#F7F7F7', '#E0A800'),   # ~35 (very robust)
    'purple_orange':   ('#5E3C99', '#F7F7F7', '#E66101'),   # ~29
    'sydnor_sa':       ('#FFA500', '#F7F7F7', '#8A2BE2'),   # ~28 (S-A axis identity map: orange -> purple)
    'blue_orange':     ('#2166AC', '#F7F7F7', '#E08214'),   # ~27
    'blue_vermillion': ('#0072B2', '#F7F7F7', '#D55E00'),   # ~22 (Okabe-Ito pair)
    'blue_red':        ('#2166AC', '#F7F7F7', '#B2182B'),   # ~21
    'blue_brown':      ('#2166AC', '#F7F7F7', '#8C510A'),   # ~21 (BrBG-style)
    'blue_pink':       ('#4C72B0', '#F7F7F7', '#C51B7D'),   # ~10 (PiYG pink; CB floor)
}
1

# Machado-2009 dichromacy simulation matrices (severity 1.0), applied to *linear* sRGB.
_CVD_MATRICES = {
    'protanopia':   [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    'deuteranopia': [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    'tritanopia':   [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def _srgb_to_linear(rgb):
    c = np.asarray(rgb, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_oklab(lin):
    r, g, b = lin
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt([l, m, s])
    return np.array([0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                     1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                     0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_])


def cvd_min_delta_e(color_a, color_b):
    """Worst-case OKLab dE (x100) between two colors under colorblind simulation.

    Simulates protanopia / deuteranopia / tritanopia (Machado 2009, full severity), converts each
    simulated color to perceptually-uniform OKLab, and returns the *minimum* Euclidean distance
    (x100) across the three -- the metric used to vet a diverging map's two poles. >= 8 means the
    pair stays distinguishable for colorblind viewers (15-25 is comfortable, 30+ very robust).
    Accepts any matplotlib color spec (hex, name, or RGB(A) tuple).

    Example
    -------
    >>> cvd_min_delta_e('#2166AC', '#B2182B')   # blue_red poles
    21.1
    """
    a = _srgb_to_linear(mcolors.to_rgb(color_a))
    b = _srgb_to_linear(mcolors.to_rgb(color_b))
    dists = []
    for mx in _CVD_MATRICES.values():
        mx = np.asarray(mx)
        dists.append(100 * np.linalg.norm(_linear_to_oklab(np.clip(mx @ a, 0, 1)) -
                                          _linear_to_oklab(np.clip(mx @ b, 0, 1))))
    return float(min(dists))


def _register_with_reverse(cmap, name):
    """Register `cmap` under `name` and its reverse under `name + '_r'` (best-effort, idempotent).

    Custom colormaps -- unlike matplotlib built-ins -- do not get an automatic '_r' variant, so we
    register it explicitly, giving every custom map the same reversible `cmap='<name>_r'` convention.
    """
    for cm, nm in ((cmap, name), (cmap.reversed(), name + '_r')):
        try:
            plt.colormaps.register(cmap=cm, name=nm)
        except Exception:
            pass


def make_diverging_cmap(name, low, mid, high, N=256, register=True):
    """Build a diverging colormap from two hue poles and a neutral midpoint.

    Parameters
    ----------
    name : str
        Colormap name; registered with matplotlib when register=True so ``cmap=name`` works anywhere.
    low, mid, high : str
        Hex colors at positions 0.0 / 0.5 / 1.0. ``mid`` should be a neutral grey (no hue).
    N : int
        Number of quantization levels.
    register : bool
        Register the colormap under ``name`` (silently skipped if the name is already registered).

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    cmap = mcolors.LinearSegmentedColormap.from_list(name, [(0.0, low), (0.5, mid), (1.0, high)], N=N)
    if register:
        _register_with_reverse(cmap, name)
    return cmap


def make_sequential_cmap(name, colors, N=256, register=True):
    """Build a sequential colormap from an ordered list of colors (light -> dark for magnitude).

    Parameters
    ----------
    name : str
    colors : list
        Hex strings (spread evenly over 0..1) or explicit (position, hex) tuples.
    N, register : see make_diverging_cmap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    if colors and not isinstance(colors[0], (tuple, list)):
        colors = list(zip(np.linspace(0, 1, len(colors)), colors))
    cmap = mcolors.LinearSegmentedColormap.from_list(name, colors, N=N)
    if register:
        _register_with_reverse(cmap, name)
    return cmap


def register_custom_colormaps():
    """Build + register every custom colormap in _DIVERGING_PRESETS (incl. the S-A axis map).

    Idempotent -- safe to call at the top of a figure notebook. Returns {name: colormap}.
    """
    return {name: make_diverging_cmap(name, lo, mid, hi)
            for name, (lo, mid, hi) in _DIVERGING_PRESETS.items()}


def show_colormaps(names=None, figsize=None):
    """Preview a horizontal gradient strip for each named colormap (default: all custom presets)."""
    register_custom_colormaps()
    names = list(names) if names is not None else list(_DIVERGING_PRESETS)
    grad = np.linspace(0, 1, 256)[None, :]
    fig, axes = plt.subplots(len(names), 1, figsize=figsize or (5, 0.35 * len(names) + 0.3))
    axes = np.atleast_1d(axes)
    for ax, nm in zip(axes, names):
        ax.imshow(grad, aspect='auto', cmap=nm)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(nm, rotation=0, ha='right', va='center', fontsize=8)
    fig.subplots_adjust(left=0.32, hspace=0.5)
    return fig

######################################################################################################################################################
# deprecated functions. Maintained for historical purposes, but have been replaced by newer functions.
######################################################################################################################################################
