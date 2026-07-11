import os, platform
import numpy as np
import pandas as pd
import scipy as sp
import nibabel as nib

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from nilearn import datasets
from nilearn import plotting


def set_plotting_params(format='png'):
    if platform.system() == 'Darwin':
        os.system('rm -rf ~/.cache/matplotlib')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['savefig.format'] = format
    plt.rcParams['font.size'] = 8

    plt.rcParams['svg.fonttype'] = 'none'
    sns.set(style='whitegrid', context='paper', font_scale=1)


def get_my_colors(normalize=True, as_list=False, cat_trio=False):
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


def compute_correlation(x, y, method='pearson'):
    """
    Compute correlation coefficient and p-value.
    
    Parameters
    ----------
    x, y : array-like
        Input variables
    method : str
        Correlation method ('pearson' or 'spearman')
        
    Returns
    -------
    dict
        Dictionary containing:
        - corr_coef : correlation coefficient
        - p_value : p-value
        - method_name : full method name
    """
    if len(x) < 3 or len(y) < 3:
        return {
            'corr_coef': np.nan,
            'p_value': np.nan,
            'method_name': method.capitalize()
        }
    
    if method.lower() == 'pearson':
        corr_coef, p_value = sp.stats.pearsonr(x, y)
        method_name = "Pearson"
    elif method.lower() == 'spearman':
        corr_coef, p_value = sp.stats.spearmanr(x, y)
        method_name = "Spearman"
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")
    
    return {
        'corr_coef': corr_coef,
        'p_value': p_value,
        'method_name': method_name
    }


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


def determine_significance(p_value):
    """
    Determine significance stars based on p-value.
    
    Parameters
    ----------
    p_value : float
        P-value
        
    Returns
    -------
    str
        Significance stars ('***', '**', '*', or 'ns')
    """
    if np.isnan(p_value):
        return 'ns'
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


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
    sig_stars = determine_significance(p_value)
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
            t_sig_stars = determine_significance(t_p_value)
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


def create_sydnor_sa_colormap():
    """Create the Sydnor S-A colormap (orange sensorimotor pole -> white -> purple association).

    Also registers it with matplotlib under the name 'sydnor_sa'.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The custom colormap.
    """
    colors = [
        (0.0, '#FFA500'),   # orange (sensorimotor pole, low rank)
        (0.5, '#FFFFFF'),   # white (midpoint)
        (1.0, '#8A2BE2'),   # purple (association pole, high rank)
    ]
    sydnor_sa = mcolors.LinearSegmentedColormap.from_list('sydnor_sa', colors, N=256)
    try:
        plt.colormaps.register(cmap=sydnor_sa, name='sydnor_sa')
    except Exception:
        pass
    return sydnor_sa

######################################################################################################################################################
# deprecated functions. Maintained for historical purposes, but have been replaced by newer functions.
######################################################################################################################################################
def get_p_val_string(p_val):
    # if np.round(p_val, 3) == 0.000:
        # p_str = "-log10($\mathit{:}$)>25".format('{p}')
    if p_val < 0.05:
        # Two significant digits in the mantissa: '{:0.0e}' rounded 1.76e-2 -> "2e-02",
        # which reads as a different p-value than the one being reported (e.g. 1.76e-02).
        p_str = '$\mathit{:}$={:.2e}'.format('{p}', p_val)
    else:
        p_str = "$\mathit{:}$={:.3f}".format('{p}', p_val)

    return p_str