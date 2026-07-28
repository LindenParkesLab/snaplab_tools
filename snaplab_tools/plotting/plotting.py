"""Figure functions for brain-map results.

Most functions take an existing matplotlib ``ax`` and draw into it, so they compose into
multi-panel figures rather than owning the figure themselves. Call
:func:`snaplab_tools.plotting.utils.set_plotting_params` first to get consistent fonts and sizing.

Correlation plots
    :func:`plot_correlation` is the workhorse -- a scatter with a fit line and an annotated
    coefficient, which can also colour points by category, fit and compare polynomial orders, and
    embed an inset showing an empirical null. :func:`plot_correlation_unity` is the variant with
    X=Y on the diagonal, for comparing two measurements of the same quantity.

Brain surfaces
    :func:`plot_brain_surface_data` and :func:`plot_brain_surface_data_single` render parcellated
    data on inflated cortical surfaces via ``surfplot`` (install the ``surface`` extra; imported
    lazily, so the rest of this module works without it). :func:`brain_scatter_plot` draws nodes
    and edges in anatomical coordinates.

Distributions and comparisons
    :func:`null_plot` overlays an observed statistic on its null distribution;
    :func:`categorical_kde_plot` and :func:`paired_line_plot` compare groups or conditions.

Surface functions expect Schaefer FreeSurfer ``.annot`` files under the directory named by the
``SCHAEFER_ANNOT_DIR`` environment variable (default ``~/Schaefer2018_LocalGlobal/...``);
:func:`snaplab_tools.utils.load_schaefer_parc` will download them for you.
"""
import io
import os
import numpy as np
import scipy as sp
import nibabel as nib
import pandas as pd
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable

# Note: no plt.ion() here. Importing a module should not switch matplotlib into interactive mode
# for the whole process -- that is the caller's decision, and it changes how plt.show() behaves
# everywhere else in their session. Call plt.ion() yourself if you want it.

from nilearn import datasets

# Default location for Schaefer FreeSurfer annotation files, used when the SCHAEFER_ANNOT_DIR
# environment variable is not set.
_DEFAULT_SCHAEFER_ANNOT_DIR = '~/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3'


def _schaefer_annot_dir():
    """Directory holding the Schaefer FreeSurfer .annot files.

    Reads SCHAEFER_ANNOT_DIR on every call rather than caching it at import. Capturing it into a
    module-level constant meant setting the variable after the first import silently had no
    effect, which is a confusing way to lose an afternoon.
    """
    return os.path.expanduser(
        os.environ.get('SCHAEFER_ANNOT_DIR', _DEFAULT_SCHAEFER_ANNOT_DIR)
    )

from snaplab_tools.stats import compute_stat, get_null_p, significance_stars
from snaplab_tools.plotting.utils import get_snap_colors, process_input_data, \
    create_correlation_text, add_stats_annotation, create_null_inset, style_correlation_axis, \
        format_pvalue

__all__ = [
    'plot_correlation',
    'plot_correlation_unity',
    'null_plot',
    'brain_scatter_plot',
    'plot_brain_surface_data',
    'plot_brain_surface_data_single',
    'categorical_kde_plot',
    'paired_line_plot',
    'YEO7_COLORS',
]


#: Canonical Yeo 7-network colours (Yeo et al. 2011, *J. Neurophysiol.*), as RGB 0-1 tuples.
#: Keys match the system names embedded in Schaefer parcel labels (e.g. the 'Vis' in
#: ``7Networks_LH_Vis_1``), so a system vector parsed from those labels indexes straight in.
#:
#: (Written as ``#:`` comments rather than a plain comment block: that is the form autodoc reads
#: as documentation for module-level data. Without it, autodoc falls back to ``dict.__doc__``.)
YEO7_COLORS = {
    'Vis':         (120/255, 18/255,  134/255),
    'SomMot':      ( 70/255, 130/255, 180/255),
    'DorsAttn':    (  0/255, 118/255,  14/255),
    'SalVentAttn': (196/255,  58/255, 250/255),
    'Limbic':      (220/255, 248/255, 164/255),
    'Cont':        (230/255, 148/255,  34/255),
    'Default':     (205/255,  62/255,  78/255),
}


def plot_correlation(x, y, ax, x_label=None, y_label=None, title=None,
                    method='pearson', color="#3B3B3B", alpha=0.6,
                    size=10, show_line=True,
                    show_confidence=True, show_stats=True,
                    stats_position='upper left', fontsize=8,
                    grid=True, grid_alpha=0.3,
                    outlier_threshold=None, highlight_outliers=False,
                    return_stats=False,
                    auto_polynomial=False, models_to_test=[1, 2, 3],
                    model_selection_metric='variance_explained',
                    data_group=None, data_group_cmap='tab10',
                    cbar_label='Data Values', show_colorbar=True,
                    custom_pvalue=None,
                    custom_inset=None):
    """
    Enhanced correlation plot with automatic polynomial model selection and data group coloring.
    
    Parameters
    -----------
    x, y : array-like, pandas.Series, or numpy.ndarray
        Input variables to correlate. If pandas Series, their names will be used 
        as default axis labels unless x_label/y_label are explicitly provided.
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on.
    x_label, y_label : str, optional
        Labels for x and y axes. If None and inputs are pandas Series, 
        the Series names will be used automatically.
    title : str, optional
        Plot title
    method : str, default 'pearson'
        Correlation method ('pearson', 'spearman')
    color : str, default '#3B3B3B'
        Color for scatter points and regression line (used when data_group=None)
    alpha : float, default 0.6
        Transparency of scatter points
    size : float, default 10
        Size of scatter points
    show_line : bool, default True
        Whether to show regression line
    show_confidence : bool, default True
        Whether to show confidence interval around regression line
    show_stats : bool, default True
        Whether to display correlation statistics on plot
    stats_position : str, default 'upper left'
        Position of statistics text box
    fontsize : int, default 8
        Base font size for labels and text
    grid : bool, default True
        Whether to show grid
    grid_alpha : float, default 0.3
        Grid transparency
    outlier_threshold : float, optional
        Z-score threshold for outlier detection (e.g., 2.5)
    highlight_outliers : bool, default False
        Whether to highlight outliers in different color
    return_stats : bool, default False
        Whether to return correlation statistics
    auto_polynomial : bool, default False
        Whether to automatically select best polynomial fit
    models_to_test : list, default [1, 2, 3]
        List of polynomial degrees to test (e.g., [1, 2, 3] for linear, quadratic, cubic)
    model_selection_metric : str, default 'variance_explained'
        Metric for model selection: 'variance_explained', 'rmse', or 'mae'
    data_group : array-like, optional
        Vector of integer values or string labels designating which group each data point belongs to.
        Must have same length as x and y. If provided, points will be colored by group.
    data_group_cmap : str, default 'tab10'
        Colormap to use for data group coloring. Good options: 'tab10', 'tab20', 'Set3', 'viridis'
    cbar_label : str, default 'Data Values'
        Label for the colorbar
    custom_pvalue : float, optional
        Custom p-value to display in the annotation box instead of the parametric p-value
        (e.g., from spatial permutation testing with BrainSMASH). Does not affect returned
        statistics. Not compatible with auto_polynomial=True.
    custom_inset : dict, optional
        Dictionary containing custom inset parameters. Supported keys:

        - 'custom_pvalue' (float): Custom p-value to display. Overridden by the top-level
          custom_pvalue parameter if both are provided.
        - 'custom_null' (array-like): Array of null distribution values (e.g., from
          spatial permutation testing). If provided, an inset KDE plot will be added
          showing the empirical null distribution with the observed correlation marked.
          If 'custom_null' is provided WITHOUT any custom_pvalue, the p-value will be
          computed automatically using a one-tailed test: for positive correlations,
          p = proportion of null >= observed; for negative correlations, p = proportion
          of null <= observed.

    Returns
    --------
    ax : matplotlib axis object
        The axis object with plot
    stats_dict : dict, optional
        Dictionary with correlation statistics (if return_stats=True)
    """
    
    # Process input data
    data_dict = process_input_data(x, y, data_group)
    x_clean = data_dict['x_clean']
    y_clean = data_dict['y_clean']
    valid_mask = data_dict['valid_mask']
    
    # Set labels from extracted names or user-provided values
    if x_label is None:
        x_label = data_dict['x_label']
    if y_label is None:
        y_label = data_dict['y_label']
    
    # Extract data_group_clean if present
    data_group_clean = data_dict.get('data_group_clean', None)
    
    # Compute correlation (returns NaN below three points -- see compute_stat)
    corr_coef, p_value = compute_stat(x_clean, y_clean, method)
    method_name = method.capitalize()
    
    # Process custom_inset parameters
    custom_null = None
    inset_pvalue = None
    if custom_inset is not None:
        if not isinstance(custom_inset, dict):
            raise ValueError("custom_inset must be a dictionary")

        custom_null = custom_inset.get('custom_null', None)
        if custom_null is not None:
            custom_null = np.array(custom_null)
            custom_null = custom_null[~np.isnan(custom_null)]

        inset_pvalue = custom_inset.get('custom_pvalue', None)
        if inset_pvalue is None and custom_null is not None and len(custom_null) > 0:
            inset_pvalue = get_null_p(corr_coef, custom_null, alternative='auto')

    # Top-level custom_pvalue takes priority over custom_inset['custom_pvalue']
    resolved_pvalue = custom_pvalue if custom_pvalue is not None else inset_pvalue

    # Check for incompatible parameters
    if resolved_pvalue is not None and auto_polynomial:
        raise ValueError(
            "custom_pvalue is not compatible with auto_polynomial=True. "
            "Please set auto_polynomial=False or remove custom_pvalue."
        )
    
    # Determine line color early for use in both regression line and inset
    line_color = color if data_group is None else '#3B3B3B'
    
    # Detect outliers if requested
    outlier_mask = np.zeros(len(x_clean), dtype=bool)
    if outlier_threshold is not None:
        z_scores_x = np.abs(sp.stats.zscore(x_clean))
        z_scores_y = np.abs(sp.stats.zscore(y_clean))
        outlier_mask = (z_scores_x > outlier_threshold) | (z_scores_y > outlier_threshold)
    
    # Plot scatter points
    if data_group is not None:
        # Color by data groups (handles both strings and integers)
        unique_groups = np.unique(data_group_clean)
        n_groups = len(unique_groups)
        
        # Create mapping from group labels to colors
        if data_group_cmap == 'yeo7':
            group_colors = {g: YEO7_COLORS.get(g, (0.5, 0.5, 0.5)) for g in unique_groups}
            colors = [group_colors[g] for g in unique_groups]
        else:
            cmap = plt.cm.get_cmap(data_group_cmap)
            if n_groups <= 10 and data_group_cmap == 'tab10':
                colors = [cmap(i) for i in range(n_groups)]
            elif n_groups <= 20 and data_group_cmap == 'tab20':
                colors = [cmap(i) for i in range(n_groups)]
            else:
                colors = [cmap(i / max(1, n_groups - 1)) for i in range(n_groups)]
            group_colors = {group: colors[i] for i, group in enumerate(unique_groups)}
        
        # Plot points for each data group
        for group in unique_groups:
            group_mask = data_group_clean == group
            
            if highlight_outliers and np.any(outlier_mask):
                # Separate normal and outlier points within this group
                normal_mask = group_mask & (~outlier_mask)
                outlier_mask_group = group_mask & outlier_mask
                
                # Plot normal points for this group
                if np.any(normal_mask):
                    ax.scatter(x_clean[normal_mask], y_clean[normal_mask], 
                              c=[group_colors[group]], alpha=alpha, s=size, 
                              edgecolors='white', linewidth=0.5, 
                              label=f'{group}')
                
                # Plot outliers for this group with different marker
                if np.any(outlier_mask_group):
                    ax.scatter(x_clean[outlier_mask_group], y_clean[outlier_mask_group], 
                              c=[group_colors[group]], alpha=alpha+0.2, s=size*1.2, 
                              edgecolors='darkred', linewidth=1, marker='^',
                              label=f'{group} (outliers)')
            else:
                # Plot all points for this group normally
                ax.scatter(x_clean[group_mask], y_clean[group_mask], 
                          c=[group_colors[group]], alpha=alpha, s=size, 
                          edgecolors='white', linewidth=0.5, 
                          label=f'{group}')
        
        # Add colorbar for data groups
        if show_colorbar and n_groups > 1:
            # For string labels, create a discrete colorbar
            if isinstance(unique_groups[0], (str, np.str_)):
                listed_cmap = ListedColormap(colors[:n_groups])
                bounds = np.arange(n_groups + 1) - 0.5
                norm = BoundaryNorm(bounds, listed_cmap.N)
                
                sm = plt.cm.ScalarMappable(cmap=listed_cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.05)
                cbar.set_label(cbar_label, fontsize=fontsize)

                # Set colorbar ticks to show string labels
                cbar.set_ticks(np.arange(n_groups))
                if n_groups <= 10:
                    cbar.set_ticklabels(unique_groups, fontsize=fontsize-2)
                else:
                    tick_indices = np.linspace(0, n_groups-1, min(5, n_groups), dtype=int)
                    cbar.set_ticks(tick_indices)
                    cbar.set_ticklabels([unique_groups[i] for i in tick_indices], 
                                       fontsize=fontsize-2)
            else:
                # Original numeric colorbar
                sm = plt.cm.ScalarMappable(
                    cmap=cmap, 
                    norm=plt.Normalize(vmin=min(unique_groups), vmax=max(unique_groups))
                )
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.05)
                cbar.set_label(cbar_label, fontsize=fontsize)
                
                # Set colorbar ticks to show group numbers
                if n_groups <= 10:
                    cbar.set_ticks(unique_groups)
                else:
                    tick_groups = unique_groups[::max(1, len(unique_groups)//5)]
                    cbar.set_ticks(tick_groups)
    else:
        # Original coloring logic when data_group is None
        if highlight_outliers and np.any(outlier_mask):
            # Plot normal points
            normal_mask = ~outlier_mask
            ax.scatter(x_clean[normal_mask], y_clean[normal_mask], 
                      c=color, alpha=alpha, s=size, edgecolors='white', 
                      linewidth=0.5, label='Data points')
            
            # Plot outliers
            ax.scatter(x_clean[outlier_mask], y_clean[outlier_mask], 
                      c='red', alpha=alpha+0.2, s=size*1.2, edgecolors='darkred', 
                      linewidth=1, marker='^', label='Outliers')
        else:
            ax.scatter(x_clean, y_clean, c=color, alpha=alpha, s=size, 
                      edgecolors='white', linewidth=0.5)
    
    # Initialize model info for stats display
    best_model_info = None
    
    # Add regression line and confidence interval
    if show_line and len(x_clean) > 2:
        
        if auto_polynomial and len(x_clean) >= (max(models_to_test) + 1):
            # Test specified polynomial degrees
            model_results = {}
            
            for degree in models_to_test:
                if len(x_clean) > degree:
                    try:
                        # Fit polynomial
                        poly_coeffs = np.polyfit(x_clean, y_clean, degree)
                        y_pred = np.polyval(poly_coeffs, x_clean)
                        
                        # Calculate model fit metrics
                        residuals = y_clean - y_pred
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                        
                        # Variance explained (R²)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        # RMSE and MAE
                        rmse = np.sqrt(np.mean(residuals ** 2))
                        mae = np.mean(np.abs(residuals))
                        
                        model_results[degree] = {
                            'coeffs': poly_coeffs,
                            'r_squared': r_squared,
                            'rmse': rmse,
                            'mae': mae,
                            'y_pred': y_pred
                        }
                    except (np.RankWarning, np.linalg.LinAlgError):
                        continue
            
            # Select best model based on chosen metric
            if model_results:
                if model_selection_metric == 'variance_explained':
                    best_degree = max(model_results.keys(), 
                                    key=lambda k: model_results[k]['r_squared'])
                elif model_selection_metric == 'rmse':
                    best_degree = min(model_results.keys(), 
                                    key=lambda k: model_results[k]['rmse'])
                elif model_selection_metric == 'mae':
                    best_degree = min(model_results.keys(), 
                                    key=lambda k: model_results[k]['mae'])
                else:
                    raise ValueError(
                        "model_selection_metric must be 'variance_explained', 'rmse', or 'mae'"
                    )
                
                best_model = model_results[best_degree]
                best_model_info = {
                    'degree': best_degree,
                    'r_squared': best_model['r_squared'],
                    'rmse': best_model['rmse'],
                    'mae': best_model['mae'],
                    'metric_used': model_selection_metric
                }
                
                # Create smooth line for plotting
                x_line = np.linspace(np.min(x_clean), np.max(x_clean), 100)
                y_line = np.polyval(best_model['coeffs'], x_line)
                
                # Plot regression line
                degree_names = {1: 'Linear', 2: 'Quadratic', 3: 'Cubic', 4: 'Quartic', 
                               5: 'Quintic', 6: 'Sextic'}
                model_name = degree_names.get(best_degree, f'Degree {best_degree}')
                
                ax.plot(x_line, y_line, color=line_color, linewidth=2.5, alpha=0.8, 
                       linestyle='-', 
                       label=f'{model_name} fit (R²={best_model["r_squared"]:.3f})')
                
                # Add confidence interval for linear only
                if show_confidence and best_degree == 1:
                    slope, intercept, r_value, p_value_reg, std_err = sp.stats.linregress(
                        x_clean, y_clean
                    )
                    
                    def predict_interval(x_new, x_data, y_data, confidence=0.95):
                        n = len(x_data)
                        x_mean = np.mean(x_data)
                        sxx = np.sum((x_data - x_mean) ** 2)
                        sxy = np.sum((x_data - x_mean) * (y_data - np.mean(y_data)))
                        syy = np.sum((y_data - np.mean(y_data)) ** 2)
                        
                        s = np.sqrt((syy - sxy**2/sxx) / (n - 2))
                        t_val = sp.stats.t.ppf((1 + confidence) / 2, n - 2)
                        
                        margin = t_val * s * np.sqrt(1/n + (x_new - x_mean)**2/sxx)
                        y_pred = slope * x_new + intercept
                        
                        return y_pred - margin, y_pred + margin
                    
                    y_lower, y_upper = predict_interval(x_line, x_clean, y_clean)
                    ax.fill_between(x_line, y_lower, y_upper, color=line_color, alpha=0.15, 
                                   label='95% Confidence Interval')
            else:
                # Fall back to linear regression if polynomial fitting fails
                auto_polynomial = False
        
        if not auto_polynomial:
            # Original linear regression
            slope, intercept, r_value, p_value_reg, std_err = sp.stats.linregress(
                x_clean, y_clean
            )
            
            # Create line points
            x_line = np.linspace(np.min(x_clean), np.max(x_clean), 100)
            y_line = slope * x_line + intercept
            
            # Plot regression line
            ax.plot(x_line, y_line, color=line_color, linewidth=2.5, alpha=0.8, 
                   linestyle='-', label='Linear regression')
            
            # Store linear model info
            y_pred_linear = slope * x_clean + intercept
            residuals_linear = y_clean - y_pred_linear
            ss_res_linear = np.sum(residuals_linear ** 2)
            ss_tot_linear = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r_squared_linear = 1 - (ss_res_linear / ss_tot_linear) if ss_tot_linear > 0 else 0
            
            best_model_info = {
                'degree': 1,
                'r_squared': r_squared_linear,
                'rmse': np.sqrt(np.mean(residuals_linear ** 2)),
                'mae': np.mean(np.abs(residuals_linear)),
                'metric_used': 'linear_only'
            }
            
            # Add confidence interval
            if show_confidence:
                def predict_interval(x_new, x_data, y_data, confidence=0.95):
                    n = len(x_data)
                    x_mean = np.mean(x_data)
                    sxx = np.sum((x_data - x_mean) ** 2)
                    sxy = np.sum((x_data - x_mean) * (y_data - np.mean(y_data)))
                    syy = np.sum((y_data - np.mean(y_data)) ** 2)
                    
                    s = np.sqrt((syy - sxy**2/sxx) / (n - 2))
                    t_val = sp.stats.t.ppf((1 + confidence) / 2, n - 2)
                    
                    margin = t_val * s * np.sqrt(1/n + (x_new - x_mean)**2/sxx)
                    y_pred = slope * x_new + intercept
                    
                    return y_pred - margin, y_pred + margin
                
                y_lower, y_upper = predict_interval(x_line, x_clean, y_clean)
                ax.fill_between(x_line, y_lower, y_upper, color=line_color, alpha=0.15, 
                               label='95% Confidence Interval')
    
    # Add statistics text box
    if show_stats:
        display_pvalue = resolved_pvalue if resolved_pvalue is not None else p_value
        
        # Prepare data group info for text creation
        data_group_info = None
        if data_group is not None:
            unique_groups = np.unique(data_group_clean)
            data_group_info = {
                'unique_groups': unique_groups,
                'group_type': 'string' if isinstance(unique_groups[0], (str, np.str_)) else 'numeric'
            }
        
        # Create statistics text
        stats_text = create_correlation_text(
            corr_coef=corr_coef,
            p_value=display_pvalue,
            method=method,
            n_outliers=np.sum(outlier_mask) if outlier_threshold else 0,
            model_info=best_model_info if auto_polynomial else None,
            data_group_info=data_group_info
        )
        
        # Add annotation to plot
        add_stats_annotation(ax, stats_text, stats_position, fontsize)
    
    # Add null distribution inset if provided
    if custom_null is not None:
        create_null_inset(ax, custom_null, corr_coef, stats_position, 
                         line_color, fontsize)
    
    # Apply axis styling
    style_correlation_axis(ax, x_label, y_label, title, fontsize, grid, grid_alpha)
    
    # Prepare return values
    if return_stats:
        stats_dict = {
            'correlation': corr_coef,
            'p_value': p_value,
            'n_samples': len(x_clean),
            'method': method_name,
            'n_outliers': np.sum(outlier_mask) if outlier_threshold else 0,
            'outlier_indices': np.where(valid_mask)[0][outlier_mask] if outlier_threshold else [],
            'model_info': best_model_info,
            'data_group_info': {
                'n_groups': len(np.unique(data_group_clean)) if data_group is not None else None,
                'unique_groups': np.unique(data_group_clean).tolist() if data_group is not None else None,
                'group_type': 'string' if data_group is not None and isinstance(
                    np.unique(data_group_clean)[0], (str, np.str_)
                ) else 'numeric'
            } if data_group is not None else None
        }
        return ax, stats_dict
    else:
        return ax


def plot_correlation_unity(x, y, ax, x_label=None, y_label=None,
                           show_marginals=False, show_unity_line=True, show_zero_lines=True,
                           color='#3B3B3B', alpha=0.5, size=10,
                           marginal_color_x='blue', marginal_color_y='red', 
                           marginal_alpha=0.25, fontsize=8,
                           grid=True, grid_alpha=0.3,
                           show_correlation=False, correlation_type='pearson'):
    """
    Plot comparison between two sets of measurements with optional marginals and statistics.
    
    Parameters
    -----------
    x : array-like or pd.Series
        First set of measurements (e.g., rest correlations)
    y : array-like or pd.Series
        Second set of measurements (e.g., task correlations)
    ax : matplotlib.axes.Axes
        Axes object to plot on
    x_label : str, optional
        Label for x-axis. If None, uses Series name if x is a Series, otherwise 'X'
    y_label : str, optional
        Label for y-axis. If None, uses Series name if y is a Series, otherwise 'Y'
    show_marginals : bool
        Whether to show marginal distributions
    show_unity_line : bool
        Whether to show unity (y=x) line
    show_zero_lines : bool
        Whether to show x=0 and y=0 reference lines
    color : str or color
        Color for scatter points (default: '#3B3B3B')
    alpha : float
        Transparency for scatter points
    size : float
        Size of scatter points (default: 10)
    marginal_color_x : str or color
        Color for x marginal distribution
    marginal_color_y : str or color
        Color for y marginal distribution
    marginal_alpha : float
        Transparency for marginal distributions
    fontsize : float
        Font size for all text elements (labels, title, tick labels) (default: 8)
    grid : bool, default True
        Whether to show grid
    grid_alpha : float, default 0.3
        Grid transparency
    show_correlation : bool
        Whether to show correlation annotation (default: False)
    correlation_type : str
        Type of correlation to compute: 'pearson' or 'spearman' (default: 'pearson')
        
    Returns
    --------
    stats_dict : dict
        Dictionary containing computed statistics:
        - n_valid: number of valid pairs
        - n_invalid: number of excluded pairs
        - x_mean, x_std: statistics for x
        - y_mean, y_std: statistics for y
        - compression_ratio: std_y / std_x
        - correlation: correlation coefficient between x and y
        - correlation_p: p-value for correlation
        - correlation_pearson: Pearson r between x and y
        - correlation_pearson_p: p-value for Pearson correlation
        - correlation_spearman: Spearman ρ between x and y
        - correlation_spearman_p: p-value for Spearman correlation
        - variance_diff_p: p-value for Levene's test of equal variances
        - ttest_t: t-statistic for paired t-test (if show_marginals=True)
        - ttest_p: p-value for paired t-test (if show_marginals=True)
    """ 
    
    # Process input data
    data_dict = process_input_data(x, y)
    x_clean = data_dict['x_clean']
    y_clean = data_dict['y_clean']
    valid_mask = data_dict['valid_mask']
    n_valid = data_dict['n_valid']
    n_invalid = data_dict['n_invalid']
    
    # Set labels from extracted names or user-provided values
    if x_label is None:
        x_label = data_dict['x_label'] or 'X'
    if y_label is None:
        y_label = data_dict['y_label'] or 'Y'
    
    # Handle case with no valid data
    if n_valid == 0:
        ax.text(0.5, 0.5, 'No valid data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=fontsize)
        return {
            'n_valid': 0,
            'n_invalid': n_invalid,
            'x_mean': np.nan,
            'x_std': np.nan,
            'y_mean': np.nan,
            'y_std': np.nan,
            'compression_ratio': np.nan,
            'correlation': np.nan,
            'correlation_p': np.nan,
            'variance_diff_p': np.nan,
            'ttest_t': np.nan,
            'ttest_p': np.nan
        }
    
    # Compute statistics
    x_mean = x_clean.mean()
    x_std = x_clean.std()
    y_mean = y_clean.mean()
    y_std = y_clean.std()
    compression_ratio = y_std / x_std if x_std > 0 else np.nan
    
    # Compute both correlations. compute_stat returns (nan, nan) below three points, so the
    # explicit small-sample branch this used to need has gone.
    r_pearson, p_pearson = compute_stat(x_clean, y_clean, 'pearson')
    r_spearman, p_spearman = compute_stat(x_clean, y_clean, 'spearman')

    # Select which correlation to use based on correlation_type
    if correlation_type.lower() == 'spearman':
        corr_coef, p_value = r_spearman, p_spearman
    else:
        corr_coef, p_value = r_pearson, p_pearson
    
    # Variance difference test
    if n_valid > 2:
        _, var_p = sp.stats.levene(x_clean, y_clean)
    else:
        var_p = np.nan
    
    # Paired t-test (if marginals will be shown)
    if show_marginals and n_valid > 2:
        t_stat, t_p_value = sp.stats.ttest_rel(x_clean, y_clean)
    else:
        t_stat, t_p_value = np.nan, np.nan
    
    # Get axis limits
    all_vals = np.concatenate([x_clean, y_clean])
    lim_min = all_vals.min() * 1.1 if all_vals.min() < 0 else all_vals.min() * 0.9
    lim_max = all_vals.max() * 1.1 if all_vals.max() > 0 else all_vals.max() * 0.9
    
    # Plot unity line
    if show_unity_line:
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 
                'k--', alpha=0.3, linewidth=1, zorder=1)
    
    # Plot zero lines
    if show_zero_lines:
        ax.axhline(0, color='gray', linestyle=':', alpha=0.3, linewidth=1, zorder=1)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.3, linewidth=1, zorder=1)
    
    # Add marginal distributions using KDE
    if show_marginals and n_valid > 5:
        kde_height = (lim_max - lim_min) * 0.10
        
        # X marginal (top)
        try:
            kde_x = sp.stats.gaussian_kde(x_clean)
            x_range = np.linspace(lim_min, lim_max, 200)
            x_density = kde_x(x_range)
            x_density_scaled = x_density / x_density.max() * kde_height
            
            ax.fill_between(x_range, 
                           lim_max - x_density_scaled,
                           lim_max,
                           color=marginal_color_x, 
                           alpha=marginal_alpha,
                           edgecolor=marginal_color_x, 
                           linewidth=1,
                           zorder=2)
        except np.linalg.LinAlgError:
            pass
        
        # Y marginal (right)
        try:
            kde_y = sp.stats.gaussian_kde(y_clean)
            y_range = np.linspace(lim_min, lim_max, 200)
            y_density = kde_y(y_range)
            y_density_scaled = y_density / y_density.max() * kde_height
            
            ax.fill_betweenx(y_range,
                            lim_max - y_density_scaled,
                            lim_max,
                            color=marginal_color_y,
                            alpha=marginal_alpha,
                            edgecolor=marginal_color_y,
                            linewidth=1,
                            zorder=2)
        except np.linalg.LinAlgError:
            pass
    
    # Scatter plot
    ax.scatter(x_clean, y_clean, 
              s=size, alpha=alpha,
              edgecolors='black', linewidths=0.5,
              c=color, zorder=3)
    
    # Add correlation annotation if requested
    if show_correlation and not np.isnan(corr_coef):
        # Find best position to minimize overlap with points
        x_mid = (lim_min + lim_max) / 2
        y_mid = (lim_min + lim_max) / 2
        
        # Count points in each quadrant
        quadrant_counts = {
            'lower left': np.sum((x_clean < x_mid) & (y_clean < y_mid)),
            'lower right': np.sum((x_clean >= x_mid) & (y_clean < y_mid)),
            'upper left': np.sum((x_clean < x_mid) & (y_clean >= y_mid)),
            'upper right': np.sum((x_clean >= x_mid) & (y_clean >= y_mid))
        }
        
        # Choose quadrant with fewest points
        best_position = min(quadrant_counts, key=quadrant_counts.get)
        
        # Create statistics text
        stats_text = create_correlation_text(
            corr_coef=corr_coef,
            p_value=p_value,
            method=correlation_type
        )
        
        # Add annotation to plot
        add_stats_annotation(ax, stats_text, best_position, fontsize)
    
    # Set labels
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    
    # Set title with t-test if marginals are shown
    if show_marginals and not np.isnan(t_stat):
        # Create title with t-test information
        t_sig_stars = significance_stars(t_p_value)
        t_p_str = format_pvalue(t_p_value)
        title_text = f't={t_stat:.2f}{t_sig_stars}, {t_p_str}'
        ax.set_title(title_text, fontsize=fontsize-2)
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    # Set axis properties
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    
    # Apply grid with custom alpha
    if grid:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5)
    
    # Compile statistics
    stats_dict = {
        'n_valid': n_valid,
        'n_invalid': n_invalid,
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'compression_ratio': compression_ratio,
        'correlation': corr_coef,
        'correlation_p': p_value,
        'correlation_pearson': r_pearson,
        'correlation_pearson_p': p_pearson,
        'correlation_spearman': r_spearman,
        'correlation_spearman_p': p_spearman,
        'variance_diff_p': var_p,
        'ttest_t': t_stat,
        'ttest_p': t_p_value
    }
    
    return stats_dict


def null_plot(observed, null, xlabel, ax, p_val=None, add_text=True, line_color=None, use_kde=False):
    """Plot a null distribution (histogram or KDE) with the observed value marked.

    Parameters
    ----------
    observed : float
        Observed statistic; drawn as a vertical line.
    null : array-like
        Null distribution values.
    xlabel : str
        X-axis label.
    ax : matplotlib Axes
        Axis to plot on.
    p_val : float or None
        If given, annotate the observed line with its p-value.
    add_text : bool
        Annotate the observed value (and p-value).
    line_color : color or None
        Colour of the observed line; defaults to the lab green.
    use_kde : bool
        Use a KDE instead of a histogram for the null.

    Notes
    -----
    Both labels run vertically alongside the observed line, in its colour: the statistic and
    p-value to its right in bold, the word 'observed' to its left. Reading bottom-to-top, so the
    text sits in the empty upper tail of the distribution rather than over the bars, and takes
    only its line height in horizontal space -- which is what lets it stay beside the line even
    when the line is hard against the edge of the axes.
    """
    if line_color is None:
        line_color = get_snap_colors()['north_sea_green']

    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]

    if use_kde is True:
        sns.kdeplot(x=null, ax=ax, color='0.45', fill=True, alpha=0.25, linewidth=1)
    else:
        # Hairline white edges separate adjacent bars without adding ink, so the distribution
        # reads as bars rather than one grey mass.
        sns.histplot(x=null, ax=ax, color='0.75', edgecolor='white', linewidth=0.4)

    ax.axvline(x=observed, linewidth=1.5, color=line_color, zorder=3)
    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)
    ax.set_xlabel(xlabel)
    # The KDE axis is a density, not a count; it used to be labelled 'counts' either way.
    ax.set_ylabel('density' if use_kde else 'counts')

    if add_text is True:
        label = f'{observed:.2f}'
        if p_val is not None:
            label += f', {format_pvalue(p_val)}'

        # Add annotation text
        shared = dict(xycoords=('data', 'axes fraction'), textcoords='offset points',
                      rotation=270, rotation_mode='anchor', color=line_color)
        ax.annotate(label, xy=(observed, 0.98), xytext=(-10, 0),
                    ha='left', va='bottom', fontweight='bold', **shared)
        ax.annotate('observed', xy=(observed, 0.98), xytext=(8, 0),
                    ha='left', va='top', fontweight='bold', **shared)


def brain_scatter_plot(parcel_coords, node_data=None, edge_data=None, fig_height=1.25, vmin=None, vmax=None, cmap=None, add_colorbar=False, ax=None):
    """Scatter brain parcels at their 2-D coordinates, optionally with edges and node colouring.

    Parameters
    ----------
    parcel_coords : (n_nodes, 2) DataFrame
        Per-parcel x/y coordinates.
    node_data : (n_nodes,) ndarray or None
        Values to colour nodes by; white if None. Signed data auto-symmetrises the limits.
    edge_data : (n_nodes, n_nodes) ndarray or None
        Adjacency matrix; nonzero entries are drawn as edges (thickness scaled by weight).
    fig_height : float
        Figure height in inches (width is 0.8x); also scales node size.
    vmin, vmax : float or None
        Colour limits for node_data.
    cmap : str or None
        Colormap for node_data.
    add_colorbar : bool
        Add a colorbar for node_data.
    ax : matplotlib Axes or None
        Axis to draw on; a new figure is created if None.

    Returns
    -------
    matplotlib Figure
        The figure containing the plot.
    """
    n_nodes = parcel_coords.shape[0]
    fig_width = fig_height * 0.8

    # determine node color
    if type(node_data) is np.ndarray:
        c = node_data
        if cmap is None:
            cmap = 'plasma'

        if vmin is None and vmax is None:
            if np.any(node_data < 0) and np.any(node_data > 0):
                vmin = -np.nanmax(np.abs(node_data))
                vmax = np.nanmax(np.abs(node_data))
                if cmap is None:
                    cmap = 'coolwarm'
            else:
                vmin = np.nanmin(node_data)
                vmax = np.nanmax(node_data)
    elif node_data is None:
        c = 'white'

    axis_level = ax is not None
    if not axis_level:
        f, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    else:
        f = ax.get_figure()

    # plot: edges
    if edge_data is not None:
        edge_density = np.count_nonzero(np.triu(edge_data)) / ((edge_data.shape[0] ** 2 - edge_data.shape[0]) / 2)

        non_zero_edges = np.where(edge_data != 0)
        n_edges = len(non_zero_edges[0])
        print('Adding {0} edges...'.format(n_edges))

        for edge_i in np.arange(n_edges):
            node_i = non_zero_edges[0][edge_i]
            node_j = non_zero_edges[1][edge_i]
            edge_thickness = edge_data[node_i, node_j] / edge_data.max()
            if edge_density > 0.05:
                edge_thickness = edge_thickness * .005
            else:
                edge_thickness = edge_thickness * .25

            ax.plot([parcel_coords.iloc[node_i, 0], parcel_coords.iloc[node_j, 0]],
                    [parcel_coords.iloc[node_i, 1], parcel_coords.iloc[node_j, 1]],
                    c='gray', linewidth=edge_thickness, alpha=0.25)

    # scatter: nodes
    s = 5
    if fig_height != 1.25:
        x = ((fig_height - 1.25) / 1.25)
        s = s + (s * x)
    sc = ax.scatter(parcel_coords.iloc[:, 0], parcel_coords.iloc[:, 1], c=c, s=s,
                    edgecolors='gray', linewidths=0.25, cmap=cmap, vmax=vmax, vmin=vmin, zorder=2)

    if add_colorbar is not False:
        if axis_level:
            f.colorbar(sc, ax=ax)
        else:
            cb_ax = f.add_axes([0.95, .125, 0.1, 0.75])
            f.colorbar(sc, cax=cb_ax)

    ax.set_axis_off()

    if not axis_level:
        f.tight_layout()
        plt.show()

    return f


def _annot_to_surf(roi_data, annot_file, mask=None):
    """Map parcel data to surface vertices via a FreeSurfer annotation file.

    Unassigned vertices (label 0 = medial wall) are left as NaN so surfplot
    renders them as the sulcal background rather than the first colormap colour.
    Per-hemisphere labels are 1-indexed (label i → roi_data[i-1]).
    """
    labels, _, _ = nib.freesurfer.read_annot(str(annot_file))
    vtx_data = np.full(labels.shape, np.nan)
    for i in np.unique(labels[labels > 0]):
        if mask is not None and mask[i - 1]:
            continue
        vtx_data[labels == i] = roi_data[i - 1]
    return vtx_data


def _draw_cbar(fig, cax, vmin, vmax, cmap, label, fontsize, orientation='horizontal'):
    """Draw a colorbar for a value range onto an existing axis.

    Shared by plot_brain_surface_data and plot_brain_surface_data_single, which differ in where
    the axis sits, not in how the bar itself is built.
    """
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation=orientation)
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize, length=2, pad=1)

    ticks, fmt = _cbar_ticks(vmin, vmax)
    cbar.set_ticks(ticks)
    if fmt:
        axis = cbar.ax.yaxis if orientation == 'vertical' else cbar.ax.xaxis
        axis.set_major_formatter(FormatStrFormatter(fmt))
    return cbar


def _cbar_ticks(vmin, vmax, n=3):
    """Return n evenly-spaced tick values; use integer format when vmin/vmax are whole numbers."""
    ticks = np.linspace(vmin, vmax, n)
    if vmin == np.round(vmin) and vmax == np.round(vmax):
        return np.round(ticks).astype(int), '%d'
    return ticks, None


def _prepare_brain_surface_data(data_vector, parcellation, surface,
                                 data_mask, vmin, vmax, threshold, symmetric_cbar):
    """Validate inputs, map parcel data to surface vertices, compute colour range.

    Uses FreeSurfer annotation files directly (no volumetric projection) so
    every cortical vertex is assigned its parcel value with no gaps.
    """
    data_vector = np.array(data_vector)
    if data_vector.ndim != 1:
        raise ValueError("data_vector must be 1-dimensional")

    if data_mask is not None and threshold is not None:
        raise ValueError("data_mask and threshold cannot be used at the same time")

    if data_mask is not None:
        data_mask = np.array(data_mask, dtype=bool)
        if data_mask.shape != data_vector.shape:
            raise ValueError("data_mask must be the same length as data_vector")

    _parcellation_map = {
        'schaefer_400-7':  (400, 7),
        'schaefer_200-7':  (200, 7),
        'schaefer_100-7':  (100, 7),
        'schaefer_400-17': (400, 17),
        'schaefer_200-17': (200, 17),
        'schaefer_100-17': (100, 17),
        # Short forms without a network suffix default to the 7-network order. These are what
        # plot_brain_surface_data's own default argument uses, so leaving them out made the
        # function raise when called with defaults.
        'schaefer_400':    (400, 7),
        'schaefer_200':    (200, 7),
        'schaefer_100':    (100, 7),
    }
    if parcellation not in _parcellation_map:
        raise ValueError(
            f"Unsupported parcellation: {parcellation!r}. "
            f"Expected one of {sorted(_parcellation_map)}")
    n_rois, yeo_networks = _parcellation_map[parcellation]

    if len(data_vector) != n_rois:
        raise ValueError(
            f"Data vector length ({len(data_vector)}) doesn't match "
            f"expected parcellation size ({n_rois})")

    if surface not in ('fsaverage5', 'fsaverage'):
        raise ValueError(f"Unsupported surface: {surface}")

    annot_dir = os.path.join(_schaefer_annot_dir(), surface, 'label')
    net_str   = f'{yeo_networks}Networks'
    lh_annot  = os.path.join(annot_dir,
                              f'lh.Schaefer2018_{n_rois}Parcels_{net_str}_order.annot')
    rh_annot  = os.path.join(annot_dir,
                              f'rh.Schaefer2018_{n_rois}Parcels_{net_str}_order.annot')

    n_hemi = n_rois // 2
    lh_mask = data_mask[:n_hemi] if data_mask is not None else None
    rh_mask = data_mask[n_hemi:] if data_mask is not None else None

    surf_data_left  = _annot_to_surf(data_vector[:n_hemi], lh_annot, lh_mask)
    surf_data_right = _annot_to_surf(data_vector[n_hemi:], rh_annot, rh_mask)

    unmasked = data_vector if data_mask is None else data_vector[~data_mask]
    if vmin is None:
        vmin = np.nanmin(unmasked)
    if vmax is None:
        vmax = np.nanmax(unmasked)
    if symmetric_cbar:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    return surf_data_left, surf_data_right, vmin, vmax


def _load_surf_meshes(surface, inflated):
    """Load brainspace-compatible surface meshes for surfplot."""
    from brainspace.mesh.mesh_io import read_surface as bs_read_surface

    surf_mesh = datasets.fetch_surf_fsaverage(mesh=surface)
    mesh_key = 'infl' if inflated else 'pial'
    surf_lh = bs_read_surface(str(surf_mesh[f'{mesh_key}_left']))
    surf_rh = bs_read_surface(str(surf_mesh[f'{mesh_key}_right']))
    return surf_lh, surf_rh


def _build_surfplot_figure(surf_data_left, surf_data_right, surf_lh, surf_rh,
                            vmin, vmax, cmap, threshold, brightness, alpha,
                            target_ratio=1.8, zoom=1.25):
    """Render brain surfaces with surfplot and return a matplotlib figure.

    Always renders WITHOUT a colorbar at a fixed internal resolution so that
    the caller can embed the image at any target size and add its own colorbar.
    Never passes a tiny figsize to surfplot — that breaks its axis layout.
    """
    from surfplot import Plot

    sd_left = surf_data_left.copy().astype(float)
    sd_right = surf_data_right.copy().astype(float)

    if threshold is not None:
        sd_left[np.abs(sd_left) < threshold] = np.nan
        sd_right[np.abs(sd_right) < threshold] = np.nan

    size_h = 800
    size_w = max(800, int(size_h * target_ratio))

    p = Plot(
        surf_lh=surf_lh, surf_rh=surf_rh,
        layout='grid', views=['lateral', 'medial'],
        size=(size_w, size_h), zoom=zoom, brightness=brightness,
    )
    p.add_layer(
        {'left': sd_left, 'right': sd_right},
        cmap=cmap,
        color_range=(vmin, vmax),
        alpha=alpha,
        cbar=False,
        zero_transparent=False,
    )
    return p.build(colorbar=False)


def _surfplot_to_array(sp_fig, dpi=200):
    """Save a surfplot figure to a cropped numpy image array and close the figure."""
    buf = io.BytesIO()
    sp_fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = np.array(Image.open(buf))
    plt.close(sp_fig)
    return img


def plot_brain_surface_data(data_vector, parcellation='schaefer_400', surface='fsaverage5',
                            cmap='viridis', vmin=None, vmax=None, threshold=None,
                            data_mask=None,
                            title=None, figsize=(1.5, 1.25),
                            colorbar=True, symmetric_cbar=False, cbar_label='Data Values',
                            cbar_orientation='right',
                            save_path=None, dpi=300, alpha=0.8, darkness=0.7,
                            show_stats=False, fontsize=8, inflated=True):
    """
    Plot brain data on cortical surface with lateral and medial views for both hemispheres.

    Parameters
    -----------
    data_vector : array-like
        1D numpy array containing brain data values. Length should match the number of parcels
        in the specified parcellation (e.g., 400 for Schaefer 400-parcel atlas).
    parcellation : str, default 'schaefer_400'
        Parcellation scheme. One of 'schaefer_100', 'schaefer_200', 'schaefer_400' (which use the
        7-network order), or an explicit '<name>-<networks>' form such as 'schaefer_400-17'.
        Requires the matching FreeSurfer .annot files under ``SCHAEFER_ANNOT_DIR``.
    surface : str, default 'fsaverage5'
        Brain surface to use. Options: 'fsaverage5', 'fsaverage'
    cmap : str, default 'viridis'
        Colormap for data visualization
    vmin, vmax : float, optional
        Min/max values for color scaling. If None, uses data range.
    threshold : float, optional
        Threshold below which values are not displayed. Cannot be used with data_mask.
    data_mask : array-like of bool, optional
        1D boolean array of the same length as data_vector. Where True, the corresponding
        parcel values are excluded from plotting. Cannot be used with threshold.
    title : str, default 'Brain Surface Data'
        Main title for the figure
    figsize : tuple, default (6, 4)
        Figure size (width, height) in inches
    colorbar : bool, default True
        Whether to show colorbar
    symmetric_cbar : bool, default False
        Whether to center colorbar at zero
    cbar_label : str, default 'Data Values'
        Label for the colorbar
    cbar_orientation : str, default 'bottom'
        Colorbar placement. Options: 'bottom', 'right'.
    save_path : str, optional
        Path to save the figure
    dpi : int, default 300
        DPI for saved figure
    alpha : float, default 0.8
        Transparency of the surface data overlay
    darkness : float, default 0.7
        Darkness of the brain surface (0=light, 1=dark)
    show_stats : bool, default False
        Whether to show data statistics text box
    fontsize : int, default 6
        Font size for all text elements in the figure

    Returns
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    axes : list
        List of matplotlib axes objects for each subplot

    Examples
    --------
    A map on the 400-parcel Schaefer atlas, one value per parcel in 7-network order:

    >>> import numpy as np
    >>> brain_map = np.random.randn(400)
    >>> fig = plot_brain_surface_data(brain_map, title='Example map')

    With a diverging colormap, symmetric limits, and small values hidden:

    >>> fig = plot_brain_surface_data(
    ...     brain_map,
    ...     parcellation='schaefer_400',
    ...     cmap='RdBu_r',
    ...     symmetric_cbar=True,
    ...     threshold=1.0,
    ...     title='Thresholded map',
    ...     show_stats=True,
    ... )
    """
    surf_data_left, surf_data_right, vmin, vmax = _prepare_brain_surface_data(
        data_vector, parcellation, surface, data_mask, vmin, vmax, threshold, symmetric_cbar)

    surf_lh, surf_rh = _load_surf_meshes(surface, inflated)

    top    = 0.06 if title else 0.01
    bottom = 0.14 if (colorbar and cbar_orientation == 'bottom') else 0.01
    right  = 0.18 if (colorbar and cbar_orientation == 'right')  else 0.0

    # Target W:H for the brain panel so surfplot renders at the right ratio
    brain_w_inches = figsize[0] * (1.0 - right)
    brain_h_inches = figsize[1] * (1.0 - bottom - top)
    target_ratio   = brain_w_inches / brain_h_inches

    brain_img = _surfplot_to_array(
        _build_surfplot_figure(
            surf_data_left, surf_data_right, surf_lh, surf_rh,
            vmin, vmax, cmap, threshold, 1.0 - darkness, alpha,
            target_ratio=target_ratio,
        )
    )

    # --- build the final matplotlib figure at the user's figsize ---
    fig = plt.figure(figsize=figsize)

    # Position the brain axes to match the image's native aspect ratio,
    # centred in the available space — no horizontal/vertical distortion.
    img_h, img_w = brain_img.shape[:2]
    img_ratio    = img_w / img_h
    avail_w      = 1.0 - right
    avail_h      = figsize[1] * (1.0 - bottom - top)

    if img_ratio >= target_ratio:
        dw = avail_w
        dh = brain_w_inches / (img_ratio * figsize[1])
    else:
        dh = avail_h / figsize[1]
        dw = avail_h * img_ratio / figsize[0]

    cx = avail_w / 2
    cy = bottom + (avail_h / figsize[1]) / 2
    ax_brain = fig.add_axes([cx - dw / 2, cy - dh / 2, dw, dh])
    ax_brain.imshow(brain_img)
    ax_brain.axis('off')

    if title:
        fig.text(0.5, 1.0 - top * 0.4, title, ha='center', va='top',
                 fontsize=fontsize, fontweight='bold')

    if colorbar:
        if cbar_orientation == 'right':
            ax_cbar = fig.add_axes([1.0 - right + 0.02, 0.15, 0.06, 0.70])
            orientation = 'vertical'
        else:
            ax_cbar = fig.add_axes([0.12, 0.02, 0.76, 0.07])
            orientation = 'horizontal'
        _draw_cbar(fig, ax_cbar, vmin, vmax, cmap, cbar_label, fontsize, orientation)

    if show_stats:
        data_vector = np.array(data_vector)
        stats_text = (
            f"[{np.min(data_vector):.2f}, {np.max(data_vector):.2f}]  "
            f"μ={np.mean(data_vector):.2f}  σ={np.std(data_vector):.2f}"
        )
        ax_brain.text(0.01, 0.01, stats_text, transform=ax_brain.transAxes,
                      fontsize=max(fontsize - 1, 5), va='bottom',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.6))

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    return fig, [ax_brain]


def plot_brain_surface_data_single(data_vector, fig, subplotspec,
                                    parcellation='schaefer_400-7', surface='fsaverage5',
                                    cmap='viridis', vmin=None, vmax=None, threshold=None,
                                    data_mask=None,
                                    colorbar=True, symmetric_cbar=False, cbar_label='Data Values',
                                    alpha=0.8, darkness=0.7, fontsize=6, inflated=True):
    """
    Axis-level variant of plot_brain_surface_data.

    Renders the 4 standard brain views (L lateral, L medial, R lateral, R medial) into a
    region of an *existing* figure defined by a matplotlib SubplotSpec, rather than
    creating its own figure. Suitable for embedding brain surface panels inside a larger
    multi-panel figure built with GridSpec.

    Parameters
    -----------
    data_vector : array-like
        1D numpy array of brain data values matching the parcellation size.
    fig : matplotlib.figure.Figure
        Existing figure to plot into.
    subplotspec : matplotlib.gridspec.SubplotSpec
        Region of the figure in which to embed the 4 views (e.g. ``gs[1, 2]``).
        The region is subdivided into a 2×2 nested GridSpec internally.
    parcellation : str, default 'schaefer_400-7'
        Parcellation scheme. Options: 'schaefer_400-7', 'schaefer_200-7', 'schaefer_100-7',
        'schaefer_400-17', 'schaefer_200-17', 'schaefer_100-17'.
    surface : str, default 'fsaverage5'
        Brain surface mesh. Options: 'fsaverage5', 'fsaverage'.
    cmap : str, default 'viridis'
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color scale bounds. If None, derived from data range.
    threshold : float, optional
        Hide values below this threshold. Cannot be used with data_mask.
    data_mask : array-like of bool, optional
        Regions where True are excluded from display. Cannot be used with threshold.
    colorbar : bool, default True
        Whether to add a colorbar. It is attached to the 4 brain axes and auto-positioned
        by matplotlib.
    symmetric_cbar : bool, default False
        Center the colorbar symmetrically around zero.
    cbar_label : str, default 'Data Values'
        Label for the colorbar.
    alpha : float, default 0.8
        Transparency of the surface data overlay.
    darkness : float, default 0.7
        Darkness of the background sulcal map (0=light, 1=dark).
    fontsize : int, default 6
        Font size for colorbar text.

    Returns
    --------
    ax : matplotlib.axes.Axes
        The axes containing the embedded brain surface image.
    """
    surf_data_left, surf_data_right, vmin, vmax = _prepare_brain_surface_data(
        data_vector, parcellation, surface, data_mask, vmin, vmax, threshold, symmetric_cbar)

    surf_lh, surf_rh = _load_surf_meshes(surface, inflated)

    bb = subplotspec.get_position(fig)
    x0, y0, w, h = bb.x0, bb.y0, bb.width, bb.height

    cbar_h_frac = 0.10
    cbar_pad    = 0.02
    brain_h     = h * (1.0 - cbar_h_frac - cbar_pad) if colorbar else h
    target_ratio = (w * fig.get_figwidth()) / (brain_h * fig.get_figheight())

    brain_img = _surfplot_to_array(
        _build_surfplot_figure(
            surf_data_left, surf_data_right, surf_lh, surf_rh,
            vmin, vmax, cmap, threshold, 1.0 - darkness, alpha,
            target_ratio=target_ratio,
        )
    )

    img_h, img_w = brain_img.shape[:2]
    img_ratio    = img_w / img_h

    if colorbar:
        cbar_abs_h = h * cbar_h_frac
        cbar_abs_p = h * cbar_pad
        avail_h    = h - cbar_abs_h - cbar_abs_p
    else:
        avail_h    = h
        cbar_abs_h = 0.0
        cbar_abs_p = 0.0

    # Centre the brain image in the available space, preserving native ratio
    if img_ratio >= target_ratio:
        dw = w
        dh = w / img_ratio
    else:
        dh = avail_h
        dw = avail_h * img_ratio

    cx  = x0 + w / 2
    cy  = y0 + cbar_abs_h + cbar_abs_p + avail_h / 2
    ax  = fig.add_axes([cx - dw/2, cy - dh/2, dw, dh])
    ax_cbar = fig.add_axes([x0 + w*0.10, y0, w*0.80, cbar_abs_h]) if colorbar else None

    ax.imshow(brain_img)
    ax.axis('off')

    if colorbar:
        _draw_cbar(fig, ax_cbar, vmin, vmax, cmap, cbar_label, fontsize, 'horizontal')

    return ax


def categorical_kde_plot(df, variable, category, fig_width=4, fig_height=1.5, category_order=None, horizontal=False, rug=True, color_palette=None):
    """Draw a categorical KDE plot

    Parameters
    ----------
    df: pd.DataFrame
        The data to plot
    variable: str
        The column in the `df` to plot (continuous variable)
    category: str
        The column in the `df` to use for grouping (categorical variable)
    horizontal: bool
        If True, draw density plots horizontally. Otherwise, draw them
        vertically.
    rug: bool
        If True, add also a sns.rugplot.
    """
    if category_order is None:
        categories = list(df[category].unique())
    else:
        categories = category_order[:]

    fig_size = (fig_width, fig_height)

    fig, axes = plt.subplots(
        nrows=len(categories) if horizontal else 1,
        ncols=1 if horizontal else len(categories),
        figsize=fig_size[::-1] if not horizontal else fig_size,
        sharex=horizontal,
        sharey=not horizontal,
    )

    for i, (cat, ax) in enumerate(zip(categories, axes)):
        plot_data = df[df[category] == cat]
        sns.kdeplot(data=plot_data, ax=ax,
                    x=variable if horizontal else None,
                    y=None if horizontal else variable,
                    color=color_palette[i] if color_palette is not None else "lightslategray",
                    bw_adjust=1, clip_on=False, fill=True, alpha=1, linewidth=1
                    )
        # ax.axvline(x=plot_data[variable].mean(), ymax=0.5, color='lightslategray')
        # ax.axvline(x=plot_data[variable].median(), ymax=0.5, color='lightslategray')
        if horizontal:
            ax.axvline(x=plot_data[variable].median(), ymax=0.5, color='white')
        else:
            ax.axhline(y=plot_data[variable].median(), xmax=0.5, color='white')

        if rug:
            sns.rugplot(data=df[df[category] == cat],
                        x=variable if horizontal else None, y=None if horizontal else variable,
                        ax=ax, color="white", height=0.15, linewidth=0.5
                        )

        keep_variable_axis = (i == len(fig.axes) - 1) if horizontal else (i == 0)
        _format_axis(ax, cat, horizontal, keep_variable_axis=keep_variable_axis)

    # plt.tight_layout()
    # plt.show()

    return fig, axes


def _format_axis(ax, category, horizontal=False, keep_variable_axis=True):
    """Style a categorical KDE sub-axis: hide spines and label the category axis.

    Parameters
    ----------
    ax : matplotlib Axes
    category : str
        Category label placed on the categorical axis.
    horizontal : bool
        Orient the category on the y-axis (else the x-axis).
    keep_variable_axis : bool
        Keep the variable (non-category) axis visible.
    """

    # Remove the axis lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', which='major')

    if horizontal:
        ax.set_ylabel(None)
        lim = ax.get_ylim()
        ax.set_yticks([(lim[0] + lim[1]) / 2])
        ax.set_yticklabels([category])
        if not keep_variable_axis:
            ax.get_xaxis().set_visible(False)
            ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel(None)
        lim = ax.get_xlim()
        ax.set_xticks([(lim[0] + lim[1]) / 2])
        ax.set_xticklabels([category])
        if not keep_variable_axis:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)


def paired_line_plot(x, y_1, y_2, y_1_label, y_2_label, ax, add_summary_line='mean', plot_diff=False):
    """Plot two families of lines over x with a mean/median summary, or their difference.

    Parameters
    ----------
    x : array-like
        Shared x values.
    y_1, y_2 : (n_x, n_series) ndarray
        Two families of series (faint individual lines plus a bold summary line).
    y_1_label, y_2_label : str
        Legend labels.
    ax : matplotlib Axes
        Axis to plot on.
    add_summary_line : {'mean', 'median'} or other
        Summary line to overlay on each family (any other value skips the summary).
    plot_diff : bool
        Plot y_2 - y_1 instead of the two families separately.
    """
    my_colors = get_snap_colors(cat_trio=True, as_list=True)

    # y_1 = y_1.mean(axis=0).mean(axis=0)
    # y_2 = y_2.mean(axis=0).mean(axis=0)
    
    if plot_diff is False:
        if add_summary_line == 'mean':
            ax.plot(x, y_1, color=my_colors[0], alpha=0.05)
            ax.plot(x, y_1.mean(axis=-1), label=y_1_label, color=my_colors[0], linewidth=1.5)
        elif add_summary_line == 'median':
            ax.plot(x, y_1, color=my_colors[0], alpha=0.05)
            ax.plot(x, np.median(y_1, axis=-1), label=y_1_label, color=my_colors[0], linewidth=1.5)
        else:
            ax.plot(x, y_1, label=y_1_label, color=my_colors[0], alpha=1)

        if add_summary_line == 'mean':
            ax.plot(x, y_2, color=my_colors[1], alpha=0.05)
            ax.plot(x, y_2.mean(axis=-1), label=y_2_label, color=my_colors[1], linewidth=1.5)
        elif add_summary_line == 'median':
            ax.plot(x, y_2, color=my_colors[1], alpha=0.05)
            ax.plot(x, np.median(y_2, axis=-1), label=y_2_label, color=my_colors[1], linewidth=1.5)
        else:
            ax.plot(x, y_2, label=y_2_label, color=my_colors[1], alpha=1)
    else:
        ax.plot(x, y_2 - y_1, label='{0}-{1}'.format(y_2_label, y_1_label), color=my_colors[0], alpha=0.05)
        if add_summary_line == 'mean':
            ax.plot(x, y_2.mean(axis=-1) - y_1.mean(axis=-1), label='{0}-{1}'.format(y_2_label, y_1_label), color=my_colors[0], linewidth=1.5)
        elif add_summary_line == 'median':
            ax.plot(x, np.median(y_2, axis=-1) - np.median(y_1, axis=-1), label='{0}-{1}'.format(y_2_label, y_1_label), color=my_colors[0], linewidth=1.5)
    # ax.set_xticks(x)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
