import os, platform
import numpy as np
import scipy as sp
import nibabel as nib
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,  BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
plt.ion()

import nibabel as nib
from nilearn import datasets
from nilearn import plotting
from nilearn.image import new_img_like
from nilearn.surface import load_surf_data, vol_to_surf

from snaplab_tools.plotting.utils import get_p_val_string, roi_to_vtx, get_my_colors


def plot_correlation(x, y, ax, x_label=None, y_label=None, title=None, 
                    method='pearson', color="#3B3B3B", alpha=0.6, 
                    size=20, show_line=True, 
                    show_confidence=True, show_stats=True,
                    stats_position='upper left', font_size=6,
                    grid=True, grid_alpha=0.3,
                    outlier_threshold=None, highlight_outliers=False, 
                    return_stats=False,
                    auto_polynomial=False, models_to_test=[1, 2, 3], 
                    model_selection_metric='variance_explained',
                    data_group=None, data_group_cmap='tab10',
                    cbar_label='Data Values'):
    """
    Enhanced correlation plot with automatic polynomial model selection and data group coloring.
    
    Parameters:
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
    size : float, default 20
        Size of scatter points
    show_line : bool, default True
        Whether to show regression line
    show_confidence : bool, default True
        Whether to show confidence interval around regression line
    show_stats : bool, default True
        Whether to display correlation statistics on plot
    stats_position : str, default 'upper left'
        Position of statistics text box
    font_size : int, default 8
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

    Returns:
    --------
    ax : matplotlib axis object
        The axis object with plot
    stats_dict : dict, optional
        Dictionary with correlation statistics (if return_stats=True)
    """
    
    # Handle pandas Series input and extract labels
    x_series_name = None
    y_series_name = None
    
    if isinstance(x, pd.Series):
        x_series_name = x.name
        x = x.values
    if isinstance(y, pd.Series):
        y_series_name = y.name
        y = y.values
    
    # Set default labels from pandas Series names if not provided
    if x_label is None and x_series_name is not None:
        x_label = str(x_series_name)
    if y_label is None and y_series_name is not None:
        y_label = str(y_series_name)
    
    # Convert to numpy arrays and handle missing data
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Handle data_group input
    if data_group is not None:
        data_group = np.array(data_group)  # Don't force dtype - allow strings or integers
        if len(data_group) != len(x):
            raise ValueError(f"data_group length ({len(data_group)}) must match x and y length ({len(x)})")
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if data_group is not None:
        data_group_clean = data_group[valid_mask]
    
    if len(x_clean) < 3:
        raise ValueError("Need at least 3 valid data points for correlation")
    
    # Calculate correlation
    if method.lower() == 'pearson':
        corr_coef, p_value = sp.stats.pearsonr(x_clean, y_clean)
        method_name = "Pearson"
    elif method.lower() == 'spearman':
        corr_coef, p_value = sp.stats.spearmanr(x_clean, y_clean)
        method_name = "Spearman"
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")
    
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
        
        # Get colormap
        cmap = plt.cm.get_cmap(data_group_cmap)
        
        # Create colors for each group
        if n_groups <= 10 and data_group_cmap == 'tab10':
            # Use discrete colors for tab10
            colors = [cmap(i) for i in range(n_groups)]
        elif n_groups <= 20 and data_group_cmap == 'tab20':
            # Use discrete colors for tab20
            colors = [cmap(i) for i in range(n_groups)]
        else:
            # Use continuous colormap
            colors = [cmap(i / max(1, n_groups - 1)) for i in range(n_groups)]
        
        # Create mapping from group labels to colors
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
        if n_groups > 1:  # Only add colorbar if more than one group
            # For string labels, we need to create a discrete colorbar
            if isinstance(unique_groups[0], (str, np.str_)):
                # Create discrete colorbar for string labels
                # Import required matplotlib components for colorbar handling
                
                # Create a listed colormap from the colors we're using
                listed_cmap = ListedColormap(colors[:n_groups])
                bounds = np.arange(n_groups + 1) - 0.5
                norm = BoundaryNorm(bounds, listed_cmap.N)
                
                sm = plt.cm.ScalarMappable(cmap=listed_cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.05)
                cbar.set_label(cbar_label, fontsize=font_size)

                # Set colorbar ticks to show string labels
                cbar.set_ticks(np.arange(n_groups))
                if n_groups <= 10:
                    cbar.set_ticklabels(unique_groups, fontsize=font_size-2)
                else:
                    # For many groups, show fewer labels
                    tick_indices = np.linspace(0, n_groups-1, min(5, n_groups), dtype=int)
                    cbar.set_ticks(tick_indices)
                    cbar.set_ticklabels([unique_groups[i] for i in tick_indices], fontsize=font_size-2)
            else:
                # Original numeric colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, 
                                          norm=plt.Normalize(vmin=min(unique_groups), 
                                                           vmax=max(unique_groups)))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.05)
                cbar.set_label(cbar_label, fontsize=font_size)
                
                # Set colorbar ticks to show group numbers
                if n_groups <= 10:
                    cbar.set_ticks(unique_groups)
                else:
                    # For many groups, show fewer ticks
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
        
        if auto_polynomial and len(x_clean) >= (max(models_to_test) + 1):  # Need more points than max degree
            # Test specified polynomial degrees
            model_results = {}
            
            for degree in models_to_test:
                if len(x_clean) > degree:  # Need more points than parameters
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
                        
                        # RMSE
                        rmse = np.sqrt(np.mean(residuals ** 2))
                        
                        # MAE
                        mae = np.mean(np.abs(residuals))
                        
                        model_results[degree] = {
                            'coeffs': poly_coeffs,
                            'r_squared': r_squared,
                            'rmse': rmse,
                            'mae': mae,
                            'y_pred': y_pred
                        }
                    except (np.RankWarning, np.linalg.LinAlgError):
                        # Skip this degree if fitting fails
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
                    raise ValueError("model_selection_metric must be 'variance_explained', 'rmse', or 'mae'")
                
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
                
                # Use the original color for regression line
                line_color = color if data_group is None else '#3B3B3B'
                ax.plot(x_line, y_line, color=line_color, linewidth=2.5, alpha=0.8, 
                       linestyle='-', 
                       label=f'{model_name} fit (R²={best_model["r_squared"]:.3f})')
                
                # Add confidence interval for polynomial (approximation)
                if show_confidence and best_degree == 1:  # Only for linear
                    # Use the original linear regression confidence interval calculation
                    slope, intercept, r_value, p_value_reg, std_err = sp.stats.linregress(x_clean, y_clean)
                    
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
            slope, intercept, r_value, p_value_reg, std_err = sp.stats.linregress(x_clean, y_clean)
            
            # Create line points
            x_line = np.linspace(np.min(x_clean), np.max(x_clean), 100)
            y_line = slope * x_line + intercept
            
            # Plot regression line
            line_color = color if data_group is None else '#3B3B3B'
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
        # Determine significance stars
        if p_value < 0.001:
            sig_stars = "***"
        elif p_value < 0.01:
            sig_stars = "**"
        elif p_value < 0.05:
            sig_stars = "*"
        else:
            sig_stars = "ns"
        
        # Create statistics text
        stats_text = f"r = {corr_coef:.2f}{sig_stars}\n"
        stats_text += f"p = {p_value:.2e}"
        
        # Add data group info if provided
        if data_group is not None:
            unique_groups = np.unique(data_group_clean)
            # Show actual group labels if they're short enough
            if len(unique_groups) <= 5:
                if isinstance(unique_groups[0], (str, np.str_)):
                    group_labels = ", ".join([str(g) for g in unique_groups])
                    if len(group_labels) <= 20:  # Only show if not too long
                        stats_text += f"\n({group_labels})"
        
        # Add model information if available and auto_polynomial was used
        if best_model_info is not None and auto_polynomial:
            degree_names = {1: 'Linear', 2: 'Quadratic', 3: 'Cubic', 4: 'Quartic', 
                           5: 'Quintic', 6: 'Sextic'}
            model_name = degree_names.get(best_model_info['degree'], f"Degree {best_model_info['degree']}")
            metric_values = {
                'variance_explained': best_model_info['r_squared'],
                'rmse': best_model_info['rmse'],
                'mae': best_model_info['mae']
            }
            metric_labels = {
                'variance_explained': 'R²',
                'rmse': 'RMSE',
                'mae': 'MAE'
            }
            metric_value = metric_values.get(model_selection_metric, 0)
            metric_label = metric_labels.get(model_selection_metric, model_selection_metric)
            stats_text += f"\n{metric_label} = {metric_value:.2f}"
        
        if np.any(outlier_mask):
            stats_text += f"\nOutliers: {np.sum(outlier_mask)}"
        
        # Position the text box
        position_dict = {
            'upper left': (0.05, 0.95),
            'upper right': (0.95, 0.95),
            'lower left': (0.05, 0.05),
            'lower right': (0.95, 0.05)
        }
        
        text_x, text_y = position_dict.get(stats_position, (0.05, 0.95))
        ha = 'left' if text_x < 0.5 else 'right'
        va = 'top' if text_y > 0.5 else 'bottom'
        
        ax.text(text_x, text_y, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor='gray', alpha=0.8),
               fontsize=font_size-2, ha=ha, va=va, family='monospace')
    
    # Customize appearance
    ax.set_xlabel(x_label or 'X Variable', fontsize=font_size)
    ax.set_ylabel(y_label or 'Y Variable', fontsize=font_size)
    
    if title:
        ax.set_title(title, fontsize=font_size, fontweight='bold', pad=10)
    
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
    ax.tick_params(axis='both', which='major', labelsize=font_size,
                  length=6, width=1.2, colors='black')
    
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
                'group_type': 'string' if data_group is not None and isinstance(np.unique(data_group_clean)[0], (str, np.str_)) else 'numeric'
            } if data_group is not None else None
        }
        return ax, stats_dict
    else:
        return ax


def null_plot(observed, null, xlabel, ax, p_val=None, add_text=True, line_color=None, use_kde=False):
    if line_color is None:
        my_colors = get_my_colors()
        line_color = my_colors['north_sea_green']
        # color_blue = sns.color_palette("Set1")[1]
        # color_red = sns.color_palette("Set1")[0]
    if use_kde is True:
        sns.kdeplot(x=null, ax=ax, color='gray')
    else:
        sns.histplot(x=null, ax=ax, color='gray')
    ax.axvline(x=observed, ymax=1, clip_on=False, linewidth=1.5, color=line_color)
    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('counts')

    if add_text is True:
        textstr = '{:.2f}'.format(observed)
        ax.text(observed, ax.get_ylim()[1], textstr,
                horizontalalignment='center', verticalalignment='bottom',
                rotation=0, c=line_color, size=6)

    if add_text is True:
        if p_val is not None:
            ax.text(observed, ax.get_ylim()[1], get_p_val_string(p_val),
                    horizontalalignment='left', verticalalignment='top',
                    rotation=270, c=line_color, size=6)

    # if p_val:
    #     textstr = '{:}'.format(get_p_val_string(p_val))
    #     ax.text(observed - (np.abs(observed)*0.0025), ax.get_ylim()[1], textstr,
    #             horizontalalignment='right', verticalalignment='top',
    #             rotation=270, c=color_red)


def brain_scatter_plot(parcel_coords, node_data=None, edge_data=None, fig_height=1.25, vmin=None, vmax=None, cmap=None, add_colorbar=False):
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
                    cmap ='coolwarm'
            else:
                vmin = np.nanmin(node_data)
                vmax = np.nanmax(node_data)
    elif node_data is None:
        c = 'white'
        
    f, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
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
        cb_ax = f.add_axes([0.95,.125,0.1,0.75])
        f.colorbar(sc, cax=cb_ax)

    ax.set_axis_off()
    f.tight_layout()
    plt.show()
    
    return f


def surface_plot(data, lh_annot_file, rh_annot_file,
                 fsaverage=datasets.fetch_surf_fsaverage(mesh='fsaverage5'),
                 order='lr', cmap='viridis', cblim=None, title_str=None):

    # project data to surface
    n_nodes = len(data)
    if order == 'lr':
        vtx_data_lh, _, _ = roi_to_vtx(data[:int(n_nodes/2)], lh_annot_file)
        vtx_data_rh, _, _ = roi_to_vtx(data[int(n_nodes/2):], rh_annot_file)
    elif order == 'rl':
        vtx_data_lh, _, _ = roi_to_vtx(data[int(n_nodes/2):], rh_annot_file)
        vtx_data_rh, _, _ = roi_to_vtx(data[:int(n_nodes/2)], lh_annot_file)

    # get colorbar axes
    if cblim is None:
        if cmap == 'coolwarm' or cmap == 'vlag' or cmap == 'icefire':
            vmax = np.round(np.nanmax(np.abs(data)), 1)
            vmin = -vmax
        else:
            vmax = np.nanmax(data)
            vmin = np.nanmin(data)
    else:
        vmax = cblim[0]
        vmin = cblim[1]

    # dummy plot for colorbar
    im = plt.imshow(np.random.random((2, 2)), cmap=cmap, vmin=vmin, vmax=vmax)
    plt.close()

    # main plot
    f, ax = plt.subplots(2, 2, figsize=(2.5, 2.5), subplot_kw={'projection': '3d'})
    plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=vtx_data_lh,
                         hemi='left', view='lateral',
                         vmin=vmin, vmax=vmax,
                         bg_map=fsaverage['sulc_left'],
                         bg_on_data=True, axes=ax[0, 0],
                         darkness=.5, cmap=cmap, colorbar=False)

    plotting.plot_surf_roi(fsaverage['infl_right'], roi_map=vtx_data_rh,
                         hemi='right', view='lateral',
                         vmin=vmin, vmax=vmax,
                         bg_map=fsaverage['sulc_right'],
                         bg_on_data=True, axes=ax[0, 1],
                         darkness=.5, cmap=cmap, colorbar=False)

    plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=vtx_data_lh,
                         hemi='left', view='medial',
                         vmin=vmin, vmax=vmax,
                         bg_map=fsaverage['sulc_left'],
                         bg_on_data=True, axes=ax[1, 0],
                         darkness=.5, cmap=cmap, colorbar=False)

    plotting.plot_surf_roi(fsaverage['infl_right'], roi_map=vtx_data_rh,
                         hemi='right', view='medial',
                         vmin=vmin, vmax=vmax,
                         bg_map=fsaverage['sulc_right'],
                         bg_on_data=True, axes=ax[1, 1],
                         darkness=.5, cmap=cmap, colorbar=False)

    plt.subplots_adjust(wspace=-0.075, hspace=-0.3)
    cb_ax = f.add_axes([0.9, 0.25, 0.05, 0.5])  # add colorbar
    f.colorbar(im, cax=cb_ax)
    if title_str:
        f.suptitle(title_str)
    plotting.show()

    return f


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
    my_colors = get_my_colors(cat_trio=True, as_list=True)

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

######################################################################################################################################################
# deprecated functions. Maintained for historical purposes, but have been replaced by newer functions.
######################################################################################################################################################
def reg_plot(x, y, ax, xlabel='X', ylabel='Y', c='gray', annotate='pearson', add_pval=True, regr_line=True, kde=True, fontsize=8, order=1):
    
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    
    if len(x.shape) > 1 and len(y.shape) > 1:
        if x.shape[0] == x.shape[1] and y.shape[0] == y.shape[1]:
            mask_x = ~np.eye(x.shape[0], dtype=bool) * ~np.isnan(x)
            mask_y = ~np.eye(y.shape[0], dtype=bool) * ~np.isnan(y)
            mask = mask_x * mask_y
            indices = np.where(mask)
        else:
            mask_x = ~np.isnan(x)
            mask_y = ~np.isnan(y)
            mask = mask_x * mask_y
            indices = np.where(mask)
    elif len(x.shape) == 1 and len(y.shape) == 1:
        mask_x = ~np.isnan(x)
        mask_y = ~np.isnan(y)
        mask = mask_x * mask_y
        indices = np.where(mask)
    else:
        print('error: input array dimension mismatch.')

    try:
        x = x[indices]
        y = y[indices]
    except:
        pass

    try:
        c = c[indices]
    except:
        pass

    # kde plot
    if kde == True:
        try:
            sns.kdeplot(x=x, y=y, ax=ax, color='gray', thresh=0.05, alpha=0.25)
        except:
            pass

    # regression line
    if regr_line == True:
        # color_blue = sns.color_palette("Set1")[1]
        my_colors = get_my_colors()
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color=my_colors['north_sea_green'], order=order)

    # scatter plot
    if type(c) == str:
        ax.scatter(x=x, y=y, c=c, s=2.5, alpha=0.5)
    else:
        ax.scatter(x=x, y=y, c=c, cmap='viridis', s=2.5, alpha=0.5)

    # axis options
    ax.set_xlabel(xlabel, labelpad=0)
    ax.set_ylabel(ylabel, labelpad=0)
    # ax.tick_params(pad=-2.5)
    # ax.grid(False)
    # sns.despine(right=True, top=True, ax=ax)
    sns.despine(offset=0, trim=False, left=False, right=True, top=True, bottom=False, ax=ax)
    ax.tick_params(left=True, bottom=True)

    # annotation
    r, r_p = sp.stats.pearsonr(x, y)
    rho, rho_p = sp.stats.spearmanr(x, y)
    if type(annotate) == str:
        if annotate == 'pearson':
            if add_pval:
                textstr = '$\mathit{:}$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p))
            else:
                textstr = '$\mathit{:}$ = {:.2f}'.format('{r}', r)
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
        elif annotate == 'spearman':
            if add_pval:
                textstr = '$\\rho$ = {:.2f}, {:}'.format(rho, get_p_val_string(rho_p))
            else:
                textstr = '$\\rho$ = {:.2f}'.format(rho)
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
        elif annotate == 'both':
            if add_pval:
                textstr = '$\mathit{:}$ = {:.2f}, {:}\n$\\rho$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p),
                                                                                    rho, get_p_val_string(rho_p))
            else:
                textstr = '$\mathit{:}$ = {:.2f}\n$\\rho$ = {:.2f}'.format('{r}', r, rho)
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
    elif type(annotate) == tuple:
        coef = annotate[0]
        p = annotate[1]
        textstr = 'coef = {:.2f}, {:}'.format(coef, get_p_val_string(p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize, verticalalignment='top')
    else:
        pass