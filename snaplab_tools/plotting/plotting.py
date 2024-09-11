import os, platform
import numpy as np
import scipy as sp
import nibabel as nib

import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
from nilearn import datasets
from nilearn import plotting

from snaplab_tools.plotting.utils import get_p_val_string, roi_to_vtx, get_my_colors


def reg_plot(x, y, ax, xlabel='X', ylabel='Y', c='gray', annotate='pearson', regr_line=True, kde=True, fontsize=8):
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
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color=my_colors['north_sea_green'])

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
            textstr = '$\mathit{:}$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p))
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
        elif annotate == 'spearman':
            textstr = '$\\rho$ = {:.2f}, {:}'.format(rho, get_p_val_string(rho_p))
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
        elif annotate == 'both':
            textstr = '$\mathit{:}$ = {:.2f}, {:}\n$\\rho$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p),
                                                                                 rho, get_p_val_string(rho_p))
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
    elif type(annotate) == tuple:
        coef = annotate[0]
        p = annotate[1]
        textstr = 'coef = {:.2f}, {:}'.format(coef, get_p_val_string(p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize, verticalalignment='top')
    else:
        pass


def null_plot(observed, null, xlabel, ax, p_val=None):
    color_blue = sns.color_palette("Set1")[1]
    color_red = sns.color_palette("Set1")[0]
    sns.histplot(x=null, ax=ax, color='gray')
    ax.axvline(x=observed, ymax=1, clip_on=False, linewidth=1, color=color_red)
    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('counts')

    if p_val:
        textstr = 'obs. = {:.2f}; {:}'.format(observed, get_p_val_string(p_val))
    else:
        textstr = 'obs. = {:.2f}'.format(observed)
    ax.text(observed, ax.get_ylim()[1], textstr,
            horizontalalignment='left', verticalalignment='top',
            rotation=270, c=color_red)

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
                 order='lr', cmap='viridis', cblim=None):

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
        if cmap == 'coolwarm':
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
        ax.axvline(x=plot_data[variable].median(), ymax=0.5, color='lightslategray')

        if rug:
            sns.rugplot(data=df[df[category] == cat],
                        x=variable if horizontal else None, y=None if horizontal else variable,
                        ax=ax, color="white", height=0.15, linewidth=0.5
                        )

        keep_variable_axis = (i == len(fig.axes) - 1) if horizontal else (i == 0)
        _format_axis(ax, cat, horizontal, keep_variable_axis=keep_variable_axis)

    plt.tight_layout()
    plt.show()

    return fig

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


def paired_line_plot(x, y_1, y_2, y_1_label, y_2_label, ax, add_mean=True, plot_diff=False):
    my_colors = get_my_colors(cat_trio=True, as_list=True)

    # y_1 = y_1.mean(axis=0).mean(axis=0)
    # y_2 = y_2.mean(axis=0).mean(axis=0)
    
    if plot_diff is False:
        if add_mean:
            ax.plot(x, y_1, color=my_colors[0], alpha=0.05)
            ax.plot(x, y_1.mean(axis=-1), label=y_1_label, color=my_colors[0], linewidth=1.5)
        else:
            ax.plot(x, y_1, label=y_1_label, color=my_colors[0], alpha=1)

        if add_mean:
            ax.plot(x, y_2, color=my_colors[1], alpha=0.05)
            ax.plot(x, y_2.mean(axis=-1), label=y_2_label, color=my_colors[1], linewidth=1.5)
        else:
            ax.plot(x, y_2, label=y_2_label, color=my_colors[1], alpha=1)
    else:
        ax.plot(x, y_2 - y_1, label='{0}-{1}'.format(y_2_label, y_1_label), color=my_colors[0], alpha=0.05)
        if add_mean:
            ax.plot(x, y_2.mean(axis=-1) - y_1.mean(axis=-1), label='{0}-{1}'.format(y_2_label, y_1_label), color=my_colors[0], linewidth=1.5)
    # ax.set_xticks(x)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
