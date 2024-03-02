import os, platform
import numpy as np
import scipy as sp
import nibabel as nib

import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
from nilearn import datasets
from nilearn import plotting

from snaplab_tools.plotting.utils import get_p_val_string, roi_to_vtx


def reg_plot(x, y, xlabel, ylabel, ax, c='gray', add_spearman=False, kdeplot=True, regplot=True):
    if x.shape == y.shape:
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

    color_blue = sns.color_palette("Set1")[1]
    if kdeplot:
        try:
            sns.kdeplot(x=x, y=y, ax=ax, color='gray', thresh=0.05, alpha=0.25)
        except:
            pass

    if regplot:
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color=color_blue)

    if type(c) == str:
        ax.scatter(x=x, y=y, c=c, s=5, alpha=0.5)
    else:
        ax.scatter(x=x, y=y, c=c, cmap='viridis', s=5, alpha=0.5)
    ax.set_xlabel(xlabel, labelpad=-0.5)
    ax.set_ylabel(ylabel, labelpad=-0.5)
    ax.tick_params(pad=-2.5)
    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)
    r, r_p = sp.stats.pearsonr(x, y)
    if add_spearman:
        rho, rho_p = sp.stats.spearmanr(x, y)
        textstr = '$\mathit{:}$ = {:.2f}, {:}\n$\\rho$ = {:.2f}, {:}' \
            .format('{r}', r, get_p_val_string(r_p), rho, get_p_val_string(rho_p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes,
                verticalalignment='top')
    else:
        textstr = '$\mathit{:}$ = {:.2f}, {:}' \
            .format('{r}', r, get_p_val_string(r_p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes,
                verticalalignment='top')


def null_plot(observed, null, xlabel, ax, p_val=None):
    color_blue = sns.color_palette("Set1")[1]
    color_red = sns.color_palette("Set1")[0]
    sns.histplot(x=null, ax=ax, color='gray')
    ax.axvline(x=observed, ymax=1, clip_on=False, linewidth=1, color=color_blue)
    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('counts')

    textstr = 'obs. = {:.2f}'.format(observed)
    ax.text(observed, ax.get_ylim()[1], textstr,
            horizontalalignment='left', verticalalignment='top',
            rotation=270, c=color_blue)

    if p_val:
        textstr = '{:}'.format(get_p_val_string(p_val))
        ax.text(observed - (np.abs(observed)*0.0025), ax.get_ylim()[1], textstr,
                horizontalalignment='right', verticalalignment='top',
                rotation=270, c=color_red)


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



