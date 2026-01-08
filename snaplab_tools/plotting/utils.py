import os, platform
import numpy as np
import scipy as sp
import nibabel as nib

import seaborn as sns
import matplotlib.pyplot as plt
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


def get_p_val_string(p_val):
    # if np.round(p_val, 3) == 0.000:
        # p_str = "-log10($\mathit{:}$)>25".format('{p}')
    if p_val < 0.05:
        p_str = '$\mathit{:}$={:0.0e}'.format('{p}', p_val)
    else:
        p_str = "$\mathit{:}$={:.3f}".format('{p}', p_val)

    return p_str


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