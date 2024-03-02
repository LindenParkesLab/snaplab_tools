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


def get_p_val_string(p_val):
    if p_val == 0.0:
        p_str = "-log10($\mathit{:}$)>25".format('{p}')
    elif p_val < 0.05:
        p_str = '$\mathit{:}$ = {:0.0e}'.format('{p}', p_val)
    else:
        p_str = "$\mathit{:}$ = {:.3f}".format('{p}', p_val)

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