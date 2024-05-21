import os, wget
import numpy as np
import pandas as pd
from git.repo.base import Repo
import nibabel as nib
from sklearn.metrics import pairwise_distances
from snaplab_tools.utils import load_schaefer_parc, get_parcelwise_average_surface

class BrainMapLoader:
    def __init__(self, research_data='~/brain_maps', parc='schaefer', n_parcels=400, order=7):
        # analysis parameters
        self.parc = parc
        self.n_parcels = n_parcels
        self.order = order

        # directories
        self.research_data = research_data
        self.bbw_dir = os.path.join(self.research_data, 'BBW_BigData')  # BigBrainData pre-downloaded August 2022
        self.glasser_dir = os.path.join(self.research_data, 'Glasser_et_al_2016_HCP_MMP1.0_kN_RVVG')  # data pre-downloaded from https://balsa.wustl.edu/mpwM

        self.outdir = os.path.join(self.research_data, 'brain_maps')

        if os.path.exists(self.outdir) == False:
            os.makedirs(self.outdir)


    def _get_parc_data(self, parc='schaefer', annot='fsaverage'):
        if parc == 'schaefer':
            self.nifti_file, self.centroids, self.lh_annot_file, self.rh_annot_file, self.hcp_file = load_schaefer_parc(
                n_parcels=self.n_parcels,
                order=self.order,
                annot=annot,
                out_dir='~/research_projects/connectome_loader/data/schaefer_parc')
            self.centroids.set_index('ROI Name', inplace=True)


    def load_cyto(self):
        self._get_parc_data(parc=self.parc, annot='fsaverage')

        lh_gifti_file = os.path.join(self.bbw_dir, 'spaces', 'tpl-fsaverage', 'tpl-fsaverage_hemi-L_den-164k_desc-Hist_G2.shape.gii')  # BigBrainData downloaded August 2022
        rh_gifti_file = os.path.join(self.bbw_dir, 'spaces', 'tpl-fsaverage', 'tpl-fsaverage_hemi-R_den-164k_desc-Hist_G2.shape.gii')  # BigBrainData downloaded August 2022

        # get average values over parcels
        data_lh = get_parcelwise_average_surface(lh_gifti_file, self.lh_annot_file)
        data_rh = get_parcelwise_average_surface(rh_gifti_file, self.rh_annot_file)

        # drop first entry (corresponds to 0)
        data_lh = data_lh[1:]
        data_rh = data_rh[1:]

        if self.parc == 'schaefer':
            self.cyto = np.hstack((data_lh, data_rh)).astype(float)
        elif self.parc == 'glasser':
            self.cyto = np.hstack((data_rh, data_lh)).astype(float)


    def load_micro(self):
        self._get_parc_data(parc=self.parc, annot='fsaverage')

        lh_gifti_file = os.path.join(self.bbw_dir, 'spaces', 'tpl-fsaverage', 'tpl-fsaverage_hemi-L_den-164k_desc-Micro_G1.curv')  # BigBrainData downloaded August 2022
        rh_gifti_file = os.path.join(self.bbw_dir, 'spaces', 'tpl-fsaverage', 'tpl-fsaverage_hemi-R_den-164k_desc-Micro_G1.curv')  # BigBrainData downloaded August 2022

        # get average values over parcels
        data_lh = get_parcelwise_average_surface(lh_gifti_file, self.lh_annot_file)
        data_rh = get_parcelwise_average_surface(rh_gifti_file, self.rh_annot_file)

        # drop first entry (corresponds to 0)
        data_lh = data_lh[1:]
        data_rh = data_rh[1:]

        if self.parc == 'schaefer':
            self.micro = np.hstack((data_lh, data_rh)).astype(float)
        elif self.parc == 'glasser':
            self.micro = np.hstack((data_rh, data_lh)).astype(float)


    def load_tau(self, return_log=False):
        self._get_parc_data(parc=self.parc)

        # download data
        remote_path = 'https://github.com/rdgao/field-echos/raw/master/data'
        file = 'df_human.csv'
        if os.path.exists(os.path.join(self.outdir, file)) == False:
            wget.download(os.path.join(remote_path, file), self.outdir)

        df_human = pd.read_csv(os.path.join(self.outdir, file), index_col=0)
        electrode_coords = df_human.loc[:, ['x', 'y', 'z']]

        D = pairwise_distances(electrode_coords, self.centroids, metric='euclidean')
        nearest_region = np.argmin(D, axis=1)

        mean_tau = pd.DataFrame(index=self.centroids.index, columns=['tau', 'log_tau'])

        for i in np.arange(self.n_parcels):
            if np.any(nearest_region == i):
                mean_tau.iloc[i, 0] = df_human.loc[nearest_region == i, 'tau'].mean()
                mean_tau.iloc[i, 1] = df_human.loc[nearest_region == i, 'log_tau'].mean()

        mean_tau['tau'] = mean_tau['tau'].astype(float)
        mean_tau['log_tau'] = mean_tau['log_tau'].astype(float)

        if return_log:
            self.tau = mean_tau['log_tau'].values
        else:
            self.tau = mean_tau['tau'].values


    def load_sa_axis(self, out_dir='~/research_projects/connectome_loader/data/S-A_ArchetypalAxis'):
        self._get_parc_data(parc=self.parc, annot='fsaverage5')

        out_dir = os.path.expanduser(out_dir)
        remote_path = 'https://github.com/PennLINC/S-A_ArchetypalAxis.git'
        if os.path.exists(out_dir) == False:
            Repo.clone_from(remote_path, out_dir)

        files = ['SensorimotorAssociation_Axis_LH.fsaverage5.func.gii',
                 'SensorimotorAssociation_Axis_RH.fsaverage5.func.gii']

        lh_gifti_file = os.path.join(out_dir, 'FSaverage5', files[0])
        rh_gifti_file = os.path.join(out_dir, 'FSaverage5', files[1])

        # get average values over parcels
        data_lh = get_parcelwise_average_surface(lh_gifti_file, self.lh_annot_file)
        data_rh = get_parcelwise_average_surface(rh_gifti_file, self.rh_annot_file)

        # drop first entry (corresponds to 0)
        data_lh = data_lh[1:]
        data_rh = data_rh[1:]

        if self.parc == 'schaefer':
            self.sa_axis = np.hstack((data_lh, data_rh)).astype(float)
        elif self.parc == 'glasser':
            self.sa_axis = np.hstack((data_rh, data_lh)).astype(float)
