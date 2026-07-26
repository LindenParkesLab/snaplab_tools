"""Loaders for published cortical maps, resampled to a common parcellation.

:class:`BrainMapLoader` fetches four widely used maps and averages each within the parcels of a
Schaefer parcellation, so they can be correlated against your own parcellated data:

=================  ==========================================================================
Map                Source
=================  ==========================================================================
``cyto``           BigBrain histological gradient G2 (BigBrainWarp)
``micro``          BigBrain microstructural profile covariance gradient G1 (BigBrainWarp)
``tau``            Intrinsic timescales from human ECoG (Gao et al. 2020, *eLife*)
``sa_axis``        Sensorimotor-Association archetypal axis (Sydnor et al. 2021, *Neuron*)
=================  ==========================================================================

Each loader stores its result on the instance rather than returning it, so the usage pattern is
call-then-read::

    loader = BrainMapLoader(n_parcels=400, order=7)
    loader.load_sa_axis()
    sa = loader.sa_axis          # (400,) ndarray

The tau and S-A maps are downloaded on first use. The two BigBrain maps are *not* -- they must
already be present under ``research_data/BBW_BigData``, pre-downloaded from BigBrainWarp.

Please cite the original sources for any map you use.
"""
import os, wget
import numpy as np
import pandas as pd
from git.repo.base import Repo
import nibabel as nib
from sklearn.metrics import pairwise_distances
from snaplab_tools.utils import load_schaefer_parc, get_parcelwise_average_surface

__all__ = ['BrainMapLoader']


class BrainMapLoader:
    """Load published cortical maps averaged within a Schaefer parcellation.

    Construction is cheap -- it only records settings and creates the output directory. The
    actual fetching happens in the ``load_*`` methods, each of which sets an attribute of the
    same name (minus the ``load_`` prefix).

    Parameters
    ----------
    research_data : str
        Root directory holding pre-downloaded source data and the download cache. Must contain
        ``BBW_BigData`` (from BigBrainWarp) if you intend to call :meth:`load_cyto` or
        :meth:`load_micro`.
    parc : {'schaefer', 'glasser'}
        Parcellation family. Only 'schaefer' is actually fetched; 'glasser' changes the
        hemisphere concatenation order (right-then-left) for maps built elsewhere.
    n_parcels : int
        Schaefer resolution (100, 200, ..., 1000).
    order : int
        Yeo network order, 7 or 17.

    Attributes
    ----------
    cyto : (n_parcels,) ndarray
        BigBrain histological gradient. Set by :meth:`load_cyto`.
    micro : (n_parcels,) ndarray
        BigBrain microstructural gradient. Set by :meth:`load_micro`.
    tau : (n_parcels,) ndarray
        ECoG intrinsic timescales, NaN in parcels with no nearby electrode. Set by
        :meth:`load_tau`.
    sa_axis : (n_parcels,) ndarray
        Sensorimotor-Association axis. Set by :meth:`load_sa_axis`.
    centroids : pandas.DataFrame
        Parcel centroid coordinates, indexed by ROI name. Set by the first ``load_*`` call.
    """

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
                out_dir='~/schaefer_parc')
            self.centroids.set_index('ROI Name', inplace=True)


    def load_cyto(self):
        """Load the BigBrain histological gradient (G2) and store it on ``self.cyto``.

        Reads the fsaverage 164k ``Hist_G2`` shape files from
        ``research_data/BBW_BigData/spaces/tpl-fsaverage`` -- these are not downloaded, and must
        already be present from BigBrainWarp. The medial wall (label 0) is dropped before the
        hemispheres are concatenated.

        Returns
        -------
        None
            Sets ``self.cyto``, a (n_parcels,) float array.
        """
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
        """Load the BigBrain microstructural profile covariance gradient (G1) into ``self.micro``.

        Reads the fsaverage 164k ``Micro_G1`` curvature files from
        ``research_data/BBW_BigData/spaces/tpl-fsaverage`` -- these are not downloaded, and must
        already be present from BigBrainWarp. The medial wall (label 0) is dropped before the
        hemispheres are concatenated.

        Returns
        -------
        None
            Sets ``self.micro``, a (n_parcels,) float array.
        """
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
        """Load ECoG intrinsic timescales into ``self.tau``.

        Downloads ``df_human.csv`` from the Gao et al. field-echos repository on first call.
        Electrodes are assigned to parcels by nearest Euclidean distance between the electrode
        coordinates and the parcel centroids, then averaged within parcel.

        Because electrode coverage is sparse and clinically determined, **parcels with no nearest
        electrode are left as NaN** -- expect substantial missingness, and use NaN-tolerant
        routines downstream (:func:`snaplab_tools.stats.partial_pearsonr` and
        :func:`snaplab_tools.nulls.generate_surrogates` both handle it).

        Parameters
        ----------
        return_log : bool
            Store log-transformed timescales instead of raw ones. Timescales are heavily
            right-skewed, so the log is usually the better choice for correlation.

        Returns
        -------
        None
            Sets ``self.tau``, a (n_parcels,) float array with NaNs where coverage is absent.
        """
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


    def load_sa_axis(self, out_dir='~/brain_maps/S-A_ArchetypalAxis'):
        """Load the Sensorimotor-Association archetypal axis into ``self.sa_axis``.

        Clones the PennLINC/S-A_ArchetypalAxis repository into `out_dir` on first call (via
        GitPython) and averages the fsaverage5 axis maps within parcels. Low values are
        sensorimotor, high values association cortex.

        Note this loader uses the fsaverage5 annotation rather than fsaverage, so it triggers a
        separate parcellation download from the other three maps.

        Parameters
        ----------
        out_dir : str
            Clone destination; '~' is expanded. Skipped if the directory already exists.

        Returns
        -------
        None
            Sets ``self.sa_axis``, a (n_parcels,) float array.
        """
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
