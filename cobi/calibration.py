"""
Calibration analysis module for LAT-SAT cross-spectrum studies.

This module provides classes for fitting calibration parameters (angles and amplitudes)
using cross-correlations between Large Aperture Telescope (LAT) and Small Aperture
Telescope (SAT) data from CMB observations.
"""

import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Dict, List, Union
from abc import ABC, abstractmethod
from cobi.simulation import CMB
from cobi.utils import Logger
import os
import healpy as hp
import pickle as pl
import emcee
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import minimize
from cobi.spectra import SpectraCross

def _oas_shrinkage(S: np.ndarray, n: int) -> np.ndarray:
    """
    Oracle Approximating Shrinkage estimator (Chen et al. 2010).

    Finds the optimal convex combination of the sample covariance S and
    a scaled identity that minimises the expected Frobenius loss.  Well-
    conditioned for any n > 1, even when n << p.
    """
    p = S.shape[0]
    tr_S  = np.trace(S)
    tr_S2 = np.trace(S @ S)
    mu       = tr_S / p
    rho_num  = (1.0 - 2.0 / p) * tr_S2 + tr_S ** 2
    rho_den  = (n + 1.0 - 2.0 / p) * (tr_S2 - tr_S ** 2 / p)
    rho      = 1.0 if rho_den == 0 else min(1.0, rho_num / rho_den)
    return (1.0 - rho) * S + rho * mu * np.eye(p)


def _format_alpha_label(tag: str) -> str:
    """
    Format map tag like 'LAT_93-1' -> r'\\alpha_{LAT,93−1}'.
    
    Args:
        tag: Map tag string to format
        
    Returns:
        Formatted alpha label for LaTeX rendering
    """
    parts = tag.replace('_', ',').replace('-', '−')
    return rf"\alpha_{{{parts}}}"


# ============================================================
# =============== SHARED SUPERCLASS ==========================
# ============================================================

class BaseSat4LatCross(ABC):
    """
    Base class for LAT-SAT cross-spectrum calibration analyses.
    
    Provides data loading, beam handling, and plotting logic.
    Subclasses must define:
        - self.__pnames__ / self.__plabels__
        - self.theory(theta)
        - self.lnprior(theta)
        
    Attributes:
        spec_cross: Cross-spectrum calculation object
        sat_err: Satellite calibration error
        sat_lrange: Satellite ell range for fitting
        lat_lrange: LAT ell range for fitting
        fit_per_split: Whether to fit per split or per frequency
        spectra_selection: Which spectra to include ('all', 'auto_only', 'cross_only')
        libdir: Directory for caching results
        binner: Binning information object
        Lmax: Maximum multipole
        maptags: List of map tags
        freq_groups: Dictionary mapping frequency bases to indices
        freq_bases: List of frequency base names
        mean_spec: Mean spectra from simulations
        std_spec: Standard deviation of spectra
        beam_arr: Beam transfer function array
    """

    def __init__(self,
                 spec_cross: 'SpectraCross',
                 sat_err: float,
                 sat_lrange: Tuple[Optional[int], Optional[int]] = (None, None),
                 lat_lrange: Tuple[Optional[int], Optional[int]] = (None, None),
                 fit_per_split: bool = True,
                 spectra_selection: str = 'all',
                 libdir_suffix: str = 'BaseSat4LatCross',
                 num_sims: int = 100,
                 verbose: bool = False,
                 cov_mode: str = 'diagonal') -> None:
        """
        cov_mode controls how the data covariance is estimated:
            'diagonal'   – per-bin variance from sims (default, always safe).
            'block_diag' – one (n_bins × n_bins) block per (i,j) map pair,
                           each block Hartlap-corrected.  Needs N_sims >> n_bins
                           (not N_sims >> n_data).  Recommended when you want
                           ell-ell correlations without full-matrix instability.
            'shrinkage'  – full (n_data × n_data) covariance shrunk toward the
                           identity via OAS.  Well-conditioned at any N_sims.
            'full'       – raw sample covariance + Hartlap correction.
                           Requires N_sims > n_data + 2; raises ValueError
                           otherwise.
        """
        if cov_mode not in ('diagonal', 'block_diag', 'shrinkage', 'full'):
            raise ValueError(f"cov_mode must be one of 'diagonal', 'block_diag', 'shrinkage', 'full'")

        self.logger = Logger(self.__class__.__name__, verbose=verbose)
        self.logger.log(f"Initializing {self.__class__.__name__}...", level="info")
        self.spec_cross = spec_cross
        self.libdir = os.path.join(spec_cross.libdir, libdir_suffix)
        os.makedirs(self.libdir, exist_ok=True)

        self.sat_err = sat_err
        self.sat_lrange = sat_lrange
        self.lat_lrange = lat_lrange
        self.fit_per_split = fit_per_split
        self.spectra_selection = spectra_selection
        self.binner = spec_cross.binInfo
        self.Lmax = spec_cross.lmax
        self.cov_mode = cov_mode
        self.use_full_cov = (cov_mode != 'diagonal')
        self._num_sims = num_sims

        # ---- Build map tags and frequency groups ----
        self.maptags: List[str] = spec_cross.maptags.copy()
        self.freq_groups: Dict[str, List[int]] = {}
        for i, tag in enumerate(self.maptags):
            base = tag.rsplit('-', 1)[0]
            self.freq_groups.setdefault(base, []).append(i)
        self.freq_bases: List[str] = list(self.freq_groups.keys())

        # ---- Masks, spectra, beams ----
        self.__likelihood_mask__: np.ndarray = self._build_likelihood_mask()
        self.mean_spec: np.ndarray
        self.std_spec: np.ndarray
        self.mean_spec, self.std_spec = self._calc_mean_std(num_sims=num_sims)
        self.beam_arr: np.ndarray = self._get_beam_arr()

        if self.use_full_cov:
            self.cov, self.cov_inv, self.logdet_cov = self._calc_covariance(num_sims=num_sims)

        # Initialize parameter names and labels (to be set by subclasses)
        self.__pnames__: List[str] = []
        self.__plabels__: List[str] = []

        self.logger.log(f"Initialized {self.__class__.__name__}.", level="info")

    # -------------------------------------------------------------------------
    # Abstract methods - must be implemented by subclasses
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def theory(self, theta: np.ndarray) -> np.ndarray:
        """
        Calculate theoretical model spectrum for given parameters.
        
        Args:
            theta: Parameter vector
            
        Returns:
            Model spectrum array with shape matching data
        """
        pass
    
    @abstractmethod
    def lnprior(self, theta: np.ndarray) -> float:
        """
        Calculate log prior probability for given parameters.
        
        Args:
            theta: Parameter vector
            
        Returns:
            Log prior probability
        """
        pass

    # -------------------------------------------------------------------------
    # Shared internals
    # -------------------------------------------------------------------------

    def _build_likelihood_mask(self) -> np.ndarray:
        """
        Build boolean mask for likelihood evaluation based on ell ranges.
        
        Returns:
            Boolean mask array with shape (n_tags, n_tags, n_bins)
        """
        ells = self.binner.get_effective_ells()
        sat_lmin, sat_lmax = self.sat_lrange
        lat_lmin, lat_lmax = self.lat_lrange
        sat_mask = (ells >= (sat_lmin or 0)) & (ells <= (sat_lmax or np.inf))
        lat_mask = (ells >= (lat_lmin or 0)) & (ells <= (lat_lmax or np.inf))
        cross_mask = sat_mask & lat_mask

        n_tags = len(self.maptags)
        n_bins = len(ells)
        mask = np.zeros((n_tags, n_tags, n_bins), dtype=bool)
        
        is_lat = np.array([tag.startswith('LAT') for tag in self.maptags])
        
        for i in range(n_tags):
            for j in range(n_tags):
                # Determine which ell mask to use
                if is_lat[i] and is_lat[j]:
                    # LAT x LAT
                    ell_mask = lat_mask
                elif not is_lat[i] and not is_lat[j]:
                    # SAT x SAT
                    ell_mask = sat_mask
                else:
                    # LAT x SAT cross
                    ell_mask = cross_mask
            
                # Apply spectra selection
                if self.spectra_selection == 'auto_only':
                    if i == j:  # Auto-spectrum
                        mask[i, j] = ell_mask
                elif self.spectra_selection == 'cross_only':
                    if i != j:  # Cross-spectrum
                        mask[i, j] = ell_mask
                else:  # 'all'
                    mask[i, j] = ell_mask

        return mask
    
    def __all_spectra__(self, num_sims: int = 100) -> np.ndarray:
        """
        Get all spectra data matrix for current ell ranges.
        
        Returns:
            Data matrix array with shape (n_tags, n_tags, n_bins)
        """
        all_specs = [
            self.spec_cross.data_matrix(i, which='EB',
                                        sat_lrange=self.sat_lrange,
                                        lat_lrange=self.lat_lrange,
                                        avg_splits=False)
            for i in tqdm(range(num_sims))
        ]
        all_specs = np.array(all_specs)
        return all_specs
        

    def _calc_mean_std(self, num_sims: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and standard deviation of spectra over simulations.
        
        Args:
            num_sims: Number of simulations to use
            
        Returns:
            Tuple of (mean_spectrum, std_spectrum) arrays
        """
        fname = os.path.join(self.libdir, f'mean_std_spec_{num_sims}_sims.pkl')
        if os.path.exists(fname):
            self.logger.log("Loading cached mean/std spectra...", level="info")
            with open(fname, 'rb') as f:
                return pl.load(f)
        self.logger.log(f"Computing mean/std over {num_sims} simulations...", level="info")
        all_specs = self.__all_spectra__(num_sims=num_sims)
        mean_spec, std_spec = np.mean(all_specs, axis=0), np.std(all_specs, axis=0)
        with open(fname, 'wb') as f:
            pl.dump((mean_spec, std_spec), f)
        return mean_spec, std_spec

    def _get_beam_arr(self) -> np.ndarray:
        """
        Calculate beam transfer function array for all map tag combinations.
        
        Returns:
            Beam array with shape (n_tags, n_tags, n_bins)
        """
        n_tags, n_bins = len(self.maptags), self.binner.get_n_bands()
        beam = np.ones((n_tags, n_tags, n_bins))
        fwhm_dict = {f'LAT_{f}': fw for f, fw in zip(self.spec_cross.lat.freqs, self.spec_cross.lat.fwhm)}
        
        # Only add SAT beams if not in LAT-only mode
        if hasattr(self.spec_cross, 'lat_only') and not self.spec_cross.lat_only:
            fwhm_dict.update({f'SAT_{f}': fw for f, fw in zip(self.spec_cross.sat.freqs, self.spec_cross.sat.fwhm)})
        
        for i, ti in enumerate(self.maptags):
            for j, tj in enumerate(self.maptags):
                bi, bj = ti.rsplit('-', 1)[0], tj.rsplit('-', 1)[0]
                b_i = hp.gauss_beam(np.radians(fwhm_dict[bi] / 60.), lmax=self.Lmax) ** 2
                b_j = hp.gauss_beam(np.radians(fwhm_dict[bj] / 60.), lmax=self.Lmax) ** 2
                beam[i, j] = self.binner.bin_cell(np.sqrt(b_i * b_j))
        return beam

    def _calc_covariance(self, num_sims: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate and cache the data covariance under the likelihood mask.

        The mode is controlled by ``self.cov_mode``:

        * ``'block_diag'``: independent (n_bins × n_bins) blocks per (i,j) map
          pair, each Hartlap-corrected.  Only needs N_sims >> n_bins_per_pair.
        * ``'shrinkage'``: full covariance regularised by OAS shrinkage toward
          the scaled identity.  Well-conditioned for any N_sims.
        * ``'full'``: raw sample covariance + Hartlap.  Raises ValueError when
          N_sims ≤ n_data + 2.

        Returns
        -------
        (cov, cov_inv, logdet_cov)
        """
        fname = os.path.join(
            self.libdir,
            f'cov_{self.cov_mode}_{num_sims}sims'
            f'_lmin{self.lat_lrange[0]}_lmax{self.lat_lrange[1]}'
            f'_slmin{self.sat_lrange[0]}_slmax{self.sat_lrange[1]}'
            f'_sel{self.spectra_selection}.pkl',
        )
        if os.path.exists(fname):
            self.logger.log(f"Loading cached covariance ({self.cov_mode})...", level="info")
            with open(fname, 'rb') as f:
                return pl.load(f)

        self.logger.log(
            f"Computing covariance (mode='{self.cov_mode}') over {num_sims} sims...", level="info"
        )
        all_specs = self.__all_spectra__(num_sims=num_sims)   # (N, n_tags, n_tags, n_bins)
        all_specs_bc = all_specs / self.beam_arr[None]        # beam-correct
        mask = self.__likelihood_mask__
        vecs = all_specs_bc[:, mask]                          # (N, n_data)
        N, n_data = vecs.shape

        self.logger.log(f"Covariance: N_sims={N}, n_data={n_data}", level="info")

        if self.cov_mode == 'full':
            if N <= n_data + 2:
                raise ValueError(
                    f"'full' covariance requires N_sims > n_data+2 = {n_data + 2}, "
                    f"but only {N} sims available.  Use cov_mode='block_diag' or 'shrinkage'."
                )
            S = np.cov(vecs.T)
            hartlap = (N - n_data - 2) / (N - 1)
            self.logger.log(f"Hartlap factor: {hartlap:.4f}", level="info")
            cov     = S
            cov_inv = hartlap * np.linalg.inv(S)
            _, logdet = np.linalg.slogdet(S)

        elif self.cov_mode == 'shrinkage':
            S   = np.cov(vecs.T)
            cov = _oas_shrinkage(S, N)
            self.logger.log(f"OAS shrinkage applied (N={N}, n_data={n_data})", level="info")
            cov_inv   = np.linalg.inv(cov)
            _, logdet = np.linalg.slogdet(cov)

        elif self.cov_mode == 'block_diag':
            # One Hartlap-corrected block per (i, j) map pair.
            # Elements from the same pair are contiguous in the C-order
            # flattening of the 3-D mask, so we can walk through vecs in order.
            n_tags = mask.shape[0]
            cov     = np.zeros((n_data, n_data))
            cov_inv = np.zeros((n_data, n_data))
            logdet  = 0.0
            k = 0
            for i in range(n_tags):
                for j in range(n_tags):
                    n_pair = int(np.sum(mask[i, j, :]))
                    if n_pair == 0:
                        continue
                    pv = vecs[:, k: k + n_pair]           # (N, n_pair)
                    S  = np.cov(pv.T) if n_pair > 1 else np.array([[float(np.var(pv, ddof=1))]])
                    if N > n_pair + 2:
                        hartlap = (N - n_pair - 2) / (N - 1)
                    else:
                        hartlap = 1.0
                        self.logger.log(
                            f"Block ({i},{j}): N ({N}) ≤ n_pair+2 ({n_pair+2}), Hartlap skipped.",
                            level="warning",
                        )
                    cov[k: k + n_pair, k: k + n_pair]     = S
                    cov_inv[k: k + n_pair, k: k + n_pair] = hartlap * np.linalg.inv(S)
                    _, ld = np.linalg.slogdet(S)
                    logdet += float(ld)
                    k += n_pair
        else:
            raise ValueError(f"Unknown cov_mode '{self.cov_mode}'")

        with open(fname, 'wb') as f:
            pl.dump((cov, cov_inv, logdet), f)
        return cov, cov_inv, logdet

    # -------------------------------------------------------------------------
    # Shared Likelihood & MCMC
    # -------------------------------------------------------------------------

    def chisq(self, theta: np.ndarray) -> float:
        """
        Calculate chi-squared statistic for given parameters.

        When use_full_cov=True uses the full inverse covariance matrix;
        otherwise uses diagonal (per-bin) variances.
        """
        model = self.theory(theta)
        data = self.mean_spec / self.beam_arr
        if self.use_full_cov:
            delta = (data - model)[self.__likelihood_mask__]
            return float(delta @ self.cov_inv @ delta)
        else:
            err = self.std_spec / self.beam_arr
            err = np.where(err == 0, np.inf, err)
            return np.sum(((data - model) / err) ** 2 * self.__likelihood_mask__)

    def ln_likelihood(self, theta: np.ndarray) -> float:
        """Calculate log likelihood for given parameters."""
        chisq_val = self.chisq(theta)
        if self.use_full_cov:
            n_data = int(np.sum(self.__likelihood_mask__))
            return -0.5 * (chisq_val + self.logdet_cov + n_data * np.log(2 * np.pi))
        return -0.5 * chisq_val

    def ln_prob(self, theta: np.ndarray) -> float:
        """Calculate log posterior probability for given parameters."""
        lp = self.lnprior(theta)
        return -np.inf if not np.isfinite(lp) else lp + self.ln_likelihood(theta)

    def run_mcmc(self, nwalkers: int = 32, nsamples: int = 2000, rerun: bool = False,
                 fiducial_params: Optional[np.ndarray] = None, fname: Optional[str] = None,
                 burnin: int = 200, thin: int = 20) -> np.ndarray:
        """
        Run (or extend/resume) MCMC using an emcee HDF5 backend.

        The backend file grows as you call run_mcmc with larger nsamples —
        no samples are ever discarded from disk.  burnin and thin only affect
        what is *returned*; you can change them freely without re-running.

        Args:
            nwalkers:  Number of walkers.
            nsamples:  Target total number of steps stored in the backend.
                       If the backend already has >= nsamples steps, the chain
                       is returned immediately.  If it has fewer, only the
                       remaining steps are run (extending the existing chain).
            rerun:     If True, wipe the backend and start from scratch.
            fiducial_params: Starting point for walkers (required on first run;
                       ignored when resuming an existing chain).
            fname:     Override the auto-generated backend path (.h5 file).
            burnin:    Steps to discard from the front when returning the chain.
            thin:      Thinning factor applied when returning the chain.
        """
        ndim = len(self.__pnames__)

        if fname is None:
            cov_tag = f"_{self.cov_mode}" if self.use_full_cov else ""
            fname = os.path.join(
                self.libdir,
                f"chain_{nwalkers}w_fitper{self.fit_per_split}_ndim{ndim}"
                f"_lmin{self.lat_lrange[0]}_lmax{self.lat_lrange[1]}"
                f"_slmin{self.sat_lrange[0]}_slmax{self.sat_lrange[1]}{cov_tag}.h5",
            )

        backend = emcee.backends.HDFBackend(fname)

        if rerun:
            backend.reset(nwalkers, ndim)

        already_run = backend.initialized and backend.iteration > 0

        if already_run and backend.iteration >= nsamples:
            self.logger.log(
                f"Returning cached chain ({backend.iteration} steps, burnin={burnin}, thin={thin})",
                level="info",
            )
            return backend.get_chain(discard=burnin, thin=thin, flat=True)

        if already_run:
            initial_state = None
            remaining = nsamples - backend.iteration
            self.logger.log(
                f"Extending chain: {backend.iteration} → {nsamples} steps", level="info"
            )
        else:
            if fiducial_params is not None:
                if len(fiducial_params) != ndim:
                    raise ValueError(
                        f"fiducial_params length {len(fiducial_params)} does not match ndim {ndim}"
                    )
            else:
                fiducial_params = np.zeros(ndim)
            initial_state = fiducial_params + 1e-3 * np.random.randn(nwalkers, ndim)
            remaining = nsamples

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, backend=backend)
        sampler.run_mcmc(initial_state, remaining, progress=True)
        return sampler.get_chain(discard=burnin, thin=thin, flat=True)

    def getdist_samples(self, nwalkers: int, nsamples: int, rerun: bool = False,
                        fiducial_params: Optional[np.ndarray] = None,
                        label: Optional[str] = None,
                        burnin: int = 200, thin: int = 20) -> MCSamples:
        samples = self.run_mcmc(nwalkers, nsamples, rerun=rerun,
                                fiducial_params=fiducial_params,
                                burnin=burnin, thin=thin)
        return MCSamples(samples=samples, names=self.__pnames__, labels=self.__plabels__, label=label)

    # -------------------------------------------------------------------------
    # Shared Plotting
    # -------------------------------------------------------------------------

    def plot_posteriors(self, nwalkers: int, nsamples: int, rerun: bool = False, label: Optional[str] = None,
                        fiducial_params: Optional[np.ndarray] = None, burnin: int = 200, thin: int = 20):
        samples = self.getdist_samples(nwalkers, nsamples, rerun=rerun, label=label,
                                       fiducial_params=fiducial_params, burnin=burnin, thin=thin)
        g = plots.get_subplot_plotter()
        g.triangle_plot([samples], filled=True, title_limit=1)
        return g

    def plot_spectra_matrix(self, theta: Optional[np.ndarray] = None, save_path: Optional[str] = None, 
                            average_split: bool = False) -> None:
        """
        Plots the mean data spectra with std deviation and optionally a theory curve
        in a matrix layout corresponding to all cross-correlations.

        Args:
            theta: Parameter vector (alphas, beta, A_EB, etc.) for theory overlay.
            save_path: If given, saves the figure instead of showing.
            average_split: If True, average spectra across splits per frequency.
        """
        import matplotlib.pyplot as plt
        ells = self.binner.get_effective_ells()
        
        # Build ell masks based on sat_lrange and lat_lrange
        n_bins = len(ells)
        
        # SAT ell mask
        sat_ell_mask = np.ones(n_bins, dtype=bool)
        if self.sat_lrange[0] is not None:
            sat_ell_mask &= (ells >= self.sat_lrange[0])
        if self.sat_lrange[1] is not None:
            sat_ell_mask &= (ells <= self.sat_lrange[1])
        
        # LAT ell mask
        lat_ell_mask = np.ones(n_bins, dtype=bool)
        if self.lat_lrange[0] is not None:
            lat_ell_mask &= (ells >= self.lat_lrange[0])
        if self.lat_lrange[1] is not None:
            lat_ell_mask &= (ells <= self.lat_lrange[1])
        
        # Cross-correlation mask (intersection)
        cross_ell_mask = sat_ell_mask & lat_ell_mask

        # ----------------------------------------------------------
        # Optionally average over splits for display only
        # ----------------------------------------------------------
        if average_split:
            freq_bases = list(self.freq_groups.keys())
            n_freqs = len(freq_bases)
            n_bins = self.binner.get_n_bands()

            data_spec = self.mean_spec / self.beam_arr
            error_spec = self.std_spec / self.beam_arr

            data_avg = np.zeros((n_freqs, n_freqs, n_bins))
            err_avg = np.zeros_like(data_avg)

            for i, base_i in enumerate(freq_bases):
                idx_i = self.freq_groups[base_i]
                for j, base_j in enumerate(freq_bases):
                    idx_j = self.freq_groups[base_j]
                    vals = [data_spec[ii, jj, :] for ii in idx_i for jj in idx_j]
                    errs = [error_spec[ii, jj, :] for ii in idx_i for jj in idx_j]
                    data_avg[i, j, :] = np.nanmean(vals, axis=0)
                    err_avg[i, j, :] = np.nanmean(errs, axis=0)

            maptags = freq_bases
            data_spec, error_spec = data_avg, err_avg
            if theta is not None:
                theory_spec = self.theory(theta)
                th_avg = np.zeros_like(data_avg)
                for i, base_i in enumerate(freq_bases):
                    idx_i = self.freq_groups[base_i]
                    for j, base_j in enumerate(freq_bases):
                        idx_j = self.freq_groups[base_j]
                        vals = [theory_spec[ii, jj, :] for ii in idx_i for jj in idx_j]
                        th_avg[i, j, :] = np.nanmean(vals, axis=0)
                theory_spec = th_avg
            else:
                theory_spec = None
        else:
            maptags = self.maptags
            data_spec = self.mean_spec / self.beam_arr
            error_spec = self.std_spec / self.beam_arr
            theory_spec = self.theory(theta) if theta is not None else None

        # ----------------------------------------------------------
        # Set up figure
        # ----------------------------------------------------------
        n_tags = len(maptags)
        fig, axes = plt.subplots(
            n_tags, n_tags,
            figsize=(n_tags * 3.0, n_tags * 3.0),
            sharex=True, sharey='row'
        )

        # ----------------------------------------------------------
        # Main plotting loop
        # ----------------------------------------------------------
        for i in range(n_tags):
            for j in range(n_tags):
                ax = axes[i, j]
                is_diagonal = (i == j)

                # Respect spectra_selection
                plot_this = False
                if self.spectra_selection == 'all':
                    plot_this = True
                elif self.spectra_selection == 'auto_only' and is_diagonal:
                    plot_this = True
                elif self.spectra_selection == 'cross_only' and not is_diagonal:
                    plot_this = True

                if not plot_this:
                    ax.set_visible(False)
                    continue

                # Determine which ell mask to use for this subplot
                is_i_sat = maptags[i].startswith('SAT')
                is_j_sat = maptags[j].startswith('SAT')
                
                if is_i_sat and is_j_sat:
                    # SAT-SAT correlation: use SAT range
                    subplot_mask = sat_ell_mask
                elif not is_i_sat and not is_j_sat:
                    # LAT-LAT correlation: use LAT range
                    subplot_mask = lat_ell_mask
                else:
                    # SAT-LAT cross-correlation: use intersection
                    subplot_mask = cross_ell_mask
                
                # Filter data for this subplot
                ells_plot = ells[subplot_mask]
                data_plot = data_spec[i, j, subplot_mask]
                error_plot = error_spec[i, j, subplot_mask]
                
                # Plot data ±1σ
                ax.errorbar(
                    ells_plot, data_plot, yerr=error_plot,
                    fmt='.', capsize=2, color='black', markersize=4, alpha=0.5,
                )

                # Overlay theory if provided (plot over full range)
                if theory_spec is not None:
                    ax.loglog(ells, theory_spec[i, j], color='red')

                # Shade region used in likelihood
                if average_split:
                    mask_1d = self.__likelihood_mask__[0, 0, :][subplot_mask]
                else:
                    mask_1d = self.__likelihood_mask__[i, j, :][subplot_mask]
                used_ells = ells_plot[mask_1d]
                # if len(used_ells) > 0:
                #     ax.axvspan(used_ells.min(), used_ells.max(), color='green', alpha=0.1, zorder=-1)

                # Formatting
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.axhline(0, color='grey', linestyle=':', linewidth=0.75)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_ylim(1e-7, 1e-4)
                ax.set_xlim(20, 3000)

                # Axes labels
                if i == n_tags - 1:
                    ax.set_xlabel(r'$\ell$')
                if j == 0:
                    ax.set_ylabel(r'$C_\ell^{EB}$')
                if i == 0:
                    ax.set_title(f"$\\rm {maptags[j].replace('_', '-')}$", fontsize=10)

        # Right-side frequency labels
        for i, tag in enumerate(maptags):
            axes[i, n_tags - 1].text(
                1.05, 0.5, f"$\\rm {tag.replace('_', '-')}$",
                transform=axes[i, n_tags - 1].transAxes,
                va='center', ha='left', fontsize=10
            )

        # Legend
        handles, labels = [], []
        for ax in axes.flatten():
            if ax.get_visible():
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
                break
        if handles:
            fig.legend(handles, labels, loc='upper right')

        plt.tight_layout(rect=(0.03, 0.03, 0.92, 0.95))

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.log(f"Figure saved to {save_path}", level="info")
            plt.close()
        else:
            #plt.show()
            return None
    
    def mle(self,nwalkers,nsamples,op=np.median):
        s = self.run_mcmc(nwalkers=nwalkers, nsamples=nsamples)
        theta = op(s,axis=0)
        chi2_min = self.chisq(theta)
        return chi2_min

# ============================================================
# =============== SUBCLASS 1: BETA FIT =======================
# ============================================================

class Sat4LatCross(BaseSat4LatCross):
    """
    Calibration model fitting for birefringence angle β.
    
    This class implements a calibration analysis that fits calibration angles (alphas)
    and the cosmic birefringence angle β using EB cross-spectra between LAT and SAT.
    
    Attributes:
        cl_len: Lensed CMB power spectra dictionary
    """
    
    def __init__(self,
                 spec_cross: 'SpectraCross',
                 sat_err: float,
                 beta_fid: float,
                 sat_lrange: Tuple[Optional[int], Optional[int]] = (None, None),
                 lat_lrange: Tuple[Optional[int], Optional[int]] = (None, None),
                 fit_per_split: bool = True,
                 spectra_selection: str = 'all',
                 num_sims: int = 100,
                 verbose: bool = False,
                 cov_mode: str = 'diagonal') -> None:
        """
        Initialize calibration analysis for birefringence angle fitting.

        Args:
            spec_cross: Cross-spectrum calculation object
            sat_err: Satellite calibration error
            beta_fid: Fiducial birefringence angle for theory calculation
            sat_lrange: Satellite ell range for fitting (lmin, lmax)
            lat_lrange: LAT ell range for fitting (lmin, lmax)
            fit_per_split: Whether to fit per split or per frequency
            spectra_selection: Which spectra to include ('all', 'auto_only', 'cross_only')
            num_sims: Number of simulations for mean/std calculation
            verbose: Whether to enable verbose output
            cov_mode: Covariance mode ('diagonal', 'block_diag', 'shrinkage', 'full')
        """
        super().__init__(spec_cross, sat_err, sat_lrange, lat_lrange,
                         fit_per_split, spectra_selection, 'CalibrationAnalysis', num_sims, verbose,
                         cov_mode=cov_mode)
        self.cl_len = CMB(spec_cross.libdir, spec_cross.nside, beta=beta_fid).get_lensed_spectra(dl=False)
        if fit_per_split:
            self.__pnames__  = [f"a_{t}" for t in self.maptags] + ['beta']
            self.__plabels__ = [_format_alpha_label(t) for t in self.maptags] + [r"\beta"]
        else:
            self.__pnames__  = [f"a_{f}" for f in self.freq_bases] + ['beta']
            self.__plabels__ = [_format_alpha_label(f) for f in self.freq_bases] + [r"\beta"]

    def theory(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute theoretical EB spectra between all map pairs
        using the exact birefringence + miscalibration formula:
            C_ell^{E_iB_j} = cos(2α_i+2β)sin(2α_j+2β)C_EE
                            - sin(2α_i+2β)cos(2α_j+2β)C_BB
        Optionally adds a dust foreground power law: A_dust * (ell/80)^alpha_dust
        """
        # Unpack angles
        # α_i = per-map miscalibration
        # β   = global cosmic birefringence
        if self.fit_per_split:
            alphas = theta[:-1]
        else:
            base_a = {b: theta[i] for i, b in enumerate(self.freq_bases)}
            alphas = np.array([base_a[t.rsplit('-', 1)[0]] for t in self.maptags])
        beta = theta[-1]

        # Get Cℓ^EE and Cℓ^BB
        cl_ee = self.cl_len["ee"][:self.Lmax+1]
        cl_bb = self.cl_len["bb"][:self.Lmax+1]

        model = np.zeros_like(self.mean_spec)

        for i in range(len(self.maptags)):          # E_i
            for j in range(len(self.maptags)):      # B_j
                # angles in radians
                ai = np.deg2rad(alphas[i])
                aj = np.deg2rad(alphas[j])
                b  = np.deg2rad(beta)

                term = (
                    np.cos(2*ai + 2*b) * np.sin(2*aj + 2*b) * cl_ee
                    - np.sin(2*ai + 2*b) * np.cos(2*aj + 2*b) * cl_bb
                )

                model[i, j] = self.binner.bin_cell(term)

        return model


    def lnprior(self, theta: np.ndarray) -> float:
        alphas, beta = theta[:-1], theta[-1]
        if np.any(np.abs(alphas) > 0.5) or not (-0.5 < beta < 0.5):
            return -np.inf
        sat_idx = [i for i, t in enumerate(self.maptags if self.fit_per_split else self.freq_bases) if t.startswith('SAT')]
        return float(-0.5 * np.sum(np.array(alphas)[sat_idx] ** 2 / self.sat_err**2 - np.log(self.sat_err * np.sqrt(2 * np.pi))))
    
    @property
    def dof(self) -> int:
        """
        Calculate degrees of freedom for chi-squared analysis.
        
        Returns:
            Number of data points minus number of fitted parameters
        """
        n_data_points = np.sum(self.__likelihood_mask__)
        n_parameters = len(self.__pnames__)
        return n_data_points - n_parameters
# ============================================================
# =============== SUBCLASS 2: AMPLITUDE FIT ==================
# ============================================================


# ============================================================
# ============ SUBCLASS 4: FG AMPLITUDE + BETA FIT ===========
# ============================================================

class Sat4LatCross_FGFit(BaseSat4LatCross):
    """
    Calibration model fitting for birefringence angle β with shared per-frequency
    foreground amplitude parameters.

    Fits calibration angles (alphas), the cosmic birefringence angle β, and
    per-frequency foreground scaling amplitudes (A_fg) using EB cross-spectra.
    Here frequency means only the band value (e.g. ``93``, ``145``), shared
    across LAT and SAT.
    Foreground EB templates are computed as full-sky power spectra from the
    saved dust Q/U maps and then binned with the same binner.

    Before initialisation the class verifies that LAT and SAT were simulated
    with the same dust model; a ``ValueError`` is raised otherwise.

    Attributes
    ----------
    cl_len : dict
        Lensed CMB power spectra dictionary.
    freq_values : List[str]
        Unique frequency values extracted from map tags (e.g. ``['93', '145']``).
    fg_templates_binned : np.ndarray
        Binned foreground EB templates, shape ``(n_freq_values, n_freq_values, n_bins)``.
    """

    def __init__(self,
                 spec_cross: 'SpectraCross',
                 sat_err: float,
                 beta_fid: float,
                 sat_lrange: Tuple[Optional[int], Optional[int]] = (None, None),
                 lat_lrange: Tuple[Optional[int], Optional[int]] = (None, None),
                 fit_per_split: bool = True,
                 spectra_selection: str = 'all',
                 num_sims: int = 100,
                 verbose: bool = False,
                 cov_mode: str = 'diagonal',
                 fg_model: str = 'geometric',
                 A_fg_max: float = 2.0) -> None:
        """
        Initialise calibration analysis with foreground amplitude fitting.

        Args:
            spec_cross: Cross-spectrum calculation object.
            sat_err: Satellite calibration angle prior width (degrees).
            beta_fid: Fiducial birefringence angle used to build the CMB template.
            sat_lrange: Satellite ell range for fitting ``(lmin, lmax)``.
            lat_lrange: LAT ell range for fitting ``(lmin, lmax)``.
            fit_per_split: Fit one alpha per split-map (True) or per frequency (False).
            spectra_selection: Which spectra to include: ``'all'``, ``'auto_only'``,
                or ``'cross_only'``.
            num_sims: Number of simulations used for mean/std estimation.
            verbose: Enable verbose logging.
            cov_mode: Covariance mode ('diagonal', 'block_diag', 'shrinkage', 'full').
            fg_model: Foreground amplitude parametrisation.
                ``'geometric'``: one amplitude per frequency; cross-frequency
                scaling via geometric mean ``sqrt(A_i * A_j)``.
                ``'per_pair'``: one independent amplitude per unique (freq_i, freq_j)
                pair — more robust, no factorizability assumption.
            A_fg_max: Hard upper bound (in absolute value) for all A_fg parameters.
                Flat prior on ``(-A_fg_max, A_fg_max)``.

        Raises
        ------
        ValueError
            If LAT and SAT use different dust foreground models.
        """
        if fg_model not in ('geometric', 'per_pair'):
            raise ValueError(f"fg_model must be 'geometric' or 'per_pair', got '{fg_model}'")

        # Foreground consistency must be checked before super().__init__ loads data.
        self._check_fg_consistency(spec_cross)

        super().__init__(spec_cross, sat_err, sat_lrange, lat_lrange,
                         fit_per_split, spectra_selection,
                         'CalibrationAnalysis_FGFit', num_sims, verbose,
                         cov_mode=cov_mode)

        self.fg_model = fg_model
        self.A_fg_max = A_fg_max

        self.cl_len = CMB(spec_cross.libdir, spec_cross.nside,
                          beta=beta_fid).get_lensed_spectra(dl=False)

        # Frequency values shared between LAT and SAT (e.g. '93', '145').
        self.freq_values: List[str] = sorted({b.split('_', 1)[1] for b in self.freq_bases})

        # Compute (or load) binned foreground templates.
        self.fg_templates_binned: np.ndarray = self._get_fg_templates()

        # Unique ordered pairs for per_pair model.
        freq_idx = {nu: k for k, nu in enumerate(self.freq_values)}
        self._fg_pairs: List[Tuple[str, str]] = [
            (fi, fj)
            for i, fi in enumerate(self.freq_values)
            for j, fj in enumerate(self.freq_values)
            if j >= i
        ]
        self._freq_idx = freq_idx

        # Build parameter names/labels: alphas + beta + A_fg parameters.
        if fg_model == 'geometric':
            fg_pnames  = [f"A_fg_{nu}" for nu in self.freq_values]
            fg_plabels = [rf"A_{{fg,{nu}}}" for nu in self.freq_values]
        else:  # per_pair
            fg_pnames  = [f"A_fg_{fi}_{fj}" for fi, fj in self._fg_pairs]
            fg_plabels = [rf"A_{{fg,{fi}\times{fj}}}" for fi, fj in self._fg_pairs]

        if fit_per_split:
            self.__pnames__  = [f"a_{t}" for t in self.maptags] + ['beta'] + fg_pnames
            self.__plabels__ = ([_format_alpha_label(t) for t in self.maptags]
                                + [r"\beta"] + fg_plabels)
        else:
            self.__pnames__  = [f"a_{f}" for f in self.freq_bases] + ['beta'] + fg_pnames
            self.__plabels__ = ([_format_alpha_label(f) for f in self.freq_bases]
                                + [r"\beta"] + fg_plabels)

    # -------------------------------------------------------------------------
    # Foreground helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _check_fg_consistency(spec_cross: 'SpectraCross') -> None:
        """Raise ValueError if LAT and SAT use different dust foreground models."""
        if spec_cross.lat_only:
            return
        if spec_cross.sat is None:
            raise ValueError("Foreground fitting requires SAT maps when lat_only is False.")
        lat_dust = spec_cross.lat.dust_model
        sat_dust = spec_cross.sat.dust_model
        if lat_dust != sat_dust:
            raise ValueError(
                f"LAT and SAT must use the same dust model for foreground fitting. "
                f"Got LAT dust_model={lat_dust}, SAT dust_model={sat_dust}."
            )

    def _get_fg_templates(self) -> np.ndarray:
        """
        Compute (or load from cache) full-sky foreground EB template spectra.

        For each pair ``(freq_i, freq_j)`` the method computes
        ``C_ell^{E_i B_j}`` full-sky from dust Q/U maps and
        bins the result with ``self.binner``.

        Returns
        -------
        np.ndarray
            Binned templates with shape ``(n_freq_values, n_freq_values, n_bins)``.
        """
        fname = os.path.join(self.libdir, 'fg_templates_binned_shared_freq_dustonly.pkl')
        if os.path.exists(fname):
            self.logger.log("Loading cached foreground EB templates...", level="info")
            with open(fname, 'rb') as f:
                return pl.load(f)

        self.logger.log("Computing full-sky dust-only foreground EB templates...", level="info")
        n_freqs = len(self.freq_values)

        # Foreground is shared between LAT/SAT for the same frequency, so build
        # one QU map per frequency using LAT foreground maps.
        fg = self.spec_cross.lat.foreground
        fg_qu: Dict[str, np.ndarray] = {}
        for freq in self.freq_values:
            dust_qu = fg.dustQU(str(freq))   # shape (2, npix): [Q, U]
            fg_qu[freq] = dust_qu

        # Precompute E/B alms once per frequency map, then use alm2cl for
        # all auto/cross combinations. This is significantly faster than
        # repeated map->spectrum transforms inside the pair loop.
        npix = fg_qu[self.freq_values[0]].shape[-1]
        nside = hp.npix2nside(npix)
        lmax_hp = min(self.Lmax, 3 * nside - 1)

        ealms: Dict[str, np.ndarray] = {}
        balms: Dict[str, np.ndarray] = {}
        for freq in tqdm(self.freq_values, desc='FG map2alm (shared freq)', leave=False):
            qmap = fg_qu[freq][0]
            umap = fg_qu[freq][1]
            _, ealm, balm = hp.map2alm([np.zeros(npix), qmap, umap], lmax=lmax_hp, pol=True)
            ealms[freq] = ealm
            balms[freq] = balm

        unbinned = np.zeros((n_freqs, n_freqs, self.Lmax + 1))
        for i, fi in tqdm(enumerate(self.freq_values), total=n_freqs, desc='FG EB auto/cross', leave=False):
            for j, fj in enumerate(self.freq_values):
                cl_eb = hp.alm2cl(ealms[fi], balms[fj], lmax=lmax_hp)
                ell_len = min(cl_eb.shape[-1], self.Lmax + 1)
                unbinned[i, j, :ell_len] = cl_eb[:ell_len]

        # Bin and cache.
        n_bins = self.binner.get_n_bands()
        binned = np.zeros((n_freqs, n_freqs, n_bins))
        for i in range(n_freqs):
            for j in range(n_freqs):
                binned[i, j] = self.binner.bin_cell(unbinned[i, j])

        with open(fname, 'wb') as f:
            pl.dump(binned, f)
        self.logger.log("Foreground EB templates computed and saved.", level="info")
        return binned

    # -------------------------------------------------------------------------
    # Abstract method implementations
    # -------------------------------------------------------------------------

    def theory(self, theta: np.ndarray) -> np.ndarray:
        """
        Theoretical EB model including miscalibration, birefringence, and foregrounds.

        .. math::

            C_\\ell^{E_i B_j} = \\cos(2\\alpha_i+2\\beta)\\sin(2\\alpha_j+2\\beta)\\,C_{EE}
                               - \\sin(2\\alpha_i+2\\beta)\\cos(2\\alpha_j+2\\beta)\\,C_{BB}
                               + \\sqrt{A_{fg,i}A_{fg,j}}\\,C_\\ell^{fg,\\,E_i B_j}

        Parameters are ordered as: ``[alphas..., beta, A_fg_per_frequency...]``.
        """        # --- Unpack parameters ---
        if self.fit_per_split:
            n_alpha  = len(self.maptags)
        else:
            n_alpha  = len(self.freq_bases)

        alphas_raw = theta[:n_alpha]
        beta       = theta[n_alpha]
        A_fg_arr   = theta[n_alpha + 1:]
        n_fg_expected = len(self.freq_values) if self.fg_model == 'geometric' else len(self._fg_pairs)
        if len(A_fg_arr) != n_fg_expected:
            raise ValueError(
                f"Expected {n_fg_expected} foreground amplitudes for fg_model='{self.fg_model}', "
                f"got {len(A_fg_arr)}."
            )

        # For fit_per_split=False, expand alphas to per-maptag.
        if self.fit_per_split:
            alphas = alphas_raw
        else:
            base_a = {b: alphas_raw[k] for k, b in enumerate(self.freq_bases)}
            alphas = np.array([base_a[t.rsplit('-', 1)[0]] for t in self.maptags])

        freq_to_idx = self._freq_idx
        cl_ee = self.cl_len["ee"][:self.Lmax + 1]
        cl_bb = self.cl_len["bb"][:self.Lmax + 1]

        # Build foreground amplitude lookup depending on model.
        if self.fg_model == 'geometric':
            freq_to_Afg = {nu: A_fg_arr[k] for k, nu in enumerate(self.freq_values)}
        else:  # per_pair
            pair_to_Afg = {pair: A_fg_arr[k] for k, pair in enumerate(self._fg_pairs)}

        model = np.zeros_like(self.mean_spec)

        for i in range(len(self.maptags)):
            for j in range(len(self.maptags)):
                ai = np.deg2rad(alphas[i])
                aj = np.deg2rad(alphas[j])
                b  = np.deg2rad(beta)

                rot_term = (
                    np.cos(2*ai + 2*b) * np.sin(2*aj + 2*b) * cl_ee
                    - np.sin(2*ai + 2*b) * np.cos(2*aj + 2*b) * cl_bb
                )

                fi = self.maptags[i].rsplit('-', 1)[0].split('_', 1)[1]
                fj = self.maptags[j].rsplit('-', 1)[0].split('_', 1)[1]
                ki = freq_to_idx[fi]
                kj = freq_to_idx[fj]

                if self.fg_model == 'geometric':
                    A_i, A_j = freq_to_Afg[fi], freq_to_Afg[fj]
                    fg_contrib = np.sqrt(max(A_i * A_j, 0.0)) * self.fg_templates_binned[ki, kj]
                else:  # per_pair — symmetric lookup
                    key = (fi, fj) if freq_to_idx[fi] <= freq_to_idx[fj] else (fj, fi)
                    fg_contrib = pair_to_Afg[key] * self.fg_templates_binned[ki, kj]

                model[i, j] = self.binner.bin_cell(rot_term) + fg_contrib

        return model

    def lnprior(self, theta: np.ndarray) -> float:
        """
        Log prior probability.

        - Gaussian prior on SAT alpha angles (width = ``sat_err``).
        - Flat prior on beta in ``(-0.5, 0.5)`` degrees.
        - Flat prior on each A_fg in ``(-A_fg_max, A_fg_max)``.
          For ``fg_model='geometric'`` A_fg is clipped to ``[0, A_fg_max]``
          because the geometric mean requires non-negative values.
        - Hard bounds ``|alpha| < 0.5`` degrees for all angles.
        """
        if self.fit_per_split:
            n_alpha = len(self.maptags)
        else:
            n_alpha = len(self.freq_bases)

        alphas   = theta[:n_alpha]
        beta     = theta[n_alpha]
        A_fg_arr = theta[n_alpha + 1:]

        n_fg_expected = len(self.freq_values) if self.fg_model == 'geometric' else len(self._fg_pairs)
        if len(A_fg_arr) != n_fg_expected:
            return -np.inf

        if np.any(np.abs(alphas) > 0.5):
            return -np.inf
        if not (-0.5 < beta < 0.5):
            return -np.inf

        if self.fg_model == 'geometric':
            # geometric mean requires non-negative amplitudes
            if np.any(A_fg_arr < 0.0) or np.any(A_fg_arr > self.A_fg_max):
                return -np.inf
        else:  # per_pair allows negative (EB foreground can have either sign)
            if np.any(np.abs(A_fg_arr) > self.A_fg_max):
                return -np.inf

        tags = self.maptags if self.fit_per_split else self.freq_bases
        sat_idx = [k for k, t in enumerate(tags) if t.startswith('SAT')]
        return float(
            -0.5 * np.sum(
                np.array(alphas)[sat_idx] ** 2 / self.sat_err ** 2
                - np.log(self.sat_err * np.sqrt(2 * np.pi))
            )
        )

    @property
    def dof(self) -> int:
        """Degrees of freedom: data points minus number of fitted parameters."""
        return int(np.sum(self.__likelihood_mask__)) - len(self.__pnames__)
class Sat4LatCross_AmplitudeFit(BaseSat4LatCross):
    """
    Calibration model fitting for amplitude parameter A_EB.
    
    This class implements a calibration analysis that fits calibration angles (alphas)
    and an amplitude parameter A_EB using EB cross-spectra between LAT and SAT.
    
    Attributes:
        eb_template_unbinned: Unbinned EB template spectrum
        binned_template: Binned EB template spectrum
        cl_len: Lensed CMB power spectra dictionary
        cl_diff_unbinned: Unbinned difference spectrum (EE - BB)/2
    """
    
    def __init__(self,
                 spec_cross: 'SpectraCross',
                 sat_err: float,
                 temp_model: str,
                 temp_value: float,
                 sat_lrange: Tuple[Optional[int], Optional[int]] = (None, None),
                 lat_lrange: Tuple[Optional[int], Optional[int]] = (None, None),
                 fit_per_split: bool = True,
                 spectra_selection: str = 'all',
                 sim_idx: int|None = None,
                 alpha_lat_prior: str = 'gaussian',
                 fix_alpha: bool = False,
                 num_sims: int = 100,
                 verbose: bool = False,
                 cov_mode: str = 'diagonal') -> None:
        """
        Initialize calibration analysis for amplitude parameter fitting.

        Args:
            spec_cross: Cross-spectrum calculation object
            sat_err: Satellite calibration error
            temp_model: Template model ('iso' or 'iso_td')
            temp_value: Template parameter value (beta for 'iso', mass for 'iso_td')
            sat_lrange: Satellite ell range for fitting (lmin, lmax)
            lat_lrange: LAT ell range for fitting (lmin, lmax)
            fit_per_split: Whether to fit per split or per frequency
            spectra_selection: Which spectra to include ('all', 'auto_only', 'cross_only')
            sim_idx: Simulation index to use for data (default: 0)
            alpha_lat_prior: Prior type for LAT alpha parameters ('gaussian' or 'flat')
            fix_alpha: Whether to fix alpha parameters (calibration angles) to zero
            num_sims: Number of simulations for mean/std calculation
            verbose: Whether to enable verbose output
            cov_mode: Covariance mode ('diagonal', 'block_diag', 'shrinkage', 'full')
        """
        self.sim_idx = sim_idx
        self.fix_alpha = fix_alpha
        suffix = f'CalibrationAnalysis_AmpFit_{temp_model}_{temp_value}_sim{sim_idx}'
        super().__init__(spec_cross, sat_err, sat_lrange, lat_lrange, fit_per_split, spectra_selection, suffix, num_sims, verbose,
                         cov_mode=cov_mode)

        if temp_model == 'iso':
            cmb = CMB(spec_cross.libdir, spec_cross.nside, beta=temp_value, verbose=verbose)
            self.eb_template_unbinned = cmb.get_cb_lensed_spectra(dl=False)['eb']
        elif temp_model == 'iso_td':
            cmb = CMB(spec_cross.libdir, spec_cross.nside, model=temp_model, mass=temp_value, verbose=verbose)
            self.eb_template_unbinned = cmb.get_cb_lensed_mass_spectra(dl=False)['eb']
        else:
            raise ValueError("temp_model must be 'iso' or 'iso_td'")
        self.binned_template = self.binner.bin_cell(self.eb_template_unbinned[:self.Lmax+1])
        self.cl_len = CMB(spec_cross.libdir, spec_cross.nside, beta=0.0).get_lensed_spectra(dl=False)
        self.cl_diff_unbinned = 0.5 * (self.cl_len["ee"][:self.Lmax + 1] -
                               self.cl_len["bb"][:self.Lmax + 1])

        if fix_alpha:
            self.__pnames__  = ['A_EB']
            self.__plabels__ = [r"A_{EB}"]
        else:
            if fit_per_split:
                self.__pnames__  = [f"a_{t}" for t in self.maptags] + ['A_EB']
                self.__plabels__ = [_format_alpha_label(t) for t in self.maptags] + [r"A_{EB}"]
            else:
                self.__pnames__  = [f"a_{f}" for f in self.freq_bases] + ['A_EB']
                self.__plabels__ = [_format_alpha_label(f) for f in self.freq_bases] + [r"A_{EB}"]
        self.alpha_lat_prior = alpha_lat_prior
        assert self.alpha_lat_prior in ['gaussian','flat'], "alpha_lat_prior must be 'gaussian' or 'flat'"
        if fit_per_split:
            self.lat_alpha_mean = np.array([0.2]*4)
            self.sat_alpha_mean = np.array([0.0]*4)
        else:
            self.lat_alpha_mean = np.array([0.2]*2)
            self.sat_alpha_mean = np.array([0.0]*2)
        self.lat_alpha_err = 0.01
        self.sat_alpha_err = sat_err



    def _calc_mean_std(self, num_sims: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Override parent method to use a specific simulation realization instead of mean.
        
        Args:
            num_sims: Number of simulations (used for std calculation only)
            
        Returns:
            Tuple of (single_realization_spectrum, std_spectrum) arrays
        """

        if self.sim_idx is None:
            fname = os.path.join(self.libdir, f'realization_spec_sim_mean_{num_sims}_sims.pkl')
        else:
            fname = os.path.join(self.libdir, f'realization_spec_sim{self.sim_idx}_{num_sims}_sims.pkl')
        if os.path.exists(fname):
            self.logger.log(f"Loading cached realization spectrum for sim_idx={self.sim_idx}...", level="info")
            with open(fname, 'rb') as f:
                return pl.load(f)
        
        self.logger.log(f"Computing realization spectrum for sim_idx={self.sim_idx} and std over {num_sims} simulations...", level="info")
        
        # Get the specific realization
        # Compute std from multiple simulations for error bars
        all_specs = [
            self.spec_cross.data_matrix(i, which='EB',
                                        sat_lrange=self.sat_lrange,
                                        lat_lrange=self.lat_lrange,
                                        avg_splits=False)
            for i in tqdm(range(num_sims))
        ]
        all_specs = np.array(all_specs)
        std_spec = np.std(all_specs, axis=0)

        if self.sim_idx is None:
            single_spec = np.mean(all_specs, axis=0)
        else:
            single_spec = all_specs[self.sim_idx]

        with open(fname, 'wb') as f:
            pl.dump((single_spec, std_spec), f)
        
        return single_spec, std_spec

    def _get_alphas(self, theta: np.ndarray) -> np.ndarray:
        """
        Extract alpha parameters from parameter vector.
        
        Args:
            theta: Full parameter vector
            
        Returns:
            Array of alpha parameters for each map tag
        """
        if self.fix_alpha:
            # Use mean values for fixed alphas
            if self.fit_per_split:
                # Combine LAT and SAT mean values in the correct order
                alphas = np.zeros(len(self.maptags))
                lat_count, sat_count = 0, 0
                for i, tag in enumerate(self.maptags):
                    if tag.startswith('LAT'):
                        alphas[i] = self.lat_alpha_mean[lat_count]
                        lat_count += 1
                    else:  # SAT
                        alphas[i] = self.sat_alpha_mean[sat_count]
                        sat_count += 1
                return alphas
            else:
                # For frequency-based fitting
                base_alphas = {}
                lat_count, sat_count = 0, 0
                for base in self.freq_bases:
                    if base.startswith('LAT'):
                        base_alphas[base] = self.lat_alpha_mean[lat_count]
                        lat_count += 1
                    else:  # SAT
                        base_alphas[base] = self.sat_alpha_mean[sat_count]
                        sat_count += 1
                return np.array([base_alphas[t.rsplit('-', 1)[0]] for t in self.maptags])
        elif self.fit_per_split:
            return theta[:-1]
        else:
            base_a = {b: theta[i] for i, b in enumerate(self.freq_bases)}
            return np.array([base_a[t.rsplit('-', 1)[0]] for t in self.maptags])

    def theory(self, theta: np.ndarray) -> np.ndarray:
        alphas = self._get_alphas(theta)
        A_EB = theta[-1] if not self.fix_alpha else theta[0]

        cl_ee = self.cl_len["ee"][:self.Lmax+1]
        cl_bb = self.cl_len["bb"][:self.Lmax+1]

        model = np.zeros_like(self.mean_spec)

        for i in range(len(self.maptags)):
            for j in range(len(self.maptags)):
                ai = np.deg2rad(alphas[i])
                aj = np.deg2rad(alphas[j])

                # physical rotation part (unbinned)
                term_unbinned = (
                    np.cos(2*ai) * np.sin(2*aj) * cl_ee
                    - np.sin(2*ai) * np.cos(2*aj) * cl_bb
                )

                # bin + add scaled template
                model[i, j] = (
                    self.binner.bin_cell(term_unbinned)
                    + (np.cos(2*ai + 2*aj) * self.binned_template / A_EB)
                )

        return model


    def lnprior(self, theta: np.ndarray) -> float:
        if self.fix_alpha:
            A_EB = theta[0]
            if not (0 < A_EB < 2):
                return -np.inf
            return 0.0  # No prior on alphas since they're fixed to zero

        alphas, A_EB = theta[:-1], theta[-1]
        if np.any(np.abs(alphas) > 0.5) or not (-1 < A_EB < 3):
            return -np.inf

        # SAT prior
        sat_idx = [i for i, t in enumerate(self.maptags if self.fit_per_split else self.freq_bases)
                if t.startswith('SAT')]
        sat_alpha_means = self.sat_alpha_mean if self.fit_per_split else self.sat_alpha_mean[:len([b for b in self.freq_bases if b.startswith('SAT')])]
        sat_prior = -0.5 * np.sum((np.array(alphas)[sat_idx] - sat_alpha_means)** 2 / self.sat_alpha_err**2 -
                                np.log(self.sat_alpha_err * np.sqrt(2 * np.pi)))
        if self.alpha_lat_prior == 'flat':
            return float(sat_prior)
        else:
            lat_idx = [i for i, t in enumerate(self.maptags if self.fit_per_split else self.freq_bases)
                    if t.startswith('LAT')]
            lat_alpha_means = self.lat_alpha_mean if self.fit_per_split else self.lat_alpha_mean[:len([b for b in self.freq_bases if b.startswith('LAT')])]
            lat_prior = -0.5 * np.sum((np.array(alphas)[lat_idx] - lat_alpha_means)** 2 / self.lat_alpha_err**2 -
                                    np.log(self.lat_alpha_err * np.sqrt(2 * np.pi)))

            return float(sat_prior + lat_prior)


# ============================================================
# =============== SUBCLASS 3: LAT-ONLY BETA FIT ==============
# ============================================================

class LatCross(BaseSat4LatCross):
    """
    Calibration model fitting for birefringence angle β using LAT-only data.
    
    This class implements a calibration analysis that fits calibration angles (alphas)
    and the cosmic birefringence angle β using EB cross-spectra from LAT only.
    
    Attributes:
        cl_len: Lensed CMB power spectra dictionary
        lat_err: LAT calibration error
    """
    
    def __init__(self,
                 spec_cross: 'SpectraCross',
                 lat_err: float,
                 beta_fid: float,
                 lat_lrange: Tuple[Optional[int], Optional[int]] = (None, None),
                 fit_per_split: bool = True,
                 spectra_selection: str = 'all',
                 num_sims: int = 100,
                 verbose: bool = False,
                 cov_mode: str = 'diagonal') -> None:
        """
        Initialize LAT-only calibration analysis for birefringence angle fitting.

        Args:
            spec_cross: Cross-spectrum calculation object (must be in LAT-only mode)
            lat_err: LAT calibration error
            beta_fid: Fiducial birefringence angle for theory calculation
            lat_lrange: LAT ell range for fitting (lmin, lmax)
            fit_per_split: Whether to fit per split or per frequency
            spectra_selection: Which spectra to include ('all', 'auto_only', 'cross_only')
            num_sims: Number of simulations for mean/std calculation
            verbose: Whether to enable verbose output
            cov_mode: Covariance mode ('diagonal', 'block_diag', 'shrinkage', 'full')
        """
        # Verify spec_cross is in LAT-only mode
        assert hasattr(spec_cross, 'lat_only') and spec_cross.lat_only, \
            "LatCross requires spec_cross to be initialized with sat=None (LAT-only mode)"

        # For LAT-only, we use the same sat_err parameter as lat_err and ignore SAT ranges
        super().__init__(spec_cross, lat_err,
                         sat_lrange=lat_lrange,  # Use LAT range for both
                         lat_lrange=lat_lrange,
                         fit_per_split=fit_per_split,
                         spectra_selection=spectra_selection,
                         libdir_suffix='LatCalibrationAnalysis',
                         num_sims=num_sims,
                         verbose=verbose,
                         cov_mode=cov_mode)

        self.lat_err = lat_err
        self.cl_len = CMB(spec_cross.libdir, spec_cross.nside, beta=beta_fid).get_lensed_spectra(dl=False)

        # Verify all maps are LAT maps (redundant check after lat_only assertion, but kept for clarity)
        if not all(tag.startswith('LAT') for tag in self.maptags):
            raise ValueError("LatCross requires all maps to be LAT maps. Found non-LAT maps in spec_cross.")

        if fit_per_split:
            self.__pnames__  = [f"a_{t}" for t in self.maptags] + ['beta']
            self.__plabels__ = [_format_alpha_label(t) for t in self.maptags] + [r"\beta"]
        else:
            self.__pnames__  = [f"a_{f}" for f in self.freq_bases] + ['beta']
            self.__plabels__ = [_format_alpha_label(f) for f in self.freq_bases] + [r"\beta"]

    def theory(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute theoretical EB spectra between all LAT map pairs
        using the exact birefringence + miscalibration formula:
            C_ell^{E_iB_j} = cos(2α_i+2β)sin(2α_j+2β)C_EE
                            - sin(2α_i+2β)cos(2α_j+2β)C_BB
        """
        if self.fit_per_split:
            alphas = theta[:-1]
        else:
            base_a = {b: theta[i] for i, b in enumerate(self.freq_bases)}
            alphas = np.array([base_a[t.rsplit('-', 1)[0]] for t in self.maptags])
        beta = theta[-1]

        cl_ee = self.cl_len["ee"][:self.Lmax+1]
        cl_bb = self.cl_len["bb"][:self.Lmax+1]

        model = np.zeros_like(self.mean_spec)

        for i in range(len(self.maptags)):          # E_i
            for j in range(len(self.maptags)):      # B_j
                ai = np.deg2rad(alphas[i])
                aj = np.deg2rad(alphas[j])
                b  = np.deg2rad(beta)

                term = (
                    np.cos(2*ai + 2*b) * np.sin(2*aj + 2*b) * cl_ee
                    - np.sin(2*ai + 2*b) * np.cos(2*aj + 2*b) * cl_bb
                )

                model[i, j] = self.binner.bin_cell(term)

        return model

    def lnprior(self, theta: np.ndarray) -> float:
        """
        Calculate log prior for LAT-only analysis.
        Uses Gaussian prior on all alphas with LAT calibration error.
        """
        alphas, beta = theta[:-1], theta[-1]
        if np.any(np.abs(alphas) > 0.5) or not (-0.5 < beta < 0.5):
            return -np.inf

        # Gaussian prior on all LAT alphas
        return float(-0.5 * np.sum(alphas ** 2 / self.lat_err**2 -
                                   np.log(self.lat_err * np.sqrt(2 * np.pi))))
    
    @property
    def dof(self) -> int:
        """
        Calculate degrees of freedom for chi-squared analysis.
        
        Returns:
            Number of data points minus number of fitted parameters
        """
        n_data_points = np.sum(self.__likelihood_mask__)
        n_parameters = len(self.__pnames__)
        return n_data_points - n_parameters

    

