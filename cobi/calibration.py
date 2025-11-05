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
                 verbose: bool = False) -> None:

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
        self.mean_spec, self.std_spec = self._calc_mean_std(num_sims=50)
        self.beam_arr: np.ndarray = self._get_beam_arr()
        
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

    def _calc_mean_std(self, num_sims: int = 50) -> Tuple[np.ndarray, np.ndarray]:
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
        all_specs = [
            self.spec_cross.data_matrix(i, which='EB',
                                        sat_lrange=self.sat_lrange,
                                        lat_lrange=self.lat_lrange,
                                        avg_splits=False)
            for i in tqdm(range(num_sims))
        ]
        all_specs = np.array(all_specs)
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
        fwhm_dict.update({f'SAT_{f}': fw for f, fw in zip(self.spec_cross.sat.freqs, self.spec_cross.sat.fwhm)})
        for i, ti in enumerate(self.maptags):
            for j, tj in enumerate(self.maptags):
                bi, bj = ti.rsplit('-', 1)[0], tj.rsplit('-', 1)[0]
                b_i = hp.gauss_beam(np.radians(fwhm_dict[bi] / 60.), lmax=self.Lmax) ** 2
                b_j = hp.gauss_beam(np.radians(fwhm_dict[bj] / 60.), lmax=self.Lmax) ** 2
                beam[i, j] = self.binner.bin_cell(np.sqrt(b_i * b_j))
        return beam

    # -------------------------------------------------------------------------
    # Shared Likelihood & MCMC
    # -------------------------------------------------------------------------

    def chisq(self, theta: np.ndarray) -> float:
        """
        Calculate chi-squared statistic for given parameters.
        
        Args:
            theta: Parameter vector
            
        Returns:
            Chi-squared value
        """
        model = self.theory(theta)
        data, err = self.mean_spec / self.beam_arr, self.std_spec / self.beam_arr
        err = np.where(err == 0, np.inf, err)
        chi2 = ((data - model) / err) ** 2
        return np.sum(chi2[self.__likelihood_mask__])

    def ln_likelihood(self, theta: np.ndarray) -> float:
        """Calculate log likelihood for given parameters."""
        return -0.5 * self.chisq(theta)

    def ln_prob(self, theta: np.ndarray) -> float:
        """Calculate log posterior probability for given parameters."""
        lp = self.lnprior(theta)
        return -np.inf if not np.isfinite(lp) else lp + self.ln_likelihood(theta)

    def run_mcmc(self, nwalkers: int = 32, nsamples: int = 2000, rerun: bool = False, 
                 fiducial_params: Optional[np.ndarray] = None, fname: str = 'samples.pkl') -> np.ndarray:
        fname = os.path.join(self.libdir,
            f"samples_cross_{nwalkers}_{nsamples}_fitper_{self.fit_per_split}_ndim_{len(self.__pnames__)}.pkl"
            )
        if os.path.exists(fname) and not rerun:
            self.logger.log(f"Loading cached samples: {fname}", level="info")
            with open(fname, 'rb') as f:
                loaded_samples = pl.load(f)
                return np.array(loaded_samples)
        ndim = len(self.__pnames__)
        if fiducial_params is not None:
            if len(fiducial_params) != ndim:
                raise ValueError(f"fiducial_params length {len(fiducial_params)} does not match ndim {ndim}")
        else:
            fiducial_params = np.zeros(ndim)
        pos = fiducial_params + 1e-3 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, threads=8)
        sampler.run_mcmc(pos, nsamples, progress=True)
        flat_samples = sampler.get_chain(discard=200, thin=20, flat=True)
        with open(fname, 'wb') as f:
            pl.dump(flat_samples, f)
        return np.array(flat_samples)

    def getdist_samples(self, nwalkers: int, nsamples: int, rerun: bool = False,
                        fiducial_params: Optional[np.ndarray] = None,
                        label: Optional[str] = None) -> MCSamples:
        samples = self.run_mcmc(nwalkers, nsamples, rerun=rerun, fiducial_params=fiducial_params)
        return MCSamples(samples=samples, names=self.__pnames__, labels=self.__plabels__, label=label)

    # -------------------------------------------------------------------------
    # Shared Plotting
    # -------------------------------------------------------------------------

    def plot_posteriors(self, nwalkers: int, nsamples: int, rerun: bool = False, label: Optional[str] = None,
                        fiducial_params: Optional[np.ndarray] = None):
        samples = self.getdist_samples(nwalkers, nsamples, rerun=rerun, label=label, fiducial_params=fiducial_params)
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

                # Plot data ±1σ
                ax.errorbar(
                    ells, data_spec[i, j], yerr=error_spec[i, j],
                    fmt='.', capsize=2, color='black', markersize=4, alpha=0.5,
                    label='Data Mean ± 1σ'
                )

                # Overlay theory if provided
                if theory_spec is not None:
                    ax.loglog(ells, theory_spec[i, j], color='red', label='Theory')

                # Shade region used in likelihood
                if average_split:
                    mask_1d = self.__likelihood_mask__[0, 0, :]
                else:
                    mask_1d = self.__likelihood_mask__[i, j, :]
                used_ells = ells[mask_1d]
                if len(used_ells) > 0:
                    ax.axvspan(used_ells.min(), used_ells.max(), color='green', alpha=0.1, zorder=-1)

                # Formatting
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.axhline(0, color='grey', linestyle=':', linewidth=0.75)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_ylim(1e-8, 1e-1)
                ax.set_xlim(10, 3000)

                # Axes labels
                if i == n_tags - 1:
                    ax.set_xlabel(r'Multipole $\ell$')
                if j == 0:
                    ax.set_ylabel(r'$C_\ell^{EB}$')
                if i == 0:
                    ax.set_title(maptags[j].replace('_', ' '), fontsize=10)

        # Right-side frequency labels
        for i, tag in enumerate(maptags):
            axes[i, n_tags - 1].text(
                1.05, 0.5, tag.replace('_', ' '),
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
            plt.savefig(save_path, dpi=150)
            self.logger.log(f"Figure saved to {save_path}", level="info")
            plt.close()
        else:
            plt.show()

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
                 verbose: bool = False) -> None:
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
            verbose: Whether to enable verbose output
        """
        super().__init__(spec_cross, sat_err, sat_lrange, lat_lrange,
                         fit_per_split, spectra_selection, 'CalibrationAnalysis', verbose)
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
        if np.any(np.abs(alphas) > 0.5) or not (0 < beta < 0.5):
            return -np.inf
        sat_idx = [i for i, t in enumerate(self.maptags if self.fit_per_split else self.freq_bases) if t.startswith('SAT')]
        return float(-0.5 * np.sum(np.array(alphas)[sat_idx] ** 2 / self.sat_err**2 - np.log(self.sat_err * np.sqrt(2 * np.pi))))

# ============================================================
# =============== SUBCLASS 2: AMPLITUDE FIT ==================
# ============================================================

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
                 verbose: bool = False) -> None:
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
            verbose: Whether to enable verbose output
        """
        self.sim_idx = sim_idx
        suffix = f'CalibrationAnalysis_AmpFit_{temp_model}_{temp_value}_sim{sim_idx}'
        super().__init__(spec_cross, sat_err, sat_lrange, lat_lrange, fit_per_split, spectra_selection, suffix, verbose)

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

        if fit_per_split:
            self.__pnames__  = [f"a_{t}" for t in self.maptags] + ['A_EB']
            self.__plabels__ = [_format_alpha_label(t) for t in self.maptags] + [r"A_{EB}"]
        else:
            self.__pnames__  = [f"a_{f}" for f in self.freq_bases] + ['A_EB']
            self.__plabels__ = [_format_alpha_label(f) for f in self.freq_bases] + [r"A_{EB}"]
        if fit_per_split:
            self.lat_alpha_mean = np.array([0.2]*4)
        else:
            self.lat_alpha_mean = np.array([0.2]*2)
        self.lat_alpha_err = 0.2

    def _calc_mean_std(self, num_sims: int = 50) -> Tuple[np.ndarray, np.ndarray]:
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
        if self.fit_per_split:
            return theta[:-1]
        base_a = {b: theta[i] for i, b in enumerate(self.freq_bases)}
        return np.array([base_a[t.rsplit('-', 1)[0]] for t in self.maptags])

    def theory(self, theta: np.ndarray) -> np.ndarray:
        alphas = self._get_alphas(theta)
        A_EB   = theta[-1]

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
        alphas, A_EB = theta[:-1], theta[-1]
        if np.any(np.abs(alphas) > 0.5) or not (0 < A_EB < 2):
            return -np.inf
        
        # Only SAT prior
        sat_idx = [i for i, t in enumerate(self.maptags if self.fit_per_split else self.freq_bases) 
                if t.startswith('SAT')]
        #lat_idx = [i for i, t in enumerate(self.maptags if self.fit_per_split else self.freq_bases) 
        #        if t.startswith('LAT')]
        sat_prior = -0.5 * np.sum((np.array(alphas)[sat_idx] - 0)** 2 / self.sat_err**2 - 
                                np.log(self.sat_err * np.sqrt(2 * np.pi)))
        #lat_prior = -0.5 * np.sum((np.array(alphas)[lat_idx]-self.lat_alpha_mean) ** 2 / self.lat_alpha_err**2 - 
        #                       np.log(self.lat_alpha_err * np.sqrt(2 * np.pi)))

        #return float(sat_prior + lat_prior)
        return float(sat_prior)


def selector(lib,lmin,lmax,):
    b = lib.binInfo.get_effective_ells()
    select = np.where((b>lmin) & (b<lmax))[0]
    return select
def selectorf(lib,avoid_freq):
    freqs = lib.lat.freqs
    select = np.where(np.array([freq not in avoid_freq for freq in freqs]))[0]
    return select

def get_sp(lib,lmax,avoid_freq=[]):
    bl_arr = []
    for i in range(2):# changed
        bl_arr.append(lib.binInfo.bin_cell(hp.gauss_beam(np.radians(lib.lat.fwhm[i]/60),lmax)**2))
    bl_arr = np.array(bl_arr)
    obs_arr = []
    # for i in range(100):
    #     sp = lib.obs_x_obs(i)
    #     obs_arr.append(sp)
    # obs_arr = np.array(obs_arr)
    # obs_arr = obs_arr[:,np.arange(2),np.arange(2),2,2:] # changed
    for i in range(100):
        sp = lib.obs_x_obs(i, False) 
        first  = (sp[1, 0, 2, 2:] + sp[0, 1, 2, 2:]) / 2
        second = (sp[2, 3, 2, 2:] + sp[3, 2, 2, 2:]) / 2
        obs_arr.append(np.stack([first, second], axis=0))
    obs_arr = np.asarray(obs_arr)         
    return obs_arr/bl_arr

def paranames(lib,name,avoid=[]):
    return [f"a{name}{fe}" for fe in lib.lat.freqs if fe not in avoid] 
def latexnames(lib, name, avoid=[]):
    return [r'\alpha_{{{}}}^{{{}}}'.format(name, fe) for fe in lib.lat.freqs if fe not in avoid]


class Sat4Lat_AmplitudeFit:
    """
    Modified class to fit for an EB amplitude parameter (A_EB) and 
    miscalibration angles (alpha) instead of the birefringence angle (beta).
    """
    def __init__(self, libdir, lmin, lmax, latlib, satlib, sat_err, temp_model,temp_value,use_diag=True,idx=None,verbose=False):
        
        latnc = latlib.lat.noise_model
        satnc = satlib.lat.noise_model
        if (latnc == 'NC') and (satnc == 'NC'):
            self.libdir = os.path.join(latlib.lat.libdir, f"CalibrationTD_AmpFit_{temp_model}_{temp_value}")
        elif (latnc == 'TOD') and (satnc == 'TOD'):
            self.libdir = os.path.join(latlib.lat.libdir, f"Calibration_TODTD_AmpFit_{temp_model}_{temp_value}")
        # ... (add other elif conditions as in your original code) ...
        else:
            raise ValueError(f"Invalid noise model {latnc} {satnc}")

        os.makedirs(self.libdir, exist_ok=True)
        self.latlib = latlib
        self.sat_err = sat_err
        self.binner = satlib.binInfo
        self.Lmax = satlib.lmax
        self.__select__ = selector(satlib, lmin, lmax)
        self.addname = "bp" if latlib.lat.bandpass else ''
        
        self.use_diag = use_diag
        self.idx = idx
        self.lat_mean_90, self.lat_icov_90 = self.calc_mean_icov(latlib, 'LAT', 0)
        self.lat_mean_150, self.lat_icov_150 = self.calc_mean_icov(latlib, 'LAT', 1)
        self.sat_mean_90, self.sat_icov_90 = self.calc_mean_icov(satlib, 'SAT', 0)
        self.sat_mean_150, self.sat_icov_150 = self.calc_mean_icov(satlib, 'SAT', 1)
        if temp_model == 'iso':
            cmb = CMB(libdir,satlib.lat.nside,beta=temp_value,verbose=verbose)
            self.eb_template_unbinned = cmb.get_cb_lensed_spectra(dl=False)['eb']
        elif temp_model == 'iso_td':
            cmb = CMB(libdir,satlib.lat.nside,model=temp_model,mass=temp_value,verbose=verbose)
            self.eb_template_unbinned = cmb.get_cb_lensed_mass_spectra(dl=False)['eb']
        else:
            raise ValueError("only 'iso' and 'iso_td' allowed")
        
        ### MODIFIED ###
        # Store the binned version of the provided EB template.
        # The template should be the C_l^EB spectrum (not D_l).
        self.binned_template = self.binner.bin_cell(self.eb_template_unbinned[:self.Lmax+1])[self.__select__]
        
        # We still need the lensed EE and BB for the alpha calculation
        self.cl_len = CMB(libdir, satlib.lat.nside, beta=0.0,verbose=verbose).get_lensed_spectra(dl=False)

        self.lmin = lmin
        self.lmax = lmax

        ### MODIFIED ###
        # Updated parameter names and labels for A_EB instead of beta
        self.__pnames__ = paranames(latlib, 'LAT') + paranames(satlib, 'SAT') + ['A_EB']
        self.__plabels__ = latexnames(latlib, 'LAT') + latexnames(satlib, 'SAT') + [r'A_{EB}']
        self.dof = 2*len(self.__select__) * 2 - len(self.__pnames__) # 2 spectra (EB) for 2 freq for lat and sat
        


    def calc_mean_icov(self, lib, name, idx):
        """
        Calculates the mean and the inverse covariance matrix of the spectra.
        """
        USE_DIAG = self.use_diag
        sp = get_sp(lib, self.Lmax)[:,idx,self.__select__]
        if self.idx is not None:
            mean = sp[self.idx]
        else:
            mean = sp.mean(axis=0)
        cov = np.cov(sp.T)
        if USE_DIAG:
            cov = np.diag(np.diag(cov))
        else:
            alpha = 1e-5 
            reg_term = alpha * np.mean(np.diag(cov)) * np.identity(cov.shape[0])
            cov += reg_term
        icov = np.linalg.inv(cov)
        if name in ['LAT', 'SAT']:
            return (mean, icov)
        else:
            raise ValueError(f"Invalid name {name}")
        
    def plot_spectra(self, tele):
        plt.figure(figsize=(4, 4))
        if tele == 'LAT' and self.latlib is not None:
            for i in range(2):
                plt.loglog(self.binner.get_effective_ells()[self.__select__], self.lat_mean[i])
        elif tele == 'SAT':
            for i in range(2):
                plt.loglog(self.binner.get_effective_ells()[self.__select__], self.sat_mean[i])
        else:
            raise ValueError(f"Invalid telescope {tele}")
    
    def theory_alpha(self, alpha_array):
        ### MODIFIED ###
        # This function now *only* calculates the effect of the miscalibration angle alpha.
        # It's kept separate for clarity in the chisq function.
        alpha_array = np.asarray(alpha_array)
        # The EB spectrum induced by E-B mixing from a miscalibration angle alpha is sin(4*alpha)*(C_l^EE - C_l^BB)/2
        th = 0.5 * (self.cl_len["ee"] - self.cl_len["bb"])[:, np.newaxis] * np.sin(np.deg2rad(4 * alpha_array))
        return np.apply_along_axis(lambda th_slice: self.binner.bin_cell(th_slice[:self.Lmax+1])[self.__select__], 0, th).T

    def chisq(self, theta):
        alpha_lat, alpha_sat, A_EB = np.array(theta[:2]), np.array(theta[2:4]), theta[-1]
        birefringence_model = self.binned_template / A_EB
        alpha_lat_90, alpha_lat_150 = alpha_lat
        alpha_sat_90, alpha_sat_150 = alpha_sat
        
        #SAT
        sat_miscal_model_90 = self.theory_alpha(np.array([alpha_sat_90]))
        sat_total_model_90 = birefringence_model + sat_miscal_model_90
        diff_90 = self.sat_mean_90 - sat_total_model_90
        sat_chi_90 = diff_90 @ self.sat_icov_90 @ diff_90.T 
        sat_miscal_model_150 = self.theory_alpha(np.array([alpha_sat_150]))
        sat_total_model_150 = birefringence_model + sat_miscal_model_150
        diff_150 = self.sat_mean_150 - sat_total_model_150
        sat_chi_150 = diff_150 @ self.sat_icov_150 @ diff_150.T 
        sat_chi = sat_chi_90 + sat_chi_150

        #LAT
        lat_miscal_model_90 = self.theory_alpha(np.array([alpha_lat_90]))
        lat_total_model_90 = birefringence_model + lat_miscal_model_90
        diff_90 = self.lat_mean_90 - lat_total_model_90
        lat_chi_90 = diff_90 @ self.lat_icov_90 @ diff_90.T 
        lat_miscal_model_150 = self.theory_alpha(np.array([alpha_lat_150]))
        lat_total_model_150 = birefringence_model + lat_miscal_model_150
        diff_150 = self.lat_mean_150 - lat_total_model_150
        lat_chi_150 = diff_150 @ self.lat_icov_150 @ diff_150.T 
        lat_chi = lat_chi_90 + lat_chi_150      

        return sat_chi + lat_chi

    
    def MLE(self,Aeb):
        
        initial_guess = np.array([0.2, 0.2, 0.0, 0.0, Aeb])  # [alpha_lat1, alpha_lat2, alpha_sat1, alpha_sat2, A_EB]

        result = minimize(self.chisq, initial_guess, method='Nelder-Mead')

        if result.success:
            print("MLE optimization successful.")
            print("Optimized parameters:", result.x)
            print("Minimum chi-squared:", result.fun)
        else:
            raise RuntimeError("MLE optimization failed.")

    def lnprior(self, theta):
        sigma = self.sat_err
        ### MODIFIED ###
        # Unpack A_EB instead of beta
        alphalat, alphasat, A_EB = np.array(theta[:2]), np.array(theta[2:4]), theta[-1]

        lnp = -0.5 * (alphasat - 0)**2 / sigma**2 - np.log(sigma * np.sqrt(2 * np.pi))
        
        ### MODIFIED ###
        # Set a flat prior for A_EB. You can change the range [-1, 1] as needed.
        if np.all(alphalat > -0.5) and np.all(alphalat < 0.5) and 0 < A_EB < 2.0:
            return np.sum(lnp)
        return -np.inf

    def ln_likelihood(self, theta):
        return -0.5 * self.chisq(theta)

    def ln_prob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta)
    
    def samples(self, nwalkers=32, nsamples=1000, rerun=True):
        ### MODIFIED ###
        # Update the 'true' values for the MCMC initialization.
        # Assuming true alphas are ~0 and true A_EB is, for example, 0.3
        true = np.array([0.2, 0.2, 0, 0, 0.3]) 
        
        fname = os.path.join(self.libdir, f"samples_{nwalkers}_{nsamples}{self.addname}_max{self.lmax}_min{self.lmin}.pkl")
        if os.path.exists(fname) and not rerun:
            return pl.load(open(fname, 'rb'))
        else:
            ndim = len(true)
            pos = true + 1e-3 * np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, threads=4)
            sampler.run_mcmc(pos, nsamples, progress=True)
            flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)
            pl.dump(flat_samples, open(fname, 'wb'))
            return flat_samples

    def getdist_samples(self, nwalkers, nsamples, rerun=False, label=None):
        flat_samples = self.samples(nwalkers, nsamples, rerun=rerun)
        return MCSamples(samples=flat_samples, names=self.__pnames__, labels=self.__plabels__, label=label)
        
    def plot_getdist(self, nwalkers, nsamples, avoid_sat=False, aeb_only=False, rerun=False):
        ### MODIFIED ###
        # Added an 'aeb_only' option for convenience, similar to 'beta_only'
        flat_samples = self.getdist_samples(nwalkers, nsamples, rerun=rerun)
        if aeb_only:
            g = plots.get_single_plotter(width_inch=4)
            g.plot_1d(flat_samples, 'A_EB', title_limit=1)
        else:
            names = self.__pnames__
            if avoid_sat:
                names = [item for item in names if 'SAT' not in item]
            g = plots.get_subplot_plotter()
            g.triangle_plot([flat_samples], names, filled=True, title_limit=1)
    
    def mle(self,nwalkers,nsamples,rerun=False):
        s = self.samples(nwalkers,nsamples,rerun=rerun)
        theta = np.median(s,axis=0)
        chi2_min = self.chisq(theta)
        return chi2_min
    
    def p_value(self,nwalkers,nsamples,rerun=False):
        chi2_min = self.mle(nwalkers,nsamples,rerun=rerun)
        p_val = chi2.sf(chi2_min, self.dof)
        return p_val