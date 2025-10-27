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
                 libdir_suffix: str = 'BaseSat4LatCross') -> None:

        print(f"Initializing {self.__class__.__name__}...")
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
        
        print(f"Initialized {self.__class__.__name__}.\n")

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

        n_tags, n_bins = len(self.maptags), self.binner.get_n_bands()
        mask = np.zeros((n_tags, n_tags, n_bins), dtype=bool)
        is_lat = np.array([tag.startswith('LAT') for tag in self.maptags])
        lat_auto = np.outer(is_lat, is_lat)
        sat_auto = np.outer(~is_lat, ~is_lat)
        cross = ~(lat_auto | sat_auto)

        mask[lat_auto] = lat_mask
        mask[sat_auto] = sat_mask
        mask[cross] = cross_mask

        if self.spectra_selection == 'auto_only':
            mask &= (lat_auto | sat_auto)[:, :, None]
        elif self.spectra_selection == 'cross_only':
            mask &= cross[:, :, None]
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
            print("Loading cached mean/std spectra...")
            with open(fname, 'rb') as f:
                return pl.load(f)
        print(f"Computing mean/std over {num_sims} simulations...")
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
            print(f"Loading cached samples: {fname}")
            with open(fname, 'rb') as f:
                loaded_samples = pl.load(f)
                return np.array(loaded_samples)
        ndim = len(self.__pnames__)
        pos = (fiducial_params or np.zeros(ndim)) + 1e-1 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, threads=8)
        sampler.run_mcmc(pos, nsamples, progress=True)
        flat_samples = sampler.get_chain(discard=200, thin=20, flat=True)
        with open(fname, 'wb') as f:
            pl.dump(flat_samples, f)
        return np.array(flat_samples)

    def getdist_samples(self, nwalkers: int, nsamples: int, rerun: bool = False, 
                        label: Optional[str] = None) -> MCSamples:
        samples = self.run_mcmc(nwalkers, nsamples, rerun=rerun)
        return MCSamples(samples=samples, names=self.__pnames__, labels=self.__plabels__, label=label)

    # -------------------------------------------------------------------------
    # Shared Plotting
    # -------------------------------------------------------------------------

    def plot_posteriors(self, nwalkers: int, nsamples: int, rerun: bool = False, label: Optional[str] = None):
        samples = self.getdist_samples(nwalkers, nsamples, rerun=rerun, label=label)
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
            print(f"Figure saved to {save_path}")
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
                 spectra_selection: str = 'all') -> None:
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
        """
        super().__init__(spec_cross, sat_err, sat_lrange, lat_lrange,
                         fit_per_split, spectra_selection, 'CalibrationAnalysis')
        self.cl_len = CMB(spec_cross.libdir, spec_cross.nside, beta=beta_fid).get_lensed_spectra(dl=False)
        if fit_per_split:
            self.__pnames__  = [f"a_{t}" for t in self.maptags] + ['beta']
            self.__plabels__ = [_format_alpha_label(t) for t in self.maptags] + [r"\beta"]
        else:
            self.__pnames__  = [f"a_{f}" for f in self.freq_bases] + ['beta']
            self.__plabels__ = [_format_alpha_label(f) for f in self.freq_bases] + [r"\beta"]

    def theory(self, theta: np.ndarray) -> np.ndarray:
        if self.fit_per_split:
            alphas = theta[:-1]
        else:
            base_a = {b: theta[i] for i, b in enumerate(self.freq_bases)}
            alphas = np.array([base_a[t.rsplit('-', 1)[0]] for t in self.maptags])
        beta = theta[-1]
        cl_diff = 0.5 * (self.cl_len["ee"] - self.cl_len["bb"])
        model = np.zeros_like(self.mean_spec)
        for i in range(len(self.maptags)):
            for j in range(len(self.maptags)):
                term = np.sin(np.deg2rad(2 * alphas[i] + 2 * alphas[j] + 4 * beta))
                model[i, j] = self.binner.bin_cell(cl_diff[:self.Lmax+1] * term)
        return model

    def lnprior(self, theta: np.ndarray) -> float:
        alphas, beta = theta[:-1], theta[-1]
        if np.any(np.abs(alphas) > 0.5) or not (0 < beta < 0.5):
            return -np.inf
        sat_idx = [i for i, t in enumerate(self.maptags if self.fit_per_split else self.freq_bases) if t.startswith('SAT')]
        return float(-0.5 * np.sum(np.array(alphas)[sat_idx] ** 2 / self.sat_err**2))

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
            verbose: Whether to enable verbose output
        """
        suffix = f'CalibrationAnalysis_AmpFit_{temp_model}_{temp_value}'
        super().__init__(spec_cross, sat_err, sat_lrange, lat_lrange, fit_per_split, spectra_selection, suffix)

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
        alphas, A_EB = self._get_alphas(theta), theta[-1]
        model = np.zeros_like(self.mean_spec)
        for i in range(len(self.maptags)):
            for j in range(len(self.maptags)):
                ang = np.sin(np.deg2rad(2 * alphas[i] + 2 * alphas[j]))
                model[i, j] = self.binner.bin_cell(self.cl_diff_unbinned * ang) + self.binned_template / A_EB
        return model

    def lnprior(self, theta: np.ndarray) -> float:
        alphas, A_EB = theta[:-1], theta[-1]
        if np.any(np.abs(alphas) > 0.5) or not (0 < A_EB < 2.0):
            return -np.inf
        sat_idx = [i for i, t in enumerate(self.maptags if self.fit_per_split else self.freq_bases) if t.startswith('SAT')]
        return float(-0.5 * np.sum(np.array(alphas)[sat_idx] ** 2 / self.sat_err**2))
