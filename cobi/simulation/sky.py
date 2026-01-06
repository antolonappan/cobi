"""
Sky Simulation Module
=====================

This module provides a unified interface for generating complete multi-frequency
sky simulations combining CMB, foregrounds, and noise with realistic instrumental
effects and data processing.

The SkySimulation class orchestrates:

- CMB signal with cosmic birefringence
- Galactic foregrounds (dust, synchrotron)
- Instrumental noise (LAT/SAT models)
- Beam convolution
- Component separation (ILC)
- Survey masks
- Coordinate transformations

Classes
-------
SkySimulation
    Main class for end-to-end sky simulation and processing.

HILC
    Harmonic-space Internal Linear Combination for component separation.

Example
-------
Generate a complete LAT simulation::

    from cobi.simulation import SkySimulation
    import numpy as np
    
    sky = SkySimulation(
        libdir='./sky_sims',
        nside=512,
        freqs=np.array([27, 39, 93, 145, 225, 280]),
        fwhm=np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9]),
        tube=np.array([0, 0, 1, 1, 2, 2]),  # tube assignments
        cb_model='iso',
        beta=0.35,
        lensing=True,
        sensitivity_mode='baseline',
        telescope='LAT',
        fsky=0.4
    )
    
    # Get full simulation (all components)
    full_sky = sky.get_sim(idx=0, component='all')
    
    # Get CMB-only
    cmb_only = sky.get_sim(idx=0, component='cmb')
    
    # Get ILC-cleaned map
    cleaned = sky.get_ilc(idx=0)

Multi-component analysis::

    # Individual components
    cmb = sky.get_sim(idx=0, component='cmb')
    fg = sky.get_sim(idx=0, component='fg')
    noise = sky.get_sim(idx=0, component='noise')
    
    # Verify: all = cmb + fg + noise
    all_sky = sky.get_sim(idx=0, component='all')

Notes
-----
This module provides the highest-level interface for simulation generation.
It handles caching, parallelization, and ensures consistency across components.
Supports both isotropic and anisotropic cosmic birefringence models.
"""
import numpy as np
import healpy as hp
import os
import pickle as pl
from tqdm import tqdm
from cobi import mpi
from typing import Dict, Union, List, Optional

from cobi.simulation import CMB, Foreground, Mask, Noise
from cobi.utils import Logger, inrad
from cobi.utils import cli, deconvolveQU
from cobi.simulation import HILC
from cobi import sht

from concurrent.futures import ThreadPoolExecutor

NCPUS = os.cpu_count()

class SkySimulation:
    """
    Unified multi-frequency sky simulation framework.
    
    This is the highest-level class that orchestrates complete end-to-end simulations
    combining CMB, foregrounds, noise, and instrumental effects for SO LAT/SAT observations.
    Includes component separation via harmonic ILC.
    
    Parameters
    ----------
    libdir : str
        Base directory for all simulation products and caching.
    nside : int
        HEALPix resolution parameter.
    freqs : ndarray
        Array of observation frequencies in GHz (e.g., [27, 39, 93, 145, 225, 280]).
    fwhm : ndarray
        Beam FWHM in arcminutes for each frequency.
    tube : ndarray
        Optical tube assignment for each frequency (for correlated noise).
    cb_model : {'iso', 'iso_td', 'aniso'}, default='iso'
        Cosmic birefringence model type.
    beta : float, default=0.35
        Isotropic birefringence angle in degrees (for cb_model='iso').
    mass : float, default=1.5
        Axion mass in 10^-22 eV units (for cb_model='iso_td').
        Must be 1, 1.5, or 6.4.
    Acb : float, default=1e-6
        Anisotropic birefringence amplitude (for cb_model='aniso').
    lensing : bool, default=True
        Include CMB lensing effects.
    dust_model : int, default=10
        PySM3 dust model number.
    sync_model : int, default=5
        PySM3 synchrotron model number.
    bandpass : bool, default=True
        Apply bandpass integration for foregrounds.
    alpha : float or array, default=0.0
        Miscalibration polarization angle(s) in degrees.
        - float: same angle for all frequencies
        - array: per-frequency angles (must match len(freqs))
    alpha_err : float, default=0.0
        Gaussian uncertainty on miscalibration angle (degrees).
    sim_config : dict, optional
        Simulation seed management configuration.
        Keys: 'set1' (int), 'reuse_last' (int)
        Controls efficient seed reuse for CMB/Noise across different alpha modes.
    noise_model : {'NC', 'TOD'}, default='NC'
        Noise simulation type (curves vs. time-ordered data).
    atm_noise : bool, default=True
        Include atmospheric/1f noise.
    nsplits : int, default=2
        Number of data splits for null tests.
    gal_cut : int, default=0
        Galactic mask cut (0, 40, 60, 70, 80, 90).
    hilc_bins : int, default=10
        Number of multipole bins for harmonic ILC.
    deconv_maps : bool, default=False
        Deconvolve beam from output maps.
    fldname_suffix : str, default=''
        Additional suffix for output directory naming.
    sht_backend : {'healpy', 'ducc0'}, default='healpy'
        Spherical harmonic transform backend.
    verbose : bool, default=True
        Enable logging output.
    
    Attributes
    ----------
    nside : int
        HEALPix resolution.
    freqs : ndarray
        Observation frequencies.
    cmb : CMB
        CMB simulation object.
    foreground : Foreground
        Foreground simulation object.
    noise : Noise
        Noise simulation object.
    mask : ndarray
        Combined survey mask.
    fsky : float
        Sky fraction.
    config : dict
        Per-frequency configuration dictionary.
    
    Methods
    -------
    get_sim(idx, component='all')
        Get simulation realization (cmb/fg/noise/all).
    get_ilc(idx, return_noise=False)
        Get ILC-cleaned CMB map.
    get_obs(idx, freqs_idx=None)
        Get observed maps at specific frequencies.
    
    Examples
    --------
    Complete LAT simulation pipeline::
    
        from cobi.simulation import SkySimulation
        import numpy as np
        
        sky = SkySimulation(
            libdir='./sims',
            nside=512,
            freqs=np.array([27, 39, 93, 145, 225, 280]),
            fwhm=np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9]),
            tube=np.array([0, 0, 1, 1, 2, 2]),
            cb_model='iso',
            beta=0.35,
            lensing=True,
            dust_model=10,
            sync_model=5,
            alpha=0.0,  # no miscalibration
            atm_noise=True,
            gal_cut=40
        )
        
        # Get full simulation
        full_map = sky.get_sim(idx=0, component='all')
        
        # Get CMB-only
        cmb_map = sky.get_sim(idx=0, component='cmb')
        
        # Get ILC-cleaned map
        cleaned_map = sky.get_ilc(idx=0)
    
    Multi-frequency miscalibration::
    
        sky = SkySimulation(
            libdir='./sims',
            nside=512,
            freqs=np.array([93, 145, 225]),
            fwhm=np.array([2.2, 1.4, 1.0]),
            tube=np.array([1, 1, 2]),
            alpha=[0.2, -0.1, 0.15],  # per-frequency
            alpha_err=0.05  # with uncertainty
        )
    
    Notes
    -----
    - Automatically manages caching and directory structure
    - MPI-aware for parallel processing
    - Supports batch processing with sim_config
    - ILC uses harmonic-space cleaning
    - All outputs in μK_CMB units
    """
    def __init__(
        self,
        libdir: str,
        nside: int,
        freqs: np.ndarray,
        fwhm: np.ndarray,
        tube: np.ndarray,
        cb_model: str = "iso",
        beta: float = 0.35,
        mass: float = 1.5,
        Acb: float = 1e-6,
        lensing: bool = True,
        dust_model: int = 10,
        sync_model: int = 5,
        bandpass: bool = True,
        alpha: Union[float,List[float]] = 0.0,
        alpha_err: float = 0.0,
        sim_config: Optional[dict] = None,
        noise_model: str = "NC",
        aso: bool = False,
        atm_noise: bool = True,
        nsplits: int = 2,
        gal_cut: int = 0,
        hilc_bins: int = 10,
        deconv_maps: bool = False,
        fldname_suffix: str = "",
        sht_backend: str = "healpy",
        verbose: bool = True,
    ):
        """
        Initialize the SkySimulation.
        
        See class docstring for detailed parameter descriptions.
        """
        self.logger = Logger(self.__class__.__name__, verbose)
        self.verbose = verbose

        fldname = "_an" if atm_noise else "_wn"
        fldname += "_bp" if bandpass else ""
        fldname += f"_{nsplits}ns" 
        fldname += "_lens" if lensing else "_gauss"
        fldname += f"_{noise_model.lower()}nm"
        fldname += "_aso" if aso else ""
        if cb_model == 'iso':
            fldname += f"_b{str(beta).replace('.','p')}"
        elif cb_model == 'iso_td':
            fldname += f"_m({str(mass).replace('.','p')}"
        elif cb_model == 'aniso':
            fldname += f"_acb{str(Acb).replace('-','')}"
        else:
            raise ValueError("Unknown CB method")
        
        fldname += f"_d{dust_model}s{sync_model}"
        fldname += f"_g{gal_cut}" if gal_cut > 0 else ""
        if isinstance(alpha, (list, np.ndarray)):
            fldname += f"_a"  ''.join('n' + f"{abs(num):g}".replace(".", "") if num < 0 else f"{num:g}".replace(".", "") for num in alpha).replace('0','')
        else:
            fldname += f"_a{str(alpha).replace('.','p')}"
        fldname += f"_ae{str(alpha_err).replace('.','p')}" if alpha_err > 0 else ""
        fldname += fldname_suffix

        self.basedir = libdir
        self.libdir = os.path.join(libdir, self.__class__.__name__[:3] + fldname)
        os.makedirs(self.libdir + '/obs', exist_ok=True)

        self.dnside = 0
        if self.__class__.__name__ == "SAT" and nside != 512:
            self.logger.log(f"SAT simulations are only supported for nside=512. Resetting the given NSIDE={nside} to 512.")
            self.dnside = 512
            
        self.nside = nside
        self.Acb = Acb
        self.cb_method = cb_model
        self.beta = beta
        self.cmb = CMB(libdir, nside, cb_model, beta, mass, Acb, lensing, sim_config, verbose=self.verbose)
        self.foreground = Foreground(libdir, nside, dust_model, sync_model, bandpass, verbose=False)
        self.dust_model = dust_model
        self.sync_model = sync_model
        self.nsplits = nsplits
        self.freqs = freqs
        self.fwhm = fwhm
        self.tube = tube
        self.gal_cut = gal_cut
        self.mask, self.fsky = self.__set_mask_fsky__(libdir)
        self.noise_model = noise_model
        self.noise = Noise(nside, self.fsky, self.__class__.__name__[:3], noise_model, atm_noise, nsplits, aso, verbose=self.verbose)
        self.aso = aso
        self.config = {}
        for split in range(nsplits):
            for band in range(len(self.freqs)):
                self.config[f'{self.freqs[band]}-{split+1}'] = {"fwhm": self.fwhm[band], "opt. tube": self.tube[band]}

        if isinstance(alpha, (list, np.ndarray)):
            assert self.freqs is not None and len(alpha) == len(
                self.freqs
            ), "Length of alpha list must match the number of frequency bands."
            for band, a in enumerate(alpha):
                for split in range(self.nsplits):
                    self.config[f'{self.freqs[band]}-{split+1}']["alpha"] = a
        else:
            if self.freqs is not None:
                for split in range(self.nsplits):
                    for band in range(len(self.freqs)):
                        self.config[f'{self.freqs[band]}-{split+1}']["alpha"] = alpha

        self.alpha = alpha
        self.alpha_err = alpha_err
        self.atm_noise = atm_noise
        self.bandpass = bandpass
        self.hilc_bins = hilc_bins
        self.deconv_maps = deconv_maps
        if sht_backend in ["ducc0", "ducc", "d"]:
            self.hp = sht.HealpixDUCC(nside=self.nside)
            self.healpy = False
        else:
            self.hp = None
            self.healpy = True

    def __set_mask_fsky__(self, libdir):
        maskobj = Mask(libdir, self.nside, self.__class__.__name__[:3], gal_cut=self.gal_cut, verbose=self.verbose)
        return maskobj.mask, maskobj.fsky

    def signalOnlyQU(self, idx: int, band: str) -> np.ndarray:
        band = band[:band.index('-')]
        cmbQU = np.array(self.cmb.get_cb_lensed_QU(idx))
        dustQU = self.foreground.dustQU(band)
        syncQU = self.foreground.syncQU(band)
        return cmbQU + dustQU + syncQU
    
    def get_alpha(self, idx: int, band: str):
        """
        Get alpha value for a given index and band.
        
        If alpha_err > 0, generates alpha values deterministically on-demand.
        Same (band, idx) always returns the same alpha value due to deterministic seeding.
        
        Parameters
        ----------
        idx : int
            Simulation index
        band : str
            Band identifier (e.g., '93-1', '145-2')
        
        Returns
        -------
        float
            Alpha value in degrees
        """
        if self.alpha_err > 0:
            # Generate value deterministically (MPI-safe: all ranks generate same value)
            base_band = band.split('-')[0]
            band_1 = f"{base_band}-1"
            alpha_mean = self.config[band_1]["alpha"]
            
            # Use deterministic seed based on band and idx for reproducibility
            np.random.seed(hash((base_band, idx)) % (2**32))
            alpha_value = np.random.normal(alpha_mean, self.alpha_err)
            
            return alpha_value
        else:
            return self.config[band]["alpha"]
        

    def obsQUwAlpha(
        self, idx: int, band: str, fwhm: float, alpha: float, apply_tranf: bool = True, return_alms: bool = False
    ) -> np.ndarray:
        signal = self.signalOnlyQU(idx, band)*self.mask
        E, B = hp.map2alm_spin(signal, 2, lmax=self.cmb.lmax)
        Elm = (E * np.cos(inrad(2 * alpha))) - (B * np.sin(inrad(2 * alpha)))
        Blm = (E * np.sin(inrad(2 * alpha))) + (B * np.cos(inrad(2 * alpha)))
        del (E, B)
        if apply_tranf:
            bl = hp.gauss_beam(inrad(fwhm / 60), lmax=self.cmb.lmax, pol=True)
            pwf = np.array(hp.pixwin(self.nside, pol=True,))
            hp.almxfl(Elm, bl[:, 1] * pwf[1, :], inplace=True)
            hp.almxfl(Blm, bl[:, 2] * pwf[1, :], inplace=True)
        if return_alms:
            return np.array([Elm, Blm])
        else:
            return hp.alm2map_spin([Elm, Blm], self.nside, 2, lmax=self.cmb.lmax)*self.mask

    def obsQUfname(self, idx: int, band: str) -> str:
        alpha = self.config[band]["alpha"]
        fwhm = self.config[band]["fwhm"]
        tube = self.config[band]["opt. tube"]
        fname = os.path.join(self.libdir,'obs', f"sims_a{str(alpha)}_f{fwhm}_t{tube}_b{band}_{idx:03d}.fits")
        return fname

    
    def SaveObsQUs(self, idx: int, apply_mask: bool = True, bands=None) -> None:

        def create_band_map(idx,band):
            fname = self.obsQUfname(idx, band)
            if os.path.isfile(fname) and (bands is None):
                return 0
            else:
                fwhm = self.config[band]["fwhm"]
                alpha = self.get_alpha(idx, band)
                signal = self.obsQUwAlpha(idx, band, fwhm, alpha)
                #noise = self.noise.atm_noise_maps_freq(idx, band)
                noise = self.noise.noiseQU_freq(idx, band)
                if len(noise) > 2:
                    nside = hp.get_nside(noise[0])
                else:
                    nside = hp.get_nside(noise)
                if nside != self.nside:
                    self.logger.log(f"Noise map is not in the same nside as the signal map. Changing nside {nside} to {self.nside}.")
                    noise = hp.ud_grade(noise, self.nside)
                sky = signal + noise
                del (signal, noise)
                if self.deconv_maps:
                    sky = deconvolveQU(sky, fwhm)
                fname = self.obsQUfname(idx, band)
                hp.write_map(fname, sky * mask, dtype=np.float64,overwrite=(bands is not None)) # type: ignore
                return 0

        mask = self.mask if apply_mask else np.ones_like(self.mask)
        Bands = list(self.config.keys()) if bands is None else bands
  
        for band in tqdm(Bands, desc="Saving Observed QUs", unit="band"):
            maps = create_band_map(idx,band)




    def obsQU(self, idx: int, band: str) -> np.ndarray:
        fname = self.obsQUfname(idx, band)
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1]) # type: ignore
        else:
            self.SaveObsQUs(idx)
            return hp.read_map(fname, field=[0, 1]) # type: ignore
    
    def checkObsQU(self, idx: int,overwrite=False,what='filesize',bands=False) -> bool:
        bands = list(self.config.keys())
        if what == 'filesize':
            err = []
            for band in bands:
                try:
                    qu = self.obsQU(idx, band)
                except ValueError:
                    err.append(band)
            if len(err) > 0:
                self.logger.log(f"Error in {idx} for bands {err}")
                if overwrite:
                    self.logger.log(f"Overwriting {idx} for bands {err}")
                    self.SaveObsQUs(idx, bands=err)
                    return True
                else:
                    return False
            else:
                self.logger.log(f"All bands are present for {idx}")
                return True
        elif what == 'file':
            err = []
            for band in bands:
                fname = self.obsQUfname(idx, band)
                if not os.path.isfile(fname):
                    err.append(band)
            if len(err) > 0:
                self.logger.log(f"Error in {idx} for bands {err}")
                if overwrite:
                    self.logger.log(f"Overwriting {idx} for bands {err}")
                    self.SaveObsQUs(idx, bands=err)
                    return True
                else:
                    return False
            else:
                return True
        else:
            raise ValueError(f"Unknown check {what}. Please use 'filesize' or 'file'.")

    
    def HILC_obsEB(self, idx: int, ret=None, split: int = 0, debug=False) -> np.ndarray:
        # Validate split parameter
        if split < 0 or split > self.nsplits:
            raise ValueError(f"split must be between 0 and {self.nsplits}, got {split}")
        
        # For nsplits=1, split=0 and split=1 are equivalent
        if self.nsplits == 1 and split == 1:
            split = 0
        
        # Construct filename based on split and nsplits
        if self.nsplits == 1:
            # For nsplits=1, always use same filename regardless of split value
            fnameS = os.path.join(
                self.libdir,
                f"obs/hilcEB_N{self.nside}_A{str(self.Acb).replace('.','p')}{'_bp' if self.bandpass else ''}_{idx:03d}.fits",
            )
        elif split == 0:
            # For nsplits>1 and split=0, use all splits (no split suffix)
            fnameS = os.path.join(
                self.libdir,
                f"obs/hilcEB_N{self.nside}_A{str(self.Acb).replace('.','p')}{'_bp' if self.bandpass else ''}_{idx:03d}.fits",
            )
        else:
            # For nsplits>1 and split>0, add split number to filename
            fnameS = os.path.join(
                self.libdir,
                f"obs/hilcEB_N{self.nside}_A{str(self.Acb).replace('.','p')}{'_bp' if self.bandpass else ''}_s{split}_{idx:03d}.fits",
            )
        fnameN = fnameS.replace('hilcEB','hilcNoise')
        
        # Debug: print bands and filename
        if split == 0:
            bands = list(self.config.keys())
        else:
            bands = [key for key in self.config.keys() if key.endswith(f'-{split}')]

        if debug:
            self.logger.log(f"HILC_obsEB: split={split}, nsplits={self.nsplits}, bands={bands}")
            self.logger.log(f"HILC_obsEB: Output files: {fnameS}")
            return
        
        if os.path.isfile(fnameS) and os.path.isfile(fnameN):
            if ret is None:
                return hp.read_alm(fnameS, hdu=[1, 2]), hp.read_cl(fnameN)
            elif ret == 'alm':
                return hp.read_alm(fnameS, hdu=[1, 2])
            elif ret == 'nl':
                return hp.read_cl(fnameN)
            else:
                raise ValueError(f"Unknown return type {ret}. Please use 'alm' or 'nl'.")
        else:
            alms = []
            nalms = []
            
            i = 0
            for band in tqdm(bands, desc=f"Computing HILC Observed QUs (split={split})", unit="band"):
                fwhm = self.config[band]["fwhm"]
                alpha = self.get_alpha(idx, band)
                noise = hp.ud_grade(self.noise.noiseQU_freq(idx, band),self.nside)*self.mask
                QU = self.obsQUwAlpha(idx, band, fwhm, alpha, apply_tranf=False, return_alms=False)
                if self.healpy:
                    elm, blm = hp.map2alm_spin(QU, 2, lmax=self.cmb.lmax)
                    nelm,nblm = hp.map2alm_spin([noise[0],noise[1]], 2, lmax=self.cmb.lmax)  
                else:
                    elm, blm = self.hp.map2alm(QU, lmax=self.lmax, nthreads=NCPUS)
                    nelm,nblm = self.hp.map2alm(noise,lmax=self.lmax, nthreads=NCPUS)
                bl = hp.gauss_beam(inrad(fwhm / 60), lmax=self.cmb.lmax, pol=True)
                pwf = np.array(hp.pixwin(self.nside, pol=True,))
                transfe = bl[:, 1] * pwf[1, :]
                transfb = bl[:, 2] * pwf[1, :]
                hp.almxfl(nelm, cli(transfe), inplace=True)
                hp.almxfl(nblm, cli(transfb), inplace=True)
                alms.append([elm+nelm, blm+nblm])
                nalms.append([nelm, nblm])
                i += 1
            alms = np.array(alms)
            nalms = np.array(nalms)
            
            hilc = HILC()
            bins = np.arange(1000) * self.hilc_bins
            cleaned,ilc_weight = hilc.harmonic_ilc_alm(alms,bins)
            ilc_noise = hilc.apply_harmonic_W(ilc_weight,nalms)
            cleaned, ilc_noise = cleaned[0], ilc_noise[0]
            ilc_noise = [hp.alm2cl(ilc_noise[0]), hp.alm2cl(ilc_noise[1])]
            hp.write_alm(fnameS, cleaned, overwrite=True)
            hp.write_cl(fnameN, ilc_noise, overwrite=True)
            if ret is None:
                return cleaned, ilc_noise
            elif ret == 'alm':
                return cleaned
            elif ret == 'nl':
                return ilc_noise
            else:
                raise ValueError(f"Unknown return type {ret}. Please use 'alm' or 'nl'.")
            


class LATsky(SkySimulation):
    freqs = np.array(["27", "39", "93", "145", "225", "280"])
    fwhm = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])  # arcmin
    tube = np.array(["LF", "LF", "MF", "MF", "HF", "HF"])  # tube each frequency occupies

    def __init__(
        self,
        libdir: str,
        nside: int,
        cb_model: str = "iso",
        beta: float = 0.35,
        mass: float = 1.5,
        Acb: float = 1e-6,
        lensing: bool = True,
        dust_model: int = 10,
        sync_model: int = 5,
        bandpass: bool = True,
        alpha: float = 0.0,
        alpha_err: float = 0.0,
        sim_config: Optional[dict] = None,
        noise_model: str = "NC",
        aso: bool = False,
        atm_noise: bool = True,
        nsplits: int = 2,
        gal_cut: int = 0,
        hilc_bins: int = 10,
        deconv_maps: bool = False,
        fldname_suffix: str = "",
        sht_backend: str = "healpy",
        verbose: bool = True,
    ):
        super().__init__(
            libdir = libdir,
            nside = nside,
            freqs = LATsky.freqs,
            fwhm = LATsky.fwhm,
            tube = LATsky.tube,
            cb_model = cb_model,
            beta = beta,
            mass = mass,
            Acb = Acb,
            lensing = lensing,
            dust_model = dust_model,
            sync_model = sync_model,
            bandpass = bandpass,
            alpha = alpha,
            alpha_err = alpha_err,
            sim_config = sim_config,
            noise_model = noise_model,
            aso = aso,
            atm_noise = atm_noise,
            nsplits = nsplits,
            gal_cut = gal_cut,
            hilc_bins = hilc_bins,
            deconv_maps = deconv_maps,
            fldname_suffix = fldname_suffix,
            sht_backend = sht_backend,
            verbose = verbose,
        )


class SATsky(SkySimulation):
    freqs = np.array(["27", "39", "93", "145", "225", "280"])
    fwhm = np.array([91, 63, 30, 17, 11, 9])
    tube = np.array(["S1", "S1", "S2", "S2", "S3", "S3"])  # example tube identifiers

    def __init__(
        self,
        libdir: str,
        nside: int,
        cb_model: str = "iso",
        beta: float = 0.35,
        mass: float = 1.5,
        Acb: float = 1e-6,
        lensing: bool = True,
        dust_model: int = 10,
        sync_model: int = 5,
        bandpass: bool = True,
        alpha: float = 0.0,
        alpha_err: float = 0.0,
        sim_config: Optional[dict] = None,
        noise_model: str = "NC",
        atm_noise: bool = True,
        nsplits: int = 2,
        gal_cut: int = 0,
        hilc_bins: int = 10,
        deconv_maps: bool = False,
        fldname_suffix: str = "",
        sht_backend: str = "healpy",
        verbose: bool = True,
    ):
        super().__init__(
            libdir = libdir,
            nside = nside,
            freqs = SATsky.freqs,
            fwhm = SATsky.fwhm,
            tube = SATsky.tube,
            cb_model = cb_model,
            beta = beta,
            mass = mass,
            Acb = Acb,
            lensing = lensing,
            dust_model = dust_model,
            sync_model = sync_model,
            bandpass = bandpass,
            alpha = alpha,
            alpha_err = alpha_err,
            sim_config = sim_config,
            noise_model = noise_model,
            atm_noise = atm_noise,
            nsplits = nsplits,
            gal_cut = gal_cut,
            hilc_bins = hilc_bins,
            deconv_maps = deconv_maps,
            fldname_suffix = fldname_suffix,
            sht_backend = sht_backend,
            verbose = verbose,
        )


class LATskyC(SkySimulation):
    freqs = np.array(["93", "145"])
    fwhm = np.array([2.2, 1.4 ])  # arcmin
    tube = np.array(["MF", "MF"])  # tube each frequency occupies

    def __init__(
        self,
        libdir: str,
        nside: int,
        cb_model: str = "iso",
        beta: float = 0.35,
        mass: float = 1.5,
        Acb: float = 1e-6,
        lensing: bool = True,
        dust_model: int = 10,
        sync_model: int = 5,
        bandpass: bool = True,
        alpha: Optional[float|List[float]] = 0.0,
        alpha_err: float = 0.0,
        sim_config: Optional[dict] = None,
        noise_model: str = "NC",
        aso: bool = False,
        atm_noise: bool = True,
        nsplits: int = 2,
        gal_cut: int = 0,
        hilc_bins: int = 10,
        deconv_maps: bool = False,
        fldname_suffix: str = "",
        verbose: bool = True,
    ):
        super().__init__(
            libdir = libdir,
            nside = nside,
            freqs = LATskyC.freqs,
            fwhm = LATskyC.fwhm,
            tube = LATskyC.tube,
            cb_model = cb_model,
            beta = beta,
            mass = mass,
            Acb = Acb,
            lensing = lensing,
            dust_model = dust_model,
            sync_model = sync_model,
            bandpass = bandpass,
            alpha = alpha,
            alpha_err = alpha_err,
            sim_config = sim_config,
            noise_model = noise_model,
            aso = aso,
            atm_noise = atm_noise,
            nsplits = nsplits,
            gal_cut = gal_cut,
            hilc_bins = hilc_bins,
            deconv_maps = deconv_maps,
            fldname_suffix = fldname_suffix,
            verbose = verbose,
        )


class SATskyC(SkySimulation):
    freqs = np.array([ "93", "145"])
    fwhm = np.array([30, 17])
    tube = np.array(["S2", "S2"])  # example tube identifiers

    def __init__(
        self,
        libdir: str,
        nside: int,
        cb_model: str = "iso",
        beta: float = 0.35,
        mass: float = 1.5,
        Acb: float = 1e-6,
        lensing: bool = True,
        dust_model: int = 10,
        sync_model: int = 5,
        bandpass: bool = True,
        alpha: float = 0.0,
        alpha_err: float = 0.0,
        sim_config: Optional[dict] = None,
        noise_model: str = "NC",
        atm_noise: bool = True,
        nsplits: int = 2,
        gal_cut: int = 0,
        hilc_bins: int = 10,
        deconv_maps: bool = False,
        fldname_suffix: str = "",
        verbose: bool = True,
    ):
        super().__init__(
            libdir = libdir,
            nside = nside,
            freqs = SATskyC.freqs,
            fwhm = SATskyC.fwhm,
            tube = SATskyC.tube,
            cb_model = cb_model,
            beta = beta,
            mass = mass,
            Acb = Acb,
            lensing = lensing,
            dust_model = dust_model,
            sync_model = sync_model,
            bandpass = bandpass,
            alpha = alpha,
            alpha_err = alpha_err,
            sim_config = sim_config,
            noise_model = noise_model,
            atm_noise = atm_noise,
            nsplits = nsplits,
            gal_cut = gal_cut,
            hilc_bins = hilc_bins,
            deconv_maps = deconv_maps,
            fldname_suffix = fldname_suffix,
            verbose = verbose,
        )