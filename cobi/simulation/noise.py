"""
Noise Simulation Module
=======================

This module simulates realistic instrumental noise for Simons Observatory
observations, including white noise, atmospheric noise, and correlated noise
between detector pairs.

Features:

- SO LAT and SAT noise models (v3.1.1)
- Sensitivity modes (baseline, goal)
- Atmospheric and 1/f noise components
- Cross-frequency correlations
- Map-based and harmonic-space realizations
- Anisotropic noise from hit maps

Classes
-------
Noise
    Main class for generating noise realizations and spectra.

Functions
---------
NoiseSpectra
    Compute noise power spectra for SO telescopes.

Example
-------
Generate noise maps for LAT baseline sensitivity::

    from cobi.simulation import Noise
    import numpy as np
    
    noise = Noise(
        libdir='./noise_sims',
        nside=512,
        lmax=3000,
        freqs=np.array([27, 39, 93, 145, 225, 280]),
        fwhm=np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9]),
        telescope='LAT',
        sensitivity_mode='baseline',
        fsky=0.4,
        atm_noise=True
    )
    
    # Get noise realization
    noise_maps = noise.get_map(idx=0)  # Shape: (nfreqs, 3, npix)

Anisotropic noise from hit maps::

    noise_aniso = Noise(
        libdir='./noise_sims',
        nside=512,
        nside_aniso=4096,  # high-res for hit maps
        isotropic=False,
        hitmap_file='hitmaps.h5'
    )

Notes
-----
Uses SO noise models v3.1.1 for realistic noise properties. Supports both
isotropic and anisotropic noise based on actual survey hit maps.
"""
import numpy as np
import healpy as hp
from pixell import enmap
from pixell.reproject import map2healpix
from typing import Dict, Optional, Any, Union, List, Tuple
from so_models_v3 import SO_Noise_Calculator_Public_v3_1_1 as so_models

from cobi import mpi
from cobi.utils import Logger, change_coord

#atm_noise or atm_corr, same for the noise map as well

def NoiseSpectra(sensitivity_mode, fsky, lmax, atm_noise, telescope, aso=False):
    if aso:
        assert telescope == "LAT", "ASO noise model is only available for LAT telescope."
    match telescope:
        case "LAT":
            if aso:
                teles = so_models.SOLatV3point1(sensitivity_mode,N_tubes=[1, 8, 4],survey_years=9, el=50)
            else:
                teles = so_models.SOLatV3point1(sensitivity_mode, el=50)
        case "SAT":
            teles = so_models.SOSatV3point1(sensitivity_mode)
    
    corr_pairs = [(0,1),(2,3),(4,5)]
    
    ell, N_ell_LA_T_full,N_ell_LA_P_full = teles.get_noise_curves(fsky, lmax, 1, full_covar=True, deconv_beam=False)
    del N_ell_LA_T_full
    bands = teles.get_bands().astype(int)
    Nbands = len(bands)
    N_ell_LA_P  = N_ell_LA_P_full[range(Nbands),range(Nbands)] #type: ignore
    N_ell_LA_Px = [N_ell_LA_P_full[i,j] for i,j in corr_pairs] #type: ignore
    Nell_dict = {}
    Nell_dict["ell"] = ell
    if atm_noise:
        for i in range(3):
            for j in range(3):
                if j < 2:
                    Nell_dict[f"{bands[i*2+j]}"] = N_ell_LA_P[i*2+j]
                else:
                    k = i*2+j
                    Nell_dict[f"{bands[k-2]}x{bands[k-1]}"] = N_ell_LA_Px[i]
    else:
        WN = np.radians(teles.get_white_noise(fsky)**.5*np.sqrt(2) / 60)**2
        for i in range(Nbands):
            Nell_dict[f"{bands[i]}"] = WN[i]*np.ones_like(ell)


    return Nell_dict


class Noise:
    """
    Instrumental noise simulator for Simons Observatory telescopes.
    
    This class generates realistic noise realizations for SO LAT and SAT observations
    based on the v3.1.1 noise models. Supports white noise, atmospheric/1f noise,
    and correlated noise between detector pairs within the same optical tube.
    
    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    fsky : float
        Sky fraction for noise power calculation (0 < fsky ≤ 1).
    telescope : {'LAT', 'SAT'}
        Telescope identifier (Large Aperture or Small Aperture).
    sim : {'NC', 'TOD'}, default='NC'
        Simulation type:
        - 'NC': Noise curves from SO models (analytic)
        - 'TOD': Time-ordered data based simulations (uses SO products)
    atm_noise : bool, default=False
        Include atmospheric and 1/f noise components.
    nsplits : int, default=2
        Number of data splits (for split-based null tests).
    verbose : bool, default=True
        Enable logging output.
    
    Attributes
    ----------
    nside : int
        HEALPix resolution.
    lmax : int
        Maximum multipole (3*nside - 1).
    sensitivity_mode : int
        SO sensitivity mode (2 = baseline).
    atm_noise : bool
        Whether atmospheric noise is included.
    Nell : dict
        Noise power spectra for each frequency and cross-frequency.
    telescope : str
        Telescope identifier.
    
    Methods
    -------
    get_map(idx, freqs, fwhm, tube)
        Generate noise map realization for given configuration.
    get_spectra(idx, freqs, fwhm, tube)
        Get noise power spectra for given configuration.
    
    Examples
    --------
    LAT baseline noise with atmosphere::
    
        noise = Noise(
            nside=512,
            fsky=0.4,
            telescope='LAT',
            sim='NC',
            atm_noise=True,
            nsplits=2
        )
        
        # Generate noise for 6 LAT frequencies
        freqs = np.array([27, 39, 93, 145, 225, 280])
        fwhm = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])
        tube = np.array([0, 0, 1, 1, 2, 2])
        
        noise_maps = noise.get_map(idx=0, freqs=freqs, fwhm=fwhm, tube=tube)
    
    SAT white noise only::
    
        noise_sat = Noise(
            nside=512,
            fsky=0.1,
            telescope='SAT',
            sim='NC',
            atm_noise=False
        )
        
        freqs = np.array([27, 39, 93, 145, 225, 280])
        noise_maps = noise_sat.get_map(idx=0, freqs=freqs, fwhm=None, tube=None)
    
    Notes
    -----
    - Uses SO v3.1.1 noise models (baseline sensitivity)
    - Correlated noise between frequency pairs in same tube
    - Supports independent realizations via idx parameter
    - MPI-aware for parallel generation
    - Output in μK_CMB units
    """

    def __init__(self, 
                 nside: int, 
                 fsky: float,
                 telescope: str,
                 sim = 'NC',
                 atm_noise: bool = False, 
                 nsplits: int = 2,
                 aso: bool = False,
                 ext_sims: bool = False,
                 sim_config: Optional[Dict[str, Any]] = None,
                 verbose: bool = True,
                 ) -> None:
        """
        Initializes the Noise class for generating noise maps with or without atmospheric noise.

        Parameters:
        nside (int): HEALPix resolution parameter.
        atm_noise (bool, optional): If True, includes atmospheric noise. Defaults to False.
        nhits (bool, optional): If True, includes hit count map. Defaults to False.
        nsplits (int, optional): Number of data splits to consider. Defaults to 2.
        ext_sims (bool, optional): If True, extends simulations by treating each split as independent (only for TOD LAT). Defaults to False.
        """
        self.nside            = nside
        self.lmax             = 3 * nside - 1
        self.sensitivity_mode = 2
        self.atm_noise        = atm_noise
        self.nsplits          = nsplits
        self.telescope = telescope
        self.ext_sims         = ext_sims
        self.Nell             = NoiseSpectra(self.sensitivity_mode, fsky, self.lmax, self.atm_noise, telescope, aso)
        self.sim = sim
        assert sim in ['NC', 'TOD'], "Invalid simulation type. Choose from 'NC' or 'TOD'."
        
        if ext_sims:
            assert sim == 'TOD', "ext_sims can only be True when sim='TOD'."
            assert telescope == 'LAT', "ext_sims is only available for LAT telescope."

        self.logger           = Logger(self.__class__.__name__, verbose)
        self.sim_config = sim_config
        if self.sim == 'NC':
            if self.atm_noise:
                self.logger.log(f"Noise Model:[{telescope}] White + 1/f noise v3.1.1")
            else:
                self.logger.log(f"Noise Model:[{telescope}] White noise v3.1.1")
        elif self.sim == 'TOD':
            self.logger.log(f"Noise Model: [{telescope}] Based on TOD and Map based simulations, directly using SO products.")
            if self.ext_sims:
                self.logger.log("Extending simulations by treating each split as independent (only for LAT).")
        else:
            raise ValueError(f"Invalid simulation type: {self.sim}", "Choose from 'NC' or 'TOD'.")
        
        self.__nseeds__ = {
            1: np.arange(7777, 7777 + 1000),
            2: np.arange(9999, 9999 + 1000),
            3: np.arange(5555, 5555 + 1000),
            4: np.arange(8888, 8888 + 1000),
        }
    
    def __get_noise_seed_idx__(self, idx: int) -> int:
        """
        Map simulation index to the actual noise seed index.
        
        For indices < set1: use idx directly
        For indices >= set1: map to last reuse_last indices of set1
        
        Parameters:
        idx (int): The simulation realization index.
        
        Returns:
        int: The seed index to use for noise generation.
        """
        if self.sim_config is None:
            return idx
        
        set1 = self.sim_config['set1']
        if idx < set1:
            return idx
        
        # Map to last reuse_last simulations from set1
        reuse_last = self.sim_config['reuse_last']
        offset = idx - set1
        base_idx = (set1 - reuse_last) + (offset % reuse_last)
        return base_idx
        


    @property
    def rand_alm(self) -> np.ndarray:
        """
        Generates random spherical harmonic coefficients (alm) with a specified power spectrum.

        Returns:
        np.ndarray: A complex array of spherical harmonic coefficients.
        """
        cl = np.repeat(1.0e3, self.lmax + 1)
        return hp.almxfl(hp.synalm(cl, lmax=self.lmax, new=True), 1 / np.sqrt(cl))

    @property
    def cholesky_matrix_elements(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Computes the Cholesky matrix elements for the noise model.

        Returns:
        Tuple of nine np.ndarray elements representing the Cholesky matrix components.
        """
        L11     = np.zeros(self.lmax, dtype=float)
        L11[2:] = np.sqrt(self.Nell["27"])
        L21     = np.zeros(self.lmax, dtype=float)
        L21[2:] = self.Nell["27x39"] / np.sqrt(self.Nell["27"])
        L22     = np.zeros(self.lmax, dtype=float)
        L22[2:] = np.sqrt(
            (self.Nell["27"] * self.Nell["39"] - self.Nell["27x39"] ** 2)
            / self.Nell["27"]
        )

        L33     = np.zeros(self.lmax, dtype=float)
        L33[2:] = np.sqrt(self.Nell["93"])
        L43     = np.zeros(self.lmax, dtype=float)
        L43[2:] = self.Nell["93x145"] / np.sqrt(self.Nell["93"])
        L44     = np.zeros(self.lmax, dtype=float)
        L44[2:] = np.sqrt(
            (self.Nell["93"] * self.Nell["145"] - self.Nell["93x145"] ** 2)
            / self.Nell["93"]
        )

        L55     = np.zeros(self.lmax, dtype=float)
        L55[2:] = np.sqrt(self.Nell["225"])
        L65     = np.zeros(self.lmax, dtype=float)
        L65[2:] = self.Nell["225x280"] / np.sqrt(self.Nell["225"])
        L66     = np.zeros(self.lmax, dtype=float)
        L66[2:] = np.sqrt(
            (self.Nell["225"] * self.Nell["280"] - self.Nell["225x280"] ** 2)
            / self.Nell["225"]
        )

        return L11, L21, L22, L33, L43, L44, L55, L65, L66


    def __white_noise__(self, band):
        n = hp.synfast(np.concatenate((np.zeros(2), self.Nell[band])), self.nside, lmax=self.lmax, pixwin=False)
        return n
        
    
    def white_noise_maps(self) -> np.ndarray: 
        return np.array([
                    self.__white_noise__('27'),
                    self.__white_noise__('39'),
                    self.__white_noise__('93'),
                    self.__white_noise__('145'),
                    self.__white_noise__('225'),
                    self.__white_noise__('280')])
    
    def white_noise_maps_freq(self, freq: str) -> np.ndarray:
        """
        Generates white noise maps for a specific frequency band.

        Returns:
        np.ndarray: An array of white noise maps for the specified frequency band.
        """
        band = freq.split('-')[0]
        n = self.__white_noise__(band)
        return n
    
    def atm_noise_maps_freq(self, idx: int, freq: str) -> np.ndarray:
        """
        Generates atmospheric noise maps using Cholesky decomposition.

        Returns:
        np.ndarray: An array of atmospheric noise maps for different frequency bands.
        """
        L11, L21, L22, L33, L43, L44, L55, L65, L66 = self.cholesky_matrix_elements 

        f = freq.split('-')[0]
        split = int(freq.split('-')[-1])

        seed_idx = self.__get_noise_seed_idx__(idx)

        def noise_map(Alm,L):
            nlm = hp.almxfl(Alm, L)
            n = hp.alm2map(nlm, self.nside, pixwin=False)
            return n
        def noise_map_cross(Alm1,Alm2,L1,L2):
            nlm = hp.almxfl(Alm1, L1) + hp.almxfl(Alm2, L2)
            n = hp.alm2map(nlm, self.nside, pixwin=False)
            return n
        
        if f == '27':
            np.random.seed(self.__nseeds__[split][seed_idx])
            alm = self.rand_alm
            return noise_map(alm, L11)*np.sqrt(self.nsplits)
        elif f == '39':
            np.random.seed(self.__nseeds__[split][seed_idx])
            alm = self.rand_alm
            np.random.seed(self.__nseeds__[split][seed_idx]+1)
            blm = self.rand_alm
            return noise_map_cross(alm,blm,L21,L22)*np.sqrt(self.nsplits)
        elif f == '93':
            np.random.seed(self.__nseeds__[split][seed_idx]+2)
            clm = self.rand_alm
            return noise_map(clm,L33)*np.sqrt(self.nsplits)
        elif f == '145':
            np.random.seed(self.__nseeds__[split][seed_idx]+2)
            clm = self.rand_alm
            np.random.seed(self.__nseeds__[split][seed_idx]+3)
            dlm = self.rand_alm
            return noise_map_cross(clm,dlm,L43,L44)*np.sqrt(self.nsplits)
        elif f == '225':
            np.random.seed(self.__nseeds__[split][seed_idx]+4)
            elm = self.rand_alm
            return noise_map(elm,L55)*np.sqrt(self.nsplits)
        elif f == '280':
            np.random.seed(self.__nseeds__[split][seed_idx]+4)
            elm = self.rand_alm
            np.random.seed(self.__nseeds__[split][seed_idx]+5)
            flm = self.rand_alm
            return noise_map_cross(elm,flm,L65,L66)*np.sqrt(self.nsplits)
        else:
            raise ValueError(f"Invalid frequency band: {f}", "Choose from '27', '39', '93', '145', '225', '280'.")
            


    def atm_noise_maps(self,split, idx) -> np.ndarray:
        """
        Generates atmospheric noise maps using Cholesky decomposition.

        Returns:
        np.ndarray: An array of atmospheric noise maps for different frequency bands.
        """
        L11, L21, L22, L33, L43, L44, L55, L65, L66 = self.cholesky_matrix_elements
        
        seed_idx = self.__get_noise_seed_idx__(idx)
        
        np.random.seed(self.__nseeds__[split][seed_idx])
        alm    = self.rand_alm
        np.random.seed(self.__nseeds__[split][seed_idx]+1)
        blm    = self.rand_alm
        nlm_27 = hp.almxfl(alm, L11)
        nlm_39 = hp.almxfl(alm, L21) + hp.almxfl(blm, L22)
        n_27   = hp.alm2map(nlm_27, self.nside, pixwin=False)
        n_39   = hp.alm2map(nlm_39, self.nside, pixwin=False)
        
        np.random.seed(self.__nseeds__[split][seed_idx]+2)
        clm     = self.rand_alm
        np.random.seed(self.__nseeds__[split][seed_idx]+3)
        dlm     = self.rand_alm
        nlm_93  = hp.almxfl(clm, L33)
        nlm_145 = hp.almxfl(clm, L43) + hp.almxfl(dlm, L44)
        n_93    = hp.alm2map(nlm_93, self.nside, pixwin=False)
        n_145   = hp.alm2map(nlm_145, self.nside, pixwin=False)
        
        np.random.seed(self.__nseeds__[split][seed_idx]+4)
        elm     = self.rand_alm
        np.random.seed(self.__nseeds__[split][seed_idx]+5)
        flm     = self.rand_alm
        nlm_225 = hp.almxfl(elm, L55)
        nlm_280 = hp.almxfl(elm, L65) + hp.almxfl(flm, L66)
        n_225   = hp.alm2map(nlm_225, self.nside, pixwin=False)
        n_280   = hp.alm2map(nlm_280, self.nside, pixwin=False)

        n = np.array([n_27, n_39, n_93, n_145, n_225, n_280])
        return n

    def noiseQU_NC(self,idx) -> np.ndarray:
        """
        Generates Q and U polarization noise maps based on the noise model.

        Returns:
        np.ndarray: An array of Q and U noise maps.
        """
        N = []
        for split in range(self.nsplits):
            if self.atm_noise:
                # TODO: q and u are the same atm noise maps
                q = self.atm_noise_maps(split+1, idx)
                u = self.atm_noise_maps(split+1, idx)
            else:
                q = self.white_noise_maps()
                u = self.white_noise_maps()            
              
            for i in range(len(q)):
                N.append([q[i], u[i]])

        return np.array(N)*np.sqrt(self.nsplits)
    
    def noiseQU_NC_freq(self, idx: int, freq: str) -> np.ndarray:
        """
        Generates Q and U polarization noise maps for a specific frequency band.

        Returns:
        np.ndarray: An array of Q and U noise maps for the specified frequency band.
        """
        if self.atm_noise:
            q = self.atm_noise_maps_freq(idx, freq)
            u = q
        else:
            q = self.white_noise_maps_freq(freq)
            u = q

        return np.array([q, u])
    
    def noiseQU_TOD_sat(self,idx: int) -> np.ndarray:
        sim_nsplits = 4
        fac = np.sqrt(self.nsplits) / np.sqrt(sim_nsplits)
        sim_no = f'{idx:04}'
        fdir = f'/global/cfs/cdirs/sobs/awg_bb/sims/BBSims/NOISE_20230531/goal_optimistic/{sim_no}'
        fbase_template = 'SO_SAT_{band}_noise_split_{split}of4_{sim_no}_goal_optimistic_20230531.fits'

        bands = ['27', '39', '93', '145', '225', '280']
        N = []
        for split in range(self.nsplits):
            for b in bands:
                fbase = fbase_template.format(band=b, split=split+1, sim_no=sim_no)
                fpath = f'{fdir}/{fbase}'
                mm = hp.read_map(fpath, field=(1, 2))
                mm = change_coord(mm)
                N.append([mm[0],mm[1]])
        return np.array(N)*fac

    def noiseQU_TOD_lat(self,idx: int) -> np.ndarray:
        sim_nsplits = 4
        fdir = '/global/cfs/cdirs/sobs/v4_sims/mbs/mbs_s0015_20240504/sims'
        fbase_template = 'so_lat_mbs_mss0002_{model}_{band}_lmax5400_4way_set{split_num}_noise_sim_map{sim_num:04}.fits'

        models = ['fdw_lf', 'fdw_mf', 'fdw_uhf']
        bands = ['lf_f030_lf_f040', 'mf_f090_mf_f150', 'uhf_f230_uhf_f290']
        
        if self.ext_sims:
            # For extended sims: treat each available split as independent
            # nsplits=1: idx=0 -> sim=0,split=1; idx=1 -> sim=0,split=2; etc. (1200 total sims)
            # nsplits=2: idx=0 -> sim=0,splits=1,2; idx=1 -> sim=0,splits=3,4; idx=2 -> sim=1,splits=1,2; etc. (600 total sims)
            # nsplits=3: idx=0 -> sim=0,splits=1,2,3; idx=1 -> sim=0,splits=4,1,2 (wraps); etc. (400 total sims)
            # nsplits=4: idx=0 -> sim=0,splits=1,2,3,4; idx=1 -> sim=1,splits=1,2,3,4; etc. (300 total sims)
            
            sims_per_original = sim_nsplits // self.nsplits  # how many extended sims per original sim
            sim_num = idx // sims_per_original
            split_group = idx % sims_per_original
            
            fac = np.sqrt(self.nsplits) / np.sqrt(sim_nsplits)
            
            N = []
            for split in range(self.nsplits):
                actual_split = split_group * self.nsplits + split  # 0-indexed (set0, set1, set2, set3)
                for i in range(len(bands)):
                    fbase = fbase_template.format(model=models[i], band=bands[i], split_num=actual_split, sim_num=sim_num)
                    fpath = f'{fdir}/{fbase}'
                    for j in range(2):
                        n = enmap.read_map(fpath,sel=np.s_[j, 0, 1:]) # the 1: will select the QU fields
                        mm = map2healpix(n, nside=2048, rot='equ,gal', spin=2)
                        del n
                        N.append([mm[0],mm[1]])
        else:
            # Original behavior
            fac = np.sqrt(self.nsplits) / np.sqrt(sim_nsplits)
            N = []
            for split in range(self.nsplits):
                for i in range(len(bands)):
                    fbase = fbase_template.format(model=models[i], band=bands[i], split_num=split, sim_num=idx)
                    fpath = f'{fdir}/{fbase}'
                    for j in range(2):
                        n = enmap.read_map(fpath,sel=np.s_[j, 0, 1:]) # the 1: will select the QU fields
                        mm = map2healpix(n, nside=2048, rot='equ,gal', spin=2)
                        del n
                        N.append([mm[0],mm[1]])
        
        return np.array(N)*fac

    def noiseQU_TOD_sat_band(self, idx: int, freq: str) -> np.ndarray:
        band = freq.split('-')[0]
        split = int(freq.split('-')[-1])
        sim_nsplits = 4
        fac = np.sqrt(self.nsplits) / np.sqrt(sim_nsplits)
        sim_no = f'{idx:04}'
        fdir = f'/global/cfs/cdirs/sobs/awg_bb/sims/BBSims/NOISE_20230531/goal_optimistic/{sim_no}'
        fbase = f'SO_SAT_{band}_noise_split_{split}of4_{sim_no}_goal_optimistic_20230531.fits'
        fpath = f'{fdir}/{fbase}'
        mm = hp.read_map(fpath, field=(1, 2)) #type: ignore
        mm = change_coord(mm)
        N = np.array([mm[0],mm[1]])
        return N*fac
    
    def noiseQU_TOD_lat_band(self, idx: int, freq: str, debug: bool = False) -> np.ndarray:
        band = freq.split('-')[0]
        split = int(freq.split('-')[-1])  # 1-indexed from user input
        sim_nsplits = 4
        fdir = '/global/cfs/cdirs/sobs/v4_sims/mbs/mbs_s0015_20240504/sims'
        
        # Validate split number
        if split < 1 or split > self.nsplits:
            raise ValueError(f"Invalid split={split}. Must be between 1 and {self.nsplits} for nsplits={self.nsplits}")

        model, band_name, j = {"27":['fdw_lf', 'lf_f030_lf_f040',0], 
                               "39":['fdw_lf', 'lf_f030_lf_f040',1], 
                               "93":['fdw_mf', 'mf_f090_mf_f150',0], 
                               "145":['fdw_mf', 'mf_f090_mf_f150',1], 
                               "225":['fdw_uhf', 'uhf_f230_uhf_f290',0], 
                               "280":['fdw_uhf', 'uhf_f230_uhf_f290',1]}[band]
        
        if self.ext_sims:
            # Apply ext_sims logic: treat each split as independent
            # For nsplits=1: each idx gets one split from the 4 available
            # For nsplits=2: each idx gets two consecutive splits
            sims_per_original = sim_nsplits // self.nsplits
            sim_num = idx // sims_per_original
            split_group = idx % sims_per_original
            
            # Map the requested split (1-indexed from freq like '93-1') to the actual file split
            # The user's split number refers to which of the nsplits they want (1 to nsplits)
            # We need to map this to the actual file split based on split_group
            # Files use 0-indexed naming (set0, set1, set2, set3)
            actual_split = split_group * self.nsplits + (split - 1)  # Convert to 0-indexed
            
            fac = np.sqrt(self.nsplits) / np.sqrt(sim_nsplits)
            fbase = f'so_lat_mbs_mss0002_{model}_{band_name}_lmax5400_4way_set{actual_split}_noise_sim_map{sim_num:04}.fits'
            
            if debug:
                self.logger.log(f"[EXT_SIMS] idx={idx}, freq={freq} -> file_idx={sim_num:04}, file_split={actual_split}")
                return None
        else:
            # Original behavior with sim_config support
            seed_idx = self.__get_noise_seed_idx__(idx)
            fac = np.sqrt(self.nsplits) / np.sqrt(sim_nsplits)
            
            if self.nsplits > 2:
                fbase = f'so_lat_mbs_mss0002_{model}_{band_name}_lmax5400_4way_set{split-1}_noise_sim_map{seed_idx:04}.fits'
            else:
                fbase = f'so_lat_mbs_mss0002_{model}_{band_name}_lmax5400_2way_set{split}_noise_sim_map{seed_idx:04}.fits'
        
        fpath = f'{fdir}/{fbase}'

        n = enmap.read_map(fpath,sel=np.s_[j, 0])
        mm = map2healpix(n, nside=2048)[1:]
        mm = change_coord(mm)
        del n
        N = np.array([mm[0],mm[1]])
        return N*fac




    def noiseQU_TOD(self, idx: int) -> np.ndarray:
        if self.telescope == 'SAT':
            return self.noiseQU_TOD_sat(idx)
        elif self.telescope == 'LAT':
            return self.noiseQU_TOD_lat(idx)
        else:
            raise ValueError(f"Invalid telescope: {self.telescope}", "Choose from 'LAT' or 'SAT'.")
        
    def noiseQU_TOD_freq(self, idx: int, freq: str) -> np.ndarray:
        if self.telescope == 'SAT':
            return self.noiseQU_TOD_sat_band(idx, freq)
        elif self.telescope == 'LAT':
            return self.noiseQU_TOD_lat_band(idx, freq)
        else:
            raise ValueError(f"Invalid telescope: {self.telescope}", "Choose from 'LAT' or 'SAT'.")

    
    def noiseQU(self, idx: Optional[int] = None) -> np.ndarray:
        """
        Generates Q and U polarization noise maps based on the noise model.

        Returns:
        np.ndarray: An array of Q and U noise maps.
        """
        if self.sim == 'NC':
            return self.noiseQU_NC(idx)
        elif self.sim == 'TOD':
            assert idx is not None, "For TOD simulation, provide the index of the simulation."
            return self.noiseQU_TOD(idx)
        else:
            raise ValueError(f"Invalid simulation type: {self.sim}", "Choose from 'NC' or 'TOD'.")
        
    def noiseQU_freq(self, idx: int, freq: str) -> np.ndarray:
        """
        Generates Q and U polarization noise maps for a specific frequency band.

        Returns:
        np.ndarray: An array of Q and U noise maps for the specified frequency band.
        """
        if self.sim == 'NC':
            return self.noiseQU_NC_freq(idx, freq)
        elif self.sim == 'TOD':
            return self.noiseQU_TOD_freq(idx, freq)
        else:
            raise ValueError(f"Invalid simulation type: {self.sim}", "Choose from 'NC' or 'TOD'.")
        
        