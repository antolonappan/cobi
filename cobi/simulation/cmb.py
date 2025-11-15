"""
CMB Simulation Module
=====================

This module provides comprehensive functionality for simulating Cosmic Microwave 
Background (CMB) temperature and polarization maps with support for:

- Lensed and unlensed CMB realizations
- Cosmic birefringence effects (isotropic and anisotropic)
- Custom power spectra from CAMB
- Auxiliary scalar fields (for patchy τ models)
- Beam convolution and coordinate transformations

Classes
-------
CMB
    Main class for generating CMB maps with optional birefringence rotation.

Functions
---------
synfast_pol
    Generate polarized CMB maps with optional auxiliary fields.
hp_alm2map_spin
    Fast spin-weighted spherical harmonic transforms.
get_camb_cls
    Compute CMB power spectra using CAMB.

Example
-------
Generate a CMB realization with isotropic birefringence::

    from cobi.simulation import CMB
    
    cmb = CMB(
        nside=512,
        lmax=1500,
        beta=0.35,  # isotropic rotation angle in degrees
        lensing=True,
        libdir='./cmb_sims'
    )
    
    # Get rotated CMB map
    cmb_map = cmb.get_map(idx=0, apply_beta=True)

Notes
-----
The module uses lenspyx for efficient lensing and supports MPI parallelization
for generating large simulation sets.
"""

# General imports
import os
import camb
import numpy as np
import pickle as pl
import healpy as hp
from typing import Dict, Optional, Any, Union, List
import lenspyx
# Local imports
from cobi import mpi
from cobi.utils import Logger, inrad
from cobi.data import CAMB_INI, SPECTRA, ISO_TD_SPECTRA

def synfast_pol(nside, spectra):
    """
    Generate polarized CMB maps (Q, U) with optional auxiliary field A.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    spectra : dict
        Power spectra dictionary with required 'EE' (and optional 'BB', 'AA', 'AE').
    
    Returns
    -------
    (Q_map, U_map) or (Q_map, U_map, A_map) if 'AA' in spectra
    """
    assert 'EE' in spectra, "The 'EE' spectrum must be provided."
    assert 'AA' in spectra, "The 'AA' spectrum must be provided."

    lmax = min(len(spectra['EE']) - 1, 3*nside -1)
    epsilon = 1e-20

    EE = spectra['EE'][:lmax+1]
    BB = spectra.get('BB', np.zeros(lmax+1))[:lmax+1]
    AE = spectra.get('AE', np.zeros(lmax+1))[:lmax+1]
    AA = spectra['AA'][:lmax+1]
    cov = np.empty((lmax+1, 2, 2))
    cov[:, 0, 0] = EE
    cov[:, 0, 1] = AE
    cov[:, 1, 0] = AE
    cov[:, 1, 1] = AA
    nfields = 2

    cov += epsilon * np.eye(nfields)[None, :, :]

    # Cholesky decomposition
    L_chol = np.empty_like(cov)
    for l in range(lmax+1):
        extra_eps = 0.0
        while True:
            try:
                L_chol[l] = np.linalg.cholesky(cov[l] + extra_eps * np.eye(nfields))
                if extra_eps > 0:
                    print(f"Warning: Increased epsilon for l={l} to {extra_eps}")
                break
            except np.linalg.LinAlgError:
                extra_eps = extra_eps * 10 if extra_eps > 0 else epsilon

    # Allocate alm arrays
    num_alm = hp.sphtfunc.Alm.getsize(lmax)
    alm_E = np.zeros(num_alm, dtype=complex)
    alm_B = np.zeros(num_alm, dtype=complex)
    alm_A = np.zeros(num_alm, dtype=complex)

    l_arr, m_arr = hp.Alm.getlm(lmax)

    # m = 0
    m0 = (m_arr == 0)
    l0 = l_arr[m0]
    z0 = np.random.normal(0.0, 1.0, size=(m0.sum(), 2))
    L0 = L_chol[l0, :, :]
    a_EA0 = np.einsum('ijk,ik->ij', L0, z0)
    alm_E[m0] = a_EA0[:, 0]
    alm_A[m0] = a_EA0[:, 1]

    # m > 0
    mpos = (m_arr > 0)
    lpos = l_arr[mpos]
    Npos = mpos.sum()
    z_real = np.random.normal(0.0, 1.0, size=(Npos, 2)) / np.sqrt(2)
    z_imag = np.random.normal(0.0, 1.0, size=(Npos, 2)) / np.sqrt(2)
    L_pos = L_chol[lpos, :, :]
    a_EA_real = np.einsum('ijk,ik->ij', L_pos, z_real)
    a_EA_imag = np.einsum('ijk,ik->ij', L_pos, z_imag)
    alm_E[mpos] = a_EA_real[:, 0] + 1j * a_EA_imag[:, 0]
    alm_A[mpos] = a_EA_real[:, 1] + 1j * a_EA_imag[:, 1]

    # B-modes
    b0 = np.random.normal(0.0, 1.0, size=m0.sum())
    alm_B[m0] = b0 * np.sqrt(BB[l_arr[m0]])
    b_real = np.random.normal(0.0, 1.0, size=Npos) / np.sqrt(2)
    b_imag = np.random.normal(0.0, 1.0, size=Npos) / np.sqrt(2)
    alm_B[mpos] = (b_real + 1j * b_imag) * np.sqrt(BB[l_arr[mpos]])

    # Maps
    Q_map, U_map = hp.alm2map_spin([alm_E, alm_B], nside, spin=2, lmax=lmax)
    A_map = hp.alm2map(alm_A, nside, lmax=lmax)

    # Always apply birefringence-like rotation
    Q_rot = Q_map * np.cos(2 * A_map) - U_map * np.sin(2 * A_map)
    U_rot = Q_map * np.sin(2 * A_map) + U_map * np.cos(2 * A_map)
    Q_map, U_map = Q_rot, U_rot

    return Q_map, U_map, A_map

class CMB:
    """
    CMB map generator with cosmic birefringence support.
    
    This class handles the generation of CMB temperature and polarization maps
    with optional cosmic birefringence rotation. Supports three birefringence models:
    
    - **iso**: Isotropic (constant) rotation angle β
    - **iso_td**: Time-dependent isotropic model with axion mass parameter
    - **aniso**: Anisotropic (spatially-varying) rotation from patchy reionization
    
    Parameters
    ----------
    libdir : str
        Directory for caching simulation products and power spectra.
    nside : int
        HEALPix resolution parameter (nside = 2^n).
    model : {'iso', 'iso_td', 'aniso'}, default='iso'
        Birefringence model type.
    beta : float, optional
        Isotropic rotation angle in degrees (required for model='iso').
    mass : float, optional
        Axion mass parameter in units of 10^-22 eV. 
        Must be 1, 1.5, or 6.4 (required for model='iso_td').
    Acb : float, optional
        Amplitude of anisotropic birefringence power spectrum
        (required for model='aniso').
    lensing : bool, default=False
        Whether to include CMB lensing effects.
    Acb_sim_config : dict, optional
        Configuration for anisotropic simulation batches with keys:
        - 'alpha_vary_index': [start, end] indices for varying α
        - 'alpha_cons_index': [start, end] indices for constant α  
        - 'null_alpha_index': [start, end] indices for null test (α=0)
    verbose : bool, default=True
        Enable logging output.
    
    Attributes
    ----------
    nside : int
        HEALPix resolution.
    lmax : int
        Maximum multipole (3*nside - 1).
    model : str
        Birefringence model type.
    beta : float or None
        Isotropic rotation angle in degrees.
    lensing : bool
        Whether lensing is enabled.
    cls : dict
        CMB power spectra (TT, EE, BB, TE).
    
    Examples
    --------
    Generate isotropic birefringence simulation::
    
        cmb = CMB(
            libdir='./cmb_sims',
            nside=512,
            model='iso',
            beta=0.35,
            lensing=True
        )
        
        # Get rotated CMB map
        cmb_map = cmb.get_map(idx=0, apply_beta=True)
    
    Generate anisotropic birefringence::
    
        cmb = CMB(
            libdir='./cmb_sims',
            nside=512,
            model='aniso',
            Acb=1e-6,
            lensing=True
        )
        
        # Get map with spatially-varying rotation
        cmb_map = cmb.get_map(idx=0, apply_beta=True)
    
    Notes
    -----
    - Uses CAMB for power spectrum generation
    - Supports efficient caching of realizations
    - MPI-aware for parallel simulation generation
    - Lensing implemented via lenspyx
    """

    def __init__(
        self,
        libdir: str,
        nside: int,
        model: str = "iso",
        beta: Optional[float]=None,
        mass: Optional[float]=None,
        Acb: Optional[float]=None,
        lensing: Optional[bool] = False,
        Acb_sim_config: Optional[Dict[str, List[int]]] = None,
        verbose: Optional[bool] = True,
    ):
        self.logger = Logger(self.__class__.__name__, verbose=verbose if verbose is not None else False)
        self.basedir = libdir

        self.nside  = nside
        self.lmax   = 3 * nside - 1
        
        self.__set_power__()
        
        self.beta   = beta
        self.mass   = mass
        self.Acb    = Acb
        assert model in ["iso", "iso_td","aniso"], "model should be 'iso' or 'aniso'"
        self.model  = model
        if self.model == "aniso":
            self.logger.log("Anisotropic cosmic birefringence model selected", level="info")
            assert self.Acb is not None, "Acb should be provided for anisotropic model"
        if self.model == "iso":
            self.logger.log("Isotropic(constant) cosmic birefringence model selected", level="info")
            assert self.beta is not None, "beta should be provided for isotropic model"
        if self.model == "iso_td":
            assert self.mass in [1,1.5,6.4], "mass should be 1, 1.59 or 6.36"
            self.logger.log("Isotropic(time dep.) cosmic birefringence model selected", level="info")

        
        self.lensing = lensing
        self.Acb_sim_config = Acb_sim_config
        self.__validate_Acb_sim_config__()

        self.__set_workspace__()
        self.__set_seeds__()
        self.verbose = verbose if verbose is not None else False
    
    def scale_invariant(self, Acb):
        ells = np.arange(self.lmax + 1)
        cl =  Acb * 2 * np.pi / ( ells**2 + ells + 1e-30)
        cl[0], cl[1] = 0, 0
        return cl
   
    def __validate_Acb_sim_config__(self) -> None:
        """
        Validate the Acb_sim_config dictionary structure.
        """
        if self.Acb_sim_config is None:
            return
        
        if self.model != "aniso":
            self.logger.log("Acb_sim_config is only applicable for anisotropic model. Ignoring.", level="warning")
            self.Acb_sim_config = None
            return
        
        valid_keys = ['alpha_vary_index', 'alpha_cons_index', 'null_alpha_index']
        for key in self.Acb_sim_config.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid key '{key}' in Acb_sim_config. Valid keys are: {valid_keys}")
            if not isinstance(self.Acb_sim_config[key], (list, tuple)) or len(self.Acb_sim_config[key]) != 2:
                raise ValueError(f"Value for '{key}' must be a list/tuple of two integers [start, end]")
            if self.Acb_sim_config[key][0] >= self.Acb_sim_config[key][1]:
                raise ValueError(f"Invalid range for '{key}': start must be less than end")
        
        self.logger.log(f"Acb simulation configuration validated: {self.Acb_sim_config}", level="info")
    
    def __get_alpha_mode__(self, idx: int) -> str:
        """
        Determine which alpha mode to use for a given index.
        
        Parameters:
        idx (int): The realization index.
        
        Returns:
        str: One of 'vary', 'constant', or 'null'
        """
        if self.Acb_sim_config is None:
            return 'vary'  # Default behavior
        
        for key, (start, end) in self.Acb_sim_config.items():
            if start <= idx < end:
                if key == 'alpha_vary_index':
                    return 'vary'
                elif key == 'alpha_cons_index':
                    return 'constant'
                elif key == 'null_alpha_index':
                    return 'null'
        
        # If idx is not in any configured range, use default
        return 'vary'

    def __set_seeds__(self) -> None:
        """
        Sets the seeds for the simulation.
        """
        nos = 700
        self.__cseeds__ = np.arange(11111,11111+nos, dtype=int)
        self.__aseeds__ = np.arange(22222,22222+nos, dtype=int)
        self.__pseeds__ = np.arange(33333,33333+nos, dtype=int)
    
    def __set_workspace__(self) -> None:
        """
        Set the workspace for the CMB simulations.
        """
        lens = "lensed" if self.lensing else "gaussian"
        if self.model == "iso":
            model = f"iso_beta_{str(self.beta).replace('.','p')}"
        elif self.model == "iso_td":
            model = f"iso_mass_{str(self.mass).replace('.','p')}"
        elif self.model == "aniso":
            model = f"aniso_alpha_{str(self.Acb)}"
        else:
            raise ValueError("Model not implemented yet, only 'iso', 'iso_td', and 'aniso' are supported")


        self.cmbdir = os.path.join(self.basedir, 'CMB', lens, model, 'cmb')
        os.makedirs(self.cmbdir, exist_ok=True)
        if self.lensing:
            self.phidir = os.path.join(self.basedir, 'CMB', lens, model, 'phi')
            os.makedirs(self.phidir, exist_ok=True)
        if self.model == "aniso":
            self.alphadir = os.path.join(self.basedir, 'CMB', lens, model, 'alpha')
            os.makedirs(self.alphadir, exist_ok=True)
    
    def __dl2cl__(self, arr: np.ndarray,unit_only=False) -> np.ndarray:
        """
        Convert Dl to Cl.
        """
        tcmb = 2.7255e6
        l = np.arange(len(arr))
        dl = l * (l + 1) / (2 * np.pi)
        if unit_only:
            return arr * tcmb ** 2 
        else:
            arr = arr * tcmb ** 2 / (dl + 1e-30)
        arr[0] = 0
        arr[1] = 0
        return arr
    
    def __td_eb__(self,dl=True) -> np.ndarray:
        """
        Read the E and B mode power spectra from the CMB power spectra data.
        """
        ISO_TD_SPECTRA.directory = self.basedir
        m = str(self.mass).replace('.','p')
        spectra = ISO_TD_SPECTRA.data[m]
        eb = np.zeros(self.lmax + 1)
        eb[2:] = spectra[:self.lmax + 1 - 2, 5]
        return self.__dl2cl__(eb,unit_only= dl)  # type: ignore

    def __td_tb__(self,dl=True) -> np.ndarray:
        ISO_TD_SPECTRA.directory = self.basedir
        m = str(self.mass).replace('.', 'p')
        spectra = ISO_TD_SPECTRA.data[m]
        tb = np.zeros(self.lmax + 1)
        tb[2:] = spectra[:self.lmax + 1 - 2, 6]
        return self.__dl2cl__(tb,unit_only= dl)      
        
    def compute_powers(self) -> Dict[str, Any]:
        """
        compute the CMB power spectra using CAMB.
        """
        CAMB_INI.directory = self.basedir
        params   = CAMB_INI.data
        results  = camb.get_results(params)
        powers   = {}
        powers["cls"] = results.get_cmb_power_spectra(
            params, CMB_unit="muK", raw_cl=True
        )
        powers["dls"] = results.get_cmb_power_spectra(
            params, CMB_unit="muK", raw_cl=False
        )
        if mpi.rank == 0:
            pl.dump(powers, open(self.__spectra_file__, "wb"))
        mpi.barrier()
        return powers

    def get_power(self, dl: bool = True) -> Dict[str, np.ndarray]:
        """
        Get the CMB power spectra.

        Parameters:
        dl (bool): If True, return the power spectra with dl factor else without dl factor.

        Returns:
        Dict[str, np.ndarray]: A dictionary containing the CMB power spectra.
        """
        return self.powers["dls"] if dl else self.powers["cls"]
    
    def __set_power__(self) -> None:
        SPECTRA.directory = self.basedir
        self.__spectra_file__ = SPECTRA.fname
        if os.path.isfile(self.__spectra_file__):
            self.logger.log("Loading CMB power spectra from file", level="info")
            self.powers = pl.load(open(self.__spectra_file__, "rb"))
        else:
            self.powers = SPECTRA.data
            lmax_infile = len(self.powers['cls']['lensed_scalar'][:, 0])
            if lmax_infile < self.lmax:
                self.logger.log("CMB power spectra file does not contain enough data", level="warning")
                self.logger.log("Computing CMB power spectra", level="info")
                self.powers = self.compute_powers()
                #TODO: feed the lmax to the compute_powers method
                self.logger.log("CMB power spectra computed doesn't guarantee the lmax", level="critical")
            else:
                self.logger.log("CMB power spectra file is up-to-date", level="info")
    
    ################ Power spectra retrieval methods ###############

    def get_unlensed_spectra(
        self, dl: bool = True, dtype: str = "d"
    ) -> Union[Dict[str, Any], np.ndarray]:
        """
        Retrieve the unlensed scalar spectra from the power spectrum data.

        Parameters:
        dl (bool, optional): If True, returns Dl (C_l * l * (l + 1) / 2π). Defaults to True.
        dtype (str, optional): Specifies the format of the returned spectra.
                               - 'd' returns a dictionary with keys 'tt', 'ee', 'bb', 'te'.
                               - 'a' returns the full array of power spectra.
                               Defaults to 'd'.

        Returns:
        Union[Dict[str, Any], np.ndarray]:
            - A dictionary containing individual power spectra for 'tt', 'ee', 'bb', 'te' if dtype is 'd'.
            - The full array of unlensed scalar power spectra if dtype is 'a'.

        Raises:
        ValueError: If `dtype` is not 'd' or 'a'.
        """
        powers = self.get_power(dl)["unlensed_scalar"]
        if dtype == "d":
            pow = {}
            pow["tt"] = powers[:, 0]
            pow["ee"] = powers[:, 1]
            pow["bb"] = powers[:, 2]
            pow["te"] = powers[:, 3]
            return pow
        elif dtype == "a":
            return powers
        else:
            raise ValueError("dtype should be 'd' or 'a'")
     
    def get_lensed_spectra(
        self, dl: bool = True, dtype: str = "d"
    ) -> Union[Dict[str, Any], np.ndarray]:
        """
        Retrieve the lensed scalar spectra from the power spectrum data.

        Parameters:
        dl (bool, optional): If True, returns Dl (C_l * l * (l + 1) / 2π). Defaults to True.
        dtype (str, optional): Specifies the format of the returned spectra.
                               - 'd' returns a dictionary with keys 'tt', 'ee', 'bb', 'te'.
                               - 'a' returns the full array of power spectra.
                               Defaults to 'd'.

        Returns:
        Union[Dict[str, Any], np.ndarray]:
            - A dictionary containing individual power spectra for 'tt', 'ee', 'bb', 'te' if dtype is 'd'.
            - The full array of lensed scalar power spectra if dtype is 'a'.

        Raises:
        ValueError: If `dtype` is not 'd' or 'a'.
        """
        powers = self.get_power(dl)["lensed_scalar"]
        if dtype == "d":
            pow = {}
            pow["tt"] = powers[:, 0]
            pow["ee"] = powers[:, 1]
            pow["bb"] = powers[:, 2]
            pow["te"] = powers[:, 3]
            return pow
        elif dtype == "a":
            return powers
        else:
            raise ValueError("dtype should be 'd' or 'a'")
     
    def get_cb_unlensed_spectra(
        self, beta: float = 0.0, dl: bool = True, dtype: str = "d", new: bool = False
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Calculate the cosmic birefringence (CB) unlensed spectra with a given rotation angle `beta`.

        Parameters:
        beta (float, optional): The rotation angle in degrees for the cosmic birefringence effect. Defaults to 0.3 degrees.
        dl (bool, optional): If True, returns Dl (C_l * l * (l + 1) / 2π). Defaults to True.
        dtype (str, optional): Specifies the format of the returned spectra.
                               - 'd' returns a dictionary with keys 'tt', 'ee', 'bb', 'te', 'eb', 'tb'.
                               - 'a' returns an array of power spectra.
                               Defaults to 'd'.
        new (bool, optional): Determines the ordering of the spectra in the array if dtype is 'a'.
                              If True, returns [TT, EE, BB, TE, EB, TB].
                              If False, returns [TT, TE, TB, EE, EB, BB].
                              Defaults to False.

        Returns:
        Union[Dict[str, np.ndarray], np.ndarray]:
            - A dictionary containing individual CB unlensed power spectra for 'tt', 'ee', 'bb', 'te', 'eb', 'tb' if dtype is 'd'.
            - An array of CB unlensed power spectra with ordering determined by the `new` parameter if dtype is 'a'.

        Raises:
        ValueError: If `dtype` is not 'd' or 'a'.

        Notes:
        The method applies a rotation by `alpha` degrees to the E and B mode spectra to account for cosmic birefringence.
        """
        powers = self.get_unlensed_spectra(dl=dl) 
        pow = {}
        pow["tt"] = powers["tt"]
        pow["te"] = powers["te"] * np.cos(2 * inrad(beta))
        pow["ee"] = (powers["ee"] * np.cos(inrad(2 * beta)) ** 2) + (powers["bb"] * np.sin(inrad(2 * beta)) ** 2)
        pow["bb"] = (powers["ee"] * np.sin(inrad(2 * beta)) ** 2) + (powers["bb"] * np.cos(inrad(2 * beta)) ** 2)
        pow["eb"] = 0.5 * (powers["ee"] - powers["bb"]) * np.sin(inrad(4 * beta))
        pow["tb"] = powers["te"] * np.sin(2 * inrad(beta))
        if dtype == "d":
            return pow
        elif dtype == "a":
            if new:
                # TT, EE, BB, TE, EB, TB
                return np.array(
                    [pow["tt"], pow["ee"], pow["bb"], pow["te"], pow["eb"], pow["tb"]]
                )
            else:
                # TT, TE, TB, EE, EB, BB
                return np.array(
                    [pow["tt"], pow["te"], pow["tb"], pow["ee"], pow["eb"], pow["bb"]]
                )
        else:
            raise ValueError("dtype should be 'd' or 'a'")
    
    def get_cb_unlensed_mass_spectra(
            self, dl: bool = True, dtype: str = "d", new: bool = False
        ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        powers = self.get_unlensed_spectra(dl=dl) 
        pow = {}
        pow["tt"] = powers["tt"]
        pow["te"] = powers["te"]
        pow["ee"] = powers["ee"]
        pow["bb"] = powers["bb"]
        internal_len = len(powers["tt"])
        _eb_ = self.__td_eb__(dl=dl)
        _tb_ = self.__td_tb__(dl=dl)
        eb = np.zeros(len(powers["tt"]))
        tb = np.zeros(len(powers["tt"]))
        eb[2:len(_eb_)] = _eb_[2:]
        tb[2:len(_eb_)] = _tb_[2:]
        pow["eb"] = eb
        pow["tb"] = tb

        if dtype == "d":
            return pow
        elif dtype == "a":
            if new:
                # TT, EE, BB, TE, EB, TB
                return np.array(
                    [pow["tt"], pow["ee"], pow["bb"], pow["te"], pow["eb"], pow["tb"]]
                )
            else:
                # TT, TE, TB, EE, EB, BB
                return np.array(
                    [pow["tt"], pow["te"], pow["tb"], pow["ee"], pow["eb"], pow["bb"]]
                )
        else:
            raise ValueError("dtype should be 'd' or 'a'")
            
    def get_cb_lensed_spectra(
        self, beta: float = 0.0, dl: bool = True, dtype: str = "d", new: bool = False
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Calculate the cosmic birefringence (CB) lensed spectra with a given rotation angle `beta`.

        Parameters:
        beta (float, optional): The rotation angle in degrees for the cosmic birefringence effect. Defaults to 0.3 degrees.
        dl (bool, optional): If True, returns Dl (C_l * l * (l + 1) / 2π). Defaults to True.
        dtype (str, optional): Specifies the format of the returned spectra.
                               - 'd' returns a dictionary with keys 'tt', 'ee', 'bb', 'te', 'eb', 'tb'.
                               - 'a' returns an array of power spectra.
                               Defaults to 'd'.
        new (bool, optional): Determines the ordering of the spectra in the array if dtype is 'a'.
                              If True, returns [TT, EE, BB, TE, EB, TB].
                              If False, returns [TT, TE, TB, EE, EB, BB].
                              Defaults to False.

        Returns:
        Union[Dict[str, np.ndarray], np.ndarray]:
            - A dictionary containing individual CB lensed power spectra for 'tt', 'ee', 'bb', 'te', 'eb', 'tb' if dtype is 'd'.
            - An array of CB lensed power spectra with ordering determined by the `new` parameter if dtype is 'a'.

        Raises:
        ValueError: If `dtype` is not 'd' or 'a'.

        Notes:
        The method applies a rotation by `alpha` degrees to the E and B mode spectra to account for cosmic birefringence.
        """

        powers = self.get_lensed_spectra(dl=dl)
        if beta == 0.0:
            if self.beta is None:
                pass
            else:
                beta = self.beta
        pow = {}
        pow["tt"] = powers["tt"]
        pow["te"] = powers["te"] * np.cos(2 * inrad(beta))  # type: ignore
        pow["ee"] = (powers["ee"] * np.cos(inrad(2 * beta)) ** 2) + (powers["bb"] * np.sin(inrad(2 * beta)) ** 2)  # type: ignore
        pow["bb"] = (powers["ee"] * np.sin(inrad(2 * beta)) ** 2) + (powers["bb"] * np.cos(inrad(2 * beta)) ** 2)  # type: ignore
        pow["eb"] = 0.5 * (powers["ee"] - powers["bb"]) * np.sin(inrad(4 * beta))  # type: ignore
        pow["tb"] = powers["te"] * np.sin(2 * inrad(beta))  # type: ignore
        if dtype == "d":
            return pow
        elif dtype == "a":
            if new:
                # TT, EE, BB, TE, EB, TB
                return np.array(
                    [pow["tt"], pow["ee"], pow["bb"], pow["te"], pow["eb"], pow["tb"]]
                )
            else:
                # TT, TE, TB, EE, EB, BB
                return np.array(
                    [pow["tt"], pow["te"], pow["tb"], pow["ee"], pow["eb"], pow["bb"]]
                )
        else:
            raise ValueError("dtype should be 'd' or 'a'")
    
    def get_cb_lensed_mass_spectra(
            self, dl: bool = True, dtype: str = "d", new: bool = False
        ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        powers = self.get_lensed_spectra(dl=dl) 
        pow = {}
        pow["tt"] = powers["tt"]
        pow["te"] = powers["te"]
        pow["ee"] = powers["ee"]
        pow["bb"] = powers["bb"]
        internal_len = len(powers["tt"])
        _eb_ = self.__td_eb__(dl=dl)
        _tb_ = self.__td_tb__(dl=dl)
        eb = np.zeros(len(powers["tt"]))
        tb = np.zeros(len(powers["tt"]))
        eb[2:len(_eb_)] = _eb_[2:]
        tb[2:len(_eb_)] = _tb_[2:]
        pow["eb"] = eb
        pow["tb"] = tb

        if dtype == "d":
            return pow
        elif dtype == "a":
            if new:
                # TT, EE, BB, TE, EB, TB
                return np.array(
                    [pow["tt"], pow["ee"], pow["bb"], pow["te"], pow["eb"], pow["tb"]]
                )
            else:
                # TT, TE, TB, EE, EB, BB
                return np.array(
                    [pow["tt"], pow["te"], pow["tb"], pow["ee"], pow["eb"], pow["bb"]]
                )
        else:
            raise ValueError("dtype should be 'd' or 'a'")
    
    def cl_aa(self):
        """
        Compute the Cl_AA power spectrum for the anisotropic model.
        """
        assert self.Acb is not None, "Acb should be provided for anisotropic model"
        return self.scale_invariant(self.Acb)
    
    def cl_pp(self):
        powers = self.get_power(dl=False)['lens_potential']
        return powers[:, 0]
    
    ################ CMB map generation methods ###############
              ###### Isotropic models ######
              ###### Real space lensed ######
    def get_iso_const_cb_real_lensed_QU(self, idx: int) -> List[np.ndarray]:
        fname = os.path.join(
            self.cmbdir,
            f"sims_nside{self.nside}_{idx:03d}.fits",
        )
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1])
        else:
            spectra = self.get_cb_unlensed_spectra(
                    beta=self.beta if self.beta is not None else 0.0,
                    dl=False,
                )
            alms = hp.synalm(
                [spectra["tt"], spectra["ee"], spectra["bb"], spectra["te"], spectra["eb"], spectra["tb"]],
                lmax=self.lmax,
                new=True,
            )
            defl = self.grad_phi_alm(idx)
            geom_info = ('healpix', {'nside':self.nside})
            Qlen, Ulen = lenspyx.alm2lenmap_spin([alms[1],alms[2]], defl, 2, geometry=geom_info, verbose=int(self.verbose))
            hp.write_map(fname, [Qlen, Ulen], dtype=np.float64)
            return [Qlen, Ulen]

    def get_iso_td_cb_real_lensed_QU(self, idx: int) -> List[np.ndarray]:
        fname = os.path.join(
            self.cmbdir,
            f"sims_nside{self.nside}_{idx:03d}.fits",
        )
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1])
        else:
            spectra = self.get_cb_unlensed_mass_spectra(
                    dl=False,
                )
            alms = hp.synalm(
                [spectra["tt"], spectra["ee"], spectra["bb"], spectra["te"], spectra["eb"], spectra["tb"]],
                lmax=self.lmax,
                new=True,
            )
            defl = self.grad_phi_alm(idx)
            geom_info = ('healpix', {'nside':self.nside})
            Qlen, Ulen = lenspyx.alm2lenmap_spin([alms[1],alms[2]], defl, 2, geometry=geom_info, verbose=int(self.verbose))
            hp.write_map(fname, [Qlen, Ulen], dtype=np.float64)
            return [Qlen, Ulen]
    
    def get_iso_model_cb_real_lensed_QU(self,idx: int) -> List[np.ndarray]:
        if self.model == "iso":
            return self.get_iso_const_cb_real_lensed_QU(idx)
        elif self.model == "iso_td":
            return self.get_iso_td_cb_real_lensed_QU(idx)
        else:
            raise NotImplementedError("Model not implemented yet")
    
    ################ CMB map generation methods ###############
              ###### Isotropic models ######
              ###### Gaussian lensed ######
    def get_iso_const_cb_gaussian_lensed_QU(self, idx: int) -> List[np.ndarray]:
        """
        Generate or retrieve the Q and U Stokes parameters after applying cosmic birefringence.

        Parameters:
        idx (int): Index for the realization of the CMB map.

        Returns:
        List[np.ndarray]: A list containing the Q and U Stokes parameter maps as NumPy arrays.

        Notes:
        The method applies a rotation to the E and B mode spherical harmonics to simulate the effect of cosmic birefringence.
        If the map for the given `idx` exists in the specified directory, it reads the map from the file.
        Otherwise, it generates the Q and U maps, applies the birefringence, and saves the resulting map to a FITS file.
        """
        fname = os.path.join(
            self.cmbdir,
            f"sims_nside{self.nside}_{idx:03d}.fits",
        )
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1])   # type: ignore
        else:
            spectra = self.get_cb_lensed_spectra(
                beta=self.beta if self.beta is not None else 0.0,
                dl=False,
            )
            # PDP: spectra start at ell=0, we are fine
            np.random.seed(self.__cseeds__[idx])
            T, E, B = hp.synalm(
                [spectra["tt"], spectra["ee"], spectra["bb"], spectra["te"], spectra["eb"], spectra["tb"]],
                lmax=self.lmax,
                new=True,
            )
            del T
            QU = hp.alm2map_spin([E, B], self.nside, 2, lmax=self.lmax)
            hp.write_map(fname, QU, dtype=np.float32)
            return QU
    
    def get_iso_td_cb_gaussian_lensed_QU(self, idx: int) -> List[np.ndarray]:
        raise NotImplementedError("Model not implemented yet")

    def get_iso_model_cb_gaussian_lensed_QU(self,idx: int) -> List[np.ndarray]:
        if self.model == "iso":
            return self.get_iso_const_cb_gaussian_lensed_QU(idx)
        elif self.model == "iso_td":
            return self.get_iso_td_cb_gaussian_lensed_QU(idx)
        else:
            raise NotImplementedError("Model not implemented yet")
    
    ################ CMB map generation methods ###############
              ###### Isotropic models ######
    def get_iso_model_cb_lensed_QU(self,idx: int) -> List[np.ndarray]:
        if self.lensing:
            return self.get_iso_model_cb_real_lensed_QU(idx)
        else:
            return self.get_iso_model_cb_gaussian_lensed_QU(idx)
        
    ################ CMB map generation methods ###############
         ###### Secondary Source Fields(alms) ######
    
    def alpha_alm(self, idx: int) -> np.ndarray:
        """
        Generate the alpha alm for the anisotropic model.

        Parameters:
        idx (int): Index for the realization of the CMB map.

        Returns:
        np.ndarray: The alpha alm as a NumPy array.

        Notes:
        The method generates the alpha alm for the anisotropic model.
        The alpha alm is generated as a random realization of the Cl_AA power spectrum.
        The behavior depends on Acb_sim_config:
        - 'vary': Each idx gets a unique seed (default)
        - 'constant': All indices in this range use the same fixed seed
        - 'null': Returns zeros (no rotation)
        """
        mode = self.__get_alpha_mode__(idx)
        
        if mode == 'null':
            # Return zero alm (no rotation)
            return np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
        
        cl_aa = self.cl_aa()
        cl_aa[0] = 0
        
        if mode == 'constant':
            # Use a fixed seed for constant alpha across this range
            # Use the first seed as the constant seed
            np.random.seed(44444)
        else:  # mode == 'vary'
            # Use index-specific seed (default behavior)
            np.random.seed(self.__aseeds__[idx])
        
        alm = hp.synalm(cl_aa, lmax=self.lmax, new=True)
        return alm
    
    def phi_alm(self, idx: int) -> np.ndarray:
        fname = os.path.join(self.phidir, f"phi_Lmax{self.lmax}_{idx:03d}.fits")
        if os.path.isfile(fname):
            return hp.read_alm(fname)
        else:
            cl_pp = self.cl_pp()
            np.random.seed(self.__pseeds__[idx])
            alm = hp.synalm(cl_pp, lmax=self.lmax, new=True)
            hp.write_alm(fname, alm)
            return alm
        
    def grad_phi_alm(self, idx: int) -> np.ndarray:
        phi_alm = self.phi_alm(idx)
        return hp.almxfl(phi_alm, np.sqrt(np.arange(self.lmax + 1, dtype=float) * np.arange(1, self.lmax + 2)), None, False)
    
    ################ CMB map generation methods ###############
                ###### Anisotropic models ######
                ###### Source Fields(map) ######
    def alpha_map(self, idx: int) -> np.ndarray:
        """
        Generate a map of the rotation angle alpha for the anisotropic model.

        Parameters:
        idx (int): Index for the realization of the CMB map.

        Returns:
        np.ndarray: A map of the rotation angle alpha as a NumPy array.

        Notes:
        The method generates a map of the rotation angle alpha for the anisotropic model.
        The map is generated as a random realization of the Cl_AA power spectrum.
        """
        fname = os.path.join(
            self.alphadir,
            f"alpha_nside{self.nside}_{idx:03d}.fits",
        )
        if os.path.isfile(fname):
            return hp.read_map(fname)
        else:
            alm = self.alpha_alm(idx)
            alpha = hp.alm2map(alm, self.nside)
            hp.write_map(fname, alpha, dtype=np.float64)
            return alpha # type: ignore
   
    ################ CMB map generation methods ###############
                ###### Anisotropic models ######
                ###### Real space lensed ######
    def get_aniso_real_lensed_QU(self, idx: int) -> List[np.ndarray]:
        fname = os.path.join(
            self.cmbdir,
            f"sims_nside{self.nside}_{idx:03d}.fits",
        )
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1])
        else:
            spectra = self.get_unlensed_spectra(dl=False)
            np.random.seed(self.__cseeds__[idx])
            Q,U = hp.synfast(
                [spectra["tt"], spectra["ee"], spectra["bb"], spectra["te"]],
                nside=self.nside,
                new=True,
            )[1:]
            
            # Check if rotation should be applied based on configuration
            mode = self.__get_alpha_mode__(idx)
            if self.Acb != 0 and mode != 'null':
                alpha = self.alpha_map(idx)
                rQ = Q * np.cos(2 * alpha) - U * np.sin(2 * alpha)
                rU = Q * np.sin(2 * alpha) + U * np.cos(2 * alpha)
                del (Q, U, alpha)
                elm, blm = hp.map2alm_spin([rQ, rU], 2)
            else:
                elm, blm = hp.map2alm_spin([Q, U], 2)
            defl = self.grad_phi_alm(idx)
            geom_info = ('healpix', {'nside':self.nside})
            Q, U = lenspyx.alm2lenmap_spin([elm,blm], defl, 2, geometry=geom_info, verbose=int(self.verbose))
            del (elm, blm, defl)
            hp.write_map(fname, [Q, U], dtype=np.float64)
            return [Q, U]
   
    ################ CMB map generation methods ###############
                ###### Anisotropic models ######
                ###### Gaussian lensed ######
    def get_aniso_gauss_lensed_QU(self, idx: int) -> List[np.ndarray]:
        fname = os.path.join(
            self.cmbdir,
            f"sims_g_nside{self.nside}_{idx:03d}.fits",
        )
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1])
        else:
            spectra = self.get_lensed_spectra(dl=False)
            np.random.seed(self.__cseeds__[idx])
            Q, U = hp.synfast(
                [spectra["tt"], spectra["ee"], spectra["bb"], spectra["te"]],
                nside=self.nside,
                new=True,
            )[1:]
            
            # Check if rotation should be applied based on configuration
            mode = self.__get_alpha_mode__(idx)
            if self.Acb != 0 and mode != 'null':
                alpha = self.alpha_map(idx)
                rQ = Q * np.cos(2 * alpha) - U * np.sin(2 * alpha)
                rU = Q * np.sin(2 * alpha) + U * np.cos(2 * alpha)
                del (Q, U, alpha)
            else:
                if mode == 'null':
                    self.logger.log(f"Index {idx} in null_alpha range, no rotation applied", level="info")
                else:
                    self.logger.log("Acb is zero, no rotation applied", level="info")
                rQ, rU = Q, U
            hp.write_map(fname, [rQ, rU], dtype=np.float64)
            return [rQ, rU]
    
    ################ CMB map generation methods ###############
                ###### Anisotropic models ######
    def get_aniso_model_cb_lensed_QU(self,idx: int) -> List[np.ndarray]:
        if self.lensing:
            return self.get_aniso_real_lensed_QU(idx)
        else:
            return self.get_aniso_gauss_lensed_QU(idx)
        
    ################ CMB map generation methods ###############
                ###### General models ######
    def get_cb_lensed_QU(self, idx: int) -> List[np.ndarray]:
        if self.model == "iso" or self.model == "iso_td":
            return self.get_iso_model_cb_lensed_QU(idx)
        elif self.model == "aniso":
            return self.get_aniso_model_cb_lensed_QU(idx)
        else:
            raise NotImplementedError("Model not implemented yet")
