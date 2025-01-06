"""
This file contains the class to handle the Cosmic Microwave Background (CMB) data and simulations.
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

class CMB:

    def __init__(
        self,
        libdir: str,
        nside: int,
        model: str = "iso",
        beta: Optional[float]=None,
        mass: Optional[float]=None,
        Acb: Optional[float]=None,
        lensing: Optional[bool] = False,
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
        assert model in ["iso", "aniso"], "model should be 'iso' or 'aniso'"
        self.model  = model
        if self.model == "aniso":
            self.logger.log("Anisotropic cosmic birefringence model selected", level="info")
            assert self.Acb is not None, "Acb should be provided for anisotropic model"
        if self.model == "iso":
            self.logger.log("Isotropic cosmic birefringence model selected", level="info")
            assert (self.beta is not None)  or (self.mass is not None), "beta or mass should be provided for isotropic model"
            if self.mass is not None:
                assert self.mass in [1,1.5,6.4], "mass should be 1, 1.59 or 6.36"
                self.logger.log(f"Time dependent component with axion mass = {self.mass} * 10^-28 eV selected", level="info")
            else:
                self.logger.log(f"Constant component with rotation angle beta = {self.beta} degrees selected", level="info")

        
        self.lensing = lensing

        self.__set_workspace__()
        self.__set_seeds__()
   
    def __set_seeds__(self) -> None:
        """
        Sets the seeds for the simulation.
        """
        nos = 500
        self.__cseeds__ = np.arange(11111,11111+nos, dtype=int)
        self.__aseeds__ = np.arange(22222,22222+nos, dtype=int)
        self.__pseeds__ = np.arange(33333,33333+nos, dtype=int)
    
    def __set_workspace__(self) -> None:
        """
        Set the workspace for the CMB simulations.
        """
        lens = "lensed" if self.lensing else "gaussian"
        if self.model == "iso":
            if self.beta is not None:
                model = f"iso_beta_{str(self.beta).replace('.','p')}"
            if self.mass is not None:
                model = f"iso_mass_{str(self.mass).replace('.','p')}"
        else:
            model = f"aniso_alpha_{str(self.Acb)}"


        self.cmbdir = os.path.join(self.basedir, 'CMB', lens, model, 'cmb')
        os.makedirs(self.cmbdir, exist_ok=True)
        if self.lensing:
            self.phidir = os.path.join(self.basedir, 'CMB', lens, model, 'phi')
            os.makedirs(self.phidir, exist_ok=True)
        if self.model == "aniso":
            self.alphadir = os.path.join(self.basedir, 'CMB', lens, model, 'alpha')
            os.makedirs(self.alphadir, exist_ok=True)
    
    def __td_eb__(self) -> np.ndarray:
        """
        Read the E and B mode power spectra from the CMB power spectra data.
        """
        ISO_TD_SPECTRA.directory = self.basedir
        m = str(self.mass).replace('.','p')
        spectra = ISO_TD_SPECTRA.data[m]
        return spectra[:self.lmax + 1,5]
        
        
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
        raise NotImplementedError("Mass model not implemented yet")
            
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
       raise NotImplementedError("Mass model not implemented yet")

    def get_cb_gaussian_lensed_QU(self,idx: int) -> List[np.ndarray]:
        if self.model == "iso":
            return self.get_iso_cb_gaussian_lensed_QU(idx)
        elif self.model == "aniso":
            return self.get_aniso_cb_gaussian_lensed_QU(idx)
        else:
            raise NotImplementedError("Model not implemented yet, only 'iso' and 'aniso' are supported")
    
    def get_iso_cb_gaussian_lensed_QU(self, idx: int) -> List[np.ndarray]:
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
        
    def cl_aa(self):
        """
        Compute the Cl_AA power spectrum for the anisotropic model.
        """
        L = np.arange(self.lmax + 1)
        assert self.Acb is not None, "Acb should be provided for anisotropic model"
        return self.Acb * 2 * np.pi / ( L**2 + L + 1e-30)
    
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
            cl_aa = self.cl_aa()
            cl_aa[0] = 0
            np.random.seed(self.__aseeds__[idx])
            alm = hp.synalm(cl_aa, lmax=self.lmax,new=True)
            alpha = hp.alm2map(alm, self.nside)
            hp.write_map(fname, alpha, dtype=np.float64)
            return alpha # type: ignore
    
    def get_aniso_cb_gaussian_lensed_QU(self, idx: int) -> List[np.ndarray]:
        """
        Generate the Q and U Stokes maps after applying cosmic birefringence for the anisotropic model.

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
            return hp.read_map(fname, field=[0, 1])
        else:
            spectra = self.get_lensed_spectra(dl=False)
            np.random.seed(self.__cseeds__[idx])
            T, Q, U = hp.synfast(
                [spectra["tt"], spectra["ee"], spectra["bb"], spectra["te"]],
                nside=self.nside,
                new=True,
            )
            del T
            alpha = self.alpha_map(idx)
            rQ = Q * np.cos(2 * alpha) - U * np.sin(2 * alpha)
            rU = Q * np.sin(2 * alpha) + U * np.cos(2 * alpha)
            del (Q, U)
            hp.write_map(fname, [rQ, rU], dtype=np.float64)
            return [rQ, rU]
    
    def cl_pp(self):
        powers = self.get_power(dl=False)['lens_potential']
        return powers[:, 0]

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

    def get_iso_cb_real_lensed_QU(self, idx: int) -> List[np.ndarray]:
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
            Qlen, Ulen = lenspyx.alm2lenmap_spin([alms[1],alms[2]], defl, 2, geometry=geom_info, verbose=1)
            hp.write_map(fname, [Qlen, Ulen], dtype=np.float64)
            return [Qlen, Ulen]

    def get_aniso_cb_real_lensed_QU(self, idx: int) -> List[np.ndarray]:
        fname = os.path.join(
            self.cmbdir,
            f"sims_nside{self.nside}_{idx:03d}.fits",
        )
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1])
        else:
            spectra = self.get_unlensed_spectra(dl=False)
            np.random.seed(self.__cseeds__[idx])
            alms = hp.synalm(
                [spectra["tt"], spectra["ee"], spectra["bb"], spectra["te"]],
                lmax=self.lmax,
                new=True,
            )
            defl = self.grad_phi_alm(idx)
            geom_info = ('healpix', {'nside':self.nside})
            Q, U = lenspyx.alm2lenmap_spin([alms[1],alms[2]], defl, 2, geometry=geom_info, verbose=1)
            alpha = self.alpha_map(idx)
            rQ = Q * np.cos(2 * alpha) - U * np.sin(2 * alpha)
            rU = Q * np.sin(2 * alpha) + U * np.cos(2 * alpha)
            del (Q, U)
            hp.write_map(fname, [rQ, rU], dtype=np.float64)
            return [rQ, rU]
        
    def get_cb_real_lensed_QU(self, idx: int) -> List[np.ndarray]:
        if self.model == "iso":
            return self.get_iso_cb_real_lensed_QU(idx)
        elif self.model == "aniso":
            return self.get_aniso_cb_real_lensed_QU(idx)
        else:
            raise NotImplementedError("Model not implemented yet, only 'iso' and 'aniso' are supported")
        
    def get_cb_lensed_QU(self, idx: int) -> List[np.ndarray]:
        if self.lensing:
            return self.get_cb_real_lensed_QU(idx)
        else:
            return self.get_cb_gaussian_lensed_QU(idx)
        

