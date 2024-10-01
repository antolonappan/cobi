import numpy as np
import healpy as hp
from typing import Dict, Optional, Any, Union, List, Tuple
from so_models_v3 import SO_Noise_Calculator_Public_v3_1_1 as so_models

from solat_cb import mpi
from solat_cb.utils import Logger

#atm_noise or atm_corr, same for the noise map as well

def NoiseSpectra(sensitivity_mode, fsky, lmax, atm_noise, telescope):
    match telescope:
        case "LAT":
            teles = so_models.SOLatV3point1(sensitivity_mode, el=50)
        case "SAT":
            teles = so_models.SOSatV3point1(sensitivity_mode)
    
    teles.get_noise_curves(fsky, lmax, 1, full_covar=True, deconv_beam=False)
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

    def __init__(self, 
                 nside: int, 
                 fsky: float,
                 telescope: str,
                 atm_noise: bool = False, 
                 nsplits: int = 2,
                 verbose: bool = True,
                 ) -> None:
        """
        Initializes the Noise class for generating noise maps with or without atmospheric noise.

        Parameters:
        nside (int): HEALPix resolution parameter.
        atm_noise (bool, optional): If True, includes atmospheric noise. Defaults to False.
        nhits (bool, optional): If True, includes hit count map. Defaults to False.
        nsplits (int, optional): Number of data splits to consider. Defaults to 2.
        """
        self.nside            = nside
        self.lmax             = 3 * nside - 1
        self.sensitivity_mode = 2
        self.atm_noise        = atm_noise
        self.nsplits          = nsplits
        self.Nell             = NoiseSpectra(self.sensitivity_mode, fsky, self.lmax, self.atm_noise, telescope)
        self.logger           = Logger(self.__class__.__name__, verbose)
        if atm_noise:
            self.logger.log("Noise Model: White + 1/f noise v3.0.0")
        else:
            self.logger.log("Noise Model: White noise v3.0.0")

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


    def atm_noise_maps(self) -> np.ndarray:
        """
        Generates atmospheric noise maps using Cholesky decomposition.

        Returns:
        np.ndarray: An array of atmospheric noise maps for different frequency bands.
        """
        L11, L21, L22, L33, L43, L44, L55, L65, L66 = self.cholesky_matrix_elements

        alm    = self.rand_alm
        blm    = self.rand_alm
        nlm_27 = hp.almxfl(alm, L11)
        nlm_39 = hp.almxfl(alm, L21) + hp.almxfl(blm, L22)
        n_27   = hp.alm2map(nlm_27, self.nside, pixwin=False)
        n_39   = hp.alm2map(nlm_39, self.nside, pixwin=False)

        clm     = self.rand_alm
        dlm     = self.rand_alm
        nlm_93  = hp.almxfl(clm, L33)
        nlm_145 = hp.almxfl(clm, L43) + hp.almxfl(dlm, L44)
        n_93    = hp.alm2map(nlm_93, self.nside, pixwin=False)
        n_145   = hp.alm2map(nlm_145, self.nside, pixwin=False)

        elm     = self.rand_alm
        flm     = self.rand_alm
        nlm_225 = hp.almxfl(elm, L55)
        nlm_280 = hp.almxfl(elm, L65) + hp.almxfl(flm, L66)
        n_225   = hp.alm2map(nlm_225, self.nside, pixwin=False)
        n_280   = hp.alm2map(nlm_280, self.nside, pixwin=False)

        n = np.array([n_27, n_39, n_93, n_145, n_225, n_280])
        return n

    def noiseQU(self) -> np.ndarray:
        """
        Generates Q and U polarization noise maps based on the noise model.

        Returns:
        np.ndarray: An array of Q and U noise maps.
        """
        N = []
        for split in range(self.nsplits):
            if self.atm_noise:
                q = self.atm_noise_maps()
                u = self.atm_noise_maps()
            else:
                q = self.white_noise_maps()
                u = self.white_noise_maps()            
              
            for i in range(len(q)):
                N.append([q[i], u[i]])

        return np.array(N)*np.sqrt(self.nsplits)