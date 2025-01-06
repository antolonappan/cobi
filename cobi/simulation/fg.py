"""
This file contains the implementation of the Foreground class for generating and handling dust and synchrotron foreground maps.
"""
# General imports
import os
import pysm3
import numpy as np
import healpy as hp
import pickle as pl
from typing import Tuple
from pysm3 import units as u
import matplotlib.pyplot as plt
from typing import List, Optional
# Local imports
from cobi import mpi
from cobi.utils import Logger
from cobi.data import BP_PROFILE



class BandpassInt:
    def __init__(
        self,
        libdir: str,
    ):
        """
        Initializes the BandpassInt class, loading bandpass profiles from a specified file.
        """
        BP_PROFILE.directory = libdir
        self.bp = BP_PROFILE.data

    def get_profile(self, band: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the frequency and bandpass profile for a specified band.

        Parameters:
        band (str): The frequency band for which to retrieve the profile.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                       - nu: Array of frequencies (GHz) where the bandpass is defined.
                                       - bp: Array of bandpass values corresponding to the frequencies.
        """
        nu, bp = self.bp[band]
        return nu[nu > 0], bp[nu > 0]

    def plot_profiles(self) -> None:
        """
        Plots the bandpass profiles for all available bands.

        The method iterates over all bands, retrieves the bandpass profile,
        and plots the profile as a function of frequency (GHz).
        """
        bands = self.bp.keys()
        plt.figure(figsize=(6, 4))
        for b in bands:
            nu, bp = self.get_profile(b)
            plt.plot(nu, bp, label=b)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Bandpass Response")
        plt.legend(title="Bands")
        plt.tight_layout()
        plt.show()

class Foreground:
    def __init__(
        self,
        libdir: str,
        nside: int,
        dust_model: int,
        sync_model: int,
        bandpass: bool = False,
        verbose: bool = True,
    ):
        """
        Initializes the Foreground class for generating and handling dust and synchrotron foreground maps.

        Parameters:
        libdir (str): Directory where the foreground maps will be stored.
        nside (int): HEALPix resolution parameter.
        dust_model (int): Model number for the dust emission.
        sync_model (int): Model number for the synchrotron emission.
        bandpass (bool, optional): If True, bandpass integration is applied. Defaults to False.
        """
        self.logger = Logger(self.__class__.__name__, verbose=verbose)
        self.basedir = libdir
        self.libdir = os.path.join(libdir, f"Foregrounds{dust_model}{sync_model}")
        if mpi.rank == 0:
            os.makedirs(self.libdir, exist_ok=True)
        mpi.barrier()
        self.nside = nside
        self.dust_model = dust_model
        self.sync_model = sync_model
        self.bandpass = bandpass
        if bandpass:
            self.bp_profile = BandpassInt(libdir)
            self.logger.log("Bandpass integration is enabled", level="info")
        else:
            self.bp_profile = None
            self.logger.log("Bandpass integration is disabled", level="info")

    def dustQU(self, band: str) -> np.ndarray:
        """
        Generates or retrieves the Q and U Stokes parameters for dust emission at a given frequency band.

        Parameters:
        band (str): The frequency band.

        Returns:
        np.ndarray: A NumPy array containing the Q and U maps.
        """
        name = (
            f"dustQU_N{self.nside}_f{band}.fits"
            if not self.bandpass
            else f"dustQU_N{self.nside}_f{band}_bp.fits"
        )
        fname = os.path.join(self.libdir, name)

        if os.path.isfile(fname):
            self.logger.log(f"Loading dust Q and U maps for band {band}", level="info")
            return hp.read_map(fname, field=[0, 1]) # type: ignore
        
        else:
            self.logger.log(f"Generating dust Q and U maps for band {band}", level="info")
            sky = pysm3.Sky(
                nside=self.nside, preset_strings=[f"d{int(self.dust_model)}"]
            )
            if self.bandpass:
                if self.bp_profile is not None:
                    nu, weights = self.bp_profile.get_profile(band)
                else:
                    raise ValueError("Bandpass profile is not initialized.")
                nu = nu * u.GHz # type: ignore
                maps = sky.get_emission(nu, weights)
                maps *= pysm3.utils.bandpass_unit_conversion(nu, weights=weights, output_unit=u.uK_CMB)
            else:
                maps = sky.get_emission(int(band) * u.GHz) # type: ignore
                maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(int(band) * u.GHz)) # type: ignore

            if mpi.rank == 0:
                hp.write_map(fname, maps[1:], dtype=np.float64) # type: ignore
            mpi.barrier()

            return maps[1:].value

    def syncQU(self, band: str) -> np.ndarray:
        """
        Generates or retrieves the Q and U Stokes parameters for synchrotron emission at a given frequency band.

        Parameters:
        band (str): The frequency band.

        Returns:
        np.ndarray: A NumPy array containing the Q and U maps.
        """
        name = (
            f"syncQU_N{self.nside}_f{band}.fits"
            if not self.bandpass
            else f"syncQU_N{self.nside}_f{band}_bp.fits"
        )
        fname = os.path.join(self.libdir, name)

        if os.path.isfile(fname):
            self.logger.log(f"Loading synchrotron Q and U maps for band {band}", level="info")
            return hp.read_map(fname, field=[0, 1]) # type: ignore
        else:
            self.logger.log(f"Generating synchrotron Q and U maps for band {band}", level="info")
            sky = pysm3.Sky(
                nside=self.nside, preset_strings=[f"s{int(self.sync_model)}"]
            )
            if self.bandpass:
                if self.bp_profile is not None:
                    nu, weights = self.bp_profile.get_profile(band)
                else:
                    raise ValueError("Bandpass profile is not initialized.")
                nu = nu * u.GHz # type: ignore
                maps = sky.get_emission(nu, weights)
                maps *= pysm3.utils.bandpass_unit_conversion(nu, weights=weights, output_unit=u.uK_CMB)
            else:
                maps = sky.get_emission(int(band) * u.GHz) # type: ignore
                maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(int(band) * u.GHz)) # type: ignore

            if mpi.rank == 0:
                hp.write_map(fname, maps[1:], dtype=np.float64) # type: ignore
            mpi.barrier()

            return maps[1:].value


class HILC:
    """
    This class is used to perform Harmonic ILC on the foregrounds.

    """

    def __init__(self):
        pass

    def harmonic_ilc_alm(self, alms : np.ndarray, lbins : Optional[List[int]] = None) -> tuple:
        """
        This method is used to perform Harmonic ILC on the foregrounds.

        :param alms: The foregrounds in alm format.
        :type alms: np.ndarray
        :param lbins: The list of ell bins.
        :type lbins: Optional[List[int]], optional
        :return: The Harmonic ILC results and the ILC filter.
        :rtype: tuple
        """

        A = np.ones((len(alms), 1))
        cov = self.empirical_harmonic_covariance(alms)
        if lbins is not None:
            for lmin, lmax in zip(lbins[:-1], lbins[1:]):
                # Average the covariances in the bin
                lmax = min(lmax, cov.shape[-1])
                dof = 2 * np.arange(lmin, lmax) + 1
                cov[..., lmin:lmax] = (
                    (dof / dof.sum() * cov[..., lmin:lmax]).sum(-1)
                    )[..., np.newaxis]
        cov = self.regularized_inverse(cov.swapaxes(-1, -3))
        ilc_filter = np.linalg.inv(A.T @ cov @ A) @ A.T @ cov
        del cov, dof
        result = self.apply_harmonic_W(ilc_filter, alms)
        return result,ilc_filter

    def empirical_harmonic_covariance(self,alms : np.ndarray) -> np.ndarray:
        """
        The method empirical_harmonic_covariance is used to compute the empirical harmonic covariance.

        :param alms: The foregrounds in alm format.
        :type alms: np.ndarray
        :return: The empirical harmonic covariance.
        :rtype: np.ndarray
        """
        alms = np.array(alms, copy=False, order='C')
        alms = alms.view(np.float64).reshape(alms.shape+(2,))
        if alms.ndim > 3:  # Shape has to be ([Stokes], freq, lm, ri)
            alms = alms.transpose(1, 0, 2, 3)
        lmax = hp.Alm.getlmax(alms.shape[-2])

        res = (alms[..., np.newaxis, :, :lmax+1, 0]
            * alms[..., :, np.newaxis, :lmax+1, 0])  # (Stokes, freq, freq, ell)


        consumed = lmax + 1
        for i in range(1, lmax+1):
            n_m = lmax + 1 - i
            alms_m = alms[..., consumed:consumed+n_m, :]
            res[..., i:] += 2 * np.einsum('...fli,...nli->...fnl', alms_m, alms_m)
            consumed += n_m

        res /= 2 * np.arange(lmax + 1) + 1
        return res

    def regularized_inverse(self,cov : np.ndarray) -> np.ndarray:
        """
        The method regularized_inverse is used to compute the regularized inverse of the covariance.

        :param cov: The covariance.
        :type cov: np.ndarray
        :return: The regularized inverse of the covariance.
        :rtype: np.ndarray
        """

        inv_std = np.einsum('...ii->...i', cov)
        inv_std = 1 / np.sqrt(inv_std)
        np.nan_to_num(inv_std, False, 0, 0, 0)
        np.nan_to_num(cov, False, 0, 0, 0)

        inv_cov = np.linalg.pinv(cov
                                * inv_std[..., np.newaxis]
                                * inv_std[..., np.newaxis, :])
        return inv_cov * inv_std[..., np.newaxis] * inv_std[..., np.newaxis, :]
    

    def apply_harmonic_W(self,W : np.ndarray, alms: np.ndarray)  ->  np.ndarray:
        """
        The method apply_harmonic_W is used to apply the harmonic weights.

        :param W: The harmonic weights.
        :type W: np.ndarray
        :param alms: The foregrounds in alm format.
        :type alms: np.ndarray

        :return: The alms after applying the harmonic weights.
        :rtype: np.ndarray
        """
    
        lmax = hp.Alm.getlmax(alms.shape[-1])
        res = np.full((W.shape[-2],) + alms.shape[1:], np.nan, dtype=alms.dtype)
        start = 0
        for i in range(0, lmax+1):
            n_m = lmax + 1 - i
            res[..., start:start+n_m] = np.einsum('...lcf,f...l->c...l',
                                                W[..., i:, :, :],
                                                alms[..., start:start+n_m])
            start += n_m
        return res
