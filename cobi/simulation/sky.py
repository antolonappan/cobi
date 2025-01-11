import numpy as np
import healpy as hp
import os
from tqdm import tqdm
from cobi import mpi
from typing import Union, List, Optional

from cobi.simulation import CMB, Foreground, Mask, Noise
from cobi.utils import Logger, inrad
from cobi.utils import cli, deconvolveQU
from cobi.simulation import HILC


class SkySimulation:
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
        alpha: float = 0.0,
        alpha_err: float = 0.0,
        noise_model: str = "NC",
        atm_noise: bool = True,
        nsplits: int = 2,
        hilc_bins: int = 10,
        deconv_maps: bool = False,
        fldname_suffix: str = "",
        verbose: bool = True,
    ):
        """
        Initializes the SkySimulation class for generating and handling sky simulations.

        Parameters:
        -----------
        libdir: str
            The directory where the simulation data will be stored.
        nside: int
            The HEALPix nside parameter for the simulation.
        freqs: np.ndarray
            The frequency bands for the simulation.
        fwhm: np.ndarray
            The full width at half maximum (FWHM) for each frequency band.
        tube: np.ndarray
            The tube identifier for each frequency band.
        """
        self.logger = Logger(self.__class__.__name__, verbose)
        self.verbose = verbose

        fldname = "_atm_noise" if atm_noise else "_white_noise"
        fldname += "_bandpass" if bandpass else ""
        fldname += f"_{nsplits}splits" + fldname_suffix
        self.basedir = libdir
        self.libdir = os.path.join(libdir, self.__class__.__name__[:3] + fldname)
        os.makedirs(self.libdir + '/obs', exist_ok=True)

        if self.__class__.__name__ == "SAT" and nside != 512:
            self.logger.log(f"SAT simulations are only supported for nside=512. Resetting the given NSIDE={nside} to 512.")
            nside = 512
            
        self.nside = nside
        self.Acb = Acb
        self.cb_method = cb_model
        self.beta = beta
        self.cmb = CMB(libdir, nside, cb_model,beta, mass, Acb, lensing, verbose=self.verbose)
        self.foreground = Foreground(libdir, nside, dust_model, sync_model, bandpass, verbose=False)
        self.dust_model = dust_model
        self.sync_model = sync_model
        self.nsplits = nsplits
        self.freqs = freqs
        self.fwhm = fwhm
        self.tube = tube
        self.mask, self.fsky = self.__set_mask_fsky__(libdir)
        self.noise = Noise(nside, self.fsky, self.__class__.__name__[:3], noise_model, atm_noise, nsplits, verbose=self.verbose)
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

    def __set_mask_fsky__(self, libdir):
        maskobj = Mask(libdir, self.nside, self.__class__.__name__[:3], verbose=self.verbose)
        return maskobj.mask, maskobj.fsky

    def signalOnlyQU(self, idx: int, band: str) -> np.ndarray:
        band = band[:band.index('-')]
        cmbQU = np.array(self.cmb.get_cb_lensed_QU(idx))
        dustQU = self.foreground.dustQU(band)
        syncQU = self.foreground.syncQU(band)
        return cmbQU + dustQU + syncQU

    def obsQUwAlpha(
        self, idx: int, band: str, fwhm: float, alpha: float, apply_tranf: bool = True, return_alms: bool = False
    ) -> np.ndarray:
        signal = self.signalOnlyQU(idx, band)
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
            return hp.alm2map_spin([Elm, Blm], self.nside, 2, lmax=self.cmb.lmax)

    def obsQUfname(self, idx: int, band: str) -> str:
        alpha = self.config[band]["alpha"]
        if self.cb_method == 'iso':
            beta = self.cmb.beta
            return os.path.join(
                self.libdir,
                f"obs/obsQU_N{self.nside}_b{str(beta).replace('.','p')}_a{str(alpha).replace('.','p')}_{band}{'_d' if self.deconv_maps else ''}{'_bp' if self.bandpass else ''}_{idx:03d}.fits",
            )
        elif self.cb_method == 'aniso':
            Acb = self.cmb.Acb
            return os.path.join(
                self.libdir,
                f"obs/obsQU_N{self.nside}_A{str(Acb).replace('.','p')}_a{str(alpha).replace('.','p')}_{band}{'_d' if self.deconv_maps else ''}{'_bp' if self.bandpass else ''}_{idx:03d}.fits",
            )
        else:
            raise ValueError("Unknown CB method")
        

    def saveObsQUs(self, idx: int, apply_mask: bool = True) -> None:
        mask = self.mask if apply_mask else np.ones_like(self.mask)
        bands = list(self.config.keys())
        signal = []
        for band in bands:
            fwhm = self.config[band]["fwhm"]
            alpha = self.config[band]["alpha"]
            signal.append(self.obsQUwAlpha(idx, band, fwhm, alpha))
        noise = self.noise.noiseQU()
        sky = np.array(signal) + noise
        
        if self.deconv_maps:
            for i in tqdm(range(len(bands)), desc='Deconvolving QUs', unit='band'):
                sky[i] = deconvolveQU(sky[i], self.config[bands[i]]['fwhm'])
            
        for i in tqdm(range(len(bands)), desc="Saving Observed QUs", unit="band"):
            fname = self.obsQUfname(idx, bands[i])
            hp.write_map(fname, sky[i] * mask, dtype=np.float64, overwrite=True) # type: ignore

    def obsQU(self, idx: int, band: str) -> np.ndarray:
        fname = self.obsQUfname(idx, band)
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1]) # type: ignore
        else:
            self.saveObsQUs(idx)
            return hp.read_map(fname, field=[0, 1]) # type: ignore
        
    
    def HILC_obsEB(self, idx: int) -> np.ndarray:
        fnameS = os.path.join(
                self.libdir,
                f"obs/hilcEB_N{self.nside}_A{str(self.Acb).replace('.','p')}{'_bp' if self.bandpass else ''}_{idx:03d}.fits",
            )
        fnameN = fnameS.replace('hilcEB','hilcNoise')
        if os.path.isfile(fnameS) and os.path.isfile(fnameN):
            return hp.read_alm(fnameS, hdu=[1, 2]), hp.read_cl(fnameN)
        else:
            noise = self.noise.noiseQU()
            alms = []
            nalms = []
            bands = list(self.config.keys())
            i = 0
            for band in tqdm(bands, desc="Computing HILC Observed QUs", unit="band"):
                fwhm = self.config[band]["fwhm"]
                alpha = self.config[band]["alpha"]
                elm,blm = self.obsQUwAlpha(idx, band, fwhm, alpha, apply_tranf=False, return_alms=True)
                nelm,nblm = hp.map2alm_spin([noise[i][0],noise[i][1]], 2, lmax=self.cmb.lmax)  
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
            return cleaned,ilc_noise
            


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
        noise_model: str = "NC",
        atm_noise: bool = True,
        nsplits: int = 2,
        hilc_bins: int = 10,
        deconv_maps: bool = False,
        fldname_suffix: str = "",
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
            noise_model = noise_model,
            atm_noise = atm_noise,
            nsplits = nsplits,
            hilc_bins = hilc_bins,
            deconv_maps = deconv_maps,
            fldname_suffix = fldname_suffix,
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
        noise_model: str = "NC",
        atm_noise: bool = True,
        nsplits: int = 2,
        hilc_bins: int = 10,
        deconv_maps: bool = False,
        fldname_suffix: str = "",
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
            noise_model = noise_model,
            atm_noise = atm_noise,
            nsplits = nsplits,
            hilc_bins = hilc_bins,
            deconv_maps = deconv_maps,
            fldname_suffix = fldname_suffix,
            verbose = verbose,
        )