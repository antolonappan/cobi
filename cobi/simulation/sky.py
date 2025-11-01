import numpy as np
import healpy as hp
import os
import pickle as pl
from tqdm import tqdm
from cobi import mpi
from typing import Union, List, Optional

from cobi.simulation import CMB, Foreground, Noise, calc_fsky
from cobi.utils import Logger, inrad
from cobi.utils import cli, deconvolveQU
from cobi.simulation import HILC

from concurrent.futures import ThreadPoolExecutor


class SkySimulation:
	def __init__(
		self,
		libdir: str,
		nside: int,
		freqs: np.ndarray,
		fwhm: np.ndarray,
		tube: np.ndarray,
		mask: str,
		telescope: str,
		cb_model: str = "iso",
		beta: float = 0.35,
		mass: float = 1.5,
		Acb: float = 1e-6,
		lensing: bool = True,
		dust_model: str = 'd10',
		sync_model: str = 's5',
		bandpass: bool = True,
		alpha: Union[float,List[float]] = 0.0,
		alpha_err: float = 0.0,
		noise_model: str = "NC",
		atm_noise: bool = True,
		nsplits: int = 2,
		gal_cut: int = 0,
		hilc_bins: int = 10,
		deconv_maps: bool = False,
		fldname_suffix: str = "",
		verbose: bool = True,
		nhits: str = None,
		nhits_fac: float = 1.0,
		fore_realization: bool = False,
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

		fldname = "_an" if atm_noise else "_wn"
		fldname += "_bp" if bandpass else ""
		fldname += f"_{nsplits}ns" 
		fldname += "_lens" if lensing else "_gauss"
		fldname += f"_{noise_model.lower()}nm"
		if cb_model == 'iso':
			fldname += f"_b{str(beta).replace('.','p')}"
		elif cb_model == 'iso_td':
			fldname += f"_m({str(mass).replace('.','p')}"
		elif cb_model == 'aniso':
			fldname += f"_acb{str(Acb).replace('-','')}"
		else:
			raise ValueError("Unknown CB method")
		
		fldname += f"_{dust_model}{sync_model}"
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
		self.telescope = telescope
		self.nside = nside
		self.Acb = Acb
		self.cb_method = cb_model
		self.beta = beta
		self.cmb = CMB(libdir, nside, cb_model,beta, mass, Acb, lensing, verbose=self.verbose)
		self.foreground = Foreground(libdir, nside, dust_model, sync_model, bandpass, verbose=False, fore_realization=fore_realization)
		self.dust_model = dust_model
		self.sync_model = sync_model
		self.nsplits = nsplits
		self.freqs = freqs
		self.fwhm = fwhm
		self.tube = tube
		self.mask = hp.read_map(mask, field=0, dtype=np.double)
		self.fsky = calc_fsky(self.mask)
		self.noise_model = noise_model
		self.noise = Noise(nside, 0.4, self.telescope, noise_model, nhits, nhits_fac, atm_noise, nsplits, verbose=self.verbose,)
		self.fore_realization = fore_realization
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
		self.__set_alpha__()

	def signalOnlyQU(self, idx: int, band: str) -> np.ndarray:
		band = band[:band.index('-')]
		cmbQU = np.array(self.cmb.get_cb_lensed_QU(idx))
		if self.fore_realization:
			dustQU = self.foreground.dustQU(band, idx)
		else:
			dustQU = self.foreground.dustQU(band)
		syncQU = self.foreground.syncQU(band)
		return cmbQU + dustQU + syncQU

	# def __gen_alpha_dict__(self):
	#     fname = os.path.join(self.libdir, 'alpha_dict.pkl')
	#     if os.path.isfile(fname):
	#         self.alpha_dict = pl.load(open(fname, 'rb'))
	#     else:
	#         self.alpha_dict = {}
	#         for band in self.config.keys():
	#             alpha = self.config[band]["alpha"]
	#             self.alpha_dict[band] = np.random.normal(alpha, self.alpha_err, 300) # given value is assumed 3 sigma
	#         if mpi.rank == 0:
	#             pl.dump(self.alpha_dict, open(fname, 'wb'))
	def __gen_alpha_dict__(self):
		fname = os.path.join(self.libdir, 'alpha_dict.pkl')
		if os.path.isfile(fname):
			self.alpha_dict = pl.load(open(fname, 'rb'))
		else:
			self.alpha_dict = {}
			# Extract base bands like '27', '39', etc.
			base_bands = set([band.split('-')[0] for band in self.config.keys()])
			for base_band in base_bands:
				# Create one random sample per base_band
				band_1 = f"{base_band}-1"
				alpha = self.config[band_1]["alpha"]
				sample = np.random.normal(alpha, self.alpha_err, 300)
				# Assign same sample to both '-1' and '-2'
				self.alpha_dict[f"{base_band}-1"] = sample
				self.alpha_dict[f"{base_band}-2"] = sample
			if mpi.rank == 0:
				pl.dump(self.alpha_dict, open(fname, 'wb'))
		

	def __set_alpha__(self):
		if self.alpha_err > 0:
			self.__gen_alpha_dict__()
		else:
			self.alpha_dict = None

	def get_alpha(self, idx: int, band: str):
		if self.alpha_err > 0:
			assert self.alpha_dict is not None, "Alpha dictionary not found. Run __gen_alpha_dict__ first."
			return self.alpha_dict[band][idx]
		else:
			return self.config[band]["alpha"]
		

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
				hp.write_map(fname, sky * mask, dtype=np.float32,overwrite=(bands is not None)) # type: ignore
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
			self.saveObsQUs(idx)
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
		mask: str,
		cb_model: str = "iso",
		beta: float = 0.35,
		mass: float = 1.5,
		Acb: float = 1e-6,
		lensing: bool = True,
		dust_model: str = 'd10',
		sync_model: str = 's5',
		bandpass: bool = True,
		alpha: float = 0.0,
		alpha_err: float = 0.0,
		noise_model: str = "NC",
		atm_noise: bool = True,
		nsplits: int = 2,
		gal_cut: int = 0,
		hilc_bins: int = 10,
		deconv_maps: bool = False,
		fldname_suffix: str = "",
		verbose: bool = True,
		nhits: str = None,
		nhits_fac: float = 1.0,
		fore_realization: bool = False,
	):
		super().__init__(
			libdir = libdir,
			nside = nside,
			mask = mask,
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
			gal_cut = gal_cut,
			hilc_bins = hilc_bins,
			deconv_maps = deconv_maps,
			fldname_suffix = fldname_suffix,
			verbose = verbose,
			nhits = nhits,
			nhits_fac = nhits_fac,
			fore_realization = fore_realization,
			telescope = 'LAT'
		)

class LiteBIRDsky(SkySimulation):
    freqs = np.array(['L40','L50','L60','L68a','L68b','L78a','L78b','L89a','L89b','L100','L119','L140','M100','M119','M140','M166','M195','H195','H235','H280','H337','H402'])
    fwhm = np.array([70.5,58.5,51.1,41.6,47.1,36.9,43.8,33.0,41.5,30.2,26.3,23.7,37.8,33.6,30.8,28.9,28.0,28.6,24.7,22.5,20.9,17.9])  # arcmin
    tube = np.array(["LF", "LF", "LF", "LF", "LF", "LF", "LF", "LF", "LF", "LF", "LF", "LF", "MF", "MF", "MF", "MF", "MF", "MF", "HF", "HF", "HF", "HF"])  # tube each frequency occupies

    def __init__(
		self,
		libdir: str,
		nside: int,
		mask: str,
		cb_model: str = "iso",
		beta: float = 0.35,
		mass: float = 1.5,
		Acb: float = 1e-6,
		lensing: bool = True,
		dust_model: str = 'd10',
		sync_model: str = 's5',
		bandpass: bool = False,
		alpha: float = 0.0,
		alpha_err: float = 0.0,
		noise_model: str = "NC",
		atm_noise: bool = True,
		nsplits: int = 2,
		gal_cut: int = 0,
		hilc_bins: int = 10,
		deconv_maps: bool = False,
		fldname_suffix: str = "",
		verbose: bool = True,
		nhits: str = None,
		nhits_fac: float = 1.0,
		fore_realization: bool = False,
	):
        super().__init__(
			libdir = libdir,
			nside = nside,
			mask = mask,
			freqs = LiteBIRDsky.freqs,
			fwhm = LiteBIRDsky.fwhm,
			tube = LiteBIRDsky.tube,
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
			gal_cut = gal_cut,
			hilc_bins = hilc_bins,
			deconv_maps = deconv_maps,
			fldname_suffix = fldname_suffix,
			verbose = verbose,
			nhits = nhits,
			nhits_fac = nhits_fac,
			fore_realization = fore_realization,
			telescope = 'LiteBIRD'
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
        gal_cut: int = 0,
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
            gal_cut = gal_cut,
            hilc_bins = hilc_bins,
            deconv_maps = deconv_maps,
            fldname_suffix = fldname_suffix,
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
        alpha: float = 0.0,
        alpha_err: float = 0.0,
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
            noise_model = noise_model,
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
            noise_model = noise_model,
            atm_noise = atm_noise,
            nsplits = nsplits,
            gal_cut = gal_cut,
            hilc_bins = hilc_bins,
            deconv_maps = deconv_maps,
            fldname_suffix = fldname_suffix,
            verbose = verbose,
        )

