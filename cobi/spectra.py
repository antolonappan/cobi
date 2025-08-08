# object oriented version of Patricia's code
import numpy as np
import healpy as hp
import pymaster as nmt
import os
from tqdm import tqdm
from cobi.simulation import LATsky, Foreground, SATsky
from cobi.utils import Logger
from cobi import mpi
from typing import Dict, Optional, Any, Union, List, Tuple
from concurrent.futures import ThreadPoolExecutor

# PDP: eventually we might want to also mask Galactic dust

class Spectra:
	def __init__(self, 
				 lat_lib: LATsky | SATsky,
				 common_dir: str,
				 aposcale: float = 2.0, 
				 template_bandpass: bool = False, 
				 pureB: bool = False,
				 CO: bool = True, 
				 PS: bool = True,
				 galcut: int | str | float = 0,
				 verbose: bool = True,
				 cache: bool = True,
				 parallel: int = 0,
				 dust_model: int = -1,
				 sync_model: int = -1,
				 binwidth: int = 1,
				 ) -> None:
		"""
		Initializes the Spectra class for computing and handling power spectra of observed CMB maps.

		Parameters:
		libdir (str): Directory where the spectra will be stored.
		lat_lib (LATsky): An instance of the LATsky class containing LAT-related configurations.
		aposcale (float, optional): Apodisation scale in degrees. Defaults to 2 deg
		template_bandpass (bool, optional): Apply bandpass integration to the foreground template. Defaults to False.
		pureB (bool, optional): Apply B-mode purification. Defaults to False
		CO (bool, optional): Mask the brightest regions of CO emission. Defautls to True.
		PS (bool, optional): Mask the brightest polarised extragalactic point sources. Defaults to True.
		"""
		self.logger = Logger(self.__class__.__name__, verbose)
		self.lat   = lat_lib
		self.nside = self.lat.nside
		libdir     = self.lat.libdir

		if dust_model != -1:
			assert sync_model != -1, "Both dust and sync models must be specified"
			self.dust_model = dust_model
			self.sync_model = sync_model
			self.logger.log(f"Evaluating special case: Simulation uses 'd{self.lat.dust_model}s{self.lat.sync_model}' FG model",'warning')
			self.logger.log(f"The template foreground is set to d{dust_model}s{sync_model}",'warning')
			fld_ext = f"_temp{dust_model}{sync_model}"
		else:
			self.dust_model = self.lat.dust_model
			self.sync_model = self.lat.sync_model
			fld_ext = ""
		self.__fld_ext__ = fld_ext
		

		deconv = self.lat.deconv_maps
		libdiri    = os.path.join(libdir, f"spectra_{self.nside}{'_d' if deconv else ''}_aposcale{str(aposcale).replace('.','p')}{'_pureB' if pureB else ''}" + fld_ext)
		comdir     = os.path.join(common_dir, f"spectra_{self.nside}{'_d' if deconv else ''}_aposcale{str(aposcale).replace('.','p')}{'_pureB' if pureB else ''}" + fld_ext)
		self.__set_dir__(libdiri, comdir)
		
		self.lmax     = 3500 #2 * self.lat.nside - 1
		
		self.temp_bp  = template_bandpass




		self.fg       = Foreground(self.lat.foreground.basedir, self.nside, self.dust_model, self.sync_model, self.temp_bp, verbose=False, fore_realization=self.lat.fore_realization)
		
		
		self.binwidth = binwidth
		self.binInfo  = nmt.NmtBin.from_lmax_linear(self.lmax, binwidth)
		self.Nell     = self.binInfo.get_n_bands()
		self.pureB    = pureB
		self.aposcale = aposcale
		self.CO       = CO
		self.PS       = PS
		self.galcut   = galcut
		self.mask     = self.lat.mask #self.get_apodised_mask()
		self.fsky     = self.lat.fsky #np.mean(self.mask**2)**2/np.mean(self.mask**4)
		
		# PDP: saving the spectra in this order makes the indexing of the mle easier
		self.freqs = self.lat.freqs
		self.Nfreq = len(self.freqs)
		self.bands = []
		for nu in self.freqs:
			for split in range(self.lat.nsplits):
				self.bands.append(f'{nu}-{split+1}')
		self.Nbands = len(self.bands)
		
		self.obs_qu_maps  = None
		self.dust_qu_maps = None
		self.sync_qu_maps = None

		self.workspace = nmt.NmtWorkspace()
		self.get_coupling_matrix()
		self.cache = cache
		self.parallel = parallel
		match self.parallel:
			case 0:
				msg = "No parallelization"
			case 1:
				msg = "Parallelized single loop"
			case 2:
				msg = "Parallelized double loop"
			case _:
				raise ValueError("Invalid parallelization option")
		self.logger.log(msg,'info')
		
		
	def get_apodised_mask(self) -> np.ndarray:
		fname = os.path.join(
			self.wdir,
			f"mask_N{self.nside}_aposcale{str(self.aposcale).replace('.','p')}{'_CO' if self.CO else ''}{'_PS' if self.PS else ''}{'_G'+str(self.galcut).replace('.','p') if self.galcut != 0 else ''}.fits",
		)
		if not os.path.isfile(fname):
			mask_str = self.lat.__class__.__name__[:3]
			if self.CO:
				mask_str += 'xCO'
			if self.PS:
				mask_str += 'xPS'
			if self.galcut != 0:
				mask_str += 'xGAL'
			maskobj = Mask(self.lat.basedir, self.nside, mask_str, self.aposcale,gal_cut=self.galcut)
			mask = maskobj.mask

			self.logger.log(f"Apodised mask saved to {fname}",'info')
			hp.write_map(fname, mask, dtype=np.float32)
			return mask
		else:
			self.logger.log(f"Reading apodised mask from {fname}",'info')
			return hp.read_map(fname)
			 
	def get_coupling_matrix(self) -> None:
		"""
		Computes or loads the coupling matrix for power spectrum estimation.
		"""
		fsky  = np.round(self.fsky, 2)
		fname = os.path.join(
			self.wdir,
			f"coupling_matrix_N{self.nside}_fsky{str(fsky).replace('.','p')}_aposcale{str(self.aposcale).replace('.','p')}_bw{self.binwidth}{'_CO' if self.CO else ''}{'_PS' if self.PS else ''}{'_pureB' if self.pureB else ''}{'_G'+str(self.galcut).replace('.','p') if self.galcut != 0 else ''}.fits",
		)
		if not os.path.isfile(fname):
			self.logger.log("Computing coupling Matrix",'info')
			mask_f = nmt.NmtField(
				self.mask, [self.mask, self.mask], lmax=self.lmax, purify_b=self.pureB
			)
			self.workspace.compute_coupling_matrix(mask_f, mask_f, self.binInfo)
			del mask_f
			self.workspace.write_to(fname)
			self.logger.log(f"Coupling Matrix saved to {fname}",'info')
		else:
			self.logger.log(f"Reading coupling Matrix from {fname}",'info')
			self.workspace.read_from(fname)

	def compute_master(self, f_a: nmt.NmtField, f_b: nmt.NmtField) -> np.ndarray:
		"""
		Computes the decoupled power spectrum using the MASTER algorithm.

		Parameters:
		f_a (nmt.NmtField): First NmtField object.
		f_b (nmt.NmtField): Second NmtField object.

		Returns:
		np.ndarray: Decoupled power spectrum.
		"""
		cl_coupled   = nmt.compute_coupled_cell(f_a, f_b)
		cl_decoupled = self.workspace.decouple_cell(cl_coupled)
		return cl_decoupled

	def __set_dir__(self, idir: str, cdir: str) -> None:
		"""
		Sets up directories for storing power spectra and workspaces.

		Parameters:
		dir (str): Directory for specific spectra.
		cdir (str): Common directory for spectra and workspaces.
		"""
		self.oxo_dir = os.path.join(idir,  "obs_x_obs")
		self.dxo_dir = os.path.join(idir,  "dust_x_obs")
		self.sxo_dir = os.path.join(idir,  "sync_x_obs")
		if self.lat.fore_realization:
			self.dxd_dir = os.path.join(idir, "dust_x_dust")
		else:
			self.dxd_dir = os.path.join(cdir, "dust_x_dust")
		self.sxs_dir = os.path.join(cdir, "sync_x_sync")
		if self.lat.fore_realization:
			self.sxd_dir = os.path.join(idir, "sync_x_dust")
		else:
			self.sxd_dir = os.path.join(cdir, "sync_x_dust")
		self.wdir    = os.path.join(cdir, "workspaces")
		if mpi.rank == 0:
			os.makedirs(self.oxo_dir, exist_ok=True)
			os.makedirs(self.dxo_dir, exist_ok=True)
			os.makedirs(self.sxo_dir, exist_ok=True)
			os.makedirs(self.dxd_dir, exist_ok=True)
			os.makedirs(self.sxs_dir, exist_ok=True)
			os.makedirs(self.sxd_dir, exist_ok=True)
			os.makedirs(self.wdir,    exist_ok=True)
		mpi.barrier()

	def load_obsQUmaps(self, idx: int) -> None:
		"""
		Loads observed Q and U Stokes parameter maps for all frequency bands.

		Parameters:
		idx (int): Index for the realization of the CMB map.
		"""
		maps = np.zeros((self.Nbands, 2, hp.nside2npix(self.nside)), dtype=np.float64)
		for i, band in enumerate(self.bands):
			maps[i] = self.lat.obsQU(idx, band)
		self.obs_qu_maps = maps

	def Obs_qu_maps(self, idx: int, ii: int) -> np.ndarray:
		return self.lat.obsQU(idx, self.bands[ii])

	def __get_fg_QUmap__(self, nu: str, fg: str, idx: int=None) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Retrieves or generates the Q and U Stokes parameter maps for dust emission for a specific frequency band.

		Parameters:
		band (str): The frequency identifier.
		fg (str): Foreground type, either 'dust' or 'sync'
		Returns:
		Tuple[np.ndarray, np.ndarray]: Q and U maps for dust emission.
		"""
		if fg not in ['dust', 'sync']:
			raise ValueError('Unknown foreground')
		if idx is None:
			fname = os.path.join(self.fg.libdir, f"{fg}QU_N{self.nside}_{nu}_template{'_bp' if self.temp_bp else ''}.fits")
		else:
			fname = os.path.join(self.fg.libdir, f"{fg}QU_N{self.nside}_{nu}_{idx:03d}_template{'_bp' if self.temp_bp else ''}.fits")
		if os.path.isfile(fname):
			m = hp.read_map(fname, field=(0, 1))
			return m[0], m[1]
		else:
			if fg=='dust':
				if idx is None:
					m = self.fg.dustQU(nu)
				else:
					m = self.fg.dustQU(nu,idx)
			elif fg=='sync':
				m = self.fg.syncQU(nu)
			E, B   = hp.map2alm_spin(m, 2, lmax=self.lmax)
			fwhm   = self.lat.fwhm[self.freqs==nu][0]
			bl     = hp.gauss_beam(np.radians(fwhm / 60), pol=True, lmax=self.lmax)
			pwf    = np.array(hp.pixwin(self.nside, pol=True, lmax=self.lmax))
			hp.almxfl(E, bl[:,1]*pwf[1,:], inplace=True)
			hp.almxfl(B, bl[:,2]*pwf[1,:], inplace=True)
			m      = hp.alm2map_spin([E, B], self.nside, 2, self.lmax)*self.mask
			hp.write_map(fname, m, dtype=np.float32)
			return m[0], m[1]

	def load_dustQUmaps(self) -> None:
		"""
		Loads dust Q and U Stokes parameter maps for all frequency bands.
		"""
		maps = np.zeros((self.Nfreq, 2, hp.nside2npix(self.nside)), dtype=np.float64)
		for i, nu in enumerate(self.freqs):
			maps[i] = self.__get_fg_QUmap__(nu, 'dust')
		self.dust_qu_maps = maps

	def Dust_qu_maps(self, ii: int, idx: int=None) -> np.ndarray:
		return self.__get_fg_QUmap__(self.freqs[ii], 'dust', idx=idx)

	def load_syncQUmaps(self) -> None:
		"""
		Loads synchrotron Q and U Stokes parameter maps for all frequency bands.
		"""
		maps = np.zeros((self.Nfreq, 2, hp.nside2npix(self.nside)), dtype=np.float32)
		for i, nu in enumerate(self.freqs):
			maps[i] = self.__get_fg_QUmap__(nu, 'sync')
		self.sync_qu_maps = maps

	def Sync_qu_maps(self, ii: int) -> np.ndarray:
		return self.__get_fg_QUmap__(self.freqs[ii], 'sync')

	def obs_x_obs_check(self, idx: int) -> None:
		"""
		Checks if the observed x observed power spectra have been computed for all frequency bands.

		Parameters:
		idx (int): Index for the realization of the CMB map.
		"""
		c = []
		for ii in range(self.Nbands):
			fname = os.path.join(
				self.oxo_dir,
				f"obs_x_obs_{self.bands[ii]}{'_obsBP' if self.lat.bandpass else ''}{'_d' if self.lat.deconv_maps else ''}_{idx:03d}.npy",
			)
			c.append(os.path.isfile(fname))
		return c

	def __obs_x_obs_helper_series__(self, ii: int, idx: int, recache: bool = False) -> np.ndarray:
		"""
		Helper function:
		Computes or loads the observed x observed power spectra for a specific frequency band.

		Parameters:
		ii (int): Index for the current frequency band.
		idx (int): Index for the realization of the CMB map.

		Returns:
		np.ndarray: Power spectra for the observed x observed fields.
		"""
		if self.__fld_ext__ != "":
			self.logger.log(f"Special case: Assumes that there exsist a previous run",'warning')
			self.logger.log(f"Special case: the default obsxobs directory is {self.oxo_dir}",'warning')
			oxo_dir = self.oxo_dir.replace(self.__fld_ext__,'')
			self.logger.log(f"Special case: the obsxobs directory is set to {oxo_dir}",'warning')
		else:
			oxo_dir = self.oxo_dir



		fname = os.path.join(
			oxo_dir,
			f"obs_x_obs_{self.bands[ii]}{'_obsBP' if self.lat.bandpass else ''}{'_d' if self.lat.deconv_maps else ''}_{idx:03d}.npy",
		)

		if os.path.isfile(fname) and not recache:
			try:
				return np.load(fname)
			except:
				self.logger.log(f"Error loading {fname}",'error')
				self.logger.log(f"Recomputing band:{ii},simulation:{idx}",'info')
				return self.__obs_x_obs_helper_series__(ii, idx, recache=True)
		else:
			cl = np.zeros(
				(self.Nbands, self.Nbands, 3, self.Nell + 2), dtype=np.float64
			)
			#assert self.obs_qu_maps is not None, "Observed Q and U maps not loaded"          #changed for the special case
			fp_i = nmt.NmtField(
				self.mask, self.Obs_qu_maps(idx,ii), lmax=self.lmax, purify_b=self.pureB,     #changed from self.obs_qu_maps[ii]
				masked_on_input=True
			)
			print(f"rank {mpi.rank} is running band {ii} and seed {idx} with OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
			for jj in range(ii, self.Nbands, 1):
				fp_j = nmt.NmtField(
					self.mask, self.Obs_qu_maps(idx,jj), lmax=self.lmax, purify_b=self.pureB, #changed from self.obs_qu_maps[jj]
					masked_on_input=True
				)

				cl_ij = self.compute_master(fp_i, fp_j)  # (EiEj, EiBj, BiEj, BiBj)

				cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
				cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
				cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj

				if ii != jj:
					cl[jj, ii, 0, 2:] = cl_ij[0, :]  # EjEi = EiEj
					cl[jj, ii, 1, 2:] = cl_ij[3, :]  # BjBi = BiBj
					cl[jj, ii, 2, 2:] = cl_ij[2, :]  # EjBi

				del fp_j
			
			if self.cache:
				np.save(fname, cl)
			return cl

	def __obs_x_obs_helper_parallel__(self, ii: int, idx: int, recache: bool = False) -> np.ndarray:
		"""
		Helper function:
		Computes or loads the observed x observed power spectra for a specific frequency band.

		Parameters:
		ii (int): Index for the current frequency band.
		idx (int): Index for the realization of the CMB map.

		Returns:
		np.ndarray: Power spectra for the observed x observed fields.
		"""
		if self.__fld_ext__ != "":
			self.logger.log(f"Special case: Assumes that there exsist a previous run",'warning')
			self.logger.log(f"Special case: the default obsxobs directory is {self.oxo_dir}",'warning')
			oxo_dir = self.oxo_dir.replace(self.__fld_ext__,'')
			self.logger.log(f"Special case: the obsxobs directory is set to {oxo_dir}",'warning')
		else:
			oxo_dir = self.oxo_dir

		fname = os.path.join(
			oxo_dir,
			f"obs_x_obs_{self.bands[ii]}{'_obsBP' if self.lat.bandpass else ''}{'_d' if self.lat.deconv_maps else ''}_{idx:03d}.npy",
		)
		if os.path.isfile(fname) and not recache:
			try:
				return np.load(fname)
			except:
				self.logger.log(f"Error loading {fname}",'error')
				self.logger.log(f"Recomputing band:{ii},simulation:{idx}",'info')
				return self.__obs_x_obs_helper_parallel__(ii, idx, recache=True)
		else:
			cl = np.zeros(
				(self.Nbands, self.Nbands, 3, self.Nell + 2), dtype=np.float64
			)
			# assert self.obs_qu_maps is not None, "Observed Q and U maps not loaded"     #changed for the special case
			fp_i = nmt.NmtField(
				self.mask, self.Obs_qu_maps(idx,ii), lmax=self.lmax, purify_b=self.pureB, #changed from self.obs_qu_maps[ii]
				masked_on_input=True
			)
			
			def compute_for_band(jj):
				# assert self.obs_qu_maps is not None, "Observed Q and U maps not loaded"   #changed for the special case
				fp_j = nmt.NmtField(
					self.mask, self.Obs_qu_maps(idx,jj), lmax=self.lmax, purify_b=self.pureB, #changed from self.obs_qu_maps[jj]
					masked_on_input=True
				)

				cl_ij = self.compute_master(fp_i, fp_j)  # (EiEj, EiBj, BiEj, BiBj)

				# Update the cl array in the appropriate positions
				cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
				cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
				cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj

				if ii != jj:
					cl[jj, ii, 0, 2:] = cl_ij[0, :]  # EjEi = EiEj
					cl[jj, ii, 1, 2:] = cl_ij[3, :]  # BjBi = BiBj
					cl[jj, ii, 2, 2:] = cl_ij[2, :]  # EjBi

				del fp_j

			# Use ThreadPoolExecutor to parallelize the loop
			with ThreadPoolExecutor() as executor:
				executor.map(compute_for_band, range(ii, self.Nbands, 1))
			
			if self.cache:
				np.save(fname, cl)
			return cl

	def __obs_x_obs_helper__(self, ii: int, idx: int) -> np.ndarray:
		if self.parallel == 2:
			return self.__obs_x_obs_helper_parallel__(ii, idx)
		else:
			return self.__obs_x_obs_helper_series__(ii, idx)

			
	def obs_x_obs(self, idx: int, progress: bool = False,) -> np.ndarray:
		"""
		Computes or loads the observed x observed power spectra for all frequency bands.

		Parameters:
		idx (int): Index for the realization of the CMB map.
		progress (bool, optional): If True, displays a progress bar. Defaults to False.
		parallel (int, optional): If 0, runs serially; otherwise, runs with multithreading. Defaults to 1.

		Returns:
		np.ndarray: Combined power spectra for the observed x observed fields across all bands.
		"""
		cl = np.zeros((self.Nbands, self.Nbands, 3, self.Nell + 2), dtype=np.float64)
		
		def process_band(ii):
			return self.__obs_x_obs_helper__(ii, idx)

		if self.parallel == 0:
			# Serial execution
			for ii in tqdm(
				range(self.Nbands),
				desc="obs x obs spectra",
				unit="band",
				disable=not progress,
			):
				cl += process_band(ii)
		else:
			# Parallel execution
			if progress:
				with ThreadPoolExecutor() as executor:
					for result in tqdm(executor.map(process_band, range(self.Nbands)),
									total=self.Nbands,
									desc="obs x obs spectra",
									unit="band"):
						cl += result
			else:
				with ThreadPoolExecutor() as executor:
					for result in executor.map(process_band, range(self.Nbands)):
						cl += result
		return cl

	def __fg_x_obs_helper_series__(self, ii: int, idx: int, fg: str, recache: bool = False) -> np.ndarray:
		"""
		Helper function:
		Computes or loads the dust x observed power spectra for a specific frequency band.

		Parameters:
		ii (int): Index for the current frequency band.
		idx (int): Index for the realization of the CMB map.
		fg (str): Type of foregrounds, either 'dust' or 'sync'
		Returns:
		np.ndarray: Power spectra for the dust x observed fields.
		"""
		if fg not in ['dust', 'sync']:
			raise ValueError('Unknown foreground')
			
		if fg=='dust':
			base_dir = self.dxo_dir
		elif fg=='sync':
			base_dir = self.sxo_dir
		fname = os.path.join(base_dir,
			f"{fg}_x_obs_{self.freqs[ii]}{'_obsBP' if self.lat.bandpass else ''}{'_tempBP' if self.temp_bp else ''}_{idx:03d}.npy",
		)
		
		if os.path.isfile(fname) and not recache:
			try:
				return np.load(fname)
			except:
				self.logger.log(f"Error loading {fname}",'error')
				self.logger.log(f"Recomputing band:{ii},simulation:{idx}, FG: {fg}",'info')
				return self.__fg_x_obs_helper_series__(ii, idx, fg, recache=True)


		else:
			cl = np.zeros((self.Nfreq, self.Nbands, 4, self.Nell + 2), dtype=np.float64)
			if fg=='dust':
				if self.lat.fore_realization:
					fp_i = nmt.NmtField(self.mask, self.Dust_qu_maps(ii,idx), lmax=self.lmax, purify_b=self.pureB,  #changed from self.dust_qu_maps[ii]
					masked_on_input=True)
				else:
					fp_i = nmt.NmtField(self.mask, self.Dust_qu_maps(ii), lmax=self.lmax, purify_b=self.pureB,  #changed from self.dust_qu_maps[ii]
					masked_on_input=True)
			elif fg=='sync':
				fp_i = nmt.NmtField(
					self.mask, self.Sync_qu_maps(ii), lmax=self.lmax, purify_b=self.pureB,  #changed from self.sync_qu_maps[ii]
					masked_on_input=True
				)
			for jj in range(0, self.Nbands, 1):
				fp_j = nmt.NmtField(
					self.mask, self.Obs_qu_maps(idx,jj), lmax=self.lmax, purify_b=self.pureB, #changed from self.obs_qu_maps[jj]
					masked_on_input=True
				)

				cl_ij = self.compute_master(fp_i,fp_j)  # (EiEj, EiBj, BiEj, BiBj)

				cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
				cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
				cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj
				cl[ii, jj, 3, 2:] = cl_ij[2, :]  # BiEj

				del fp_j
			
			if self.cache:
				np.save(fname, cl)
			return cl

	def __fg_x_obs_helper_parallel__(self, ii: int, idx: int, fg: str, recache: bool = False) -> np.ndarray:
		"""
		Helper function:
		Computes or loads the dust x observed power spectra for a specific frequency band.

		Parameters:
		ii (int): Index for the current frequency band.
		idx (int): Index for the realization of the CMB map.
		fg (str): Type of foregrounds, either 'dust' or 'sync'
		
		Returns:
		np.ndarray: Power spectra for the dust x observed fields.
		"""
		if fg not in ['dust', 'sync']:
			raise ValueError('Unknown foreground')
			
		base_dir = self.dxo_dir if fg == 'dust' else self.sxo_dir

		fname = os.path.join(
			base_dir,
			f"{fg}_x_obs_{self.freqs[ii]}{'_obsBP' if self.lat.bandpass else ''}{'_tempBP' if self.temp_bp else ''}_{idx:03d}.npy",
		)
		
		if os.path.isfile(fname) and not recache:
			try:
				return np.load(fname)
			except:
				self.logger.log(f"Error loading {fname}",'error')
				self.logger.log(f"Recomputing band:{ii},simulation:{idx}, FG: {fg}",'info')
				return self.__fg_x_obs_helper_parallel__(ii, idx, fg, recache=True)
		else:
			cl = np.zeros((self.Nfreq, self.Nbands, 4, self.Nell + 2), dtype=np.float64)

			# Choose the field based on the foreground type
			fp_i = nmt.NmtField(
				self.mask, 
				self.Dust_qu_maps(ii) if fg == 'dust' else self.Sync_qu_maps(ii),  #changed from self.dust_qu_maps[ii] and self.sync_qu_maps[ii]
				lmax=self.lmax, 
				purify_b=self.pureB,
				masked_on_input=True
			)

			def compute_for_band(jj):
				# Inner function to process each band in parallel
				fp_j = nmt.NmtField(
					self.mask, self.Obs_qu_maps(idx,jj), lmax=self.lmax, purify_b=self.pureB,  #changed from self.obs_qu_maps[jj]
					masked_on_input=True
				)

				cl_ij = self.compute_master(fp_i, fp_j)  # (EiEj, EiBj, BiEj, BiBj)

				# Update the cl array in the appropriate positions
				cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
				cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
				cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj
				cl[ii, jj, 3, 2:] = cl_ij[2, :]  # BiEj

				del fp_j



			# Use ThreadPoolExecutor to parallelize the inner loop
			with ThreadPoolExecutor() as executor:
				executor.map(compute_for_band, range(0, self.Nbands, 1))

			if self.cache:
				np.save(fname, cl)
			return cl

	def __fg_x_obs_helper__(self, ii: int, idx: int, fg: str) -> np.ndarray:
		if self.parallel == 2:
			return self.__fg_x_obs_helper_parallel__(ii, idx, fg)
		else:
			return self.__fg_x_obs_helper_series__(ii, idx, fg)

	def dust_x_obs(self, idx: int, progress: bool = False) -> np.ndarray:
		"""
		Computes or loads the dust x observed power spectra for all frequency bands.

		Parameters:
		idx (int): Index for the realization of the CMB map.
		progress (bool, optional): If True, displays a progress bar. Defaults to False.
		parallel (int, optional): If 0, runs serially; otherwise, runs with multithreading. Defaults to 1.

		Returns:
		np.ndarray: Combined power spectra for the dust x observed fields across all bands.
		"""
		cl = np.zeros((self.Nfreq, self.Nbands, 4, self.Nell + 2), dtype=np.float64)

		def process_band(ii):
			return self.__fg_x_obs_helper__(ii, idx, 'dust')

		if self.parallel == 0:
			# Serial execution
			for ii in tqdm(
				range(self.Nfreq),
				desc="dust x obs spectra",
				unit="band",
				disable=not progress,
			):
				cl += process_band(ii)
		else:
			# Parallel execution
			
			if progress:
				with ThreadPoolExecutor() as executor:
					for result in tqdm(executor.map(process_band, range(self.Nfreq)),
									total=self.Nfreq,
									desc="dust x obs spectra",
									unit="band"):
						cl += result
			else:
				with ThreadPoolExecutor() as executor:
					for result in executor.map(process_band, range(self.Nfreq)):
						cl += result
		
		return cl

	def dust_x_obs_check(self, idx: int) -> None:
		"""
		Checks if the dust x observed power spectra have been computed for all frequency bands.

		Parameters:
		idx (int): Index for the realization of the CMB map.
		"""
		c = []
		for ii in range(self.Nfreq):
			fname = os.path.join(
				self.dxo_dir,
				f"dust_x_obs_{self.freqs[ii]}{'_obsBP' if self.lat.bandpass else ''}{'_tempBP' if self.temp_bp else ''}_{idx:03d}.npy",
			)
			c.append(os.path.isfile(fname))
		return c

	def sync_x_obs_check(self, idx: int) -> None:
		"""
		Checks if the synchrotron x observed power spectra have been computed for all frequency bands.

		Parameters:
		idx (int): Index for the realization of the CMB map.
		"""
		c = []
		for ii in range(self.Nfreq):
			fname = os.path.join(
				self.sxo_dir,
				f"sync_x_obs_{self.freqs[ii]}{'_obsBP' if self.lat.bandpass else ''}{'_tempBP' if self.temp_bp else ''}_{idx:03d}.npy",
			)
			c.append(os.path.isfile(fname))
		return c



	def sync_x_obs(self, idx: int, progress: bool = False) -> np.ndarray:
		"""
		Computes or loads the synchrotron x observed power spectra for all frequency bands.

		Parameters:
		idx (int): Index for the realization of the CMB map.
		progress (bool, optional): If True, displays a progress bar. Defaults to False.
		parallel (int, optional): Controls parallelization.
								0 = serial, 2 = multithreading.
								Defaults to using self.parallel.

		Returns:
		np.ndarray: Combined power spectra for the synchrotron x observed fields across all bands.
		"""
		cl = np.zeros((self.Nfreq, self.Nbands, 4, self.Nell + 2), dtype=np.float64)

		def process_band(ii):
			return self.__fg_x_obs_helper__(ii, idx, 'sync')

		if self.parallel == 0:
			# Serial execution
			for ii in tqdm(
				range(self.Nfreq),
				desc="sync x obs spectra",
				unit="band",
				disable=not progress,
			):
				cl += process_band(ii)
		else:
			# Parallel execution using multithreading
			
			if progress:
				with ThreadPoolExecutor() as executor:
					for result in tqdm(executor.map(process_band, range(self.Nfreq)),
									total=self.Nfreq,
									desc="sync x obs spectra",
									unit="band"):
						cl += result
			else:
				with ThreadPoolExecutor() as executor:
					for result in executor.map(process_band, range(self.Nfreq)):
						cl += result



		return cl

	def __fg_x_fg_helper_series__(self, ii: int, fg: str, recache:bool = False, idx: int = None) -> np.ndarray:
		"""
		Helper function:
		Computes or loads the synchrotron x synchrotron power spectra for a specific frequency band.

		Parameters:
		ii (int): Index for the current frequency band.
		fg (str): Type of foregrounds, either 'dust' or 'sync'
		Returns:
		np.ndarray: Power spectra for the synchrotron x synchrotron fields.
		"""
		if fg not in ['dust', 'sync']:
			raise ValueError('Unknown foreground')
			
		if fg=='dust':
			base_dir = self.dxd_dir
			model = self.dust_model
		elif fg=='sync':
			base_dir = self.sxs_dir
			model = self.sync_model
		if self.lat.fore_realization:
			fname_help = f"{fg}_x_{fg}_{model}_{self.freqs[ii]}_{idx:03d}{'_tempBP' if self.temp_bp else ''}.npy"
		else:
			fname_help = f"{fg}_x_{fg}_{model}_{self.freqs[ii]}_{idx:03d}{'_tempBP' if self.temp_bp else ''}.npy"
		fname = os.path.join(base_dir,fname_help,)
		
		if os.path.isfile(fname):
			try:
				return np.load(fname)
			except:
				self.logger.log(f"Error loading {fname}",'error')
				if self.lat.fore_realization:
					self.logger.log(f"Recomputing band:{ii}, FG: {fg}, index: {idx}",'info')
					return self. __fg_x_fg_helper_series__(ii, fg, recache=True, idx=idx)
				else:
					self.logger.log(f"Recomputing band:{ii}, FG: {fg}",'info')
					return self. __fg_x_fg_helper_series__(ii, fg, recache=True)
		else:
			cl = np.zeros(
				(self.Nfreq, self.Nfreq, 3, self.Nell + 2), dtype=np.float64
			)
			if fg=='dust':
				if self.lat.fore_realization:
					fp_i = nmt.NmtField(self.mask, self.Dust_qu_maps(ii,idx=idx), lmax=self.lmax, purify_b=self.pureB, #changed from self.dust_qu_maps[ii]
					masked_on_input=True)
				else:
					fp_i = nmt.NmtField(self.mask, self.Dust_qu_maps(ii), lmax=self.lmax, purify_b=self.pureB, #changed from self.dust_qu_maps[ii]
						masked_on_input=True)
			elif fg=='sync':
				fp_i = nmt.NmtField(
					self.mask, self.Sync_qu_maps(ii), lmax=self.lmax, purify_b=self.pureB, #changed from self.sync_qu_maps[ii]
					masked_on_input=True
				)
				
			for jj in range(ii, self.Nfreq, 1):
				if fg=='dust':
					if self.lat.fore_realization:
						fp_j = nmt.NmtField(self.mask, self.Dust_qu_maps(jj, idx=idx), lmax=self.lmax, purify_b=self.pureB, #changed from self.dust_qu_maps[jj]
						masked_on_input=True)
					else:
						fp_j = nmt.NmtField(self.mask, self.Dust_qu_maps(jj), lmax=self.lmax, purify_b=self.pureB, #changed from self.dust_qu_maps[jj]
						masked_on_input=True)
				elif fg=='sync':
					fp_j = nmt.NmtField(
						self.mask, self.Sync_qu_maps(jj), lmax=self.lmax, purify_b=self.pureB, #changed from self.sync_qu_maps[jj]
						masked_on_input=True
					)

				cl_ij = self.compute_master(fp_i, fp_j)

				cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
				cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
				cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj

				if ii != jj:
					cl[jj, ii, 0, 2:] = cl_ij[0, :]  # EjEi = EiEj
					cl[jj, ii, 1, 2:] = cl_ij[3, :]  # BjBi = BiBj
					cl[jj, ii, 2, 2:] = cl_ij[2, :]  # EjBi

				del fp_j
			if self.cache:
				np.save(fname, cl)
			return cl

	def __fg_x_fg_helper_parallel__(self, ii: int, fg: str, recache:bool = False) -> np.ndarray:
		"""
		Helper function:
		Computes or loads the synchrotron x synchrotron power spectra for a specific frequency band.

		Parameters:
		ii (int): Index for the current frequency band.
		fg (str): Type of foregrounds, either 'dust' or 'sync'

		Returns:
		np.ndarray: Power spectra for the synchrotron x synchrotron fields.
		"""
		if fg not in ['dust', 'sync']:
			raise ValueError('Unknown foreground')
		
		base_dir = self.dxd_dir if fg == 'dust' else self.sxs_dir
		model = self.dust_model if fg == 'dust' else self.sync_model
		fname = os.path.join(
			base_dir,
			f"{fg}_x_{fg}_{model}{self.freqs[ii]}{'_tempBP' if self.temp_bp else ''}.npy",
		)
		
		if os.path.isfile(fname):
			try:
				return np.load(fname)
			except:
				self.logger.log(f"Error loading {fname}",'error')
				self.logger.log(f"Recomputing band:{ii}, FG: {fg}",'info')
				return self.__fg_x_fg_helper_parallel__(ii, fg, recache=True)
		else:
			cl = np.zeros((self.Nfreq, self.Nfreq, 3, self.Nell + 2), dtype=np.float64)

			if fg == 'dust':
				fp_i = nmt.NmtField(
					self.mask, self.Dust_qu_maps(ii), lmax=self.lmax, purify_b=self.pureB,  #changed from self.dust_qu_maps[ii]
					masked_on_input=True
				)
			elif fg == 'sync':
				fp_i = nmt.NmtField(
					self.mask, self.Sync_qu_maps(ii), lmax=self.lmax, purify_b=self.pureB, #changed from self.sync_qu_maps[ii]
					masked_on_input=True
				)

			def process_jj(jj):
				if fg == 'dust':
					fp_j = nmt.NmtField(
						self.mask, self.Dust_qu_maps(jj), lmax=self.lmax, purify_b=self.pureB, #changed from self.dust_qu_maps[jj]
						masked_on_input=True
					)
				elif fg == 'sync':
					fp_j = nmt.NmtField(
						self.mask, self.Sync_qu_maps(jj), lmax=self.lmax, purify_b=self.pureB, #changed from self.sync_qu_maps[jj]
						masked_on_input=True
					)
				
				cl_ij = self.compute_master(fp_i, fp_j)
				
				# Update cl for the given indices
				cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
				cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
				cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj

				if ii != jj:
					cl[jj, ii, 0, 2:] = cl_ij[0, :]  # EjEi = EiEj
					cl[jj, ii, 1, 2:] = cl_ij[3, :]  # BjBi = BiBj
					cl[jj, ii, 2, 2:] = cl_ij[2, :]  # EjBi

				del fp_j

			# Parallelize the loop over jj using ThreadPoolExecutor
			with ThreadPoolExecutor() as executor:
				executor.map(process_jj, range(ii, self.Nfreq, 1))
			
			if self.cache:
				np.save(fname, cl)
			return cl

	def __fg_x_fg_helper__(self, ii: int, fg: str, idx: int = None) -> np.ndarray:
		if self.parallel == 2:
			return self.__fg_x_fg_helper_parallel__(ii, fg)
		else:
			return self.__fg_x_fg_helper_series__(ii, fg, idx=idx)

	def sync_x_sync(self, progress: bool = False) -> np.ndarray:
		"""
		Computes or loads the synchrotron x synchrotron power spectra for all frequency bands.

		Parameters:
		progress (bool, optional): If True, displays a progress bar. Defaults to False.

		Returns:
		np.ndarray: Combined power spectra for the synchrotron x synchrotron fields across all bands.
		"""
		cl = np.zeros((self.Nfreq, self.Nfreq, 3, self.Nell + 2), dtype=np.float64)

		def process_band(ii):
			return self.__fg_x_fg_helper__(ii, 'sync')

		if self.parallel == 0:
			# Serial execution
			for ii in tqdm(
				range(self.Nfreq),
				desc="sync x sync spectra",
				unit="band",
				disable=not progress,
			):
				cl += process_band(ii)
		else:
			# Parallel execution using multithreading

			if progress:
				with ThreadPoolExecutor() as executor:
					for result in tqdm(executor.map(process_band, range(self.Nfreq)),
									total=self.Nfreq,
									desc="sync x sync spectra",
									unit="band"):
						cl += result
			else:
				with ThreadPoolExecutor() as executor:
					for result in executor.map(process_band, range(self.Nfreq)):
						cl += result


		return cl

	def dust_x_dust(self, progress: bool = False, idx: int = None) -> np.ndarray:
		"""
		Computes or loads the dust x dust power spectra for all frequency bands.

		Parameters:
		progress (bool, optional): If True, displays a progress bar. Defaults to False.
		
		idx (int, optional): Only needed if the foregrounds are different realizations

		Returns:
		np.ndarray: Combined power spectra for the dust x dust fields across all bands.
		"""
		cl = np.zeros((self.Nfreq, self.Nfreq, 3, self.Nell + 2), dtype=np.float64)

		def process_band(ii):
			return self.__fg_x_fg_helper__(ii, 'dust', idx=idx)

		if self.parallel == 0:
			# Serial execution
			for ii in tqdm(
				range(self.Nfreq),
				desc="dust x dust spectra",
				unit="band",
				disable=not progress,
			):
				cl += process_band(ii)
		else:
			# Parallel execution using multithreading

			if progress:
				with ThreadPoolExecutor() as executor:
					for result in tqdm(executor.map(process_band, range(self.Nfreq)),
									total=self.Nfreq,
									desc="dust x dust spectra",
									unit="band"):
						cl += result
			else:
				with ThreadPoolExecutor() as executor:
					for result in executor.map(process_band, range(self.Nfreq)):
						cl += result


		return cl

	def __sync_x_dust_helper_series__(self, ii: int) -> np.ndarray:
		"""
		Helper function:
		Computes or loads the synchrotron x dust power spectra for a specific frequency band.

		Parameters:
		ii (int): Index for the current frequency band.

		Returns:
		np.ndarray: Power spectra for the synchrotron x dust fields.
		"""
		fname = os.path.join(self.sxd_dir, f"sync{self.sync_model}_x_dust{self.dust_model}_{self.freqs[ii]}{'_tempBP' if self.temp_bp else ''}.npy")
		if os.path.isfile(fname):
			return np.load(fname)
		else:
			cl = np.zeros(
				(self.Nfreq, self.Nfreq, 4, self.Nell + 2), dtype=np.float64
			)
			fp_i = nmt.NmtField(
				self.mask, self.Sync_qu_maps(ii), lmax=self.lmax, purify_b=self.pureB, #changed from self.sync_qu_maps[ii]
				masked_on_input=True
			)
			for jj in range(0, self.Nfreq, 1):
				fp_j = nmt.NmtField(
					self.mask, self.Dust_qu_maps(jj), lmax=self.lmax, purify_b=self.pureB, #changed from self.dust_qu_maps[jj]
					masked_on_input=True
				)

				cl_ij = self.compute_master(fp_i,fp_j)  # (EiEj, EiBj, BiEj, BiBj)

				cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
				cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
				cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj
				cl[ii, jj, 3, 2:] = cl_ij[2, :]  # BiEj

				del fp_j
			
			if self.cache:
				np.save(fname, cl)
			return cl 

	def __sync_x_dust_helper_parallel__(self, ii: int) -> np.ndarray:
		"""
		Helper function:
		Computes or loads the synchrotron x dust power spectra for a specific frequency band.

		Parameters:
		ii (int): Index for the current frequency band.

		Returns:
		np.ndarray: Power spectra for the synchrotron x dust fields.
		"""
		fname = os.path.join(self.sxd_dir, f"sync{self.sync_model}_x_dust{self.dust_model}_{self.freqs[ii]}{'_tempBP' if self.temp_bp else ''}.npy")
		
		if os.path.isfile(fname):
			return np.load(fname)
		else:
			cl = np.zeros((self.Nfreq, self.Nfreq, 4, self.Nell + 2), dtype=np.float64)
			
			fp_i = nmt.NmtField(
				self.mask, self.Sync_qu_maps(ii), lmax=self.lmax, purify_b=self.pureB, #changed from self.sync_qu_maps[ii]
				masked_on_input=True
			)

			def process_jj(jj):
				fp_j = nmt.NmtField(
					self.mask, self.Dust_qu_maps(jj), lmax=self.lmax, purify_b=self.pureB, #changed from self.dust_qu_maps[jj]
					masked_on_input=True
				)

				cl_ij = self.compute_master(fp_i, fp_j)  # (EiEj, EiBj, BiEj, BiBj)

				# Update cl for the given indices
				cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
				cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
				cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj
				cl[ii, jj, 3, 2:] = cl_ij[2, :]  # BiEj

				del fp_j

			# Parallelize the loop over jj using ThreadPoolExecutor
			num_workers = os.cpu_count()  # Utilize all available CPU cores
			with ThreadPoolExecutor(max_workers=num_workers) as executor:
				executor.map(process_jj, range(self.Nfreq))
			
			if self.cache:
				np.save(fname, cl)
			return cl

	def __sync_x_dust_helper__(self, ii: int) -> np.ndarray:
		if self.parallel == 2:
			return self.__sync_x_dust_helper_parallel__(ii)
		else:
			return self.__sync_x_dust_helper_series__(ii)
		
	def sync_x_dust(self, progress: bool = False) -> np.ndarray:
		"""
		Computes or loads the synchrotron x dust power spectra for all frequency bands.

		Parameters:
		progress (bool, optional): If True, displays a progress bar. Defaults to False.

		Returns:
		np.ndarray: Combined power spectra for the synchrotron x dust fields across all bands.
		"""
		cl = np.zeros((self.Nfreq, self.Nfreq, 4, self.Nell + 2), dtype=np.float64)

		def process_band(ii):
			return self.__sync_x_dust_helper__(ii)

		if self.parallel == 0:
			# Serial execution
			for ii in tqdm(
				range(self.Nfreq),
				desc="sync x dust spectra",
				unit="band",
				disable=not progress,
			):
				cl += process_band(ii)
		else:
			# Parallel execution using multithreading
			if progress:
				with ThreadPoolExecutor() as executor:
					for result in tqdm(executor.map(process_band, range(self.Nfreq)),
									total=self.Nfreq,
									desc="sync x dust spectra",
									unit="band"):
						cl += result
			else:
				with ThreadPoolExecutor() as executor:
					for result in executor.map(process_band, range(self.Nfreq)):
						cl += result
	  


		return cl

	def clear_obs_qu_maps(self) -> None:
		"""Clears the loaded observed Q and U maps to free up memory."""
		self.obs_qu_maps = None

	def clear_dust_qu_maps(self) -> None:
		"""Clears the loaded dust Q and U maps to free up memory."""
		self.dust_qu_maps = None

	def clear_sync_qu_maps(self) -> None:
		"""Clears the loaded synchrotron Q and U maps to free up memory."""
		self.sync_qu_maps = None

	def compute(self, idx: int, sync: bool = False) -> None:
		"""
		Computes and stores all relevant spectra for a given realization index.

		Parameters:
		idx (int): Index for the realization of the CMB map.
		sync (bool, optional): If True, calculate also synchrotron power spectra. Defaults to False.
		"""
		self.load_dustQUmaps()
		dxd = self.dust_x_dust(progress=True)
		self.load_obsQUmaps(idx)
		oxo = self.obs_x_obs(idx, progress=True)
		dxo = self.dust_x_obs(idx, progress=True)
		if sync:
			self.load_syncQUmaps()
			sxd = self.sync_x_dust(progress=True)
			self.clear_dust_qu_maps()
			sxs = self.sync_x_sync(progress=True)
			sxo = self.sync_x_obs(idx, progress=True)
			self.clear_obs_qu_maps()
			self.clear_sync_qu_maps()
			del (oxo, dxo, dxd, sxd, sxs, sxo)
		else:
			self.clear_dust_qu_maps()
			self.clear_obs_qu_maps()
			del (oxo, dxo, dxd)

	def Compute(self, idx: int, sync: bool = False) -> None:
		dxd = self.dust_x_dust(progress=True)
		oxo = self.obs_x_obs(idx, progress=True)
		dxo = self.dust_x_obs(idx, progress=True)
		if sync:
			sxd = self.sync_x_dust(progress=True)
			sxs = self.sync_x_sync(progress=True)
			sxo = self.sync_x_obs(idx, progress=True)
			del (oxo, dxo, dxd, sxd, sxs, sxo)
		else:
			del (oxo, dxo, dxd)

	def get_spectra(self, idx: int, 
					sync: bool = False
	) -> Dict:
		"""
		Retrieves all relevant spectra for a given realization index.

		Parameters:
		idx (int): Index for the realization of the CMB map.
		sync (bool, optional): If True, calculate also synchrotron power spectra. Defaults to False.
		Returns:
		Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
			Tuple containing the power spectra (oxo, dxo, dxd, sxd, sxs, sxo).
		"""
		oxo = self.obs_x_obs(idx)
		dxo = self.dust_x_obs(idx)
		if self.lat.fore_realization:
			dxd = self.dust_x_dust(idx=idx)
		else:
			dxd = self.dust_x_dust()
		if sync:
			sxo = self.sync_x_obs(idx)
			sxs = self.sync_x_sync()
			sxd = self.sync_x_dust()
			return {'oxo':oxo, 'dxd':dxd, 'sxs':sxs, 'dxo':dxo, 'sxo':sxo, 'sxd':sxd}
		else:
			return {'oxo':oxo, 'dxd':dxd, 'dxo':dxo}
