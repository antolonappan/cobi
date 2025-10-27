import numpy as np
from tqdm import tqdm
from dataclasses import dataclass,field
from typing import Any, Optional
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

def selector(lib,lmin,lmax,):
    b = lib.binInfo.get_effective_ells()
    select = np.where((b>lmin) & (b<lmax))[0]
    return select
def selectorf(lib,avoid_freq):
    freqs = lib.lat.freqs
    select = np.where(np.array([freq not in avoid_freq for freq in freqs]))[0]
    return select

def get_sp(lib,lmax,avoid_freq=[]):
    bl_arr = []
    for i in range(2):# changed
        bl_arr.append(lib.binInfo.bin_cell(hp.gauss_beam(np.radians(lib.lat.fwhm[i]/60),lmax)**2))
    bl_arr = np.array(bl_arr)
    obs_arr = []
    # for i in range(100):
    #     sp = lib.obs_x_obs(i)
    #     obs_arr.append(sp)
    # obs_arr = np.array(obs_arr)
    # obs_arr = obs_arr[:,np.arange(2),np.arange(2),2,2:] # changed
    for i in range(100):
        sp = lib.obs_x_obs(i, False) 
        first  = (sp[1, 0, 2, 2:] + sp[0, 1, 2, 2:]) / 2
        second = (sp[2, 3, 2, 2:] + sp[3, 2, 2, 2:]) / 2
        obs_arr.append(np.stack([first, second], axis=0))
    obs_arr = np.asarray(obs_arr)         
    return obs_arr/bl_arr

def paranames(lib,name,avoid=[]):
    return [f"a{name}{fe}" for fe in lib.lat.freqs if fe not in avoid] 
def latexnames(lib, name, avoid=[]):
    return [r'\alpha_{{{}}}^{{{}}}'.format(name, fe) for fe in lib.lat.freqs if fe not in avoid]

class Sat4Lat:
    
    def __init__(self,libdir,lmin,lmax,latlib,satlib,sat_err,beta):
        
        latnc = latlib.lat.noise_model
        satnc = satlib.lat.noise_model
        if (latnc == 'NC') and (satnc == 'NC'):
            self.libdir = os.path.join(latlib.lat.libdir,'Calibration')
        elif (latnc == 'TOD') and (satnc == 'TOD'):
            self.libdir = os.path.join(latlib.lat.libdir,'Calibration_TOD')
        elif (latnc == 'TOD') and (satnc == 'NC'):
            self.libdir = os.path.join(latlib.lat.libdir,'Calibration_TOD_NC')
        elif (latnc == 'NC') and (satnc == 'TOD'):
            self.libdir = os.path.join(latlib.lat.libdir,'Calibration_NC_TOD')
        else:
            raise ValueError(f"Invalid noise model {latnc} {satnc}")

        os.makedirs(self.libdir,exist_ok=True)
        self.latlib = latlib
        self.sat_err = sat_err
        self.binner = satlib.binInfo
        self.Lmax = satlib.lmax
        self.__select__ = selector(satlib,lmin,lmax)
        self.addname = "bp" if latlib.lat.bandpass else ''

        self.lat_mean, self.lat_std = self.calc_mean_std(latlib,'LAT') 
        self.sat_mean, self.sat_std = self.calc_mean_std(satlib,'SAT')
        self.cl_len = CMB(libdir,satlib.lat.nside,beta=beta).get_lensed_spectra(dl=False)

        self.lmin = lmin
        self.lmax = lmax

 
        self.__pnames__ = paranames(latlib,'LAT') + paranames(satlib,'SAT') + ['beta']
        self.__plabels__ = latexnames(latlib,'LAT') + latexnames(satlib,'SAT')  + [r'\beta']

    def calc_mean_std(self,lib,name):
        sp = get_sp(lib,self.Lmax)
        if name == 'LAT':
            return ( sp.mean(axis=0)[:,self.__select__],
                     sp.std(axis=0)[:,self.__select__] )
        elif name == 'SAT':
            return ( sp.mean(axis=0)[:,self.__select__],
                     sp.std(axis=0)[:,self.__select__] )
        else:
            raise ValueError(f"Invalid name {name}")
    
    def plot_spectra(self,tele):
        plt.figure(figsize=(4,4))
        if tele == 'LAT' and self.latlib is not None:
            for i in range(2): # changed
                plt.loglog(self.binner.get_effective_ells()[self.__select__],self.lat_mean[i])
        elif tele == 'SAT': 
            for i in range(2): # changed
                plt.loglog(self.binner.get_effective_ells()[self.__select__],self.sat_mean[i])
        else:
            raise ValueError(f"Invalid telescope {tele}")
    
    def theory(self,beta_array):
        beta_array = np.asarray(beta_array)
        th = 0.5 * (self.cl_len["ee"] - self.cl_len["bb"])[:, np.newaxis] * np.sin(np.deg2rad(4 * beta_array))
        return np.apply_along_axis(lambda th_slice: self.binner.bin_cell(th_slice[:self.Lmax+1])[self.__select__], 0, th).T
    
    def chisq(self,theta):

        alpha_lat,alpha_sat,beta = np.array(theta[:2]), np.array(theta[2:4]), theta[-1]
        

        #diff_mean = self.lat_mean - self.sat_mean
        #diff_std = np.sqrt(self.lat_std**2 + self.sat_std**2)  
        #diff_model = self.theory(alpha_lat-alpha_sat)
        #diff_chi = np.sum(((diff_mean - diff_model)/diff_std)**2)

        sat_model = self.theory(np.ones(len(alpha_sat))*beta + alpha_sat)
        sat_chi = np.sum(((self.sat_mean - sat_model)/self.sat_std)**2)
        
        lat_model = self.theory(np.ones(len(alpha_lat))*beta + alpha_lat)
        lat_chi = np.sum(((self.lat_mean - lat_model)/self.lat_std)**2)

        return  sat_chi + lat_chi #+ diff_chi
    
    def lnprior(self,theta):
        sigma = self.sat_err
        alphalat,alphasat,beta = np.array(theta[:2]), np.array(theta[2:4]), theta[-1] # changed


        lnp = -0.5 * (alphasat - 0 )**2 / sigma**2 - np.log(sigma*np.sqrt(2*np.pi))
        

        if np.all(alphalat > -0.5) and np.all(alphalat < 0.5) and -0.1 < beta < 0.5:
            return np.sum(lnp)
        return -np.inf


    def ln_likelihood(self,theta):
        return -0.5 * self.chisq(theta)

    def ln_prob(self,theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta)
    
    
    def samples(self,nwalkers=32,nsamples=1000,rerun=False):
        true = np.array([0.2,0.2,0,0,0.35]) #changed
        fname = os.path.join(self.libdir,f"samples_{nwalkers}_{nsamples}{self.addname}.pkl")
        if os.path.exists(fname) and not rerun:
            return pl.load(open(fname,'rb'))
        else:
            ndim = len(true)
            pos = true + 1e-3 * np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, threads=4)
            sample = sampler.run_mcmc(pos, nsamples, progress=True)
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
            pl.dump(flat_samples,open(fname,'wb'))
            return flat_samples

    def getdist_samples(self,nwalkers,nsamples, rerun=False,label=None):
        flat_samples = self.samples(nwalkers,nsamples, rerun=rerun)
        return MCSamples(samples=flat_samples,names = self.__pnames__, labels = self.__plabels__,label=label)
        

    def plot_getdist(self,nwalkers,nsamples,avoid_sat=False,beta_only=False, rerun=False):
    
        flat_samples = self.getdist_samples(nwalkers,nsamples, rerun=rerun)
        if beta_only:
            g = plots.get_single_plotter(width_inch=4)
            g.plot_1d(flat_samples, 'beta', title_limit=1)
        else:
            names = self.__pnames__
            if avoid_sat:
                names = [item for item in names if 'SAT' not in item]
            g = plots.get_subplot_plotter()
            g.triangle_plot([flat_samples], names, filled=True,title_limit=1)

class Sat4Lat_AmplitudeFit:
    """
    Modified class to fit for an EB amplitude parameter (A_EB) and 
    miscalibration angles (alpha) instead of the birefringence angle (beta).
    """
    def __init__(self, libdir, lmin, lmax, latlib, satlib, sat_err, temp_model,temp_value,use_diag=True,idx=None,verbose=False):
        
        latnc = latlib.lat.noise_model
        satnc = satlib.lat.noise_model
        if (latnc == 'NC') and (satnc == 'NC'):
            self.libdir = os.path.join(latlib.lat.libdir, f"CalibrationTD_AmpFit_{temp_model}_{temp_value}")
        elif (latnc == 'TOD') and (satnc == 'TOD'):
            self.libdir = os.path.join(latlib.lat.libdir, f"Calibration_TODTD_AmpFit_{temp_model}_{temp_value}")
        # ... (add other elif conditions as in your original code) ...
        else:
            raise ValueError(f"Invalid noise model {latnc} {satnc}")

        os.makedirs(self.libdir, exist_ok=True)
        self.latlib = latlib
        self.sat_err = sat_err
        self.binner = satlib.binInfo
        self.Lmax = satlib.lmax
        self.__select__ = selector(satlib, lmin, lmax)
        self.addname = "bp" if latlib.lat.bandpass else ''
        
        self.use_diag = use_diag
        self.idx = idx
        self.lat_mean_90, self.lat_icov_90 = self.calc_mean_icov(latlib, 'LAT', 0)
        self.lat_mean_150, self.lat_icov_150 = self.calc_mean_icov(latlib, 'LAT', 1)
        self.sat_mean_90, self.sat_icov_90 = self.calc_mean_icov(satlib, 'SAT', 0)
        self.sat_mean_150, self.sat_icov_150 = self.calc_mean_icov(satlib, 'SAT', 1)
        if temp_model == 'iso':
            cmb = CMB(libdir,satlib.lat.nside,beta=temp_value,verbose=verbose)
            self.eb_template_unbinned = cmb.get_cb_lensed_spectra(dl=False)['eb']
        elif temp_model == 'iso_td':
            cmb = CMB(libdir,satlib.lat.nside,model=temp_model,mass=temp_value,verbose=verbose)
            self.eb_template_unbinned = cmb.get_cb_lensed_mass_spectra(dl=False)['eb']
        else:
            raise ValueError("only 'iso' and 'iso_td' allowed")
        
        ### MODIFIED ###
        # Store the binned version of the provided EB template.
        # The template should be the C_l^EB spectrum (not D_l).
        self.binned_template = self.binner.bin_cell(self.eb_template_unbinned[:self.Lmax+1])[self.__select__]
        
        # We still need the lensed EE and BB for the alpha calculation
        self.cl_len = CMB(libdir, satlib.lat.nside, beta=0.0,verbose=verbose).get_lensed_spectra(dl=False)

        self.lmin = lmin
        self.lmax = lmax

        ### MODIFIED ###
        # Updated parameter names and labels for A_EB instead of beta
        self.__pnames__ = paranames(latlib, 'LAT') + paranames(satlib, 'SAT') + ['A_EB']
        self.__plabels__ = latexnames(latlib, 'LAT') + latexnames(satlib, 'SAT') + [r'A_{EB}']
        self.dof = 2*len(self.__select__) * 2 - len(self.__pnames__) # 2 spectra (EB) for 2 freq for lat and sat
        


    def calc_mean_icov(self, lib, name, idx):
        """
        Calculates the mean and the inverse covariance matrix of the spectra.
        """
        USE_DIAG = self.use_diag
        sp = get_sp(lib, self.Lmax)[:,idx,self.__select__]
        if self.idx is not None:
            mean = sp[self.idx]
        else:
            mean = sp.mean(axis=0)
        cov = np.cov(sp.T)
        if USE_DIAG:
            cov = np.diag(np.diag(cov))
        else:
            alpha = 1e-5 
            reg_term = alpha * np.mean(np.diag(cov)) * np.identity(cov.shape[0])
            cov += reg_term
        icov = np.linalg.inv(cov)
        if name in ['LAT', 'SAT']:
            return (mean, icov)
        else:
            raise ValueError(f"Invalid name {name}")
        
    def plot_spectra(self, tele):
        plt.figure(figsize=(4, 4))
        if tele == 'LAT' and self.latlib is not None:
            for i in range(2):
                plt.loglog(self.binner.get_effective_ells()[self.__select__], self.lat_mean[i])
        elif tele == 'SAT':
            for i in range(2):
                plt.loglog(self.binner.get_effective_ells()[self.__select__], self.sat_mean[i])
        else:
            raise ValueError(f"Invalid telescope {tele}")
    
    def theory_alpha(self, alpha_array):
        ### MODIFIED ###
        # This function now *only* calculates the effect of the miscalibration angle alpha.
        # It's kept separate for clarity in the chisq function.
        alpha_array = np.asarray(alpha_array)
        # The EB spectrum induced by E-B mixing from a miscalibration angle alpha is sin(4*alpha)*(C_l^EE - C_l^BB)/2
        th = 0.5 * (self.cl_len["ee"] - self.cl_len["bb"])[:, np.newaxis] * np.sin(np.deg2rad(4 * alpha_array))
        return np.apply_along_axis(lambda th_slice: self.binner.bin_cell(th_slice[:self.Lmax+1])[self.__select__], 0, th).T

    def chisq(self, theta):
        alpha_lat, alpha_sat, A_EB = np.array(theta[:2]), np.array(theta[2:4]), theta[-1]
        birefringence_model = self.binned_template / A_EB
        alpha_lat_90, alpha_lat_150 = alpha_lat
        alpha_sat_90, alpha_sat_150 = alpha_sat
        
        #SAT
        sat_miscal_model_90 = self.theory_alpha(np.array([alpha_sat_90]))
        sat_total_model_90 = birefringence_model + sat_miscal_model_90
        diff_90 = self.sat_mean_90 - sat_total_model_90
        sat_chi_90 = diff_90 @ self.sat_icov_90 @ diff_90.T 
        sat_miscal_model_150 = self.theory_alpha(np.array([alpha_sat_150]))
        sat_total_model_150 = birefringence_model + sat_miscal_model_150
        diff_150 = self.sat_mean_150 - sat_total_model_150
        sat_chi_150 = diff_150 @ self.sat_icov_150 @ diff_150.T 
        sat_chi = sat_chi_90 + sat_chi_150

        #LAT
        lat_miscal_model_90 = self.theory_alpha(np.array([alpha_lat_90]))
        lat_total_model_90 = birefringence_model + lat_miscal_model_90
        diff_90 = self.lat_mean_90 - lat_total_model_90
        lat_chi_90 = diff_90 @ self.lat_icov_90 @ diff_90.T 
        lat_miscal_model_150 = self.theory_alpha(np.array([alpha_lat_150]))
        lat_total_model_150 = birefringence_model + lat_miscal_model_150
        diff_150 = self.lat_mean_150 - lat_total_model_150
        lat_chi_150 = diff_150 @ self.lat_icov_150 @ diff_150.T 
        lat_chi = lat_chi_90 + lat_chi_150      

        return sat_chi + lat_chi

    
    def MLE(self,Aeb):
        
        initial_guess = np.array([0.2, 0.2, 0.0, 0.0, Aeb])  # [alpha_lat1, alpha_lat2, alpha_sat1, alpha_sat2, A_EB]

        result = minimize(self.chisq, initial_guess, method='Nelder-Mead')

        if result.success:
            print("MLE optimization successful.")
            print("Optimized parameters:", result.x)
            print("Minimum chi-squared:", result.fun)
        else:
            raise RuntimeError("MLE optimization failed.")

    def lnprior(self, theta):
        sigma = self.sat_err
        ### MODIFIED ###
        # Unpack A_EB instead of beta
        alphalat, alphasat, A_EB = np.array(theta[:2]), np.array(theta[2:4]), theta[-1]

        lnp = -0.5 * (alphasat - 0)**2 / sigma**2 - np.log(sigma * np.sqrt(2 * np.pi))
        
        ### MODIFIED ###
        # Set a flat prior for A_EB. You can change the range [-1, 1] as needed.
        if np.all(alphalat > -0.5) and np.all(alphalat < 0.5) and 0 < A_EB < 2.0:
            return np.sum(lnp)
        return -np.inf

    def ln_likelihood(self, theta):
        return -0.5 * self.chisq(theta)

    def ln_prob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta)
    
    def samples(self, nwalkers=32, nsamples=1000, rerun=True):
        ### MODIFIED ###
        # Update the 'true' values for the MCMC initialization.
        # Assuming true alphas are ~0 and true A_EB is, for example, 0.3
    
        true = np.array([0.2, 0.2, 0, 0, 0.3]) 
        
        fname = os.path.join(self.libdir, f"samples_{nwalkers}_{nsamples}{self.addname}_max{self.lmax}_min{self.lmin}.pkl")
        if os.path.exists(fname) and not rerun:
            return pl.load(open(fname, 'rb'))
        else:
            ndim = len(true)
            pos = true + 1e-3 * np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, threads=4)
            sampler.run_mcmc(pos, nsamples, progress=True)
            flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)
            pl.dump(flat_samples, open(fname, 'wb'))
            return flat_samples

    def getdist_samples(self, nwalkers, nsamples, rerun=False, label=None):
        flat_samples = self.samples(nwalkers, nsamples, rerun=rerun)
        return MCSamples(samples=flat_samples, names=self.__pnames__, labels=self.__plabels__, label=label)
        
    def plot_getdist(self, nwalkers, nsamples, avoid_sat=False, aeb_only=False, rerun=False):
        ### MODIFIED ###
        # Added an 'aeb_only' option for convenience, similar to 'beta_only'
        flat_samples = self.getdist_samples(nwalkers, nsamples, rerun=rerun)
        if aeb_only:
            g = plots.get_single_plotter(width_inch=4)
            g.plot_1d(flat_samples, 'A_EB', title_limit=1)
        else:
            names = self.__pnames__
            if avoid_sat:
                names = [item for item in names if 'SAT' not in item]
            g = plots.get_subplot_plotter()
            g.triangle_plot([flat_samples], names, filled=True, title_limit=1)
    
    def mle(self,nwalkers,nsamples,rerun=False):
        s = self.samples(nwalkers,nsamples,rerun=rerun)
        theta = np.median(s,axis=0)
        chi2_min = self.chisq(theta)
        return chi2_min
    
    def p_value(self,nwalkers,nsamples,rerun=False):
        chi2_min = self.mle(nwalkers,nsamples,rerun=rerun)
        p_val = chi2.sf(chi2_min, self.dof)
        return p_val



class Sat4LatCross:
    """
    Performs a calibration analysis using cross-power spectra between LAT and SAT.

    This class takes a SpectraCross object, which contains the machinery to compute
    cross-spectra between different telescope channels, and uses them to constrain
    instrumental calibration parameters (alpha) and a global sky rotation angle (beta).
    """

    def __init__(self, spec_cross: 'SpectraCross', sat_err: float, beta_fid: float,
                 sat_lrange: tuple = (None, None), lat_lrange: tuple = (None, None),
                 fit_per_split: bool = True, spectra_selection: str = 'all'):
        """
        Initializes the analysis pipeline.

        Args:
            spec_cross (SpectraCross): An initialized SpectraCross object.
            sat_err (float): The standard deviation of the Gaussian prior on SAT alpha parameters.
            beta_fid (float): Fiducial beta used for generating the CMB theory spectra.
            sat_lrange (tuple, optional): (lmin, lmax) range for SAT spectra.
            lat_lrange (tuple, optional): (lmin, lmax) range for LAT spectra.
            fit_per_split (bool, optional): If True, fit alpha for each split.
                                            If False, use one alpha per frequency across all splits.
            spectra_selection (str, optional): 'all', 'auto_only', or 'cross_only'.
        """
        print("Initializing Sat4LatCross analysis...")
        self.spec_cross = spec_cross
        self.libdir = os.path.join(spec_cross.libdir, 'CalibrationAnalysis')
        os.makedirs(self.libdir, exist_ok=True)

        self.sat_err = sat_err
        self.sat_lrange = sat_lrange
        self.lat_lrange = lat_lrange
        self.fit_per_split = fit_per_split
        if spectra_selection not in ['all', 'auto_only', 'cross_only']:
            raise ValueError("spectra_selection must be one of 'all', 'auto_only', or 'cross_only'")
        self.spectra_selection = spectra_selection
        self.binner = spec_cross.binInfo
        self.Lmax = spec_cross.lmax

        # --- Build map tags and frequency grouping ---
        self.maptags = self._get_maptags()
        self.freq_groups = {}  # map freq base -> list of indices
        for i, tag in enumerate(self.maptags):
            base = tag.rsplit('-', 1)[0]  # e.g. LAT_93
            if base not in self.freq_groups:
                self.freq_groups[base] = []
            self.freq_groups[base].append(i)
        self.freq_bases = list(self.freq_groups.keys())

        # --- Parameter naming ---
        if self.fit_per_split:
            self.__pnames__ = [f"a_{tag}" for tag in self.maptags] + ['beta']
            self.__plabels__ = [r'\alpha_{{{}}}'.format(tag.replace('_', ' ')) for tag in self.maptags] + [r'\beta']
        else:
            self.__pnames__ = [f"a_{base}" for base in self.freq_bases] + ['beta']
            self.__plabels__ = [r'\alpha_{{{}}}'.format(base.replace('_', ' ')) for base in self.freq_bases] + [r'\beta']

        print(f"Parameters to constrain: {', '.join(self.__pnames__)}")

        # --- Build mask and load spectra ---
        self.__likelihood_mask__ = self._build_likelihood_mask()
        self.mean_spec, self.std_spec = self._calc_mean_std(num_sims=50)
        self.beam_arr = self._get_beam_arr()

        # --- Load CMB theory spectra ---
        self.cl_len = CMB(spec_cross.libdir, spec_cross.nside, beta=beta_fid).get_lensed_spectra(dl=False)
        print("Initialization complete.")

    # -------------------------------------------------------------------------
    def _get_maptags(self):
        """Helper function to get map tags."""
        return self.spec_cross.maptags.copy()

    # -------------------------------------------------------------------------
    def _build_likelihood_mask(self):
        ells = self.binner.get_effective_ells()
        sat_lmin, sat_lmax = self.sat_lrange
        lat_lmin, lat_lmax = self.lat_lrange
        sat_ell_mask = np.ones_like(ells, dtype=bool)
        if sat_lmin is not None: sat_ell_mask &= (ells >= sat_lmin)
        if sat_lmax is not None: sat_ell_mask &= (ells <= sat_lmax)
        lat_ell_mask = np.ones_like(ells, dtype=bool)
        if lat_lmin is not None: lat_ell_mask &= (ells >= lat_lmin)
        if lat_lmax is not None: lat_ell_mask &= (ells <= lat_lmax)
        cross_ell_mask = lat_ell_mask & sat_ell_mask

        n_tags = len(self.maptags)
        n_bins = self.binner.get_n_bands()
        mask = np.zeros((n_tags, n_tags, n_bins), dtype=bool)
        is_lat = np.array([tag.startswith('LAT') for tag in self.maptags])
        lat_auto = is_lat[:, None] & is_lat[None, :]
        sat_auto = ~is_lat[:, None] & ~is_lat[None, :]
        cross = ~(lat_auto | sat_auto)
        mask[lat_auto] = lat_ell_mask
        mask[sat_auto] = sat_ell_mask
        mask[cross] = cross_ell_mask
        if self.spectra_selection == 'auto_only':
            mask &= (lat_auto | sat_auto)[:, :, None]
        elif self.spectra_selection == 'cross_only':
            mask &= cross[:, :, None]
        return mask

    # -------------------------------------------------------------------------
    def _calc_mean_std(self, num_sims: int):
        fname = os.path.join(self.libdir, f'mean_std_spec_{num_sims}_sims.pkl')
        if os.path.exists(fname):
            print("Loading pre-computed mean and std deviation...")
            with open(fname, 'rb') as f:
                return pl.load(f)

        print(f"Calculating mean and std deviation over {num_sims} simulations...")
        all_spectra = []
        for i in tqdm(range(num_sims), desc="Processing Simulations"):
            spec_matrix = self.spec_cross.data_matrix(
                i, which='EB',
                sat_lrange=self.sat_lrange,
                lat_lrange=self.lat_lrange,
                avg_splits=False  # keep splits, never average
            )
            all_spectra.append(spec_matrix)
        all_spectra = np.array(all_spectra)
        mean_spec = np.mean(all_spectra, axis=0)
        std_spec = np.std(all_spectra, axis=0)
        with open(fname, 'wb') as f:
            pl.dump((mean_spec, std_spec), f)
        return mean_spec, std_spec

    # -------------------------------------------------------------------------
    def _get_beam_arr(self):
        n_tags = len(self.maptags)
        n_bins = self.binner.get_n_bands()
        beam_matrix = np.ones((n_tags, n_tags, n_bins))
        fwhm_dict = {
            f'LAT_{freq}': fwhm for freq, fwhm in zip(self.spec_cross.lat.freqs, self.spec_cross.lat.fwhm)
        }
        fwhm_dict.update({
            f'SAT_{freq}': fwhm for freq, fwhm in zip(self.spec_cross.sat.freqs, self.spec_cross.sat.fwhm)
        })
        for i, tag_i in enumerate(self.maptags):
            for j, tag_j in enumerate(self.maptags):
                base_i = tag_i.rsplit('-', 1)[0]
                base_j = tag_j.rsplit('-', 1)[0]
                beam_i = hp.gauss_beam(np.radians(fwhm_dict[base_i] / 60.), lmax=self.Lmax)**2
                beam_j = hp.gauss_beam(np.radians(fwhm_dict[base_j] / 60.), lmax=self.Lmax)**2
                cross_beam = np.sqrt(beam_i * beam_j)
                beam_matrix[i, j, :] = self.binner.bin_cell(cross_beam)
        return beam_matrix

    # -------------------------------------------------------------------------
    def theory(self, theta: np.ndarray):
        """Computes theoretical EB cross-power spectra matrix for given parameters."""
        if self.fit_per_split:
            alphas = theta[:-1]
        else:
            # One alpha per frequency; replicate across splits
            base_alphas = {base: theta[i] for i, base in enumerate(self.freq_bases)}
            alphas = np.zeros(len(self.maptags))
            for i, tag in enumerate(self.maptags):
                base = tag.rsplit('-', 1)[0]
                alphas[i] = base_alphas[base]
        beta = theta[-1]

        cl_diff = 0.5 * (self.cl_len["ee"] - self.cl_len["bb"])
        theory_matrix = np.zeros_like(self.mean_spec)
        for i in range(len(self.maptags)):
            for j in range(len(self.maptags)):
                angle_term = np.sin(np.deg2rad(2 * alphas[i] + 2 * alphas[j] + 4 * beta))
                th_cl = cl_diff * angle_term
                theory_matrix[i, j, :] = self.binner.bin_cell(th_cl[:self.Lmax+1])
        return theory_matrix

    # -------------------------------------------------------------------------
    def chisq(self, theta: np.ndarray):
        model = self.theory(theta)
        data = self.mean_spec / self.beam_arr
        error = self.std_spec / self.beam_arr
        error_safe = np.where(error == 0, np.inf, error)
        chi_sq = ((data - model) / error_safe)**2
        return np.sum(chi_sq[self.__likelihood_mask__])

    # -------------------------------------------------------------------------
    def lnprior(self, theta: np.ndarray):
        num_alphas = len(theta) - 1
        beta = theta[-1]
        if self.fit_per_split:
            alphas = theta[:-1]
            tag_list = self.maptags
        else:
            alphas = theta[:-1]
            tag_list = self.freq_bases

        sat_indices = [i for i, tag in enumerate(tag_list) if tag.startswith('SAT')]
        alphas_sat = alphas[np.array(sat_indices)]
        lnp_sat = -0.5 * np.sum(alphas_sat**2 / self.sat_err**2)
        if np.all(np.abs(alphas) < 0.5) and 0 < beta < 0.5:
            return lnp_sat
        return -np.inf

    # -------------------------------------------------------------------------
    def ln_likelihood(self, theta: np.ndarray):
        return -0.5 * self.chisq(theta)

    # -------------------------------------------------------------------------
    def ln_prob(self, theta: np.ndarray):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta)

    # -------------------------------------------------------------------------
    def run_mcmc(self, nwalkers: int = 32, nsamples: int = 2000, rerun: bool = False):
        fname = os.path.join(
            self.libdir,
            f"samples_cross_{nwalkers}_{nsamples}_fit_per_split_{self.fit_per_split}.pkl"
        )
        if os.path.exists(fname) and not rerun:
            print(f"Loading existing samples from {fname}")
            with open(fname, 'rb') as f:
                return pl.load(f)

        ndim = len(self.__pnames__)
        print(f"Running MCMC with {nwalkers} walkers for {nsamples} steps...")
        fiducial_params = np.zeros(ndim)
        pos = fiducial_params + 1e-1 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, threads=8)
        sampler.run_mcmc(pos, nsamples, progress=True)
        flat_samples = sampler.get_chain(discard=200, thin=20, flat=True)
        with open(fname, 'wb') as f:
            pl.dump(flat_samples, f)
        return flat_samples

    # -------------------------------------------------------------------------
    def getdist_samples(self, nwalkers: int, nsamples: int, rerun: bool = False, label: str = None):
        flat_samples = self.run_mcmc(nwalkers, nsamples, rerun=rerun)
        return MCSamples(samples=flat_samples, names=self.__pnames__,
                         labels=self.__plabels__, label=label)

    # -------------------------------------------------------------------------
    def plot_posteriors(self, nwalkers: int, nsamples: int, rerun: bool = False):
        print("Generating posterior triangle plot...")
        samples = self.getdist_samples(nwalkers, nsamples, rerun=rerun, label='Cross-Spectra Analysis')
        g = plots.get_subplot_plotter()
        g.triangle_plot([samples], filled=True, title_limit=1)
        print("Plotting complete.")
        return g
    

    def plot_spectra_matrix(self, theta=None, save_path=None, average_split=False):
        """
        Plots the mean data spectra with std deviation and optionally a theory curve
        in a matrix layout corresponding to all cross-correlations.

        Args:
            theta (np.ndarray, optional): Parameter vector (alphas, beta) for theory overlay.
            save_path (str, optional): If given, saves the figure instead of showing.
            average_split (bool, optional): If True, average spectra across splits per frequency
                                            before plotting. Default is False.
        """
        ells = self.binner.get_effective_ells()

        # Optionally average over splits for plotting only
        if average_split:
            freq_bases = list(self.freq_groups.keys())
            n_freqs = len(freq_bases)
            n_bins = self.binner.get_n_bands()

            data_spec = self.mean_spec / self.beam_arr
            error_spec = self.std_spec / self.beam_arr

            # build averaged matrix
            data_avg = np.zeros((n_freqs, n_freqs, n_bins))
            err_avg = np.zeros_like(data_avg)

            for i, base_i in enumerate(freq_bases):
                idx_i = self.freq_groups[base_i]
                for j, base_j in enumerate(freq_bases):
                    idx_j = self.freq_groups[base_j]
                    # collect all split-pairs
                    vals = [data_spec[ii, jj, :] for ii in idx_i for jj in idx_j]
                    errs = [error_spec[ii, jj, :] for ii in idx_i for jj in idx_j]
                    data_avg[i, j, :] = np.nanmean(vals, axis=0)
                    err_avg[i, j, :] = np.nanmean(errs, axis=0)

            maptags = freq_bases
            data_spec, error_spec = data_avg, err_avg
            if theta is not None:
                theory_spec = self.theory(theta)
                # average theory too
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

        # Start plotting
        n_tags = len(maptags)
        fig, axes = plt.subplots(n_tags, n_tags, figsize=(n_tags * 3, n_tags * 3),
                                 sharex=True, sharey='row')

        for i in range(n_tags):
            for j in range(n_tags):
                ax = axes[i, j]
                is_diagonal = (i == j)

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

                ax.errorbar(ells, data_spec[i, j], yerr=error_spec[i, j],
                            fmt='.', capsize=2, color='black', markersize=4, alpha=0.5,
                            label='Data Mean ± 1σ')

                if theory_spec is not None:
                    ax.loglog(ells, theory_spec[i, j], color='red', label='Theory')

                mask_1d = self.__likelihood_mask__[0, 0, :] if average_split else self.__likelihood_mask__[i, j, :]
                used_ells = ells[mask_1d]
                if len(used_ells) > 0:
                    ax.axvspan(used_ells.min(), used_ells.max(), color='green', alpha=0.1, zorder=-1)

                if i == n_tags - 1:
                    ax.set_xlabel(r'Multipole $\ell$')
                if j == 0:
                    ax.set_ylabel(r'$C_\ell^{EB}$')
                if i == 0:
                    ax.set_title(maptags[j].replace('_', ' '), fontsize=10)

                ax.axhline(0, color='grey', linestyle=':', linewidth=0.75)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_ylim(1e-8, 1e-1)
                ax.set_xlim(10, 3000)

        # right-side labels
        for i, tag in enumerate(maptags):
            axes[i, n_tags - 1].text(1.05, 0.5, tag.replace('_', ' '),
                                     transform=axes[i, n_tags - 1].transAxes,
                                     va='center', ha='left', fontsize=10)

        handles, labels = [], []
        for ax in axes.flatten():
            if ax.get_visible():
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
                break

        if handles:
            fig.legend(handles, labels, loc='upper right')

        plt.tight_layout(rect=[0.03, 0.03, 0.92, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Figure saved to {save_path}")
            plt.close()
        else:
            plt.show()


class Sat4LatCross_AmplitudeFit:
    """
    Performs a calibration analysis using cross-power spectra between LAT and SAT.

    Modified class to fit for an EB amplitude parameter (A_EB) and 
    miscalibration angles (alpha) instead of the birefringence angle (beta).
    """

    def __init__(self, spec_cross: 'SpectraCross', sat_err: float,
                 temp_model: str, temp_value: float,
                 sat_lrange: tuple = (None, None), lat_lrange: tuple = (None, None),
                 fit_per_split: bool = True, spectra_selection: str = 'all',
                 verbose: bool = False):
        """
        Initializes the analysis pipeline.

        Args:
            spec_cross (SpectraCross): An initialized SpectraCross object.
            sat_err (float): The standard deviation of the Gaussian prior on SAT alpha parameters.
            temp_model (str): The model for the EB template (e.g., 'iso', 'iso_td').
            temp_value (float): The value for the template model (e.g., beta or mass).
            sat_lrange (tuple, optional): (lmin, lmax) range for SAT spectra.
            lat_lrange (tuple, optional): (lmin, lmax) range for LAT spectra.
            fit_per_split (bool, optional): If True, fit alpha for each split.
                                            If False, use one alpha per frequency across all splits.
            spectra_selection (str, optional): 'all', 'auto_only', or 'cross_only'.
            verbose (bool, optional): Verbosity flag for CMB class.
        """
        print("Initializing Sat4LatCross_AmplitudeFit analysis...")
        self.spec_cross = spec_cross
        self.libdir = os.path.join(spec_cross.libdir, f'CalibrationAnalysis_AmpFit_{temp_model}_{temp_value}')
        os.makedirs(self.libdir, exist_ok=True)

        self.sat_err = sat_err
        self.sat_lrange = sat_lrange
        self.lat_lrange = lat_lrange
        self.fit_per_split = fit_per_split
        if spectra_selection not in ['all', 'auto_only', 'cross_only']:
            raise ValueError("spectra_selection must be one of 'all', 'auto_only', or 'cross_only'")
        self.spectra_selection = spectra_selection
        self.binner = spec_cross.binInfo
        self.Lmax = spec_cross.lmax

        # --- Build map tags and frequency grouping ---
        self.maptags = self._get_maptags()
        self.freq_groups = {}  # map freq base -> list of indices
        for i, tag in enumerate(self.maptags):
            base = tag.rsplit('-', 1)[0]  # e.g. LAT_93
            if base not in self.freq_groups:
                self.freq_groups[base] = []
            self.freq_groups[base].append(i)
        self.freq_bases = list(self.freq_groups.keys())

        # --- Parameter naming ---
        if self.fit_per_split:
            self.__pnames__ = [f"a_{tag}" for tag in self.maptags] + ['A_EB']
            self.__plabels__ = [r'\alpha_{{{}}}'.format(tag.replace('_', ' ')) for tag in self.maptags] + [r'A_{EB}']
        else:
            self.__pnames__ = [f"a_{base}" for base in self.freq_bases] + ['A_EB']
            self.__plabels__ = [r'\alpha_{{{}}}'.format(base.replace('_', ' ')) for base in self.freq_bases] + [r'A_{EB}']

        print(f"Parameters to constrain: {', '.join(self.__pnames__)}")

        # --- Build mask and load spectra ---
        self.__likelihood_mask__ = self._build_likelihood_mask()
        self.mean_spec, self.std_spec = self._calc_mean_std(num_sims=50)
        self.beam_arr = self._get_beam_arr()

        # --- Load CMB theory spectra ---
        # 1. Load EB template based on model and value
        print(f"Loading EB template: model={temp_model}, value={temp_value}")
        if temp_model == 'iso':
            cmb_template = CMB(spec_cross.libdir, spec_cross.nside, beta=temp_value, verbose=verbose)
            self.eb_template_unbinned = cmb_template.get_cb_lensed_spectra(dl=False)['eb']
        elif temp_model == 'iso_td':
            cmb_template = CMB(spec_cross.libdir, spec_cross.nside, model=temp_model, mass=temp_value, verbose=verbose)
            self.eb_template_unbinned = cmb_template.get_cb_lensed_mass_spectra(dl=False)['eb']
        else:
            raise ValueError("only 'iso' and 'iso_td' allowed")
        
        # Bin the template
        self.binned_template = self.binner.bin_cell(self.eb_template_unbinned[:self.Lmax+1])
        
        # 2. Load lensed spectra (with beta=0) for alpha calculation
        self.cl_len = CMB(spec_cross.libdir, spec_cross.nside, beta=0.0, verbose=verbose).get_lensed_spectra(dl=False)
        self.cl_diff_unbinned = 0.5 * (self.cl_len["ee"] - self.cl_len["bb"])
        
        print("Initialization complete.")

    # -------------------------------------------------------------------------
    def _get_maptags(self):
        """Helper function to get map tags."""
        return self.spec_cross.maptags.copy()

    # -------------------------------------------------------------------------
    def _build_likelihood_mask(self):
        ells = self.binner.get_effective_ells()
        sat_lmin, sat_lmax = self.sat_lrange
        lat_lmin, lat_lmax = self.lat_lrange
        sat_ell_mask = np.ones_like(ells, dtype=bool)
        if sat_lmin is not None: sat_ell_mask &= (ells >= sat_lmin)
        if sat_lmax is not None: sat_ell_mask &= (ells <= sat_lmax)
        lat_ell_mask = np.ones_like(ells, dtype=bool)
        if lat_lmin is not None: lat_ell_mask &= (ells >= lat_lmin)
        if lat_lmax is not None: lat_ell_mask &= (ells <= lat_lmax)
        cross_ell_mask = lat_ell_mask & sat_ell_mask

        n_tags = len(self.maptags)
        n_bins = self.binner.get_n_bands()
        mask = np.zeros((n_tags, n_tags, n_bins), dtype=bool)
        is_lat = np.array([tag.startswith('LAT') for tag in self.maptags])
        lat_auto = is_lat[:, None] & is_lat[None, :]
        sat_auto = ~is_lat[:, None] & ~is_lat[None, :]
        cross = ~(lat_auto | sat_auto)
        mask[lat_auto] = lat_ell_mask
        mask[sat_auto] = sat_ell_mask
        mask[cross] = cross_ell_mask
        if self.spectra_selection == 'auto_only':
            mask &= (lat_auto | sat_auto)[:, :, None]
        elif self.spectra_selection == 'cross_only':
            mask &= cross[:, :, None]
        return mask

    # -------------------------------------------------------------------------
    def _calc_mean_std(self, num_sims: int):
        fname = os.path.join(self.libdir, f'mean_std_spec_{num_sims}_sims.pkl')
        if os.path.exists(fname):
            print("Loading pre-computed mean and std deviation...")
            with open(fname, 'rb') as f:
                return pl.load(f)

        print(f"Calculating mean and std deviation over {num_sims} simulations...")
        all_spectra = []
        for i in tqdm(range(num_sims), desc="Processing Simulations"):
            spec_matrix = self.spec_cross.data_matrix(
                i, which='EB',
                sat_lrange=self.sat_lrange,
                lat_lrange=self.lat_lrange,
                avg_splits=False  # keep splits, never average
            )
            all_spectra.append(spec_matrix)
        all_spectra = np.array(all_spectra)
        mean_spec = np.mean(all_spectra, axis=0)
        std_spec = np.std(all_spectra, axis=0)
        with open(fname, 'wb') as f:
            pl.dump((mean_spec, std_spec), f)
        return mean_spec, std_spec

    # -------------------------------------------------------------------------
    def _get_beam_arr(self):
        n_tags = len(self.maptags)
        n_bins = self.binner.get_n_bands()
        beam_matrix = np.ones((n_tags, n_tags, n_bins))
        fwhm_dict = {
            f'LAT_{freq}': fwhm for freq, fwhm in zip(self.spec_cross.lat.freqs, self.spec_cross.lat.fwhm)
        }
        fwhm_dict.update({
            f'SAT_{freq}': fwhm for freq, fwhm in zip(self.spec_cross.sat.freqs, self.spec_cross.sat.fwhm)
        })
        for i, tag_i in enumerate(self.maptags):
            for j, tag_j in enumerate(self.maptags):
                base_i = tag_i.rsplit('-', 1)[0]
                base_j = tag_j.rsplit('-', 1)[0]
                beam_i = hp.gauss_beam(np.radians(fwhm_dict[base_i] / 60.), lmax=self.Lmax)**2
                beam_j = hp.gauss_beam(np.radians(fwhm_dict[base_j] / 60.), lmax=self.Lmax)**2
                cross_beam = np.sqrt(beam_i * beam_j)
                beam_matrix[i, j, :] = self.binner.bin_cell(cross_beam)
        return beam_matrix

    # -------------------------------------------------------------------------
    # --- THEORY COMPUTATION ---
    # -------------------------------------------------------------------------

    def _get_alphas(self, theta: np.ndarray) -> np.ndarray:
        """Helper to expand alphas if not fitting per split."""
        if self.fit_per_split:
            return theta[:-1]
        else:
            # One alpha per frequency; replicate across splits
            base_alphas = {base: theta[i] for i, base in enumerate(self.freq_bases)}
            alphas = np.zeros(len(self.maptags))
            for i, tag in enumerate(self.maptags):
                base = tag.rsplit('-', 1)[0]
                alphas[i] = base_alphas[base]
            return alphas

    def theory_miscal_matrix(self, alphas: np.ndarray):
        """Computes the EB spectrum matrix from miscalibration (alpha) only."""
        theory_matrix = np.zeros_like(self.mean_spec)
        for i in range(len(self.maptags)):
            for j in range(len(self.maptags)):
                # Model is 0.5 * (Cl_EE - Cl_BB) * sin(2*alpha_i + 2*alpha_j)
                angle_term = np.sin(np.deg2rad(2 * alphas[i] + 2 * alphas[j]))
                th_cl = self.cl_diff_unbinned * angle_term
                theory_matrix[i, j, :] = self.binner.bin_cell(th_cl[:self.Lmax+1])
        return theory_matrix

    def theory(self, theta: np.ndarray):
        """Computes total theoretical EB cross-power spectra matrix."""
        alphas = self._get_alphas(theta)
        A_EB = theta[-1]

        # 1. Miscalibration part
        miscal_model = self.theory_miscal_matrix(alphas)
        
        # 2. Birefringence template part
        # Following Sat4Lat_AmplitudeFit logic: template / A_EB
        # This has shape (n_bins,)
        birefringence_1d = self.binned_template / A_EB
        
        # Broadcast to (n_tags, n_tags, n_bins)
        birefringence_model = birefringence_1d[None, None, :]

        # 3. Total model is the sum
        return miscal_model + birefringence_model

    # -------------------------------------------------------------------------
    # --- LIKELIHOOD AND MCMC ---
    # -------------------------------------------------------------------------

    def chisq(self, theta: np.ndarray):
        """Computes the chi-squared value."""
        model = self.theory(theta)
        data = self.mean_spec / self.beam_arr
        error = self.std_spec / self.beam_arr
        error_safe = np.where(error == 0, np.inf, error)
        chi_sq = ((data - model) / error_safe)**2
        return np.sum(chi_sq[self.__likelihood_mask__])

    # -------------------------------------------------------------------------
    def lnprior(self, theta: np.ndarray):
        num_alphas = len(theta) - 1
        A_EB = theta[-1]  # Changed from beta

        if self.fit_per_split:
            alphas = theta[:-1]
            tag_list = self.maptags
        else:
            alphas = theta[:-1]
            tag_list = self.freq_bases

        sat_indices = [i for i, tag in enumerate(tag_list) if tag.startswith('SAT')]
        alphas_sat = alphas[np.array(sat_indices)]
        lnp_sat = -0.5 * np.sum(alphas_sat**2 / self.sat_err**2)
        
        # Changed prior range from beta to A_EB
        if np.all(np.abs(alphas) < 0.5) and 0 < A_EB < 2.0:
            return lnp_sat
        return -np.inf

    # -------------------------------------------------------------------------
    def ln_likelihood(self, theta: np.ndarray):
        return -0.5 * self.chisq(theta)

    # -------------------------------------------------------------------------
    def ln_prob(self, theta: np.ndarray):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta)

    # -------------------------------------------------------------------------
    def run_mcmc(self, nwalkers: int = 32, nsamples: int = 2000, rerun: bool = False):
        fname = os.path.join(
            self.libdir,
            f"samples_cross_ampfit_{nwalkers}_{nsamples}_fit_per_split_{self.fit_per_split}.pkl"
        )
        if os.path.exists(fname) and not rerun:
            print(f"Loading existing samples from {fname}")
            with open(fname, 'rb') as f:
                return pl.load(f)

        ndim = len(self.__pnames__)
        print(f"Running MCMC with {nwalkers} walkers for {nsamples} steps...")
        
        # Updated fiducial parameters for A_EB
        fiducial_params = np.zeros(ndim)
        fiducial_params[-1] = 0.3  # Set fiducial A_EB (e.g., 0.3 or 1.0)
        
        pos = fiducial_params + 1e-1 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, threads=8)
        sampler.run_mcmc(pos, nsamples, progress=True)
        flat_samples = sampler.get_chain(discard=200, thin=20, flat=True)
        with open(fname, 'wb') as f:
            pl.dump(flat_samples, f)
        return flat_samples

    # -------------------------------------------------------------------------
    def getdist_samples(self, nwalkers: int, nsamples: int, rerun: bool = False, label: str = None):
        flat_samples = self.run_mcmc(nwalkers, nsamples, rerun=rerun)
        if label is None:
            label = 'Cross-Spectra AmpFit Analysis'
        return MCSamples(samples=flat_samples, names=self.__pnames__,
                         labels=self.__plabels__, label=label)

    # -------------------------------------------------------------------------
    # --- PLOTTING ---
    # -------------------------------------------------------------------------

    def plot_posteriors(self, nwalkers: int, nsamples: int, rerun: bool = False):
        print("Generating posterior triangle plot...")
        samples = self.getdist_samples(nwalkers, nsamples, rerun=rerun)
        g = plots.get_subplot_plotter()
        g.triangle_plot([samples], filled=True, title_limit=1)
        print("Plotting complete.")
        return g
    
    def plot_spectra_matrix(self, theta=None, save_path=None, average_split=False):
        """
        Plots the mean data spectra with std deviation and optionally a theory curve
        in a matrix layout corresponding to all cross-correlations.

        Args:
            theta (np.ndarray, optional): Parameter vector (alphas, A_EB) for theory overlay.
            save_path (str, optional): If given, saves the figure instead of showing.
            average_split (bool, optional): If True, average spectra across splits per frequency
                                            before plotting. Default is False.
        """
        ells = self.binner.get_effective_ells()

        # Optionally average over splits for plotting only
        if average_split:
            freq_bases = list(self.freq_groups.keys())
            n_freqs = len(freq_bases)
            n_bins = self.binner.get_n_bands()

            data_spec = self.mean_spec / self.beam_arr
            error_spec = self.std_spec / self.beam_arr

            # build averaged matrix
            data_avg = np.zeros((n_freqs, n_freqs, n_bins))
            err_avg = np.zeros_like(data_avg)

            for i, base_i in enumerate(freq_bases):
                idx_i = self.freq_groups[base_i]
                for j, base_j in enumerate(freq_bases):
                    idx_j = self.freq_groups[base_j]
                    # collect all split-pairs
                    vals = [data_spec[ii, jj, :] for ii in idx_i for jj in idx_j]
                    errs = [error_spec[ii, jj, :] for ii in idx_i for jj in idx_j]
                    data_avg[i, j, :] = np.nanmean(vals, axis=0)
                    err_avg[i, j, :] = np.nanmean(errs, axis=0)

            maptags = freq_bases
            data_spec, error_spec = data_avg, err_avg
            if theta is not None:
                theory_spec = self.theory(theta)
                # average theory too
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

        # Start plotting
        n_tags = len(maptags)
        fig, axes = plt.subplots(n_tags, n_tags, figsize=(n_tags * 3, n_tags * 3),
                                 sharex=True, sharey='row')

        for i in range(n_tags):
            for j in range(n_tags):
                ax = axes[i, j]
                is_diagonal = (i == j)

                plot_this = False
                if self.spectra_selection == 'all':
                    plot_this = True
                elif self.spectra_selection == 'auto_only' and is_diagonal:
                    plot_this = True
                elif self.spectra_selection == 'cross_only' and not is_diagonal:
                    plot_this = True
                
                # Determine the correct mask index for this plot
                if average_split:
                    # Find original indices to check mask
                    orig_i = self.freq_groups[maptags[i]][0]
                    orig_j = self.freq_groups[maptags[j]][0]
                    mask_1d = self.__likelihood_mask__[orig_i, orig_j, :]
                else:
                    mask_1d = self.__likelihood_mask__[i, j, :]
                
                # Hide plots that are not selected and have no data in the mask
                if not plot_this or not np.any(mask_1d):
                    ax.set_visible(False)
                    continue

                ax.errorbar(ells, data_spec[i, j], yerr=error_spec[i, j],
                            fmt='.', capsize=2, color='black', markersize=4, alpha=0.5,
                            label='Data Mean ± 1σ')

                if theory_spec is not None:
                    ax.plot(ells, theory_spec[i, j], color='red', label='Theory') # Use plot for line

                
                used_ells = ells[mask_1d]
                if len(used_ells) > 0:
                    ax.axvspan(used_ells.min(), used_ells.max(), color='green', alpha=0.1, zorder=-1)

                if i == n_tags - 1:
                    ax.set_xlabel(r'Multipole $\ell$')
                if j == 0:
                    ax.set_ylabel(r'$C_\ell^{EB}$')
                if i == 0:
                    ax.set_title(maptags[j].replace('_', ' '), fontsize=10)

                ax.axhline(0, color='grey', linestyle=':', linewidth=0.75)
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # Set log scale
                ax.set_yscale('log')
                ax.set_xscale('log')
                
                # Set reasonable y-limits, avoiding non-positive values
                y_min = np.nanmin(data_spec[i,j][data_spec[i,j] > 0]) / 10 if np.any(data_spec[i,j] > 0) else 1e-9
                y_max = np.nanmax(data_spec[i,j]) * 10 if np.any(data_spec[i,j] > 0) else 1e-5
                if theory_spec is not None:
                     y_min = min(y_min, np.nanmin(theory_spec[i,j][theory_spec[i,j] > 0]) / 10) if np.any(theory_spec[i,j] > 0) else y_min
                     y_max = max(y_max, np.nanmax(theory_spec[i,j]) * 10) if np.any(theory_spec[i,j] > 0) else y_max
                
                ax.set_ylim(max(y_min, 1e-9), min(y_max, 1e-1)) # Set practical bounds
                ax.set_xlim(ells[ells>0].min()*0.8, ells.max()*1.2)


        # right-side labels
        for i, tag in enumerate(maptags):
            if axes[i, n_tags - 1].get_visible():
                axes[i, n_tags - 1].text(1.05, 0.5, tag.replace('_', ' '),
                                         transform=axes[i, n_tags - 1].transAxes,
                                         va='center', ha='left', fontsize=10)

        handles, labels = [], []
        for ax in axes.flatten():
            if ax.get_visible():
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
                break # Get legend from first visible plot

        if handles:
            # Avoid duplicate labels
            by_label = dict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.tight_layout(rect=[0.03, 0.03, 0.92, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Figure saved to {save_path}")
            plt.close()
        else:
            plt.show()