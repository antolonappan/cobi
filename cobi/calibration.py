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
    def __init__(self, libdir, lmin, lmax, latlib, satlib, sat_err, temp_model,temp_value,verbose=False):
        
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

        self.lat_mean, self.lat_std = self.calc_mean_std(latlib, 'LAT')
        self.sat_mean, self.sat_std = self.calc_mean_std(satlib, 'SAT')
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


    def calc_mean_std(self, lib, name):
        sp = get_sp(lib, self.Lmax)
        if name in ['LAT', 'SAT']:
            return (sp.mean(axis=0)[:, self.__select__],
                    sp.std(axis=0)[:, self.__select__])
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
        ### MODIFIED ###
        # The parameters in theta are now the alphas and A_EB
        alpha_lat, alpha_sat, A_EB = np.array(theta[:2]), np.array(theta[2:4]), theta[-1]

        # The theoretical model is now the sum of two parts:
        # 1. The birefringence signal: A_EB * template
        # 2. The miscalibration signal: calculated from alpha
        birefringence_model =  self.binned_template/A_EB

        # Calculate chi-squared for SAT
        sat_miscal_model = self.theory_alpha(alpha_sat)
        sat_total_model = birefringence_model + sat_miscal_model
        sat_chi = np.sum(((self.sat_mean - sat_total_model) / self.sat_std)**2)
        
        # Calculate chi-squared for LAT
        lat_miscal_model = self.theory_alpha(alpha_lat)
        lat_total_model = birefringence_model + lat_miscal_model
        lat_chi = np.sum(((self.lat_mean - lat_total_model) / self.lat_std)**2)

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

class Sat4Lat_AmplitudeFit_test:
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

    def __init__(self, spec_cross: 'SpectraCross', sat_err: float, beta_fid: float, sat_lrange: tuple = (None, None), lat_lrange: tuple = (None, None), avg_splits: bool = True, spectra_selection: str = 'all'):
        """
        Initializes the analysis pipeline.

        Args:
            spec_cross (SpectraCross): An initialized SpectraCross object.
            sat_err (float): The standard deviation of the Gaussian prior on the SAT alpha parameters (in radians).
            beta_fid (float): The fiducial value of beta used for generating the CMB theory spectra.
            sat_lrange (tuple, optional): The (lmin, lmax) range for SAT spectra. Defaults to (None, None).
            lat_lrange (tuple, optional): The (lmin, lmax) range for LAT spectra. Defaults to (None, None).
            avg_splits (bool, optional): If True, average spectra over data splits before analysis. Defaults to True.
            spectra_selection (str, optional): Which spectra to use: 'all', 'auto_only', 'cross_only'. Defaults to 'all'.
        """
        print("Initializing Sat4LatCross analysis...")
        self.spec_cross = spec_cross
        self.libdir = os.path.join(spec_cross.libdir, 'CalibrationAnalysis')
        os.makedirs(self.libdir, exist_ok=True)

        self.sat_err = sat_err
        self.sat_lrange = sat_lrange
        self.lat_lrange = lat_lrange
        self.avg_splits = avg_splits
        if spectra_selection not in ['all', 'auto_only', 'cross_only']:
            raise ValueError("spectra_selection must be one of 'all', 'auto_only', or 'cross_only'")
        self.spectra_selection = spectra_selection
        self.binner = spec_cross.binInfo
        self.Lmax = spec_cross.lmax

        # Get parameter names from the map tags (averaged over splits or not)
        self.maptags = self._get_maptags()
        self.__pnames__ = [f"a_{tag}" for tag in self.maptags] + ['beta']
        self.__plabels__ = [r'\alpha_{{{}}}'.format(tag.replace('_', ' ')) for tag in self.maptags] + [r'\beta']
        print(f"Parameters to constrain: {', '.join(self.__pnames__)}")

        # Build the likelihood mask from the ell ranges
        self.__likelihood_mask__ = self._build_likelihood_mask()

        # Calculate mean and standard deviation of spectra over 100 simulations
        self.mean_spec, self.std_spec = self._calc_mean_std(num_sims=50)

        # Pre-compute beam and CMB theory spectra
        self.beam_arr = self._get_beam_arr()
        # This assumes a CMB class similar to the one in the first code snippet is available
        self.cl_len = CMB(spec_cross.libdir, spec_cross.nside, beta=beta_fid).get_lensed_spectra(dl=False)
        print("Initialization complete.")

    def _get_maptags(self):
        """Helper function to get map tags, optionally averaging over splits."""
        if self.avg_splits:
            unique_tags = []
            for tag in self.spec_cross.maptags:
                base_tag = tag.rsplit('-', 1)[0]
                if base_tag not in unique_tags:
                    unique_tags.append(base_tag)
            return unique_tags
        else:
            return self.spec_cross.maptags.copy()

    def _build_likelihood_mask(self):
        """
        Constructs a 3D boolean mask for the chi-squared calculation based on
        instrument-specific ell ranges and spectra selection type.
        """
        ells = self.binner.get_effective_ells()
        
        # Create 1D ell masks for each instrument type
        sat_lmin, sat_lmax = self.sat_lrange
        lat_lmin, lat_lmax = self.lat_lrange
        
        sat_ell_mask = np.ones_like(ells, dtype=bool)
        if sat_lmin is not None: sat_ell_mask &= (ells >= sat_lmin)
        if sat_lmax is not None: sat_ell_mask &= (ells <= sat_lmax)
        
        lat_ell_mask = np.ones_like(ells, dtype=bool)
        if lat_lmin is not None: lat_ell_mask &= (ells >= lat_lmin)
        if lat_lmax is not None: lat_ell_mask &= (ells <= lat_lmax)

        cross_ell_mask = lat_ell_mask & sat_ell_mask

        # Create a 3D mask with shape (n_tags, n_tags, n_bins)
        n_tags = len(self.maptags)
        n_bins = self.binner.get_n_bands()
        likelihood_mask = np.zeros((n_tags, n_tags, n_bins), dtype=bool)

        # Create 2D masks for correlation types
        is_lat = np.array([tag.startswith('LAT') for tag in self.maptags])
        lat_auto_mask_2d = is_lat[:, np.newaxis] & is_lat[np.newaxis, :]
        sat_auto_mask_2d = ~is_lat[:, np.newaxis] & ~is_lat[np.newaxis, :]
        cross_mask_2d = ~(lat_auto_mask_2d | sat_auto_mask_2d)

        # Populate the 3D mask using broadcasting
        likelihood_mask[lat_auto_mask_2d] = lat_ell_mask
        likelihood_mask[sat_auto_mask_2d] = sat_ell_mask
        likelihood_mask[cross_mask_2d] = cross_ell_mask

        # Filter the mask based on the spectra selection ('auto_only', 'cross_only')
        if self.spectra_selection == 'auto_only':
            selection_mask_2d = lat_auto_mask_2d | sat_auto_mask_2d
            likelihood_mask &= selection_mask_2d[:, :, np.newaxis]
        elif self.spectra_selection == 'cross_only':
            selection_mask_2d = cross_mask_2d
            likelihood_mask &= selection_mask_2d[:, :, np.newaxis]
        
        return likelihood_mask

    def _calc_mean_std(self, num_sims: int):
        """
        Calculates the mean and standard deviation of the data matrix over multiple simulations.
        """
        sat_lmin, sat_lmax = self.sat_lrange
        lat_lmin, lat_lmax = self.lat_lrange
        fname = os.path.join(self.libdir, f'mean_std_spec_{num_sims}_sims_sat_{sat_lmin}_{sat_lmax}_lat_{lat_lmin}_{lat_lmax}_avg_splits_{self.avg_splits}.pkl')
        if os.path.exists(fname):
            print("Loading pre-computed mean and std deviation...")
            with open(fname, 'rb') as f:
                return pl.load(f)

        print(f"Calculating mean and std deviation over {num_sims} simulations...")
        all_spectra = []
        for i in tqdm(range(num_sims), desc="Processing Simulations"):
            spec_matrix = self.spec_cross.data_matrix(
                i,
                which='EB',
                avg_splits=self.avg_splits,
                sat_lrange=self.sat_lrange,
                lat_lrange=self.lat_lrange
            )
            all_spectra.append(spec_matrix)

        all_spectra = np.array(all_spectra)
        mean_spec = np.mean(all_spectra, axis=0)
        std_spec = np.std(all_spectra, axis=0)
        
        with open(fname, 'wb') as f:
            pl.dump((mean_spec, std_spec), f)
            
        return mean_spec, std_spec

    def _get_beam_arr(self):
        """
        Computes the binned beam window function matrix for all cross-correlations.
        The beam for a cross-spectrum is the geometric mean of the two individual beams.
        """
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
                # Get base tag to look up FWHM, e.g., 'LAT_93' from 'LAT_93-1'
                base_tag_i = tag_i.rsplit('-', 1)[0]
                base_tag_j = tag_j.rsplit('-', 1)[0]
                
                beam_i = hp.gauss_beam(np.radians(fwhm_dict[base_tag_i] / 60.), lmax=self.Lmax)**2
                beam_j = hp.gauss_beam(np.radians(fwhm_dict[base_tag_j] / 60.), lmax=self.Lmax)**2

                cross_beam_cl = np.sqrt(beam_i * beam_j)
                beam_matrix[i, j, :] = self.binner.bin_cell(cross_beam_cl)
                
        return beam_matrix

    def theory(self, theta: np.ndarray):
        """
        Computes the theoretical EB cross-power spectrum matrix for a given set of parameters.

        The EB signal is generated by a rotation of the polarization plane. For a cross-spectrum
        between two detectors i and j, the signal is proportional to sin(2*(psi_i + psi_j)),
        where psi is the total rotation angle for each detector.
        """
        num_alphas = len(self.maptags)
        alphas = theta[:num_alphas]
        beta = theta[num_alphas]

        cl_diff = 0.5 * (self.cl_len["ee"] - self.cl_len["bb"])
        theory_matrix = np.zeros_like(self.mean_spec)
        
        for i in range(num_alphas):
            for j in range(num_alphas):
                # Total rotation angle for the cross-spectrum term
                # psi_i = alpha_i + beta, psi_j = alpha_j + beta
                # The EB term is sin(2*(psi_i + psi_j)) = sin(2*alpha_i + 2*alpha_j + 4*beta)
                angle_term = np.sin(np.deg2rad(2 * alphas[i] + 2 * alphas[j] + 4 * beta))
                th_cl = cl_diff * angle_term
                theory_matrix[i, j, :] = self.binner.bin_cell(th_cl[:self.Lmax+1])
                
        return theory_matrix

    def chisq(self, theta: np.ndarray):
        """Calculates the chi-squared value for a given set of parameters."""
        model = self.theory(theta)
        
        data = self.mean_spec / self.beam_arr
        error = self.std_spec / self.beam_arr
        
        # Avoid division by zero if std is zero
        error_safe = np.where(error == 0, np.inf, error)
        
        chi_sq_matrix = ((data - model) / error_safe)**2
        
        # Apply the pre-computed 3D likelihood mask
        return np.sum(chi_sq_matrix[self.__likelihood_mask__])

    def lnprior(self, theta: np.ndarray):
        """Defines the log-prior for the parameters."""
        num_alphas = len(self.maptags)
        alphas = theta[:num_alphas]
        beta = theta[num_alphas]

        # Gaussian priors for SAT alphas centered at 0
        sat_indices = [i for i, tag in enumerate(self.maptags) if tag.startswith('SAT')]
        alphas_sat = alphas[np.array(sat_indices)]
        lnp_sat = -0.5 * np.sum(alphas_sat**2 / self.sat_err**2)

        # Flat priors for LAT alphas and beta within reasonable bounds
        if np.all(np.abs(alphas) < 0.5) and 0 < beta < 0.5:
             return lnp_sat
        return -np.inf

    def ln_likelihood(self, theta: np.ndarray):
        """Log-likelihood function."""
        return -0.5 * self.chisq(theta)

    def ln_prob(self, theta: np.ndarray):
        """Log-probability function (prior + likelihood)."""
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta)

    def run_mcmc(self, nwalkers: int = 32, nsamples: int = 2000, rerun: bool = False):
        """
        Runs the MCMC sampler to get posterior samples of the parameters.
        """
        fname = os.path.join(self.libdir, f"samples_cross_{nwalkers}_{nsamples}_avg_splits_{self.avg_splits}_selection_{self.spectra_selection}.pkl")
        if os.path.exists(fname) and not rerun:
            print(f"Loading existing samples from {fname}")
            with open(fname, 'rb') as f:
                return pl.load(f)
        
        print(f"Running MCMC with {nwalkers} walkers for {nsamples} steps...")
        # A sensible starting point for the walkers
        if self.avg_splits:
            fiducial_params = np.array([0.2,0.2,0.0,0.0,0.35])
        else:
            fiducial_params = np.array([0.2,0.2,0.2,0.2,0.0,0.0,0.0,0.0,0.35])
        ndim = len(fiducial_params)
        
        pos = fiducial_params + 1e-1 * np.random.randn(nwalkers, ndim)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, threads=8)
        sampler.run_mcmc(pos, nsamples, progress=True)
        
        flat_samples = sampler.get_chain(discard=200, thin=20, flat=True)
        
        with open(fname, 'wb') as f:
            pl.dump(flat_samples, f)
            
        return flat_samples

    def getdist_samples(self, nwalkers: int, nsamples: int, rerun: bool = False, label: str = None):
        """
        Gets the MCMC samples in a GetDist MCSamples object for plotting.
        """
        flat_samples = self.run_mcmc(nwalkers, nsamples, rerun=rerun)
        return MCSamples(samples=flat_samples, names=self.__pnames__, labels=self.__plabels__, label=label)

    def plot_posteriors(self, nwalkers: int, nsamples: int, rerun: bool = False):
        """
        Generates and shows a triangle plot of the posterior distributions.
        """
        print("Generating posterior triangle plot...")
        samples = self.getdist_samples(nwalkers, nsamples, rerun=rerun, label='Cross-Spectra Analysis')
        g = plots.get_subplot_plotter()
        g.triangle_plot([samples], filled=True, title_limit=1)
        print("Plotting complete.")
        return g
    def plot_spectra_matrix(self, theta=None, save_path=None):
        """
        Plots the mean data spectra with std deviation and optionally a theory curve
        in a matrix layout corresponding to all cross-correlations.

        Args:
            theta (np.ndarray, optional): The parameter vector (alphas, beta) to compute
                                          and overlay the theory model. Defaults to None.
            save_path (str, optional): If provided, saves the figure to this path.
                                       Defaults to None (shows the plot).
        """
        n_tags = len(self.maptags)
        ells = self.binner.get_effective_ells()

        fig, axes = plt.subplots(n_tags, n_tags, figsize=(n_tags * 3, n_tags * 3),
                                 sharex=True, sharey='row')

        theory_spec = None
        if theta is not None:
            theory_spec = self.theory(theta)

        data_spec = self.mean_spec / self.beam_arr
        error_spec = self.std_spec / self.beam_arr
        
        for i in range(n_tags):
            for j in range(n_tags):
                ax = axes[i, j]
                is_diagonal = (i == j)

                # Determine if the current subplot should be visible based on selection
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

                # Plot data mean and std
                ax.errorbar(ells, data_spec[i, j], yerr=error_spec[i, j], fmt='.', capsize=2,
                            label='Data Mean +/- 1σ', color='black', markersize=4,alpha=0.5)

                # Plot theory if provided
                if theory_spec is not None:
                    ax.loglog(ells, theory_spec[i, j], color='red', label='Theory')

                # Apply the likelihood mask to visually grey out unused data
                mask_1d = self.__likelihood_mask__[i, j, :]
                used_ells = ells[mask_1d]
                if len(used_ells) > 0:
                    ax.axvspan(used_ells.min(), used_ells.max(), color='green', alpha=0.1, zorder=-1)

                # Formatting
                if i == n_tags - 1:
                    ax.set_xlabel(r'Multipole $\ell$')
                if j == 0:
                    ax.set_ylabel(r'$C_\ell^{EB}$')

                # Set subplot titles
                if i == 0:
                    ax.set_title(self.maptags[j].replace('_', ' '), fontsize=10)
                
                ax.axhline(0, color='grey', linestyle=':', linewidth=0.75)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_ylim(1e-8, 1e-1)
                ax.set_xlim(10, 3000)

        # Use the row labels on the right side
        for i, tag in enumerate(self.maptags):
            axes[i, n_tags-1].text(1.05, 0.5, tag.replace('_', ' '),
                                   transform=axes[i, n_tags-1].transAxes,
                                   va='center', ha='left', fontsize=10)


        handles, labels = [], []
        for ax in axes.flatten():
            if ax.get_visible():
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
                break # Get legend from the first visible plot
        
        if handles:
            fig.legend(handles, labels, loc='upper right')

        plt.tight_layout(rect=[0.03, 0.03, 0.92, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Figure saved to {save_path}")
            plt.close()
        else:
            plt.show()