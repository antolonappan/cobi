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
        self.libdir = os.path.join(libdir,'Calibration')
        
        latnc = latlib.lat.noise_model
        satnc = satlib.lat.noise_model
        if (latnc == 'NC') and (satnc == 'NC'):
            self.libdir = os.path.join(libdir,'Calibration')
        elif (latnc == 'TOD') and (satnc == 'TOD'):
            self.libdir = os.path.join(libdir,'Calibration_TOD')
        elif (latnc == 'TOD') and (satnc == 'NC'):
            self.libdir = os.path.join(libdir,'Calibration_TOD_NC')
        elif (latnc == 'NC') and (satnc == 'TOD'):
            self.libdir = os.path.join(libdir,'Calibration_NC_TOD')
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
    
    
    def samples(self,nwalkers=32,nsamples=1000,rerun=True):
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

    def getdist_samples(self,nwalkers,nsamples):
        flat_samples = self.samples(nwalkers,nsamples)
        return MCSamples(samples=flat_samples,names = self.__pnames__, labels = self.__plabels__)
        
    
    def plot_getdist(self,nwalkers,nsamples,avoid_sat=False,beta_only=False):
        flat_samples = self.getdist_samples(nwalkers,nsamples)
        if beta_only:
            g = plots.get_single_plotter(width_inch=4)
            g.plot_1d(flat_samples, 'beta', title_limit=1)
        else:
            names = self.__pnames__
            if avoid_sat:
                names = [item for item in names if 'SAT' not in item]
            g = plots.get_subplot_plotter()
            g.triangle_plot([flat_samples], names, filled=True,title_limit=1)

