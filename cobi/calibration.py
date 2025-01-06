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

def get_sp(lib,lmax):
    bl_arr = []
    for i in range(6):
        bl_arr.append(lib.binInfo.bin_cell(hp.gauss_beam(np.radians(lib.lat.fwhm[i]/60),lmax)**2))
    bl_arr = np.array(bl_arr)
    obs_arr = []
    for i in range(20):
        sp = lib.obs_x_obs(i)
        obs_arr.append(sp)
    obs_arr = np.array(obs_arr)
    obs_arr = obs_arr[:,np.arange(6),np.arange(6),2,2:]
    return obs_arr/bl_arr

def paranames(lib,name,avoid=[]):
    return [f"a{name}{fe}" for fe in lib.lat.freqs if fe not in avoid] 
def latexnames(lib, name, avoid=[]):
    return [r'\alpha_{{{}}}^{{{}}}'.format(name, fe) for fe in lib.lat.freqs if fe not in avoid]

class Sat4Lat:
    
    def __init__(self,libdir,satlib,lmin,lmax,sat_err,alpha_sat,beta,latlib=None,alpha_lat=None,avoid_freq_s=[],avoid_freq_l=[]):
        self.libdir = os.path.join(libdir,'Calibration')
        os.makedirs(self.libdir,exist_ok=True)
        self.latlib = latlib
        self.sat_err = sat_err
        self.binner = satlib.binInfo
        self.Lmax = satlib.lmax
        self.__select__ = selector(satlib,lmin,lmax)
        self.__selectfsat__ = selectorf(satlib,avoid_freq_s)
        self.__selectflat__ = selectorf(latlib,avoid_freq_l) if latlib is not None else []

        alpha_sat = list(np.array(alpha_sat)[self.__selectfsat__])
        alpha_lat = list(np.array(alpha_lat)[self.__selectflat__]) if alpha_lat is not None else []


        self.lat_mean, self.lat_std = self.calc_mean_std(latlib,'LAT') if latlib is not None else (None,None)
        self.sat_mean, self.sat_std = self.calc_mean_std(satlib,'SAT')
        self.cl_len = CMB(libdir,satlib.lat.nside,).get_lensed_spectra(dl=False)
        self.true =  np.concatenate([alpha_lat,alpha_sat,[beta]]) if latlib is not None else np.concatenate([alpha_sat,[beta]])
        self.__asat__ = alpha_sat
        self.__alat__ = alpha_lat
        self.lmin = lmin
        self.lmax = lmax

        if len(avoid_freq_l) > 0:
            local_type = type(avoid_freq_l[0])
            assert local_type == str, f"avoid_freq elements should be of type str"
            for af in avoid_freq_l:
                assert af in satlib.lat.freqs, f"{af} not in {satlib.lat.freqs}"
        if len(avoid_freq_s) > 0:
            local_type = type(avoid_freq_s[0])
            assert local_type == str, f"avoid_freq elements should be of type str"
            for af in avoid_freq_s:
                assert af in satlib.lat.freqs, f"{af} not in {satlib.lat.freqs}"
        self.avoid_freq_s = avoid_freq_s
        self.lasat = len(avoid_freq_s)
        self.avoid_freq_l = avoid_freq_l
        self.lalat = len(avoid_freq_l)

        self.__used_freqs_sat__ = [freq for freq in satlib.lat.freqs if freq not in avoid_freq_s]
        self.__used_freqs_lat__ = [freq for freq in latlib.lat.freqs if freq not in avoid_freq_l] if latlib is not None else []

        if latlib is None:
            self.__pnames__ = paranames(satlib,'SAT',avoid_freq_s) + ['beta']
            self.__plabels__ = latexnames(satlib,'SAT',avoid_freq_s)  + [r'\beta']
        else:
            self.__pnames__ = paranames(latlib,'LAT',avoid_freq_l) + paranames(satlib,'SAT',avoid_freq_s) + ['beta']
            self.__plabels__ = latexnames(latlib,'LAT',avoid_freq_l) + latexnames(satlib,'SAT',avoid_freq_s)  + [r'\beta']

    def calc_mean_std(self,lib,name):
        sp = get_sp(lib,self.Lmax)
        if name == 'LAT':
            return ( sp.mean(axis=0)[:,self.__select__][self.__selectflat__],
                     sp.std(axis=0)[:,self.__select__][self.__selectflat__] )
        elif name == 'SAT':
            return ( sp.mean(axis=0)[:,self.__select__][self.__selectfsat__],
                     sp.std(axis=0)[:,self.__select__][self.__selectfsat__] )
        else:
            raise ValueError(f"Invalid name {name}")
    
    def plot_spectra(self,tele):
        plt.figure(figsize=(4,4))
        if tele == 'LAT' and self.latlib is not None:
            for i in range(6-self.lalat):
                plt.loglog(self.binner.get_effective_ells()[self.__select__],self.lat_mean[i])
        elif tele == 'SAT':
            for i in range(6-self.lasat):
                plt.loglog(self.binner.get_effective_ells()[self.__select__],self.sat_mean[i])
        else:
            raise ValueError(f"Invalid telescope {tele}")
    
    def theory(self,beta_array):
        beta_array = np.asarray(beta_array)
        th = 0.5 * (self.cl_len["ee"] - self.cl_len["bb"])[:, np.newaxis] * np.sin(np.deg2rad(4 * beta_array))
        return np.apply_along_axis(lambda th_slice: self.binner.bin_cell(th_slice[:self.Lmax+1])[self.__select__], 0, th).T
    
    def chisq(self,theta):
        if self.latlib is None:
            alpha_sat,beta = np.array(theta[:6-self.lasat]), theta[-1]
        else:
            alpha_lat,alpha_sat,beta = np.array(theta[:6-self.lalat]), np.array(theta[6-self.lalat:12-self.lasat-self.lalat]), theta[-1]
        

        #diff_mean = self.lat_mean - self.sat_mean
        #diff_std = np.sqrt(self.lat_std**2 + self.sat_std**2)  
        #diff_model = self.theory(alpha_lat-alpha_sat)
        #diff_chi = np.sum(((diff_mean - diff_model)/diff_std)**2)

        sat_model = self.theory(np.ones(len(alpha_sat))*beta + alpha_sat)
        sat_chi = np.sum(((self.sat_mean - sat_model)/self.sat_std)**2)

        if self.latlib is None:
            return sat_chi
        
        lat_model = self.theory(np.ones(len(alpha_lat))*beta + alpha_lat)
        lat_chi = np.sum(((self.lat_mean - lat_model)/self.lat_std)**2)

        return  sat_chi + lat_chi #+ diff_chi
    
    def lnprior(self,theta):
        sigma = self.sat_err
        if self.latlib is not None:
            alphalat,alphasat,beta = np.array(theta[:6-self.lalat]), np.array(theta[6-self.lalat:12-self.lasat-self.lalat]), theta[-1]
        else:
            alphasat,beta = np.array(theta[:6-self.lasat]), theta[-1]

        lnp = -0.5 * (alphasat - self.__asat__ )**2 / sigma**2 - np.log(sigma*np.sqrt(2*np.pi))
        
        if self.latlib is not None:
            if np.all(alphalat > -0.5) and np.all(alphalat < 0.5) and -0.1 < beta < 0.5:
                return np.sum(lnp)
            return -np.inf
        else:
            if 0 < beta < 0.5:
                return np.sum(lnp)
            return -np.inf


    def ln_likelihood(self,theta):
        return -0.5 * self.chisq(theta)

    def ln_prob(self,theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta)
    
    
    def samples(self,nwalkers=32,nsamples=1000):
        fused_sat = 'sat' + '_'.join(self.__used_freqs_sat__)
        fused_lat = 'lat' + '_'.join(self.__used_freqs_lat__)

        if self.latlib is None:
            fname = os.path.join(self.libdir,f'samples_f{fused_sat}_li{self.lmin}_le{self.lmax}_w{nwalkers}_n{nsamples}_sat.pkl')
        else:
            fname = os.path.join(self.libdir,f'samples_fs{fused_sat}_fl{fused_lat}_li{self.lmin}_le{self.lmax}_w{nwalkers}_n{nsamples}.pkl')
        if os.path.isfile(fname):
            flat_samples = pl.load(open(fname,'rb'))
        else:
            ndim = len(self.true)
            pos = self.true + 1e-3 * np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, threads=4)
            sample = sampler.run_mcmc(pos, nsamples, progress=True)
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
            pl.dump(flat_samples,open(fname,'wb'))
        return np.nan_to_num(flat_samples)
    
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

