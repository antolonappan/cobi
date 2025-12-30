"""
Quadratic Estimator for Cosmic Birefringence Module
====================================================

This module implements quadratic estimator (QE) methods for reconstructing
cosmic birefringence rotation angle fields from CMB polarization observations.

The module provides:
- Wiener filtering of E and B modes using C⁻¹ filters
- Quadratic estimator reconstruction of rotation angle alms
- N0 and RDN0 bias estimation
- Mean field subtraction
- Likelihood analysis for cosmic birefringence amplitude

Classes
-------
FilterEB
    Implements C⁻¹ Wiener filtering for E and B mode polarization using
    the curvedsky library. Handles component-separated maps with noise
    and beam deconvolution.

QE
    Quadratic estimator class for cosmic birefringence reconstruction.
    Computes rotation angle power spectra with various bias correction
    methods (analytical N0, realization-dependent N0, mean field).
    
    Simulation Index Structure (sim_config is required):
    
    From sim_config={'set1': 400, 'reuse_last': 100} and nsims_mf=100:
    - stat_index: 0-299 (set1 - nsims_mf) - Statistics and OCL computation
    - mf_index: 300-399 (last nsims_mf from set1) - Mean field simulations
    - vary_index: 300-399 (last reuse_last from set1) - Varying alpha (CMB mode='vary')
    - const_index: 400-499 (reuse_last sims) - Constant alpha (CMB mode='constant')
    - null_index: 500-599 (reuse_last sims) - Null alpha (CMB mode='null')
    
    Note: When nsims_mf equals reuse_last, mf_index and vary_index are identical.
    
    Bias Estimation:
    - MCN0('stat'): Uses stat_index (0-299)
    - MCN0('vary'): Uses vary_index (300-399)
    - MCN0('const'): Uses const_index (400-499)
    - N1 = MCN0('const') - MCN0('vary')
    - Nlens: Uses null_index (500-599) and MCN0('vary')

AcbLikelihood
    Likelihood analysis class for constraining the cosmic birefringence
    amplitude A_CB from reconstructed rotation angle power spectra.

Algorithm
---------
The quadratic estimator exploits EB correlations induced by cosmic birefringence:

1. Apply C⁻¹ Wiener filter to observed E and B maps
2. Compute quadratic combination: qlm ∝ Ẽlm B̃*lm
3. Apply normalization from Fisher information
4. Estimate and subtract biases (N0, mean field)
5. Compare to theoretical prediction for A_CB

Requirements
------------
Requires the curvedsky library for efficient curved-sky operations.
Install from: https://github.com/toshiyan/cmblensplus

Example
-------
    from cobi.quest import FilterEB, QE
    from cobi.simulation import LATsky, Mask
    
    # Set up sky and mask
    sky = LATsky(libdir, nside=512)
    mask = Mask(libdir, nside=512, mask_str='LAT')
    
    # Initialize Wiener filter
    filter_eb = FilterEB(
        sky=sky,
        mask=mask,
        lmax=3000,
        fwhm=2.0,  # arcmin
        sht_backend='ducc0'
    )
    
    # Create quadratic estimator
    qe = QE(
        filter=filter_eb,
        lmin=30,
        lmax=300,
        recon_lmax=100,
        norm_nsim=100
    )
    
    # Reconstruct rotation angle for simulation 0
    qlm = qe.qlm(idx=0)
    
    # Get bias-corrected power spectrum
    cl_aa = qe.qcl(idx=0, rm_bias=True, rdn0=True)
    
    # Run likelihood analysis
    likelihood = AcbLikelihood(qe, lmin=2, lmax=50)
    samples = likelihood.samples(nwalkers=100, nsamples=2000)

Notes
-----
The curvedsky library must be installed to use this module.
MPI parallelization is supported for RDN0 computation.
"""

import os
try:
    import curvedsky as cs
except ImportError:
    cs = None
    print("Install curvedsky to use the QE module.")
import numpy as np
import healpy as hp
import pickle as pl
import matplotlib.pyplot as plt
import pymaster as nmt
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from typing import Dict, Optional, Any, Union, List

from cobi.simulation import LATsky, Mask
from cobi.utils import cli, slice_alms
from cobi import sht
from cobi import mpi
from time import time


Tcmb  = 2.726e6
NCPUS = os.cpu_count()

class FilterEB:

    def __init__(self, sky: LATsky, mask: Mask, lmax: int, fwhm: float = 2, sht_backend: str = "healpy"):
        self.sky = sky
        self.nside = sky.nside
        self.lmax = min(lmax, 3*self.nside -1)
        self.mask = mask.mask
        self.fsky = mask.fsky
        self.ninv = np.reshape(np.array((self.mask,self.mask)),(2,1,hp.nside2npix(self.nside)))
        self.bl = hp.gauss_beam(fwhm=np.radians(fwhm/60), lmax=self.lmax)
        if sht_backend in ['ducc0', 'ducc', 'd']:
            self.hp = sht.HealpixDUCC(nside=self.nside)
            self.healpy = False
        else:
            self.hp = None
            self.healpy = True
        self.cl_len = sky.cmb.get_lensed_spectra(dl=False,dtype='a').T/ Tcmb**2
        self.lib_dir = os.path.join(sky.libdir, 'cinv')
        if mpi.rank == 0:
            os.makedirs(self.lib_dir, exist_ok=True)
        

    @property
    def Bl(self):
        return np.reshape(self.bl,(1,self.lmax+1))

    def convolved_EB(self,idx,split=0):
        """
        convolve the component separated map with the beam

        Parameters
        ----------
        idx : int : index of the simulation
        """
        E,B = self.sky.HILC_obsEB(idx,ret='alm',split=split)
        hp.almxfl(E,self.bl,inplace=True)
        hp.almxfl(B,self.bl,inplace=True)
        return E,B
    
    def NL(self,idx,split=0):
        """
        array manipulation of noise spectra obtained by ILC weight
        for the filtering process
        """
        ne,nb = self.sky.HILC_obsEB(idx, ret='nl',split=split)/Tcmb**2
        return np.reshape(np.array((cli(ne[:self.lmax+1]*self.bl**2),
                          cli(nb[:self.lmax+1]*self.bl**2))),(2,1,self.lmax+1))
    
    
    def QU(self, idx, split=0):
        """
        deconvolve the beam from the QU map

        Parameters
        ----------
        idx : int : index of the simulation
        """
        E, B = self.convolved_EB(idx, split=split)
        E, B = slice_alms(np.array([E, B]), self.lmax)
        if self.healpy:
            QU = hp.alm2map([E*0,E,B], self.nside)[1:]/Tcmb
        else:
            QU = self.hp.alm2map([E, B], lmax=self.lmax,nthreads=NCPUS)/Tcmb
        QU = QU*self.mask
        QU[QU == -0] = 0
        return QU
    
    def __cinv_eb__(self, idx, fname, test=False, split=0):
        QU = self.QU(idx, split=split)
        QU = np.reshape(QU,(2,1,hp.nside2npix(self.nside)))
        
        iterations = [200]
        stat_file = '' 
        if test:
            iterations = [10]
            stat_file = os.path.join('test_stat.txt')

        E,B = cs.cninv.cnfilter_freq(2,1,self.nside,self.lmax,self.cl_len[1:3,:self.lmax+1],
                                    self.Bl, self.ninv,QU,chn=1,itns=iterations,filter="",
                                    eps=[1e-5],ro=10,inl=self.NL(idx,split=split),stat=stat_file)
        if not test:
            pl.dump((E,B), open(fname,'wb'))
        return E, B

    def cinv_EB(self,idx,split=0,test=False):
        """
        C inv Filter for the component separated maps

        Parameters
        ----------
        idx : int : index of the simulation
        test : bool : if True, run the filter for 10 iterations (not cached)
        """
        fsky = f"{self.fsky:.2f}".replace('.','p')
        if split == 0:
            fname = os.path.join(self.lib_dir,f"cinv_EB_{idx:04d}_fsky_{fsky}.pkl")
        else:
            fname = os.path.join(self.lib_dir,f"cinv_EB_{idx:04d}_fsky_{fsky}_split{split}.pkl")
        if os.path.isfile(fname):
            try:
                E,B = pl.load(open(fname,'rb'))
            except:
                E,B = self.__cinv_eb__(idx,fname,test=test,split=split)
        else:
            E,B = self.__cinv_eb__(idx,fname,test=test,split=split)
        return E,B
    
    def check_file_exist(self,nsims=600,split=0):
        missing = []
        file_err = []
        for idx in tqdm(range(nsims),desc='Checking cinv files'):
            fsky = f"{self.fsky:.2f}".replace('.','p')
            if split == 0:
                fname = os.path.join(self.lib_dir,f"cinv_EB_{idx:04d}_fsky_{fsky}.pkl")
            else:
                fname = os.path.join(self.lib_dir,f"cinv_EB_{idx:04d}_fsky_{fsky}_split{split}.pkl")
            if not os.path.isfile(fname):
                missing.append(idx)
            else:
                try:
                    E,B = pl.load(open(fname,'rb'))
                except:
                    file_err.append(idx)
        print(f"Total missing files: {len(missing)}")
        print(f"Total file errors: {len(file_err)}")
        return missing, file_err


    def plot_cinv(self,idx,split=0,lmin=2,lmax=3000):
        """
        plot the cinv filtered Cls for a given idx

        Parameters
        ----------
        idx : int : index of the simulation
        """
        E,_ = self.cinv_EB(idx, split=split)
        ne,_ = self.sky.HILC_obsEB(idx,split=split, ret='nl')/Tcmb**2
        cle = cs.utils.alm2cl(self.lmax,E)/self.fsky
        plt.figure(figsize=(4,4))
        plt.loglog(cle,label='Cinv E mode')
        plt.loglog(1/(self.cl_len[1,:len(ne)]  + ne), label='1/(S+N)')
        plt.xlim(lmin,lmax)
        plt.legend()


class QE:
    def __init__(self, filter: FilterEB, lmin: int, lmax: int, recon_lmax: int, nsims_mf=100, nlb=10, lmax_bin=1024):
        self.basedir = os.path.join(filter.sky.libdir, 'qe')
        self.recdir = os.path.join(self.basedir, f'reco_min{lmin}_max{lmax}_rmax{recon_lmax}')
        self.rdn0dir = os.path.join(self.basedir, f'rdn0_min{lmin}_max{lmax}_rmax{recon_lmax}')
        if mpi.rank == 0:
            os.makedirs(self.basedir, exist_ok=True)
            os.makedirs(self.recdir, exist_ok=True)
            os.makedirs(self.rdn0dir, exist_ok=True)

        self.filter = filter
        self.sim_config = filter.sky.cmb.sim_config
        self.lmin = lmin
        self.lmax = lmax
        self.recon_lmax = recon_lmax
        self.cl_len = filter.cl_len
        self.nsims_mf = nsims_mf
        
        # sim_config is required
        if self.sim_config is None:
            raise ValueError("sim_config must be set in CMB initialization. QE requires sim_config to define simulation ranges.")
        
        
        set1 = self.sim_config['set1']
        reuse_last = self.sim_config['reuse_last']
        
        # Statistics simulations: first (set1 - nsims_mf)
        self.stat_index = np.arange(0, set1 - nsims_mf)
        
        # Mean field simulations: last nsims_mf from set1
        self.mf_index = np.arange(set1 - nsims_mf, set1)
        
        # Varying alpha range: last reuse_last from set1
        self.vary_index = np.arange(set1 - reuse_last, set1)
        
        # Constant alpha range: set1 to set1+reuse_last
        self.const_index = np.arange(set1, set1 + reuse_last)
        
        # Null alpha range: set1+reuse_last to set1+2*reuse_last
        self.null_index = np.arange(set1 + reuse_last, set1 + 2 * reuse_last)
        
        self.norm = self.__norm__
        self.lmax_bin = lmax_bin
        self.binner = nmt.NmtBin.from_lmax_linear(lmax_bin,nlb)
        self.b = self.binner.get_effective_ells()
        self.nlb = nlb

    @property
    def __norm__(self):
        fname = os.path.join(self.basedir, f'norm_min{self.lmin}_max{self.lmax}_rmax{self.recon_lmax}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            ocl = self.__ocl__
            norm = cs.norm_quad.qeb('rot',self.recon_lmax,self.lmin,self.lmax,self.cl_len[1,:self.lmax+1],ocl[1,:self.lmax+1],ocl[2,:self.lmax+1])[0]
            pl.dump(norm, open(fname,'wb'))
            return norm

    @property
    def __ocl__(self):
        fname = os.path.join(self.basedir, f'ocl_min{self.lmin}_max{self.lmax}_rmax{self.recon_lmax}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            ocl_len = self.cl_len.copy()
            ne,nb = [],[]
            # Use stat_index for computing OCL
            for i in tqdm(self.stat_index, desc='Computing OCL'):
                e,b = self.filter.sky.HILC_obsEB(i, ret='nl')
                ne.append(e[:self.lmax+1]/Tcmb**2)
                nb.append(b[:self.lmax+1]/Tcmb**2)
            ne, nb = np.array(ne).mean(axis=0), np.array(nb).mean(axis=0)
            ocl_len[1,:self.lmax+1] += ne
            ocl_len[2,:self.lmax+1] += nb
            pl.dump(ocl_len, open(fname,'wb'))
        return ocl_len
    
    @property
    def cl_aa(self):
        return self.filter.sky.cmb.cl_aa()[:self.recon_lmax+1]
    
    def qlm(self, idx):
        fname = os.path.join(self.recdir, f'qlm_fsky{self.filter.fsky:.2f}_{idx:04d}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname, 'rb'))
        else:
            E,B = self.filter.cinv_EB(idx)
            alm = cs.rec_rot.qeb(self.recon_lmax,self.lmin,self.lmax,self.cl_len[1,:self.lmax+1],E[:self.lmax+1,:self.lmax+1],B[:self.lmax+1,:self.lmax+1])
            nalm = alm * self.norm[:,None]
            pl.dump(nalm, open(fname, 'wb'))
            return nalm
        
    def check_file_exist(self):
        missing = []
        for idx in range(300):
            fname = os.path.join(self.recdir, f'qlm_fsky{self.filter.fsky:.2f}_{idx:04d}.pkl')
            if not os.path.isfile(fname):
                missing.append(idx)
        return missing

    def __qcl__(self,idx,n0=None, mf=False, nlens=False, n1=False):
        qcl = cs.utils.alm2cl(self.recon_lmax,self.qlm(idx))/self.filter.fsky
        if n0 is None:
            N0 = np.zeros_like(qcl)
        elif n0 == 'norm':
            N0 = self.norm
        elif n0 == 'mcn0':
            N0 = self.MCN0('stat')
        elif n0 == 'rdn0':
            N0 = self.RDN0(idx)
        else:
            raise ValueError("n0 must be 'norm', 'rdn0', or 'mcn0'")
        qcl = qcl - N0
        if mf:
            qcl = qcl - self.mean_field_cl()
        if nlens:
            qcl = qcl - self.Nlens()
        if n1:
            qcl = qcl - self.N1()
        return qcl
        
    def qcl(self,idx, n0=None, mf=False, nlens=False, n1=False, binned=False):
        cl = self.__qcl__(idx,n0=n0,mf=mf,nlens=nlens,n1=n1)
        if binned:
            bcl = self.binner.bin_cell(cl[:self.lmax_bin+1])
            return bcl
        else:
            return cl

    def mean_field(self):
        fname = os.path.join(self.basedir, f'mf{self.nsims_mf}_fsky{self.filter.fsky:.2f}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname, 'rb'))
        else:
            mf = np.zeros_like(self.qlm(self.mf_index[0]))
            for i in tqdm(self.mf_index, desc='Computing mean field'):
                mf += self.qlm(i)
            mf /= self.nsims_mf
            pl.dump(mf, open(fname, 'wb'))
            return mf
    
    def mean_field_cl(self):
        return cs.utils.alm2cl(self.recon_lmax, self.mean_field()) / self.filter.fsky
    
    def RDN0(self,idx):
        fname = os.path.join(self.rdn0dir,f"RDN0_{self.filter.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            # Use stat_index for cycling
            myidx = self.stat_index.copy()

            E0,B0 = self.filter.cinv_EB(idx)

            mean_rdn0 = []

            for i in tqdm(range(100),desc=f'RDN0 for simulation {idx}', unit='sim'):
                # Cycle through n0_index
                i1 = myidx[i % len(myidx)]
                i2 = myidx[(i + 1) % len(myidx)]
                
                E1,B1 = self.filter.cinv_EB(i1)
                E2,B2 = self.filter.cinv_EB(i2)
                # E_0,B_1
                glm1 = cs.rec_rot.qeb(self.recon_lmax,self.lmin,self.lmax,self.filter.cl_len[1,:self.lmax+1], E0[:self.lmax+1,:self.lmax+1], B1[:self.lmax+1,:self.lmax+1])

                # E_1,B_0
                glm2 = cs.rec_rot.qeb(self.recon_lmax,self.lmin,self.lmax,self.filter.cl_len[1,:self.lmax+1], E1[:self.lmax+1,:self.lmax+1], B0[:self.lmax+1,:self.lmax+1])

                # E_1,B_2
                glm3 = cs.rec_rot.qeb(self.recon_lmax,self.lmin,self.lmax,self.filter.cl_len[1,:self.lmax+1], E1[:self.lmax+1,:self.lmax+1], B2[:self.lmax+1,:self.lmax+1])

                # E_2,B_1
                glm4 = cs.rec_rot.qeb(self.recon_lmax,self.lmin,self.lmax,self.filter.cl_len[1,:self.lmax+1], E2[:self.lmax+1,:self.lmax+1], B1[:self.lmax+1,:self.lmax+1])

                del (E1,B1,E2,B2)


                glm1 *= self.norm[:,None]
                glm2 *= self.norm[:,None]
                glm3 *= self.norm[:,None]
                glm4 *= self.norm[:,None]

                first_four = cs.utils.alm2cl(self.recon_lmax, glm1 + glm2)/(self.filter.fsky) #type: ignore
                del (glm1,glm2)
                second_last = cs.utils.alm2cl(self.recon_lmax, glm3)/(self.filter.fsky) #type: ignore
                last = cs.utils.alm2cl(self.recon_lmax, glm3,glm4)/(self.filter.fsky) #type: ignore
                del (glm3,glm4)

                mean_rdn0.append(first_four - second_last - last)
                del (first_four,second_last,last)

            del (E0,B0)
            rdn0 = np.mean(mean_rdn0,axis=0)
            pl.dump(rdn0,open(fname,'wb'))
            return rdn0

    def RDN0_mpi(self, idx):
        """
        MPI-parallel version of RDN0(...) using mpi4py.

        Parallelizes the 100 Monte-Carlo iterations; safe for any number of ranks.
        Only rank 0 touches the on-disk cache.
        """
        MPI = mpi.mpi
        comm  = mpi.com
        rank  = mpi.rank
        size  = mpi.size

        fname = os.path.join(self.rdn0dir, f"RDN0_{self.filter.fsky:.2f}_{idx:04d}.pkl")

        # 1) Try to load cached result on rank 0, then broadcast.
        rdn0 = None
        if rank == 0 and os.path.isfile(fname):
            with open(fname, 'rb') as f:
                rdn0 = pl.load(f)
        rdn0 = comm.bcast(rdn0, root=0)
        if rdn0 is not None:
            return rdn0

        # 2) Build the index cycling array on all ranks (cheap & deterministic).
        myidx = self.stat_index.copy()

        # 3) Compute (or broadcast) the fixed E0, B0 for this idx.
        if rank == 0:
            E0, B0 = self.filter.cinv_EB(idx)
        else:
            E0 = B0 = None
        E0 = comm.bcast(E0, root=0)
        B0 = comm.bcast(B0, root=0)

        # 4) Distribute the 100 iterations across ranks.
        tasks = np.arange(100, dtype=int)
        # chunking keeps i and i+1 local in the same process most of the time
        chunks = np.array_split(tasks, size)
        my_tasks = chunks[rank]

        # We know alm2cl returns an array of length (recon_lmax+1)
        L = self.recon_lmax + 1
        local_sum = np.zeros(L, dtype=np.float64)

        # Helper to compute one contribution (vector length L)
        def _one_contrib(i):
            # Cycle through n0_index
            i1 = int(myidx[i % len(myidx)])
            i2 = int(myidx[(i + 1) % len(myidx)])

            E1, B1 = self.filter.cinv_EB(i1)
            E2, B2 = self.filter.cinv_EB(i2)

            # Quadratic estimators
            glm1 = cs.rec_rot.qeb(
                self.recon_lmax, self.lmin, self.lmax,
                self.filter.cl_len[1, :self.lmax+1],
                E0[:self.lmax+1, :self.lmax+1],
                B1[:self.lmax+1, :self.lmax+1],
            )
            glm2 = cs.rec_rot.qeb(
                self.recon_lmax, self.lmin, self.lmax,
                self.filter.cl_len[1, :self.lmax+1],
                E1[:self.lmax+1, :self.lmax+1],
                B0[:self.lmax+1, :self.lmax+1],
            )
            glm3 = cs.rec_rot.qeb(
                self.recon_lmax, self.lmin, self.lmax,
                self.filter.cl_len[1, :self.lmax+1],
                E1[:self.lmax+1, :self.lmax+1],
                B2[:self.lmax+1, :self.lmax+1],
            )
            glm4 = cs.rec_rot.qeb(
                self.recon_lmax, self.lmin, self.lmax,
                self.filter.cl_len[1, :self.lmax+1],
                E2[:self.lmax+1, :self.lmax+1],
                B1[:self.lmax+1, :self.lmax+1],
            )

            # Normalization per-ℓ
            glm1 *= self.norm[:, None]
            glm2 *= self.norm[:, None]
            glm3 *= self.norm[:, None]
            glm4 *= self.norm[:, None]

            first_four  = cs.utils.alm2cl(self.recon_lmax, glm1 + glm2) / (self.filter.fsky)  # type: ignore
            second_last = cs.utils.alm2cl(self.recon_lmax, glm3)        / (self.filter.fsky)  # type: ignore
            last        = cs.utils.alm2cl(self.recon_lmax, glm3, glm4)  / (self.filter.fsky)  # type: ignore

            return first_four - second_last - last

        # 5) Local accumulation with an optional progress bar on rank 0.
        iterator = tqdm(my_tasks, desc=f'RDN0 (rank {rank}) for simulation {idx}', unit='sim') if rank == 0 else my_tasks
        for i in iterator:
            local_sum += _one_contrib(i)

        # 6) Global reduction (sum over ranks), then average over the 100 draws.
        global_sum = np.zeros_like(local_sum)
        comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)

        if rank == 0:
            rdn0 = global_sum / float(len(tasks))  # divide by 100
            # Cache to disk
            os.makedirs(self.rdn0dir, exist_ok=True)
            with open(fname, 'wb') as f:
                pl.dump(rdn0, f)

        # 7) Broadcast the final result so all ranks return the same array.
        rdn0 = comm.bcast(rdn0, root=0)
        return rdn0
      
    def N0_sim(self,idx,which='vary',debug=False):
        """
        Calculate the N0 bias from the Reconstructed potential using filtered Fields
        with different CMB fields. If E modes is from ith simulation then B modes is 
        from (i+1)th simulation

        idx: int : index of the N0
        which: str : 'stat', 'vary', or 'const' to select which index range to use
        debug: bool : if True, print the indices used for computation and return None
        
        Index ranges and wrapping:
        - 'stat': stat_index, wraps within stat range
        - 'vary': vary_index, wraps within vary range
        - 'const': const_index, wraps within const range
        
        Requires sim_config to be set, or manually set the corresponding index array.
        """
        if which == 'stat':
            index_range = self.stat_index
            label = 'stat'
        elif which == 'vary':
            index_range = self.vary_index
            label = 'vary'
        elif which == 'const':
            index_range = self.const_index
            label = 'const'
        else:
            raise ValueError("which must be 'stat', 'vary', or 'const'")
        
        assert idx in index_range, f"The requested idx {idx} is not in the {which} index range"
            
        # Simple increment with wrapping within the index_range bounds
        idx1 = idx
        min_idx = min(index_range)
        max_idx = max(index_range)
        # If at the end of range, wrap to beginning of range
        if idx == max_idx:
            idx2 = min_idx
        else:
            idx2 = idx + 1
        
        if debug:
            print(f"N0_sim debug mode:")
            print(f"  which: {which}")
            print(f"  index_range: [{min_idx}, {max_idx}]")
            print(f"  idx1 (E1B1): {idx1}")
            print(f"  idx2 (E2B2): {idx2}")
            return None
            
        fname = os.path.join(self.rdn0dir,f"N0_{label}_{self.filter.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            E1,B1 = self.filter.cinv_EB(idx1)
            E2,B2 = self.filter.cinv_EB(idx2)
            glm1 = cs.rec_rot.qeb(self.recon_lmax,self.lmin,self.lmax,
                                       self.cl_len[1,:self.lmax+1],
                                       E1[:self.lmax+1,:self.lmax+1],
                                       B2[:self.lmax+1,:self.lmax+1]) 
            glm2 = cs.rec_rot.qeb(self.recon_lmax,self.lmin,self.lmax,
                                        self.cl_len[1,:self.lmax+1],
                                        E2[:self.lmax+1,:self.lmax+1],
                                        B1[:self.lmax+1,:self.lmax+1]) 
            glm1 *= self.norm[:,None]
            glm2 *= self.norm[:,None]
            
            glm = glm1 + glm2
            n0cl = cs.utils.alm2cl(self.recon_lmax,glm)/(2*self.filter.fsky) # type: ignore
            pl.dump(n0cl,open(fname,'wb'))
            return n0cl
    
    def MCN0(self, which='vary'):
        """
        Monte Carlo average of N0_sim over specified index range
        
        which: str : 'stat', 'vary', or 'const' to select which index range to use
                     'stat' uses stat_index
                     'vary' uses vary_index
                     'const' uses const_index
        
        Requires sim_config to be set, or manually set the corresponding index arrays.
        Note: vary_index overlaps with mf_index only when nsims_mf equals reuse_last.
        """
        fname = os.path.join(self.basedir, f'MCN0_{which}_fsky{self.filter.fsky:.2f}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            if which == 'stat':
                index = self.stat_index
            elif which == 'vary':
                index = self.vary_index
            elif which == 'const':
                index = self.const_index
            else:
                raise ValueError("which must be 'stat', 'vary', or 'const'")
            
            n0_list = []
            for idx in tqdm(index, desc=f'Computing MCN0 ({which})'):
                n0_list.append(self.N0_sim(idx, which=which))
            
            mcn0 = np.array(n0_list).mean(axis=0)
            pl.dump(mcn0, open(fname,'wb'))
            return mcn0
    
    def N1(self,binned=False):
        """
        N1 bias: difference between MCN0 for constant and varying alpha
        N1 = MCN0('const') - MCN0('vary')
        """
        fname = os.path.join(self.basedir, f'N1_fsky{self.filter.fsky:.2f}.pkl')
        if os.path.isfile(fname):
            n1 = pl.load(open(fname,'rb'))
        else:
            n1 = self.MCN0('const') - self.MCN0('vary')
            pl.dump(n1, open(fname,'wb'))
        if binned:
            bn1 = self.binner.bin_cell(n1[:self.lmax_bin+1])
            return bn1
        else:
            return n1
    
    def Nlens(self,MCN0=True,binned=False):
        """
        Lensing bias: average qcl on null_index minus MCN0('vary')
        Nlens = <qcl(null_alpha)> - MCN0('vary')
        """
        fname = os.path.join(self.basedir, f'Nlens_fsky{self.filter.fsky:.2f}_mcn0{MCN0}.pkl')
        if os.path.isfile(fname):
            nlens = pl.load(open(fname,'rb'))
        else:
            qcl_list = []
            for idx in tqdm(self.null_index, desc='Computing Nlens'):
                # Get qcl without any bias subtraction
                qcl_list.append(cs.utils.alm2cl(self.recon_lmax, self.qlm(idx))/self.filter.fsky)
            avg_qcl = np.array(qcl_list).mean(axis=0)
            if MCN0:
                nlens = avg_qcl - self.MCN0('vary')
            else:
                nlens = avg_qcl - self.norm
            pl.dump(nlens, open(fname,'wb'))
        if binned:
            bnlen = self.binner.bin_cell(nlens[:self.lmax_bin+1])
            return bnlen
        else:
            return nlens
        
    def RDN0_stat(self):
        """
        RDN0 for all stat_index simulations
        """
        fname = os.path.join(self.basedir, f'RDN0_stat_fsky{self.filter.fsky:.2f}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            rdn0_list = []
            for i in tqdm(self.stat_index, desc='Computing RDN0 statistics'):
                rdn0_list.append(self.RDN0(i))
            rdn0_array = np.array(rdn0_list).mean(axis=0)
            pl.dump(rdn0_array, open(fname,'wb'))
            return rdn0_array
          
    def qcl_stat(self, n0=None, mf=False, nlens=False, n1=False,binned=True):
        st = ''
        if n0 is None:
            st += '_noN0'
        elif n0 == 'norm':
            st += '_n0anal'
        elif n0 == 'rdn0':
            st += '_n0rdn0'
        elif n0 == 'mcn0':
            st += '_n0mcn0'
        else:
            raise ValueError("n0 must be 'norm', 'rdn0', or 'mcn0'")
        if mf:
            st += '_mf'
        if nlens:
            st += '_nlens'
        if n1:
            st += '_n1'
        fname = os.path.join(self.basedir, f'qcl_min{self.lmin}_max{self.lmax}_rmax{self.recon_lmax}_nlb{self.nlb}_{st}_{binned}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            cl = []
            for i in tqdm(self.stat_index,desc='Computing cl statistics',unit='sim'):
                cl.append(self.qcl(i,n0=n0,mf=mf,nlens=nlens,n1=n1,binned=binned))
            cl = np.array(cl)
            pl.dump(cl,open(fname,'wb'))
            return cl
        

class CrossQE:
    def __init__(self, filter: FilterEB, lmin: int, lmax: int, recon_lmax: int,nsims_mf=100, nlb=10, lmax_bin=1024):
        assert filter.sky.nsplits == 4, "CrossQE requires 4 splits in the FilterEB sky object."
        self.basedir = os.path.join(filter.sky.libdir, 'qecross')
        self.recdir = os.path.join(self.basedir, f'reco_min{lmin}_max{lmax}_rmax{recon_lmax}')
        self.n0dir = os.path.join(self.basedir, f'rdn0_min{lmin}_max{lmax}_rmax{recon_lmax}')
        self.mdir = os.path.join(self.basedir, f'misc_min{lmin}_max{lmax}_rmax{recon_lmax}')
        if mpi.rank == 0:
            os.makedirs(self.basedir, exist_ok=True)
            os.makedirs(self.recdir, exist_ok=True)
            os.makedirs(self.n0dir, exist_ok=True)
            os.makedirs(self.mdir, exist_ok=True)
        self.filter = filter
        self.sim_config = filter.sky.cmb.sim_config
        self.lmin = lmin
        self.lmax = lmax
        self.recon_lmax = recon_lmax
        self.cl_len = filter.cl_len
        
        self.lmax_bin = lmax_bin


        self.sim_config = filter.sky.cmb.sim_config

        if self.sim_config is None:
            raise ValueError("sim_config must be set in CMB initialization. QE requires sim_config to define simulation ranges.")
        # Default simulation ranges
        set1 = self.sim_config['set1']
        reuse_last = self.sim_config['reuse_last']
        
        # Statistics simulations: first (set1 - nsims_mf)
        self.stat_index = np.arange(0, set1 - nsims_mf)
        
        # Mean field simulations: last nsims_mf from set1
        self.mf_index = np.arange(set1 - nsims_mf, set1)
        
        # Varying alpha range: last reuse_last from set1
        self.vary_index = np.arange(set1 - reuse_last, set1)
        
        # Constant alpha range: set1 to set1+reuse_last
        self.const_index = np.arange(set1, set1 + reuse_last)
        
        # Null alpha range: set1+reuse_last to set1+2*reuse_last
        self.null_index = np.arange(set1 + reuse_last, set1 + 2 * reuse_last)
        self.binner = nmt.NmtBin.from_lmax_linear(lmax_bin,nlb)
        self.b = self.binner.get_effective_ells()
        self.nlb = nlb

    @property
    def cl_aa(self):
        return self.filter.sky.cmb.cl_aa()[:self.recon_lmax+1]
    
    def precompute_split_ocl(self, splits=(1,2,3,4)):
        """
        Computes split-level total spectra:
        oclE[s] = ClEE_lensed + <N_EE_split_s>
        oclB[s] = ClBB_lensed + <N_BB_split_s>
        Saves dict {s: (oclE, oclB)}.
        """
        fsky_tag = f"{self.filter.fsky:.2f}".replace('.', 'p')
        fname = os.path.join(
            self.basedir,
            f"ocl_splits_min{self.lmin}_max{self.lmax}_rmax{self.recon_lmax}_fsky_{fsky_tag}.pkl"
        )
        if os.path.isfile(fname):
            return pl.load(open(fname, "rb"))

        # Start from theory (lensed) spectra
        clE_th = self.cl_len[1, :self.lmax+1].copy()
        clB_th = self.cl_len[2, :self.lmax+1].copy()

        ocl = {}
        for s in splits:
            ne_acc = np.zeros(self.lmax+1, dtype=np.float64)
            nb_acc = np.zeros(self.lmax+1, dtype=np.float64)
            ncount = 0

            for i in tqdm(self.stat_index, desc=f"Split OCL s={s}"):
                ne, nb = self.filter.sky.HILC_obsEB(i, ret="nl", split=s)
                ne_acc += ne[:self.lmax+1] / Tcmb**2
                nb_acc += nb[:self.lmax+1] / Tcmb**2
                ncount += 1

            ne_mean = ne_acc / ncount
            nb_mean = nb_acc / ncount

            oclE = clE_th + ne_mean
            oclB = clB_th + nb_mean
            ocl[s] = (oclE, oclB)

        pl.dump(ocl, open(fname, "wb"))
        return ocl
    
    def precompute_pair_norms(self, pairs=((1,2),(3,4),(1,3),(2,4),(1,4),(2,3))):
        """
        Precomputes normalization arrays for each split pair.

        Returns dict:
        norms[(i,j)] = {"EiBj": norm_ij, "EjBi": norm_ji}
        where norm_ij normalizes Q(E^i, B^j).
        """
        fsky_tag = f"{self.filter.fsky:.2f}".replace('.', 'p')
        fname = os.path.join(
            self.basedir,
            f"norm_pairs_min{self.lmin}_max{self.lmax}_rmax{self.recon_lmax}_fsky_{fsky_tag}.pkl"
        )
        if os.path.isfile(fname):
            return pl.load(open(fname, "rb"))

        ocl = self.precompute_split_ocl(splits=(1,2,3,4))
        clE_th = self.cl_len[1, :self.lmax+1]

        norms = {}
        for (i, j) in pairs:
            oclE_i, oclB_i = ocl[i]
            oclE_j, oclB_j = ocl[j]

            # norm for Q(E^i, B^j)
            norm_ij = cs.norm_quad.qeb(
                'rot',
                self.recon_lmax, self.lmin, self.lmax,
                clE_th,
                oclE_i,   # total EE for E leg from split i
                oclB_j    # total BB for B leg from split j
            )[0]

            # norm for Q(E^j, B^i)
            norm_ji = cs.norm_quad.qeb(
                'rot',
                self.recon_lmax, self.lmin, self.lmax,
                clE_th,
                oclE_j,
                oclB_i
            )[0]

            norms[(i, j)] = {"EiBj": norm_ij, "EjBi": norm_ji}

        pl.dump(norms, open(fname, "wb"))
        return norms
    
    def qlm_pair(self, idx, si: int, sj: int):
        assert si != sj
        i, j = (si, sj) if si < sj else (sj, si)

        norms = self.precompute_pair_norms(
            pairs=((1,2),(3,4),(1,3),(2,4),(1,4),(2,3))
        )
        n_ij = norms[(i, j)]["EiBj"] if (si, sj) == (i, j) else norms[(i, j)]["EjBi"]
        n_ji = norms[(i, j)]["EjBi"] if (si, sj) == (i, j) else norms[(i, j)]["EiBj"]

        Ei, Bi = self.filter.cinv_EB(idx, split=si)
        Ej, Bj = self.filter.cinv_EB(idx, split=sj)

        alm_EiBj = cs.rec_rot.qeb(
            self.recon_lmax, self.lmin, self.lmax,
            self.cl_len[1, :self.lmax+1],
            Ei[:self.lmax+1, :self.lmax+1],
            Bj[:self.lmax+1, :self.lmax+1]
        )
        alm_EjBi = cs.rec_rot.qeb(
            self.recon_lmax, self.lmin, self.lmax,
            self.cl_len[1, :self.lmax+1],
            Ej[:self.lmax+1, :self.lmax+1],
            Bi[:self.lmax+1, :self.lmax+1]
        )

        # Apply pair-direction-specific norms, then symmetrize
        phi = 0.5 * (alm_EiBj * n_ij[:, None] + alm_EjBi * n_ji[:, None])
        return phi
    
    def mean_field_sim(self, idx, splits=(1,2,3,4)):
        """
        Cross-only lensing power estimate:
        average over cross-spectra of reconstructions from disjoint split pairs.
        For 4 splits, uses the 3 disjoint pairings.
        """

        fname = os.path.join(
            self.mdir,
            f"mean_min{self.lmin}_max{self.lmax}_rmax{self.recon_lmax}_fsky{self.filter.fsky:.2f}_{idx:04d}.pkl"
        )
        if os.path.isfile(fname):
            return pl.load(open(fname, "rb"))
        else:
            s1, s2, s3, s4 = splits
            
            # three disjoint pairings
            pairings = [
                ((s1, s2), (s3, s4)),
                ((s1, s3), (s2, s4)),
                ((s1, s4), (s2, s3)),
            ]

            m = None
            for (i, j), (k, l) in pairings:
                phi_ij = self.qlm_pair(idx, i, j)
                phi_kl = self.qlm_pair(idx, k, l)

                phi = 0.5 * (phi_ij + phi_kl)
                if m is None:
                    m = phi
                else:
                    m += phi
            m /= 3.0
            
            pl.dump(m, open(fname, "wb"))

            return m
    
    def mean_field(self):
        m = []
        for idx in tqdm(self.mf_index, desc='Computing cross-only mean field'):
            self.mean_field_sim(idx)
            m.append(self.mean_field_sim(idx))
        return np.mean(m, axis=0)
    
    def mean_field_cl(self):
        fname = os.path.join(
            self.mdir,
            f"meanfield_cl_min{self.lmin}_max{self.lmax}_rmax{self.recon_lmax}_fsky{self.filter.fsky:.2f}.pkl"
        )
        if os.path.isfile(fname):
            return pl.load(open(fname, "rb"))
        else:
            mf = self.mean_field()
            mf_cl = cs.utils.alm2cl(self.recon_lmax, mf) / self.filter.fsky
            pl.dump(mf_cl, open(fname, "wb"))
            return mf_cl
        
    def qcl_cross_only(self, idx, splits=(1,2,3,4)):
        """
        Cross-only lensing power estimate:
        average over cross-spectra of reconstructions from disjoint split pairs.
        For 4 splits, uses the 3 disjoint pairings.
        """

        fname = os.path.join(
            self.recdir,
            f"qcl_crossonly_min{self.lmin}_max{self.lmax}_rmax{self.recon_lmax}_fsky{self.filter.fsky:.2f}_{idx:04d}.pkl"
        )
        if os.path.isfile(fname):
            return pl.load(open(fname, "rb"))
        else:
            s1, s2, s3, s4 = splits
            
            # three disjoint pairings
            pairings = [
                ((s1, s2), (s3, s4)),
                ((s1, s3), (s2, s4)),
                ((s1, s4), (s2, s3)),
            ]

            cls = []
            for (i, j), (k, l) in pairings:
                phi_ij = self.qlm_pair(idx, i, j)
                phi_kl = self.qlm_pair(idx, k, l)

                # cross-spectrum of the two phi maps
                cl = cs.utils.alm2cl(self.recon_lmax, phi_ij, phi_kl) / self.filter.fsky
                cls.append(cl)
            
            pl.dump(np.mean(cls, axis=0), open(fname, "wb"))

            return np.mean(cls, axis=0)
       
    def Nlens(self,binned=False):
        """
        Lensing bias: average qcl on null_index minus MCN0('vary')
        Nlens = <qcl(null_alpha)> - MCN0('vary')
        """
        fname = os.path.join(self.basedir, f'Nlens_fsky{self.filter.fsky:.2f}.pkl')
        if os.path.isfile(fname):
            nlens = pl.load(open(fname,'rb'))
        else:
            qcl_list = []
            for idx in tqdm(self.null_index, desc='Computing Nlens'):
                # Get qcl without any bias subtraction
                qcl_list.append(self.qcl_cross_only(idx))
            avg_qcl = np.array(qcl_list).mean(axis=0)
            nlens = avg_qcl - self.MCN0('stat')
            pl.dump(nlens, open(fname,'wb'))
        if binned:
            bnlen = self.binner.bin_cell(nlens[:self.lmax_bin+1])
            return bnlen
        else:
            return nlens
    
    def N1(self,binned=False):
        """
        N1 bias: difference between MCN0 for constant and varying alpha
        N1 = MCN0('const') - MCN0('vary')
        """
        fname = os.path.join(self.basedir, f'N1_fsky{self.filter.fsky:.2f}.pkl')
        if os.path.isfile(fname):
            n1 = pl.load(open(fname,'rb'))
        else:
            n1 = self.MCN0('const') - self.MCN0('vary')
            pl.dump(n1, open(fname,'wb'))
        if binned:
            bn1 = self.binner.bin_cell(n1[:self.lmax_bin+1])
            return bn1
        else:
            return n1  
         
    def qcl_stat(self, n0=None, mf=False, nlens=False, binned=True):
        """
        Compute statistics of qcl over stat_sims with various bias corrections.
        
        Parameters
        ----------
        n0 : str or None
            N0 bias correction: None (no correction), 'mcn0', or 'rdn0'
        mf : bool
            If True, subtract mean field bias
        nlens : bool
            If True, subtract lensing bias
        binned : bool
            If True, return binned spectra
            
        Returns
        -------
        cl : array
            Array of shape (nsims, nlmax+1) or (nsims, nbins) if binned
        """
        st = ''
        if n0 is None:
            st += '_noN0'
        elif n0 == 'mcn0':
            st += '_n0mcn0'
        elif n0 == 'rdn0':
            st += '_n0rdn0'
        else:
            raise ValueError("n0 must be None, 'mcn0', or 'rdn0'")
        if mf:
            st += '_mf'
        if nlens:
            st += '_nlens'
        fname = os.path.join(
            self.basedir, 
            f'qcl_crossonly_min{self.lmin}_max{self.lmax}_rmax{self.recon_lmax}_nlb{self.nlb}_{st}_{binned}.pkl'
        )
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            # Determine bias corrections
            if n0 is None:
                N0 = np.zeros(self.recon_lmax + 1)
            elif n0 == 'mcn0':
                N0 = self.MCN0()
            elif n0 == 'rdn0':
                # For cross-only, RDN0 should be computed per simulation
                # This will be handled in the loop below
                N0 = None
            
            if mf:
                MF = self.mean_field_cl()
            else:
                MF = np.zeros(self.recon_lmax + 1)
            
            if nlens:
                NLENS = self.Nlens(binned=False)
            else:
                NLENS = np.zeros(self.recon_lmax + 1)
            
            cl = []
            for i in tqdm(range(self.stat_sims[0], self.stat_sims[1]), desc='Computing cross-only cl statistics', unit='sim'):
                qcl_i = self.qcl_cross_only(i, splits=(1,2,3,4))
                
                # Apply RDN0 correction if requested
                if n0 == 'rdn0':
                    N0_i = self.RDN0(i)
                else:
                    N0_i = N0
                
                # Apply all bias corrections
                qcl_i = qcl_i - N0_i - MF - NLENS
                
                if binned:
                    qcl_i = self.binner.bin_cell(qcl_i[:self.lmax_bin+1])
                
                cl.append(qcl_i)
            
            cl = np.array(cl)
            pl.dump(cl, open(fname, 'wb'))
            return cl
    
    def _cl_cross_only_between(self, A: dict, B: dict):
        """
        Cross-only spectrum for 4 splits:
        (12×34 + 13×24 + 14×23)/3
        where A[(i,j)] are recon alms from split pair (i,j).
        """
        c12_34 = cs.utils.alm2cl(self.recon_lmax, A[(1,2)], B[(3,4)]) / self.filter.fsky
        c13_24 = cs.utils.alm2cl(self.recon_lmax, A[(1,3)], B[(2,4)]) / self.filter.fsky
        c14_23 = cs.utils.alm2cl(self.recon_lmax, A[(1,4)], B[(2,3)]) / self.filter.fsky
        return (c12_34 + c13_24 + c14_23) / 3.0
    
    def _qlm_pair_tag_sym(self, d_idx: int, s_idx: int, s0_idx: int,
                      si: int, sj: int, tagE: str, tagB: str):
        """
        Symmetric EB estimator (lensing-like), compatible with qlm_pair():
        0.5 * [ Q(E^si, B^sj) + Q(E^sj, B^si) ]
        with direction-dependent norms.
        tagE/tagB in {'d','s','s0'} select which realization supplies E and B legs.
        """
        assert si != sj

        def pick(tag: str) -> int:
            if tag == "d":  return d_idx
            if tag == "s":  return s_idx
            if tag == "s0": return s0_idx
            raise ValueError("tag must be one of {'d','s','s0'}")

        idxE = pick(tagE)
        idxB = pick(tagB)

        # direction-dependent norms (same logic as qlm_pair)
        norms = self.precompute_pair_norms(pairs=((1,2),(3,4),(1,3),(2,4),(1,4),(2,3)))
        i, j = (si, sj) if si < sj else (sj, si)

        if (si, sj) == (i, j):
            n_EiBj = norms[(i, j)]["EiBj"]
            n_EjBi = norms[(i, j)]["EjBi"]
        else:
            n_EiBj = norms[(i, j)]["EjBi"]
            n_EjBi = norms[(i, j)]["EiBj"]

        # Pull E from idxE, B from idxB, with the requested split indices
        E_si, _ = self.filter.cinv_EB(idxE, split=si)
        E_sj, _ = self.filter.cinv_EB(idxE, split=sj)
        _, B_si = self.filter.cinv_EB(idxB, split=si)
        _, B_sj = self.filter.cinv_EB(idxB, split=sj)

        # Q(E_si, B_sj)
        alm_ij = cs.rec_rot.qeb(
            self.recon_lmax, self.lmin, self.lmax,
            self.cl_len[1, :self.lmax+1],
            E_si[:self.lmax+1, :self.lmax+1],
            B_sj[:self.lmax+1, :self.lmax+1]
        )
        # Q(E_sj, B_si)
        alm_ji = cs.rec_rot.qeb(
            self.recon_lmax, self.lmin, self.lmax,
            self.cl_len[1, :self.lmax+1],
            E_sj[:self.lmax+1, :self.lmax+1],
            B_si[:self.lmax+1, :self.lmax+1]
        )

        # SYM: symmetric combination + direction norms
        phi = 0.5 * (alm_ij * n_EiBj[:, None] + alm_ji * n_EjBi[:, None])
        return phi

    def _build_six_sym(self, d_idx: int, s_idx: int, s0_idx: int, tagE: str, tagB: str):
        """
        Build the six split-pair reconstructions (12,34,13,24,14,23) using the symmetric estimator.
        """
        pairs = [(1,2),(3,4),(1,3),(2,4),(1,4),(2,3)]
        out = {}
        for (i, j) in pairs:
            out[(i, j)] = self._qlm_pair_tag_sym(d_idx, s_idx, s0_idx, i, j, tagE, tagB)
        return out
    
    def N0_sim(self,idx,which='vary',debug=False):
        """
        Calculate the N0 bias from the Reconstructed potential using filtered Fields
        with different CMB fields. If E modes is from ith simulation then B modes is 
        from (i+1)th simulation

        idx: int : index of the N0
        which: str : 'stat', 'vary', or 'const' to select which index range to use
        debug: bool : if True, print the indices used for computation and return None
        
        Index ranges and wrapping:
        - 'stat': stat_index, wraps within stat range
        - 'vary': vary_index, wraps within vary range
        - 'const': const_index, wraps within const range
        
        Requires sim_config to be set, or manually set the corresponding index array.
        """
        if which == 'stat':
            index_range = self.stat_index
            label = 'stat'
        elif which == 'vary':
            index_range = self.vary_index
            label = 'vary'
        elif which == 'const':
            index_range = self.const_index
            label = 'const'
        else:
            raise ValueError("which must be 'stat', 'vary', or 'const'")
        
        assert idx in index_range, f"The requested idx {idx} is not in the {which} index range"
            
        # Simple increment with wrapping within the index_range bounds
        idx1 = idx
        min_idx = min(index_range)
        max_idx = max(index_range)
        # If at the end of range, wrap to beginning of range
        if idx == max_idx:
            idx2 = min_idx
        else:
            idx2 = idx + 1
        
        if debug:
            print(f"N0_sim debug mode:")
            print(f"  which: {which}")
            print(f"  index_range: [{min_idx}, {max_idx}]")
            print(f"  idx1 (E1B1): {idx1}")
            print(f"  idx2 (E2B2): {idx2}")
            return None
            
        # choose (a,b) as consecutive sims, like your lensing codeims[0],self.stat_sims[1]), (0,1), mode="wrap")
        # pad/wrap to avoid idx+1 overflo
        a = idx1
        b = idx2

        fname = os.path.join(self.n0dir, f"N0_{label}_{self.filter.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname, "rb"))

        # Build six recon alms for each split-pair:
        # ab = Q(E_a, B_b), ba = Q(E_b, B_a)
        # We can use your (d_idx, s_idx, s0_idx) interface with tags:
        # tagE="s", tagB="s0"  -> E from s_idx, B from s0_idx
        # tagE="s0", tagB="s"  -> E from s0_idx, B from s_idx
        ab = self._build_six_sym(d_idx=0, s_idx=a, s0_idx=b, tagE="s",  tagB="s0")
        ba = self._build_six_sym(d_idx=0, s_idx=a, s0_idx=b, tagE="s0", tagB="s")

        # Cross-only operator already divides by fsky internally.
        n0 = self._cl_cross_only_between(ab, ab) + self._cl_cross_only_between(ab, ba)

        pl.dump(n0, open(fname, "wb"))
        return n0
    
    def MCN0(self, which='vary'):
        fname = os.path.join(self.basedir, f'MCN0_{which}_fsky{self.filter.fsky:.2f}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            if which == 'stat':
                index = self.stat_index
            elif which == 'vary':
                index = self.vary_index
            elif which == 'const':
                index = self.const_index
            else:
                raise ValueError("which must be 'stat', 'vary', or 'const'")
            
            n0_list = []
            for idx in tqdm(index, desc=f'Computing MCN0 ({which})'):
                n0_list.append(self.N0_sim(idx, which=which))
            
            mcn0 = np.array(n0_list).mean(axis=0)
            pl.dump(mcn0, open(fname,'wb'))
            return mcn0
    
    def RDN0(self, idx: int, navg: int = 100):
        """
        Realization-dependent N0 (RDN0) compatible with qcl_cross_only() (symmetric estimator).

        Uses the same Eq.(43)-style structure:
        RDN0 = Cx(ds,ds) + Cx(ds,sd) + Cx(sd,ds) + Cx(sd,sd) - Cx(ss0,ss0) - Cx(ss0,s0s)
        where Cx is the cross-only spectrum operator (12×34+13×24+14×23)/3.

        Returns: array C_L[0..recon_lmax]
        """
        fsky_tag = f"{self.filter.fsky:.2f}".replace('.', 'p')
        fname = os.path.join(self.n0dir, f"RDN0x_{fsky_tag}_{idx:04d}_navg{navg}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname, "rb"))

        # pool of sim indices excluding idx
        myidx = np.append(np.arange(100), np.arange(2))
        myidx = np.delete(myidx, np.where(myidx == idx)[0])

        d_idx = idx
        acc = np.zeros(self.recon_lmax + 1, dtype=np.float64)
        nused = 0

        for i in tqdm(range(navg), desc=f"RDN0x for sim {idx}", unit="mc"):
            s_idx  = int(myidx[i])
            s0_idx = int(myidx[i+1])

            # Build the needed recon sets
            ds  = self._build_six_sym(d_idx, s_idx, s0_idx, tagE="d",  tagB="s")
            sd  = self._build_six_sym(d_idx, s_idx, s0_idx, tagE="s",  tagB="d")
            ss0 = self._build_six_sym(d_idx, s_idx, s0_idx, tagE="s",  tagB="s0")
            s0s = self._build_six_sym(d_idx, s_idx, s0_idx, tagE="s0", tagB="s")

            cl_ds_ds   = self._cl_cross_only_between(ds,  ds)
            cl_ds_sd   = self._cl_cross_only_between(ds,  sd)
            cl_sd_ds   = self._cl_cross_only_between(sd,  ds)
            cl_sd_sd   = self._cl_cross_only_between(sd,  sd)
            cl_ss0_ss0 = self._cl_cross_only_between(ss0, ss0)
            cl_ss0_s0s = self._cl_cross_only_between(ss0, s0s)

            rdn0_i = (cl_ds_ds + cl_ds_sd + cl_sd_ds + cl_sd_sd
                    - cl_ss0_ss0 - cl_ss0_s0s)

            acc += rdn0_i
            nused += 1

        rdn0 = acc / max(nused, 1)
        pl.dump(rdn0, open(fname, "wb"))
        return rdn0
    
    def RDN0_mpi(self, idx: int, navg: int = 100):
        """
        MPI-parallel cross-only RDN0 for SYM estimator (compatible with qcl_cross_only()).

        Implements the Eq.(43)-style combination (same algebra as your rot version),
        but with SYM estimator:
        RDN0 = Cx(ds,ds)+Cx(ds,sd)+Cx(sd,ds)+Cx(sd,sd) - Cx(ss0,ss0) - Cx(ss0,s0s)

        Parallelizes navg Monte-Carlo draws over (s, s0) pairs.
        Only rank 0 reads/writes the cache.
        """
        MPI  = mpi.mpi
        comm = mpi.com
        rank = mpi.rank
        size = mpi.size

        fsky_tag = f"{self.filter.fsky:.2f}".replace('.', 'p')
        fname = os.path.join(self.n0dir, f"RDN0x_{fsky_tag}_{idx:04d}_navg{navg}.pkl")

        # 1) Try to load cached result on rank 0, then broadcast.
        rdn0 = None
        if rank == 0 and os.path.isfile(fname):
            with open(fname, "rb") as f:
                rdn0 = pl.load(f)
        rdn0 = comm.bcast(rdn0, root=0)
        if rdn0 is not None:
            return rdn0

        # 2) Build deterministic index pool excluding idx (same trick).
        nsim = 100  # or self.sim_config.nsim if available and consistent
        myidx = np.append(np.arange(nsim), np.arange(2))
        myidx = np.delete(myidx, np.where(myidx == idx)[0])

        # 3) Distribute tasks.
        tasks = np.arange(navg, dtype=int)
        chunks = np.array_split(tasks, size)
        my_tasks = chunks[rank]

        L = self.recon_lmax + 1
        local_sum = np.zeros(L, dtype=np.float64)

        # 4) One MC draw.
        def _one_draw(i: int):
            s_idx  = int(myidx[i % len(myidx)])
            s0_idx = int(myidx[(i + 1) % len(myidx)])
            d_idx = idx

            # Build the mixed estimators needed in Eq.(43)
            ds  = self._build_six_sym(d_idx, s_idx, s0_idx, tagE="d",  tagB="s")
            sd  = self._build_six_sym(d_idx, s_idx, s0_idx, tagE="s",  tagB="d")
            ss0 = self._build_six_sym(d_idx, s_idx, s0_idx, tagE="s",  tagB="s0")
            s0s = self._build_six_sym(d_idx, s_idx, s0_idx, tagE="s0", tagB="s")

            cl_ds_ds   = self._cl_cross_only_between(ds,  ds)
            cl_ds_sd   = self._cl_cross_only_between(ds,  sd)
            cl_sd_ds   = self._cl_cross_only_between(sd,  ds)
            cl_sd_sd   = self._cl_cross_only_between(sd,  sd)
            cl_ss0_ss0 = self._cl_cross_only_between(ss0, ss0)
            cl_ss0_s0s = self._cl_cross_only_between(ss0, s0s)

            return (cl_ds_ds + cl_ds_sd + cl_sd_ds + cl_sd_sd
                    - cl_ss0_ss0 - cl_ss0_s0s)

        # 5) Local accumulation (progress only on rank 0).
        iterator = tqdm(my_tasks, desc=f'RDN0x(sym) rank {rank} for sim {idx}', unit='mc') if rank == 0 else my_tasks
        for t in iterator:
            local_sum += _one_draw(int(t))

        # 6) Reduce to rank 0 and normalize by navg.
        global_sum = np.zeros_like(local_sum)
        comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)

        if rank == 0:
            rdn0 = global_sum / float(len(tasks))
            os.makedirs(self.n0dir, exist_ok=True)
            with open(fname, "wb") as f:
                pl.dump(rdn0, f)

        # 7) Broadcast to all ranks.
        rdn0 = comm.bcast(rdn0, root=0)
        return rdn0

    def RDN0_stat(self):
        """
        RDN0 for all stat_index simulations
        """
        fname = os.path.join(self.basedir, f'RDN0_stat_fsky{self.filter.fsky:.2f}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            rdn0_list = []
            for i in tqdm(range(100), desc='Computing RDN0 statistics'):
                rdn0_list.append(self.RDN0(i))
            rdn0_array = np.array(rdn0_list).mean(axis=0)
            pl.dump(rdn0_array, open(fname,'wb'))
            return rdn0_array

