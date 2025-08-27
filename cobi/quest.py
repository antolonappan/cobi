import os
import curvedsky as cs
import numpy as np
import healpy as hp
import pickle as pl
import matplotlib.pyplot as plt
import pymaster as nmt
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import emcee

from cobi.simulation import LATsky, Mask
from cobi.utils import cli, slice_alms
from cobi import sht
from cobi import mpi


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

    def convolved_EB(self,idx):
        """
        convolve the component separated map with the beam

        Parameters
        ----------
        idx : int : index of the simulation
        """
        E,B = self.sky.HILC_obsEB(idx,ret='alm')
        hp.almxfl(E,self.bl,inplace=True)
        hp.almxfl(B,self.bl,inplace=True)
        return E,B
    
    def NL(self,idx):
        """
        array manipulation of noise spectra obtained by ILC weight
        for the filtering process
        """
        ne,nb = self.sky.HILC_obsEB(idx, ret='nl')/Tcmb**2
        return np.reshape(np.array((cli(ne[:self.lmax+1]*self.bl**2),
                          cli(nb[:self.lmax+1]*self.bl**2))),(2,1,self.lmax+1))
    
    
    def QU(self, idx):
        """
        deconvolve the beam from the QU map

        Parameters
        ----------
        idx : int : index of the simulation
        """
        E, B = self.convolved_EB(idx)
        E, B = slice_alms(np.array([E, B]), self.lmax)
        if self.healpy:
            QU = hp.alm2map([E*0,E,B], self.nside)[1:]/Tcmb
        else:
            QU = self.hp.alm2map([E, B], lmax=self.lmax,nthreads=NCPUS)/Tcmb
        QU = QU*self.mask
        QU[QU == -0] = 0
        return QU

    def cinv_EB(self,idx,test=False):
        """
        C inv Filter for the component separated maps

        Parameters
        ----------
        idx : int : index of the simulation
        test : bool : if True, run the filter for 10 iterations
        """
        fsky = f"{self.fsky:.2f}".replace('.','p')
        fname = os.path.join(self.lib_dir,f"cinv_EB_{idx:04d}_fsky_{fsky}.pkl")
        if not os.path.isfile(fname):
            QU = self.QU(idx)
            QU = np.reshape(QU,(2,1,hp.nside2npix(self.nside)))
            
            iterations = [200]
            stat_file = '' 
            if test:
                iterations = [10]
                stat_file = os.path.join('test_stat.txt')

            E,B = cs.cninv.cnfilter_freq(2,1,self.nside,self.lmax,self.cl_len[1:3,:self.lmax+1],
                                        self.Bl, self.ninv,QU,chn=1,itns=iterations,filter="",
                                        eps=[1e-5],ro=10,inl=self.NL(idx),stat=stat_file)
            if not test:
                pl.dump((E,B),open(fname,'wb'))
        else:
            E,B = pl.load(open(fname,'rb'))
        
        return E,B


    def plot_cinv(self,idx,lmin=2,lmax=3000):
        """
        plot the cinv filtered Cls for a given idx

        Parameters
        ----------
        idx : int : index of the simulation
        """
        E,_ = self.cinv_EB(idx)
        ne,_ = self.sky.HILC_obsEB(idx, ret='nl')/Tcmb**2
        cle = cs.utils.alm2cl(self.lmax,E)/self.fsky
        plt.figure(figsize=(4,4))
        plt.loglog(cle,label='Cinv E mode')
        plt.loglog(1/(self.cl_len[1,:len(ne)]  + ne), label='1/(S+N)')
        plt.xlim(lmin,lmax)
        plt.legend()




        
    

class QE:
    def __init__(self, filter: FilterEB, lmin: int, lmax: int, recon_lmax: int, norm_nsim=100, nlb=2, lmax_bin=100):
        self.basedir = os.path.join(filter.sky.libdir, 'qe')
        self.recdir = os.path.join(self.basedir, f'reco_min{lmin}_max{lmax}_rmax{recon_lmax}')
        self.rdn0dir = os.path.join(self.basedir, f'rdn0_min{lmin}_max{lmax}_rmax{recon_lmax}')
        if mpi.rank == 0:
            os.makedirs(self.basedir, exist_ok=True)
            os.makedirs(self.recdir, exist_ok=True)
            os.makedirs(self.rdn0dir, exist_ok=True)

        self.filter = filter
        self.lmin = lmin
        self.lmax = lmax
        self.recon_lmax = recon_lmax
        self.cl_len = filter.cl_len
        self.norm_nsim = norm_nsim
        self.norm = self.__norm__
        self.lmax_bin = lmax_bin
        self.binner = nmt.NmtBin.from_lmax_linear(lmax_bin,nlb)
        self.b = self.binner.get_effective_ells()
       


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
            for i in tqdm(range(self.norm_nsim), desc='Computing OCL'):
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

    def __qcl__(self,idx, rm_bias=False, rdn0=False):
        if rdn0:
            N0 = self.RDN0(idx)
        else:
            N0 = self.norm
        if rm_bias:
            return (cs.utils.alm2cl(self.recon_lmax,self.qlm(idx))/self.filter.fsky) - self.mean_field_cl() - N0
        else:
            return (cs.utils.alm2cl(self.recon_lmax,self.qlm(idx))/self.filter.fsky)
        
    def qcl(self,idx,rm_bias=False,rdn0=False,binned=False):
        if binned:
            cl = self.__qcl__(idx,rm_bias=rm_bias,rdn0=rdn0)
            bcl = self.binner.bin_cell(cl[:self.lmax_bin+1])
            return bcl
        else:
            return self.__qcl__(idx,rm_bias=rm_bias)

    def mean_field(self):
        fname = os.path.join(self.basedir, f'mf100_fsky{self.filter.fsky:.2f}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname, 'rb'))
        else:
            mf = np.zeros_like(self.qlm(0))
            for i in tqdm(range(200,300), desc='Computing mean field'):
                mf += self.qlm(i)
            mf /= 100
            pl.dump(mf, open(fname, 'wb'))
            return mf
    
    def mean_field_cl(self):
        return cs.utils.alm2cl(self.recon_lmax, self.mean_field()) / self.filter.fsky
    
    def RDN0(self,idx):
        fname = os.path.join(self.rdn0dir,f"RDN0_{self.filter.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            myidx = np.append(np.arange(200),np.arange(2))
            sel = np.where(myidx == idx)[0]
            myidx = np.delete(myidx,sel)

            E0,B0 = self.filter.cinv_EB(idx)

            mean_rdn0 = []

            for i in tqdm(range(100),desc=f'RDN0 for simulation {idx}', unit='sim'):
                E1,B1 = self.filter.cinv_EB(myidx[i])
                E2,B2 = self.filter.cinv_EB(myidx[i+1])
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
        myidx = np.append(np.arange(200), np.arange(2))
        sel = np.where(myidx == idx)[0]
        myidx = np.delete(myidx, sel)

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
            # Pull the paired indices needed for this Monte-Carlo draw
            i1 = int(myidx[i])
            i2 = int(myidx[i+1])

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
    
    def N0_sim(self,idx):
        """
        Calculate the N0 bias from the Reconstructed potential using filtered Fields
        with different CMB fields. If E modes is from ith simulation then B modes is 
        from (i+1)th simulation

        idx: int : index of the N0
        """
        myidx = np.pad(np.arange(300),(0,1),'constant',constant_values=(0,0))
        fname = os.path.join(self.rdn0dir,f"N0_{self.filter.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            E1,B1 = self.filter.cinv_EB(myidx[idx])
            E2,B2 = self.filter.cinv_EB(myidx[idx+1])
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
    
    def qcl_stat(self,rm_bias=True,rdn0=False,binned=True):
        st = ''
        if rm_bias:
            st += '_rb'
        if rdn0:
            st += '_rdn0'
        if binned:
            st += '_b'
        fname = os.path.join(self.basedir, f'qcl_min{self.lmin}_max{self.lmax}_rmax{self.recon_lmax}{st}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            cl = []
            for i in tqdm(range(200),desc='Computing cl statistics',unit='sim'):
                cl.append(self.qcl(i,rm_bias=rm_bias,rdn0=rdn0,binned=binned))
            cl = np.array(cl)
            pl.dump(cl,open(fname,'wb'))
            return cl
        
    
class AcbLikelihood:

    def __init__(self, qelib: QE, lmin=2,lmax=50):
        self.qe = qelib
        self.lmax = lmax
        qcl = self.qe.qcl_stat()*1e7
        self.binner = self.qe.binner
        self.b = self.qe.b
        self.sel = np.where((self.b >= lmin) & (self.b <= lmax))[0]
        self.mean = qcl.mean(axis=0)[self.sel]
        self.cov = np.cov(qcl.T)[self.sel][:,self.sel]
        self.std = qcl.std(axis=0)[self.sel]
        self.icov = np.linalg.inv(self.cov)

    def theory(self, Acb):
        l = np.arange(self.qe.lmax_bin+1)
        cl =  Acb * 2 * np.pi / ( l**2 + l + 1e-30)*1e7
        cl[0], cl[1] = 0, 0
        return self.binner.bin_cell(cl)[self.sel]
    
    def plot(self,Acb):
        plt.figure(figsize=(6,6))
        plt.errorbar(self.b[self.sel], self.mean, yerr=self.std[self.sel], fmt='o')
        plt.loglog(self.b[self.sel], self.theory(Acb), label='Theory')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$C_l^{\mathrm{BB}}$')
        plt.title(r'$C_l^{\mathrm{BB}}$ vs $l$')
        plt.grid()
        plt.legend()
        plt.show()

    def ln_prior(self, Acb):
        if 0 < Acb < 1e-5:
            return 0.0
        else:
            return -np.inf
        
    def ln_likelihood(self, Acb):
        theory = self.theory(Acb)
        delta = theory - self.mean
        chisq = (delta/self.std)**2
        return -0.5 * chisq.sum()

    def ln_posterior(self, Acb):
        lp = self.ln_prior(Acb)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(Acb)
    
    def sampler(self,nwalkers=32,nsamples=1000):
        pos = np.array([3.5e-6]) * np.random.randn(nwalkers, 1)
        sampler = emcee.EnsembleSampler(nwalkers, 1, self.ln_posterior)
        sampler.run_mcmc(pos, nsamples, progress=True)
        return sampler

    def samples(self, nwalkers=100, nsamples=2000, discard=300):
        sampler = self.sampler(nwalkers=nwalkers, nsamples=nsamples)
        flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
        return flat_samples
