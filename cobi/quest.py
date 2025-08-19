import os
import curvedsky as cs
import numpy as np
import healpy as hp
import pickle as pl
import matplotlib.pyplot as plt

from cobi.simulation import LATsky
from cobi.utils import cli, slice_alms
from cobi import sht
from cobi import mpi


Tcmb  = 2.726e6
NCPUS = os.cpu_count()

class FilterEB:

    def __init__(self, sky: LATsky, lmax: int, mask: np.ndarray, fwhm: float = 2, sht_backend: str = "healpy"):
        self.sky = sky
        self.nside = sky.nside
        self.lmax = min(lmax, 3*self.nside -1)
        self.mask = mask
        self.fsky     = np.mean(self.mask**2)**2/np.mean(self.mask**4)
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


    def plot_cinv(self,idx):
        """
        plot the cinv filtered Cls for a given idx

        Parameters
        ----------
        idx : int : index of the simulation
        """
        E,_ = self.cinv_EB(idx)
        ne,_ = self.sky.HILC_obsEB(idx, ret='nl')/Tcmb**2
        cle = cs.utils.alm2cl(self.lmax,E)
        plt.figure(figsize=(8,8))
        plt.loglog(cle,label='B')
        plt.loglog(1/(self.cl_len[1,:len(ne)]  + ne))




        
    

