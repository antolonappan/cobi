import os
import curvedsky as cs
import numpy as np
import healpy as hp
import pickle as pl

class QE:

    def __init__(self,libdir,simlib,lmax=2048,cmb_lmin=50,cmb_lmax=2048,filt='iso'):
        self.libdir = libdir
        self.simlib = simlib
        self.lmax = lmax
        self.cmb_lmin = cmb_lmin
        self.cmb_lmax = cmb_lmax
        self.libdir = os.path.join(libdir,'QE')
        self.filt = filt
        self.Tcmb = 2.726e6
        self.lcl = (self.simlib.cmb.get_lensed_spectra(dl=False,dtype='a')/self.Tcmb**2).T
        os.makedirs(self.libdir,exist_ok=True)

    def norm(self,idx):
        lcl = self.lcl.copy()
        neb = self.ilc_noise_spectra(idx)
        nl  = np.zeros((4,self.lmax+1))
        nl[1,:] = neb[0][:self.lmax+1]
        nl[2,:] = neb[1][:self.lmax+1]
        ocl = self.lcl[:,:self.lmax+1] + nl

        Al = np.zeros((2,self.lmax+1))
        Al[0,:] = cs.norm_quad.qeb('rot',self.lmax,self.cmb_lmin,self.cmb_lmax,lcl[1,:self.cmb_lmax+1],ocl[1,:self.cmb_lmax+1],ocl[2,:self.cmb_lmax+1])[0]
        Al[1,:] = cs.norm_quad.qtb('rot',self.lmax,self.cmb_lmin,self.cmb_lmax,lcl[3,:self.cmb_lmax+1],ocl[0,:self.cmb_lmax+1],ocl[2,:self.cmb_lmax+1])[0]
        return Al

    def filteredEB(self,idx):
        if self.filt == 'iso':
            return self.iso_filt_EB(idx)
        elif self.filt == 'aniso':
            #return aniso_filt_EB(idx)
            raise NotImplementedError('Anisotropic filtering not implemented yet')
        else:
            raise ValueError('Invalid filter type, must be iso or aniso')
        
    def ilc_noise_spectra(self,idx):
        _, neb = self.simlib.HILC_obsEB(idx)
        del _
        return neb/self.Tcmb**2
        
    def iso_filt_EB(self,idx):
        
        EB, neb = self.simlib.HILC_obsEB(idx)
        lmax = hp.Alm.getlmax(len(EB[0]))
        Ealm = cs.utils.lm_healpy2healpix(EB[0]/self.Tcmb,lmax)
        Balm = cs.utils.lm_healpy2healpix(EB[1]/self.Tcmb,lmax)
        nl  = np.zeros((4,self.lmax+1))
        nl[1,:] = neb[0][:self.lmax+1]/self.Tcmb**2
        nl[2,:] = neb[1][:self.lmax+1]/self.Tcmb**2
        ocl = self.lcl[:,:self.lmax+1] + nl
        Fl = np.zeros((3,self.lmax+1,self.lmax+1))
        for l in range(self.cmb_lmin,self.cmb_lmax):
            Fl[:,l,0:l+1] = 1./ocl[:3,l,None]
        Ealm = Ealm[:self.lmax+1,:self.lmax+1]
        Balm = Balm[:self.lmax+1,:self.lmax+1]
        Ealm *= Fl[1,:,:]
        Balm *= Fl[2,:,:]
        Ealm = Ealm[:self.cmb_lmax+1,:self.cmb_lmax+1]
        Balm = Balm[:self.cmb_lmax+1,:self.cmb_lmax+1]
        return Ealm,Balm
    
    def reconstruct(self,idx):
        Ealm,Balm = self.filteredEB(idx)
        Al = self.norm(idx)
        return cs.rec_rot.qeb(self.lmax,
                            self.cmb_lmin,
                            self.cmb_lmax,
                            self.lcl[1,:self.cmb_lmax+1],
                            Ealm,Balm)
    
    def norm_reconstruct(self,idx):
        aalm = self.reconstruct(idx)
        Al = self.norm(idx)
        return aalm[:self.lmax+1,:self.lmax+1] * Al[0][:,None]
    

    def input_alpha(self,idx):
        input_aalm = cs.utils.hp_map2alm(self.simlib.cmb.nside,self.lmax,self.lmax,self.simlib.cmb.alpha_map(idx))
        return input_aalm
    
    
    