import numpy as np
import os
from cobi.spectra import Spectra
from cobi.simulation import LATsky
from cobi.mle import MLE, LinearSystem, Result, effective_ell



class S2N:
    
    def __init__(self, libdir, mode, nside, atm_noise, nsplits, dust, sync, 
                 template_bandpass, fit, bmin, bmax, alpha_per_split, rm_same_tube,
                 bandpass=True, aposcale=2.0, CO=True, PS=True, pureB=True,
                 window=5, binwidth=20,parallel=0):
        
        self.libdir    = libdir+'/S2N'
        os.makedirs(self.libdir, exist_ok=True)
        self.mode      = mode
        self.window    = window
        self.nside     = nside
        self.atm_noise = atm_noise
        self.dust      = dust
        self.sync      = sync
        self.nsplits   = nsplits
        self.bp        = bandpass
        self.temp_bp   = template_bandpass
        self.apo       = aposcale
        self.CO        = CO
        self.PS        = PS
        # create specific simulations for s/n calculation
        # fixed values for the calculation of the s/n
        self.beta            = 0.3 # deg
        self.alpha           = 0.3 # deg
        self.idx             = 0
        self.sky             = LATsky(self.libdir, nside, 'iso', dust, sync, 
                                      self.alpha,self.beta, atm_noise=atm_noise,
                                      nsplits=nsplits, bandpass=self.bp)

        #self.sky.saveObsQUs(self.idx)
        self.spec            = Spectra(self.sky, template_bandpass=template_bandpass,
                                       pureB=pureB, aposcale=aposcale, CO=CO, PS=PS,
                                       parallel=parallel)
        self.fit             = fit
        self.alpha_per_split = alpha_per_split
        self.rm_same_tube    = rm_same_tube
        self.bmin            = bmin
        self.bmax            = bmax
        self.mle             = MLE(self.libdir, self.spec, fit, 
                                   alpha_per_split=alpha_per_split, 
                                   rm_same_tube=rm_same_tube, 
                                   bmax=bmax, bmin=bmin, binwidth=binwidth)
     
          
    def s2n_ell(self, iC, Niter, res):
        if self.fit=="alpha":
            return self.__s2n_ell_alpha__(iC, Niter, res)
        elif self.fit=="Ad + alpha":
            return self.__s2n_ell_Ad_alpha__(iC, Niter, res) 
        elif self.fit=="beta + alpha":
            return self.__s2n_ell_beta_alpha__(iC, Niter, res)
        elif self.fit=="As + Ad + alpha":
            return self.__s2n_ell_As_Ad_alpha__(iC, Niter, res)
        elif self.fit=="Ad + beta + alpha":
            return self.__s2n_ell_Ad_beta_alpha__(iC, Niter, res)
        elif self.fit=="As + Ad + beta + alpha":
            return self.__s2n_ell_As_Ad_beta_alpha__(iC, Niter, res)
        elif self.fit=="As + Asd + Ad + alpha":
            return self.__s2n_ell_As_Asd_Ad_alpha__(iC, Niter, res)
        elif self.fit=="As + Asd + Ad + beta + alpha":
            return self.__s2n_ell_As_Asd_Ad_beta_alpha__(iC, Niter, res)

    def __s2n_ell_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as alpha_i
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = {}
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
   
    def __s2n_ell_Ad_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as Ad, alpha_i
        sys_mat[:, 0, 0] = linsys.R  # Ad - Ad        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*Ad_ai            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,0]}
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
            
    def __s2n_ell_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as beta, alpha_i
        sys_mat[:, 0, 0] = 4*(linsys.G + linsys.F - 2*linsys.C) # beta - beta
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii,:], axis=0) + np.sum(linsys.epsilon_ij[ii,:,:], axis=0) - np.sum(linsys.varphi_ij[:,ii,:], axis=0) - np.sum(linsys.ene_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 4*b_a
            sys_mat[:, idx_i+res.ext_par, 0] += 4*b_a
            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "beta":res.ml[f"Iter {Niter}"]["beta"]/std_ell[:,3] }
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
            
    def __s2n_ell_As_Ad_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as As, Ad, alpha_i
        sys_mat[:, 0, 0] = linsys.S                              # As - As
        sys_mat[:, 0, 1] = linsys.W; sys_mat[:, 1, 0] = linsys.W # As - Ad

        sys_mat[:, 1, 1] = linsys.R                              # Ad - Ad
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii,:], axis=0) - np.sum(linsys.psi_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 1] += 2*Ad_ai

            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "As":  res.ml[f"Iter {Niter}"]["As"]  /std_ell[:,0],
                "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,1]}
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
                    
    def __s2n_ell_Ad_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as Ad, beta, alpha_i
        sys_mat[:, 0, 0] = linsys.R                              # Ad - Ad
        sys_mat[:, 0, 1] = 2*(linsys.LAMBDA - linsys.mu)         # Ad - beta
        sys_mat[:, 1, 0] = 2*(linsys.LAMBDA - linsys.mu) 
        
        sys_mat[:, 1, 1] = 4*(linsys.G + linsys.F - 2*linsys.C) # beta - beta
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*Ad_ai

            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii,:], axis=0) + np.sum(linsys.epsilon_ij[ii,:,:], axis=0) - np.sum(linsys.varphi_ij[:,ii,:], axis=0) - np.sum(linsys.ene_ij[ii,:,:], axis=0)
            sys_mat[:, 1, idx_i+res.ext_par] += 4*b_a
            sys_mat[:, idx_i+res.ext_par, 1] += 4*b_a
            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,1],
                "beta":res.ml[f"Iter {Niter}"]["beta"]/std_ell[:,3] }
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n

    def __s2n_ell_As_Ad_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as As, Ad, beta, alpha_i
        sys_mat[:, 0, 0] = linsys.S                              # As - As
        sys_mat[:, 0, 1] = linsys.W; sys_mat[:, 1, 0] = linsys.W # As - Ad
        sys_mat[:, 0, 2] = 2*(linsys.X - linsys.Y)               # As - beta
        sys_mat[:, 2, 0] = 2*(linsys.X - linsys.Y)

        sys_mat[:, 1, 1] = linsys.R                              # Ad - Ad
        sys_mat[:, 1, 2] = 2*(linsys.LAMBDA - linsys.mu)         # Ad - beta
        sys_mat[:, 2, 1] = 2*(linsys.LAMBDA - linsys.mu) 
        
        sys_mat[:, 2, 2] = 4*(linsys.G + linsys.F - 2*linsys.C) # beta - beta
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii,:], axis=0) - np.sum(linsys.psi_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 1] += 2*Ad_ai

            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii,:], axis=0) + np.sum(linsys.epsilon_ij[ii,:,:], axis=0) - np.sum(linsys.varphi_ij[:,ii,:], axis=0) - np.sum(linsys.ene_ij[ii,:,:], axis=0)
            sys_mat[:, 2, idx_i+res.ext_par] += 4*b_a
            sys_mat[:, idx_i+res.ext_par, 2] += 4*b_a
            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "As":  res.ml[f"Iter {Niter}"]["As"]  /std_ell[:,0],
                "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,1],
                "beta":res.ml[f"Iter {Niter}"]["beta"]/std_ell[:,2] }
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
      
    def __s2n_ell_As_Asd_Ad_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as As, Ad, Asd, alpha_i
        sys_mat[:, 0, 0] = linsys.S                              # As - As
        sys_mat[:, 0, 1] = linsys.W; sys_mat[:, 1, 0] = linsys.W # As - Ad
        sys_mat[:, 0, 2] = linsys.Q + linsys.V                   # As - Asd
        sys_mat[:, 2, 0] = linsys.Q + linsys.V     

        sys_mat[:, 1, 1] = linsys.R                              # Ad - Ad
        sys_mat[:, 1, 2] = linsys.K + linsys.xi                  # Ad - Asd   
        sys_mat[:, 2, 1] = linsys.K + linsys.xi           
        
        sys_mat[:, 2, 2] = linsys.T + linsys.U + 2*linsys.Z      # Asd - Asd
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii,:], axis=0) - np.sum(linsys.psi_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 1] += 2*Ad_ai

            # Asd - alpha_i
            Asd_ai = np.sum(linsys.pi_ij[:,ii,:], axis=0) + np.sum(linsys.rho_ij[:,ii,:], axis=0) - np.sum(linsys.phi_ij[ii,:,:], axis=0) - np.sum(linsys.OMEGA_ij[ii,:,:], axis=0)
            sys_mat[:, 2, idx_i+res.ext_par] += 2*Asd_ai
            sys_mat[:, idx_i+res.ext_par, 2] += 2*Asd_ai
            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "As":  res.ml[f"Iter {Niter}"]["As"]  /std_ell[:,0],
                "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,1],
                "Asd": res.ml[f"Iter {Niter}"]["Asd"] /std_ell[:,2]}
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
    
    def __s2n_ell_As_Asd_Ad_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as As, Ad, Asd, beta, alpha_i
        sys_mat[:, 0, 0] = linsys.S                              # As - As
        sys_mat[:, 0, 1] = linsys.W; sys_mat[:, 1, 0] = linsys.W # As - Ad
        sys_mat[:, 0, 2] = linsys.Q + linsys.V                   # As - Asd
        sys_mat[:, 2, 0] = linsys.Q + linsys.V     
        sys_mat[:, 0, 3] = 2*(linsys.X - linsys.Y)               # As - beta
        sys_mat[:, 3, 0] = 2*(linsys.X - linsys.Y)

        sys_mat[:, 1, 1] = linsys.R                              # Ad - Ad
        sys_mat[:, 1, 2] = linsys.K + linsys.xi                  # Ad - Asd   
        sys_mat[:, 2, 1] = linsys.K + linsys.xi           
        sys_mat[:, 1, 3] = 2*(linsys.LAMBDA - linsys.mu)         # Ad - beta
        sys_mat[:, 3, 1] = 2*(linsys.LAMBDA - linsys.mu) 
        
        sys_mat[:, 2, 2] = linsys.T + linsys.U + 2*linsys.Z      # Asd - Asd
        sys_mat[:, 2, 3] = 2*(linsys.DELTA - linsys.delta + linsys.eta - linsys.theta) # Asd - beta
        sys_mat[:, 3, 2] = 2*(linsys.DELTA - linsys.delta + linsys.eta - linsys.theta) 
                                                           
        sys_mat[:, 3, 3] = 4*(linsys.G + linsys.F - 2*linsys.C) # beta - beta
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii,:], axis=0) - np.sum(linsys.psi_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 1] += 2*Ad_ai

            # Asd - alpha_i
            Asd_ai = np.sum(linsys.pi_ij[:,ii,:], axis=0) + np.sum(linsys.rho_ij[:,ii,:], axis=0) - np.sum(linsys.phi_ij[ii,:,:], axis=0) - np.sum(linsys.OMEGA_ij[ii,:,:], axis=0)
            sys_mat[:, 2, idx_i+res.ext_par] += 2*Asd_ai
            sys_mat[:, idx_i+res.ext_par, 2] += 2*Asd_ai

            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii,:], axis=0) + np.sum(linsys.epsilon_ij[ii,:,:], axis=0) - np.sum(linsys.varphi_ij[:,ii,:], axis=0) - np.sum(linsys.ene_ij[ii,:,:], axis=0)
            sys_mat[:, 3, idx_i+res.ext_par] += 4*b_a
            sys_mat[:, idx_i+res.ext_par, 3] += 4*b_a
            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "As":  res.ml[f"Iter {Niter}"]["As"]  /std_ell[:,0],
                "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,1],
                "Asd": res.ml[f"Iter {Niter}"]["Asd"] /std_ell[:,2],
                "beta":res.ml[f"Iter {Niter}"]["beta"]/std_ell[:,3] }
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n

    def calculate(self, fit=None, bmin=None, bmax=None, binwidth=None,
                  alpha_per_split=None, rm_same_tube=None):
        
        # reuse the object to calculate a new fit 
        if fit!=None or bmin!=None or bmax!=None or binwidth!=None or alpha_per_split!=None or rm_same_tube!=None: 
            if fit!=None:  self.fit = fit
            if bmin!=None: self.bmin = bmin
            if bmax!=None: self.bmax = bmax
            nlb = binwidth if binwidth!=None else self.mle.nlb
            if alpha_per_split!=None: self.alpha_per_split = alpha_per_split
            if rm_same_tube!=None:    self.rm_same_tube    = rm_same_tube
            self.mle = MLE(self.libdir, self.spec, self.fit, 
                           alpha_per_split=self.alpha_per_split, 
                           rm_same_tube=self.rm_same_tube, 
                           bmax=self.bmax, bmin=self.bmin, binwidth=nlb)
            
        niter = 0
        true_params = Result(self.spec, self.fit, self.idx, self.alpha_per_split, 
                             self.rm_same_tube, self.mle.nlb, self.bmin, self.bmax,
                             beta_ini=np.deg2rad(self.beta),
                             alpha_ini=np.deg2rad(self.alpha))
        # read the input spectra 
        try:
            input_cls = self.spec.get_spectra(self.idx, sync='As' in self.fit)
        except TypeError:
            self.spec.compute(self.idx, sync='As' in self.fit)
            input_cls = self.spec.get_spectra(self.idx, sync='As' in self.fit)
        
        # format cls and calculate elements of covariance matrix 
        self.mle.process_cls(input_cls)
        del input_cls # free memory
        cov    = self.mle.build_cov(niter, true_params)
        invcov = np.linalg.inv(cov/self.spec.fsky)
        # calculate signal-to-noise per multipole
        try:
            s2n = self.s2n_ell(invcov, niter, true_params)
        except StopIteration:
            print('NaN in covariance')
        # effective ell corresponding to each bin
        eff_ell = effective_ell(self.spec.lmax, self.mle.bin_conf)
        if self.mode=="ell":
            eff_ell = eff_ell[self.window//2:-self.window//2+1]
        return eff_ell, s2n