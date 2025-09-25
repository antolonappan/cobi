import numpy as np
import healpy as hp
import os
import pickle as pl
from cobi.simulation import CMB
from cobi import mpi
from cobi.utils import Logger
from typing import Dict, Any, Tuple,Optional, List

RAD2ARCMIN = 180 * 60 / np.pi

def bin_from_edges(start: np.ndarray, end: np.ndarray) -> Tuple:
    lmax = np.amax(end) - 1 if len(end) > 0 else 0
    ells, bpws, weights = [], [], []
    for ib, (li, le) in enumerate(zip(start, end)):
        nlb = int(le - li)
        ells.extend(range(li, le))
        bpws.extend([ib] * nlb)
        weights.extend([1.0 / nlb] * nlb)
    
    ells, bpws, weights = np.array(ells), np.array(bpws), np.array(weights)
    n_bands = bpws[-1] + 1 if len(bpws) > 0 else 0
    nell_array = np.zeros(n_bands, dtype=int)
    valid_indices = (ells <= lmax) & (bpws >= 0)
    
    unique_bpws, counts = np.unique(bpws[valid_indices], return_counts=True)
    if len(unique_bpws) > 0:
        nell_array[unique_bpws] = counts

    ell_list = [np.zeros(n, dtype=int) for n in nell_array]
    w_list = [np.zeros(n, dtype=np.float64) for n in nell_array]
    
    counts_so_far = np.zeros(n_bands, dtype=int)
    for i in range(len(ells)):
        if valid_indices[i]:
            b = bpws[i]
            idx = counts_so_far[b]
            ell_list[b][idx] = ells[i]
            w_list[b][idx] = weights[i]
            counts_so_far[b] += 1

    for i in range(n_bands):
        norm = np.sum(w_list[i])
        if norm > 0:
            w_list[i] /= norm
            
    return n_bands, nell_array, ell_list, w_list

def bin_configuration(info: Tuple) -> Tuple:
    n_bands, nell_array, ell_list, w_list = info
    max_nell = np.max(nell_array) if len(nell_array) > 0 else 0
    ib_grid, il_grid = np.meshgrid(np.arange(n_bands,dtype=int), np.arange(max_nell,dtype=int), indexing='ij')
    
    w_array = np.array([np.pad(l, (0, max_nell - len(l))) for l in w_list])
    ell_array = np.array([np.pad(l, (0, max_nell - len(l)), 'constant', constant_values=-1) for l in ell_list], dtype=int)
    return ib_grid, il_grid, w_array, ell_array

def bin_spec_matrix(spec: np.ndarray, info: Tuple) -> np.ndarray:
    (ib_grid, il_grid, w_array, ell_array) = info
    n_bands = ib_grid.shape[0]
    binned_spec = np.zeros((spec.shape[0], spec.shape[1], n_bands))
    for b in range(n_bands):
        valid_ells_mask = ell_array[b, :] >= 0
        ells_in_bin = ell_array[b, valid_ells_mask]
        weights_in_bin = w_array[b, valid_ells_mask]
        if len(ells_in_bin) > 0:
            binned_spec[:, :, b] = np.sum(weights_in_bin * spec[:, :, ells_in_bin], axis=2)
    return binned_spec

def bin_cov_matrix(cov: np.ndarray, info: Tuple) -> np.ndarray:
    (ib_grid, il_grid, w_array, ell_array) = info
    n_bands = ib_grid.shape[0]
    binned_cov = np.zeros((cov.shape[0], cov.shape[1], cov.shape[2], n_bands))
    for b in range(n_bands):
        valid_ells_mask = ell_array[b, :] >= 0
        ells_in_bin = ell_array[b, valid_ells_mask]
        weights_in_bin = w_array[b, valid_ells_mask]
        if len(ells_in_bin) > 0:
            binned_cov[:, :, :, b] = np.sum(weights_in_bin**2 * cov[:, :, :, ells_in_bin], axis=3)
    return binned_cov


class _Result:
    _PARAM_CONFIG = {
        "alpha": {}, "Ad + alpha": {'Ad': 1.0}, "beta + alpha": {'beta': 0.0},
        "As + Ad + alpha": {'As': 1.0, 'Ad': 1.0}, "Ad + beta + alpha": {'Ad': 1.0, 'beta': 0.0},
        "As + Ad + beta + alpha": {'As': 1.0, 'Ad': 1.0, 'beta': 0.0},
        "As + Asd + Ad + alpha": {'As': 1.0, 'Asd': 1.0, 'Ad': 1.0},
        "As + Asd + Ad + beta + alpha": {'As': 1.0, 'Asd': 1.0, 'Ad': 1.0, 'beta': 0.0}
    }

    def __init__(self, spec, fit, sim_idx, alpha_per_split, rm_same_tube, 
                 binwidth, bmin, bmax, bands, freqs):
        self.specdir, self.nside, self.fsky = spec.lat.libdir, spec.nside, spec.fsky
        self.temp_bp, self.aposcale, self.CO, self.PS, self.pureB = spec.temp_bp, spec.aposcale, spec.CO, spec.PS, spec.pureB
        self.sim_idx = sim_idx
        self.nlb, self.bmin, self.bmax = binwidth, bmin, bmax
        self.fit, self.rm_same_tube, self.alpha_per_split = fit, rm_same_tube, alpha_per_split
        self.ml, self.std_fisher, self.cov_fisher = {}, {}, {"Iter 0": None}
        initial_params = self._PARAM_CONFIG.get(fit, {})
        self.ml["Iter 0"] = initial_params.copy()
        self.std_fisher["Iter 0"] = {key: None for key in initial_params}
        self.variables = list(initial_params.keys())

        if alpha_per_split: 
            self.Nalpha, self.alpha_keys = len(bands), bands
        else: 
            self.Nalpha, self.alpha_keys = len(freqs), freqs

        for key in self.alpha_keys:
            self.ml["Iter 0"][str(key)] = 0.0
            self.std_fisher["Iter 0"][str(key)] = None
        self.variables.extend([str(k) for k in self.alpha_keys])
        self.variables = ", ".join(self.variables)
        self.ext_par = len(initial_params)
        self.Nvar = self.Nalpha + self.ext_par

class MLE:
    FIT_OPTIONS = ["alpha", "Ad + alpha", "As + Ad + alpha", "As + Asd + Ad + alpha",
                   "beta + alpha", "Ad + beta + alpha", "As + Ad + beta + alpha", "As + Asd + Ad + beta + alpha"]
    
    def __init__(self, libdir, spec_lib, fit, alpha_per_split=False, rm_same_tube=False,
                 binwidth=20, bmin=51, bmax=1000, verbose=True, 
                 avoid_spectra: Optional[List[str]] = None):
        if fit not in self.FIT_OPTIONS: raise ValueError(f"Invalid fit option. Choose from: {self.FIT_OPTIONS}")
        self.logger = Logger(self.__class__.__name__, verbose)
        self.niter_max = 100; self.tol = 0.5; self.spec = spec_lib; self.fit = fit
        self.alpha_per_split = alpha_per_split; self.rm_same_tube = rm_same_tube
        self.nside = self.spec.nside; self.fsky = self.spec.fsky
        self.avoid_spectra = [str(s) for s in avoid_spectra] if avoid_spectra else None

        if self.avoid_spectra:
            avoid_set = set(self.avoid_spectra)
            self.logger.log(f"Avoiding frequency channels: {avoid_set}", 'info')
            # The spec_lib.bands are strings like '27-1', freqs are numbers like 27
            self.bands = [b for b in self.spec.bands if b.split('-')[0] not in avoid_set]
            self.freqs = [f for f in self.spec.freqs if str(f) not in avoid_set]
        else:
            self.bands = self.spec.bands
            self.freqs = self.spec.freqs
        
        self.Nbands = len(self.bands)
        self.Nfreq = len(self.freqs)
        if self.Nbands == 0: raise ValueError("All frequency channels have been filtered out.")

        fld_ext = f"{spec_lib.dust_model}{spec_lib.sync_model}" if spec_lib.lat.dust_model != spec_lib.dust_model else ""
        self.libdir = os.path.join(spec_lib.lat.libdir, 'mle' + fld_ext)
        if mpi.rank == 0: os.makedirs(self.libdir, exist_ok=True)
        mpi.barrier()

        self.cmb = CMB(libdir, self.nside, beta=0, model='iso')
        self.cmb_cls = self.cmb.get_lensed_spectra(dl=False, dtype='d')
        assert bmax <= self.spec.lmax, "bmax must be <= lmax"
        self.nlb, self.bmin, self.bmax = binwidth, bmin, bmax
        lower = np.arange(self.bmin, self.bmax - self.nlb, self.nlb)
        upper = np.arange(self.bmin + self.nlb, self.bmax, self.nlb)
        bin_def = bin_from_edges(lower, upper)
        self.bin_conf = bin_configuration(bin_def); self.Nbins = bin_def[0]
        
        self.inst = {b:{"fwhm":spec_lib.lat.config[b]['fwhm'],"opt. tube":spec_lib.lat.config[b]['opt. tube'],"cl idx":i} for i,b in enumerate(self.bands)}
        if alpha_per_split:
            for i, band in enumerate(self.bands): self.inst[band]["alpha idx"] = i
        else:
            counter = 0
            # Use a dictionary to handle non-contiguous frequency numbers
            freq_map = {freq: i for i, freq in enumerate(self.freqs)}
            for freq_val, idx in freq_map.items():
                matching_bands = [b for b in self.bands if b.startswith(str(freq_val))]
                for band_name in matching_bands:
                    self.inst[band_name]["alpha idx"] = idx
        
        self._setup_indexing()

    def calculate(self, idx: int, return_result: bool = False) -> Any:
        res = _Result(self.spec, self.fit, idx, self.alpha_per_split, self.rm_same_tube, 
                      self.nlb, self.bmin, self.bmax, self.bands, self.freqs)
        
        try:
            input_cls = self.spec.get_spectra(idx, sync='As' in self.fit, avoid_bands=self.avoid_spectra)
        except TypeError:
            self.spec.compute(idx, sync='As' in self.fit)
            input_cls = self.spec.get_spectra(idx, sync='As' in self.fit, avoid_bands=self.avoid_spectra)
        
        self._process_cls(input_cls); del input_cls
        converged = False; niter = 0
        while not converged and niter < self.niter_max:
            try:
                cov = self._build_cov(niter, res)
                invcov = np.linalg.pinv(cov / self.fsky)
                self._solve_linear_system(invcov, niter, res)
                
                ang_now = self._get_ml_alphas(niter+1, res, add_beta='beta' in self.fit)
                ang_prev = self._get_ml_alphas(niter, res, add_beta='beta' in self.fit)
                
                diff = np.abs(ang_now - ang_prev) * RAD2ARCMIN
                if np.all(diff < self.tol) or np.sum(diff >= self.tol) <= 1: converged = True
                niter += 1
            except (np.linalg.LinAlgError, StopIteration, KeyError, IndexError) as e:
                self.logger.log(f"Iteration {niter} failed: {e}. Stopping.", 'error'); break
        
        with open(self.result_name(idx), "wb") as f: pl.dump(res, f, protocol=pl.HIGHEST_PROTOCOL)
        if return_result: return res

    def estimate_angles(self, idx: int, overwrite: bool = False, Niter: int = -1, to_degrees: bool = True) -> Dict:
        file = self.result_name(idx)
        if (not os.path.isfile(file)) or overwrite:
            res = self.calculate(idx, return_result=True)
            if res is None: return {}
        else: res = pl.load(open(file, "rb"))
        max_iter = len(res.ml.keys())
        if max_iter <= 1:
             self.logger.log(f"Calculation for index {idx} did not converge or failed.", 'warning'); return {}
        iter_key = f"Iter {max_iter - 1 if Niter == -1 else Niter}"
        final_params = res.ml[iter_key]
        if to_degrees:
            return {k: (np.rad2deg(v) if k not in ["As","Ad","Asd"] else v) for k,v in final_params.items()}
        return final_params

    def _setup_indexing(self):
        self.avoid = 4 if self.rm_same_tube else 1
        IJidx = []
        for i, band_i in enumerate(self.bands):
            for j, band_j in enumerate(self.bands):
                if self.rm_same_tube:
                    if self.inst[band_i]["opt. tube"] != self.inst[band_j]["opt. tube"]: IJidx.append((i, j))
                elif i != j: IJidx.append((i, j))
        self.IJidx = np.array(IJidx, dtype=np.uint8); self.num_pairs = len(self.IJidx)
        m, n = np.meshgrid(range(self.num_pairs), range(self.num_pairs), indexing='ij')
        self.MNi, self.MNj, self.MNp, self.MNq = self.IJidx[m,0], self.IJidx[m,1], self.IJidx[n,0], self.IJidx[n,1]

    def _process_cls(self, incls: Dict):
        lmax = self.spec.lmax
        self.bin_terms, self.cov_terms = {}, {}
        raw = {} 
        cl_shape = (self.Nbands, self.Nbands, lmax + 1)
        
        raw['EEo_ij']=incls['oxo'][:,:,0,:lmax+1]; raw['BBo_ij']=incls['oxo'][:,:,1,:lmax+1]
        raw['EBo_ij']=incls['oxo'][:,:,2,:lmax+1]; raw['EiEj_o']=incls['oxo'][:,:,0,:lmax+1]
        raw['BiBj_o']=incls['oxo'][:,:,1,:lmax+1]; raw['EiBj_o']=incls['oxo'][:,:,2,:lmax+1]
        raw['BiEj_o']=incls['oxo'][:,:,2,:lmax+1].transpose(1,0,2)

        str_freqs = [str(f) for f in self.freqs]

        for ii, band_i in enumerate(self.bands):
            freq_i_str = band_i.split('-')[0]
            freq_i = str_freqs.index(freq_i_str)

            for jj, band_j in enumerate(self.bands):
                freq_j_str = band_j.split('-')[0]
                freq_j = str_freqs.index(freq_j_str)

                if 'beta' in self.fit:
                    if ii==0 and jj==0:
                        raw['EEcmb_ij'] = np.zeros(cl_shape)
                        raw['BBcmb_ij'] = np.zeros(cl_shape)
                    fwhm_i, fwhm_j = self.inst[band_i]['fwhm'], self.inst[band_j]['fwhm']
                    raw['EEcmb_ij'][ii,jj,:] = self._convolve_gaussBeams_pwf("ee", fwhm_i, fwhm_j, lmax)
                    raw['BBcmb_ij'][ii,jj,:] = self._convolve_gaussBeams_pwf("bb", fwhm_i, fwhm_j, lmax)
                
                if 'Ad' in self.fit:
                    if ii==0 and jj==0:
                        for k in ['EBd_ij','EiEj_d','BiBj_d','EiBj_d','BiEj_d','Eid_Ejo','Bid_Bjo','Eid_Bjo','Bid_Ejo']:
                            raw[k]=np.zeros(cl_shape)
                    raw['EBd_ij'][ii,jj,:]=incls['dxd'][freq_i,freq_j,2,:lmax+1]
                    raw['EiEj_d'][ii,jj,:]=incls['dxd'][freq_i,freq_j,0,:lmax+1]
                    raw['BiBj_d'][ii,jj,:]=incls['dxd'][freq_i,freq_j,1,:lmax+1]
                    raw['EiBj_d'][ii,jj,:]=incls['dxd'][freq_i,freq_j,2,:lmax+1]
                    raw['BiEj_d'][ii,jj,:]=incls['dxd'][freq_j,freq_i,2,:lmax+1]
                    raw['Eid_Ejo'][ii,jj,:]=incls['dxo'][freq_i,jj,0,:lmax+1]
                    raw['Bid_Bjo'][ii,jj,:]=incls['dxo'][freq_i,jj,1,:lmax+1]
                    raw['Eid_Bjo'][ii,jj,:]=incls['dxo'][freq_i,jj,2,:lmax+1]
                    raw['Bid_Ejo'][ii,jj,:]=incls['dxo'][freq_i,jj,3,:lmax+1]

        for k, v in raw.items(): self.bin_terms[k + "_b"] = bin_spec_matrix(v, self.bin_conf)
        self.cov_terms["C_oxo"] = self._C_oxo(raw['EiEj_o'],raw['BiBj_o'],raw['EiBj_o'],raw['BiEj_o'])
        if 'beta' in self.fit: self.cov_terms["C_cmb"] = self._C_cmb()
        if 'Ad' in self.fit:
            self.cov_terms["C_dxd"] = self._C_fgxfg(raw['EiEj_d'],raw['BiBj_d'],raw['EiBj_d'],raw['BiEj_d'])
            self.cov_terms["C_dxo"] = self._C_fgxo(raw['Eid_Ejo'],raw['Bid_Bjo'],raw['Eid_Bjo'],raw['Bid_Ejo'])


    def _build_cov(self, niter, res):
        p=res.ml[f"Iter {niter}"]; ai,aj,ap,aq=self._get_alpha_blocks(niter,res)
        with np.errstate(divide='ignore',invalid='ignore'):
            c4ij=np.cos(4*ai)+np.cos(4*aj); c4pq=np.cos(4*ap)+np.cos(4*aq)
            Aij=np.sin(4*aj)/c4ij; Apq=np.sin(4*aq)/c4pq; Bij=np.sin(4*ai)/c4ij; Bpq=np.sin(4*ap)/c4pq
            Dij=2*np.cos(2*ai)*np.cos(2*aj)/c4ij; Dpq=2*np.cos(2*ap)*np.cos(2*aq)/c4pq
            Eij=2*np.sin(2*ai)*np.sin(2*aj)/c4ij; Epq=2*np.sin(2*ap)*np.sin(2*aq)/c4pq
        cov=self.cov_terms['C_oxo'][0]+Apq*Aij*self.cov_terms['C_oxo'][1]+Bpq*Bij*self.cov_terms['C_oxo'][2]
        if "beta" in self.fit:
            with np.errstate(divide='ignore',invalid='ignore'): Cij=np.sin(4*p["beta"])/(2*np.cos(2*ai+2*aj)); Cpq=np.sin(4*p["beta"])/(2*np.cos(2*ap+2*aq))
            cov+=-2*Cij*Cpq*(self.cov_terms['C_cmb'][0]+self.cov_terms['C_cmb'][1])
        if "Ad" in self.fit:
            Ad=p["Ad"]; Td=self.cov_terms['C_dxd']; Tdo=self.cov_terms['C_dxo']
            cov+=Dij*Dpq*Ad**2*Td[0]+Eij*Epq*Ad**2*Td[1]+Dij*Epq*Ad**2*Td[2]+Eij*Dpq*Ad**2*Td[3]
            cov+=-Dij*Ad*Tdo[0]-Dpq*Ad*Tdo[1]-Eij*Ad*Tdo[2]-Epq*Ad*Tdo[3]
        return np.nan_to_num(cov)

    def _solve_linear_system(self, invcov, niter, res):
        terms=self._compute_linear_system_terms(invcov); ext_params=[p for p in ['As','Ad','Asd','beta'] if p in self.fit]
        sys_mat=np.zeros((res.Nvar,res.Nvar)); ind_term=np.zeros(res.Nvar)
        for i,p1 in enumerate(ext_params):
            sys_mat[i,i]=self._SYSTEM_MATRIX_CONFIG[(p1,p1)](terms); ind_term[i]=self._IND_TERM_CONFIG[p1](terms)
            for j,p2 in enumerate(ext_params[i+1:],start=i+1):
                key=(p1,p2) if (p1,p2) in self._SYSTEM_MATRIX_CONFIG else (p2,p1)
                val=self._SYSTEM_MATRIX_CONFIG.get(key,lambda t:0)(terms); sys_mat[i,j]=sys_mat[j,i]=val
            for k in range(self.Nbands):
                idx_a=self.inst[self.bands[k]]['alpha idx']+res.ext_par
                val=self._SYSTEM_MATRIX_CONFIG[(p1,'alpha')](terms,k); sys_mat[i,idx_a]+=val; sys_mat[idx_a,i]+=val
        for i_band in range(self.Nbands):
            idx_i=self.inst[self.bands[i_band]]['alpha idx']+res.ext_par
            ind_term[idx_i]+=self._IND_TERM_CONFIG['alpha'](terms,i_band)
            for j_band in range(self.Nbands):
                idx_j=self.inst[self.bands[j_band]]['alpha idx']+res.ext_par
                sys_mat[idx_i,idx_j]+=self._SYSTEM_MATRIX_CONFIG[('alpha','alpha')](terms,i_band,j_band)
        solution=np.linalg.solve(sys_mat,ind_term); cov_now=np.linalg.pinv(sys_mat); std_now=np.sqrt(np.diagonal(cov_now))
        if np.any(np.isnan(std_now)): raise StopIteration("NaN in std")
        iter_key=f"Iter {niter+1}"; res.ml[iter_key],res.std_fisher[iter_key]={},{}
        res.cov_fisher[iter_key]=cov_now
        for i,p in enumerate(ext_params): res.ml[iter_key][p]=solution[i]; res.std_fisher[iter_key][p]=std_now[i]
        for i,key in enumerate(res.alpha_keys): res.ml[iter_key][str(key)]=solution[i+res.ext_par]; res.std_fisher[iter_key][str(key)]=std_now[i+res.ext_par]

    def _summation(self, v_ij, v_pq, invcov_T):
        v1=v_ij[self.MNi,self.MNj,:]; v2=v_pq[self.MNp,self.MNq,:]
        return np.sum(v1*invcov_T*v2,axis=2).reshape(self.MNi.shape)

    def _compute_linear_system_terms(self, invcov):
        terms={}; invcov_T=np.moveaxis(invcov,0,2)
        recipes={'B_ijpq':('ijpq','BBo_ij_b','BBo_ij_b'),'E_ijpq':('ijpq','EEo_ij_b','EEo_ij_b'),'I_ijpq':('ijpq','BBo_ij_b','EEo_ij_b'),
            'D_ij':('ij','EEo_ij_b','EBo_ij_b'),'H_ij':('ij','BBo_ij_b','EBo_ij_b'),'A':('_','EBo_ij_b','EBo_ij_b'),
            'tau_ij':('ij','EEo_ij_b','EEcmb_ij_b'),'varphi_ij':('ij','EEo_ij_b','BBcmb_ij_b'),'ene_ij':('ij','BBo_ij_b','EEcmb_ij_b'),
            'epsilon_ij':('ij','BBo_ij_b','BBcmb_ij_b'),'C':('_','EEcmb_ij_b','BBcmb_ij_b'),'F':('_','EEcmb_ij_b','EEcmb_ij_b'),
            'G':('_','BBcmb_ij_b','BBcmb_ij_b'),'O':('_','EBo_ij_b','EEcmb_ij_b'),'P':('_','EBo_ij_b','BBcmb_ij_b'),
            'sigma_ij':('ij','EEo_ij_b','EBd_ij_b'),'omega_ij':('ij','BBo_ij_b','EBd_ij_b'),'R':('_','EBd_ij_b','EBd_ij_b'),'N':('_','EBo_ij_b','EBd_ij_b'),
            'LAMBDA':('_','EBd_ij_b','EEcmb_ij_b'),'mu':('_','EBd_ij_b','BBcmb_ij_b')} # etc.
        for name,(shape,s1,s2) in recipes.items():
            if s1 not in self.bin_terms or s2 not in self.bin_terms: continue
            summed=self._summation(self.bin_terms[s1],self.bin_terms[s2],invcov_T)
            if shape=='_': terms[name]=np.atleast_1d(np.sum(summed))
            elif shape=='ijpq':
                r=np.zeros((self.Nbands,self.Nbands,self.Nbands,self.Nbands)); r[self.MNi.ravel(),self.MNj.ravel(),self.MNp.ravel(),self.MNq.ravel()]=summed.ravel(); terms[name]=r
            elif shape=='ij':
                r=np.zeros((self.Nbands,self.Nbands)); np.add.at(r,(self.IJidx[:,0],self.IJidx[:,1]),np.sum(summed,axis=1)); terms[name]=r
        return terms

    _IND_TERM_CONFIG={'Ad':lambda t:t.get('N',[0])[0],'beta':lambda t:2*(t.get('O',[0])[0]-t.get('P',[0])[0]),'alpha':lambda t,i:2*(np.sum(t.get('D_ij',0)[:,i])-np.sum(t.get('H_ij',0)[i,:]))}
    _SYSTEM_MATRIX_CONFIG={
        ('Ad','Ad'):lambda t:t.get('R',[0])[0],('beta','beta'):lambda t:4*(t.get('G',[0])[0]+t.get('F',[0])[0]-2*t.get('C',[0])[0]),
        ('Ad','beta'):lambda t:2*(t.get('LAMBDA',[0])[0]-t.get('mu',[0])[0]),
        ('Ad','alpha'):lambda t,i:2*(np.sum(t.get('sigma_ij',0)[:,i])-np.sum(t.get('omega_ij',0)[i,:])),
        ('beta','alpha'):lambda t,i:4*(np.sum(t.get('tau_ij',0)[:,i])+np.sum(t.get('epsilon_ij',0)[i,:])-np.sum(t.get('varphi_ij',0)[:,i])-np.sum(t.get('ene_ij',0)[i,:])),
        ('alpha','alpha'):lambda t,i,j:2*(np.sum(t.get('E_ijpq',0)[:,j,:,i])+np.sum(t.get('E_ijpq',0)[:,i,:,j])+np.sum(t.get('B_ijpq',0)[j,:,i,:])+np.sum(t.get('B_ijpq',0)[i,:,j,:])-2*(np.sum(t.get('I_ijpq',0)[j,:,:,i])+np.sum(t.get('I_ijpq',0)[i,:,:,j])))}

    def _convolve_gaussBeams_pwf(self, mode, fwhm1, fwhm2, lmax):
        _,pwf=hp.pixwin(self.nside,pol=True,lmax=lmax); b1=hp.gauss_beam(fwhm1/RAD2ARCMIN,lmax=lmax,pol=True); b2=hp.gauss_beam(fwhm2/RAD2ARCMIN,lmax=lmax,pol=True)
        bl=b1[:,1]*b2[:,1] if mode=='ee' else b1[:,2]*b2[:,2]
        return self.cmb_cls[mode][:lmax+1]*bl*pwf**2

    def _C_cmb(self):
        lmax=self.spec.lmax; ell=np.arange(lmax+1); bl_EE=np.zeros((self.num_pairs,self.num_pairs,lmax+1)); bl_BB=np.zeros_like(bl_EE)
        for m in range(self.num_pairs):
            for n in range(self.num_pairs):
                b=[hp.gauss_beam(self.inst[self.bands[idx]]['fwhm']/RAD2ARCMIN,lmax=lmax,pol=True) for idx in [self.MNi[m,n],self.MNj[m,n],self.MNp[m,n],self.MNq[m,n]]]
                bl_EE[m,n,:]=b[0][:,1]*b[1][:,1]*b[2][:,1]*b[3][:,1]; bl_BB[m,n,:]=b[0][:,2]*b[1][:,2]*b[2][:,2]*b[3][:,2]
        _,pwf=hp.pixwin(self.nside,pol=True,lmax=lmax); Tcmb=np.zeros((2,self.num_pairs,self.num_pairs,lmax+1))
        with np.errstate(divide='ignore',invalid='ignore'): Tcmb[0]=pwf**4*bl_EE*self.cmb_cls['ee'][:lmax+1]**2/(2*ell+1); Tcmb[1]=pwf**4*bl_BB*self.cmb_cls['bb'][:lmax+1]**2/(2*ell+1)
        Tcmb[np.isnan(Tcmb)]=0; return np.moveaxis(bin_cov_matrix(Tcmb,self.bin_conf),3,1)
    
    def _C_oxo(self,EiEjo,BiBjo,EiBjo,BiEjo):
        lmax=self.spec.lmax; ell=np.arange(lmax+1); C=np.zeros((3,self.num_pairs,self.num_pairs,lmax+1))
        with np.errstate(divide='ignore',invalid='ignore'):
            C[0]=(EiEjo[self.MNi,self.MNp,:]*BiBjo[self.MNj,self.MNq,:]+EiBjo[self.MNi,self.MNq,:]*BiEjo[self.MNj,self.MNp,:])/(2*ell+1)
            C[1]=(EiEjo[self.MNi,self.MNp,:]*EiEjo[self.MNj,self.MNq,:]+EiEjo[self.MNi,self.MNq,:]*EiEjo[self.MNj,self.MNp,:])/(2*ell+1)
            C[2]=(BiBjo[self.MNi,self.MNp,:]*BiBjo[self.MNj,self.MNq,:]+BiBjo[self.MNi,self.MNq,:]*BiBjo[self.MNj,self.MNp,:])/(2*ell+1)
        C[np.isnan(C)]=0; return np.moveaxis(bin_cov_matrix(C,self.bin_conf),3,1)
    
    def _C_fgxfg(self,EiEj,BiBj,EiBj,BiEj):
        lmax=self.spec.lmax; ell=np.arange(lmax+1); C=np.zeros((4,self.num_pairs,self.num_pairs,lmax+1))
        with np.errstate(divide='ignore',invalid='ignore'):
            C[0]=(EiEj[self.MNi,self.MNp,:]*BiBj[self.MNj,self.MNq,:]+EiBj[self.MNi,self.MNq,:]*BiEj[self.MNj,self.MNp,:])/(2*ell+1)
            C[1]=BiBj[self.MNi,self.MNp,:]*EiEj[self.MNj,self.MNq,:]/(2*ell+1)
            C[2]=EiEj[self.MNi,self.MNq,:]*BiBj[self.MNj,self.MNp,:]/(2*ell+1)
            C[3]=BiBj[self.MNi,self.MNq,:]*EiEj[self.MNj,self.MNp,:]/(2*ell+1)
        C[np.isnan(C)]=0; return np.moveaxis(bin_cov_matrix(C,self.bin_conf),3,1)

    def _C_fgxo(self,Eifg_Ejo,Bifg_Bjo,Eifg_Bjo,Bifg_Ejo):
        lmax=self.spec.lmax; ell=np.arange(lmax+1); C=np.zeros((4,self.num_pairs,self.num_pairs,lmax+1))
        with np.errstate(divide='ignore',invalid='ignore'):
            C[0]=(Eifg_Ejo[self.MNi,self.MNp,:]*Bifg_Bjo[self.MNj,self.MNq,:]+Eifg_Bjo[self.MNi,self.MNq,:]*Bifg_Ejo[self.MNj,self.MNp,:])/(2*ell+1)
            C[1]=(Eifg_Ejo[self.MNp,self.MNi,:]*Bifg_Bjo[self.MNq,self.MNj,:]+Eifg_Bjo[self.MNp,self.MNj,:]*Bifg_Ejo[self.MNq,self.MNi,:])/(2*ell+1)
            C[2]=Bifg_Bjo[self.MNi,self.MNq,:]*Eifg_Ejo[self.MNj,self.MNp,:]/(2*ell+1)
            C[3]=Bifg_Bjo[self.MNp,self.MNj,:]*Eifg_Ejo[self.MNq,self.MNi,:]/(2*ell+1)
        C[np.isnan(C)]=0; return np.moveaxis(bin_cov_matrix(C,self.bin_conf),3,1)

    def _get_alpha_blocks(self, niter, res):
        alphas=np.zeros(self.Nbands); iter_key=f"Iter {niter}"
        for band in self.bands:
            key=str(band) if self.alpha_per_split else str(band[:-2])
            alphas[self.inst[band]['cl idx']]=res.ml[iter_key][key]
        return alphas[self.MNi],alphas[self.MNj],alphas[self.MNp],alphas[self.MNq]

    def _get_ml_alphas(self, niter, res, add_beta=False):
        iter_key=f"Iter {niter}"; alphas=np.zeros(res.Nalpha)
        for i,key in enumerate(res.alpha_keys):
            alphas[i]=res.ml[iter_key][str(key)]
        if add_beta: alphas+=res.ml[iter_key].get('beta',0.0)
        return alphas

    def result_name(self, idx: int) -> str:
        fit_tag=self.fit.replace(' + ','_'); alpha_tag='alphaPerSplit' if self.alpha_per_split else 'alphaPerFreq'
        tube_tag='_rmSameTube' if self.rm_same_tube else ''; spec_flags=f"{'_tempBP' if self.spec.temp_bp else ''}"
        bin_tag=f"Nb{self.nlb}_bmin{self.bmin}_bmax{self.bmax}"; spec_tag=f"aposcale{str(self.spec.aposcale).replace('.','p')}"
        fname=f"ml_params_{fit_tag}_{alpha_tag}{tube_tag}{spec_flags}_{bin_tag}_{spec_tag}_{idx:03d}.pkl"
        return os.path.join(self.libdir,fname)