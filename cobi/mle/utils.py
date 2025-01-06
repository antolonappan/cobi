
import numpy as np

def effective_ell(lmax, bin_conf):
    (ib_grid, il_grid, w_array, ell_array) = bin_conf
    ell = np.arange(0, lmax+1, 1)
    return np.sum(w_array[ib_grid,il_grid]*ell[ell_array[ib_grid,il_grid]], axis=1)

def bin_from_edges(start, end):
    nls  = np.amax(end)
    lmax = nls-1
    ells, bpws, weights = [], [], []
    for ib, (li, le) in enumerate(zip(start, end)):
        nlb      = int(le - li)
        ells    += list(range(li, le))
        bpws    += [ib] * nlb
        weights += [1./nlb] * nlb
    ells = np.array(ells); bpws = np.array(bpws); weights = np.array(weights)

    nell       = len(ells)
    n_bands    = bpws[-1]+1
    nell_array = np.zeros(n_bands, dtype=int)
    for ii in range(0, nell, 1):
        if ells[ii]<=lmax and bpws[ii]>=0:
            nell_array[bpws[ii]]+=1   
            
    ell_list, w_list = [], []
    for ii in range(0, n_bands, 1):
        ell_list.append(np.zeros(nell_array[ii], dtype=int))
        w_list.append(np.zeros(nell_array[ii], dtype=np.float64))
    
    nell_array *= 0
    for ii in range(0, nell, 1):
        if ells[ii]<=lmax and bpws[ii]>=0: 
            ell_list[bpws[ii]][nell_array[bpws[ii]]] = ells[ii]
            w_list[bpws[ii]][nell_array[bpws[ii]]]   = weights[ii]
            nell_array[bpws[ii]]+=1

    for ii in range(0, n_bands, 1):
        norm=0
        for jj in range(0, nell_array[ii], 1):
            norm += w_list[ii][jj]
        if norm<=0:
            print(f"Weights in band {ii} are wrong\n")
        for jj in range(0, nell_array[ii], 1):
            w_list[ii][jj]/=norm

    return (n_bands, nell_array, ell_list, w_list)

def bin_configuration(info):
    (n_bands, nell_array, ell_list, w_list) = info
    ib_array          = np.arange(0, n_bands,       1, dtype=int)
    il_array          = np.arange(0, nell_array[0], 1, dtype=int)
    (ib_grid,il_grid) = np.meshgrid(ib_array, il_array, indexing='ij')
    return (ib_grid, il_grid, np.array(w_list), np.array(ell_list))

#TODO might change the shape of the output array when optimising linear system terms
def bin_spec_matrix(spec, info):
    (ib_grid, il_grid, w_array, ell_array) = info
    return np.sum(w_array[ib_grid,il_grid]*spec[:,:,ell_array[ib_grid,il_grid]], axis=3)

#TODO might change the shape of the output array when optimising linear system terms
def bin_cov_matrix(cov, info):
    (ib_grid, il_grid, w_array, ell_array) = info
    return np.sum(w_array[ib_grid,il_grid]**2*cov[:,:,:,ell_array[ib_grid,il_grid]], axis=4)

def moving_sum(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] 


class Result:
    
    def __init__(self, spec, fit, sim, 
                 alpha_per_split, rm_same_tube,
                 binwidth, bmin, bmax,
                 beta_ini=0.0, alpha_ini=0.0):

        # information I should record to remember how the result was calculated
        # information about the spectra
        self.specdir  = spec.lat.libdir
        self.temp_bp  = spec.temp_bp
        self.aposcale = spec.aposcale
        self.CO       = spec.CO
        self.PS       = spec.PS
        self.pureB    = spec.pureB
        self.nside    = spec.nside
        self.fsky     = spec.fsky
        self.sim_idx  = sim 
        # information about the fit
        self.nlb             = binwidth
        self.bmin            = bmin
        self.bmax            = bmax
        self.fit             = fit
        self.rm_same_tube    = rm_same_tube
        self.alpha_per_split = alpha_per_split
        # parameters to calculate
        self.ml         = {}
        self.std_fisher = {}
        self.cov_fisher = {"Iter 0":None }
        self.variables  = ''
        if fit=="alpha":
            ext_par = 0
            # add them later once you know how many are there
            self.ml["Iter 0"]         = {}
            self.std_fisher["Iter 0"] = {}
        elif fit=="Ad + alpha":
            ext_par = 1
            self.ml["Iter 0"]         = { 'Ad':1.0  } 
            self.std_fisher["Iter 0"] = { 'Ad':None }
            self.variables           += 'Ad'
        elif fit=="beta + alpha":
            ext_par = 1
            self.ml["Iter 0"]         = { 'beta':beta_ini  }
            self.std_fisher["Iter 0"] = { 'beta':None }
            self.variables           += 'beta'
        elif fit=="As + Ad + alpha":
            ext_par = 2
            self.ml["Iter 0"]         = { 'As':1.0,  'Ad':1.0  }
            self.std_fisher["Iter 0"] = { 'As':None, 'Ad':None }
            self.variables           += 'As, Ad'
        elif fit=="Ad + beta + alpha":
            ext_par = 2
            self.ml["Iter 0"]         = { 'Ad':1.0,  'beta':beta_ini}
            self.std_fisher["Iter 0"] = { 'Ad':None, 'beta':0.0 }
            self.variables           += 'Ad, beta'
        elif fit=="As + Ad + beta + alpha":
            ext_par = 3
            self.ml["Iter 0"]         = { 'As':1.0, 'Ad':1.0, 'beta':beta_ini}
            self.std_fisher["Iter 0"] = { 'As':None,'Ad':None,'beta':None }
            self.variables           += 'As, Ad, beta'
        elif fit=="As + Asd + Ad + alpha":
            ext_par = 3
            self.ml["Iter 0"]         = { 'As':1.0,  'Asd':1.0,  'Ad':1.0 }
            self.std_fisher["Iter 0"] = { 'As':None, 'Asd':None, 'Ad':None}
            self.variables           += 'As, Asd, Ad' 
        elif fit=="As + Asd + Ad + beta + alpha":
            ext_par = 4
            self.ml["Iter 0"]         = { 'As':1.0, 'Asd':1.0,  'Ad':1.0, 'beta':beta_ini}
            self.std_fisher["Iter 0"] = { 'As':None,'Asd':None, 'Ad':None,'beta':None }
            self.variables           += 'As, Asd, Ad, beta'
            
        if alpha_per_split:
            self.Nalpha = spec.Nbands
            for ii, band in enumerate(spec.bands):
                self.ml["Iter 0"][band]         = alpha_ini
                self.std_fisher["Iter 0"][band] = None
                self.variables                 += f'{band}'if (ii==0 and fit=="alpha") else f', {band}'
        else:
            self.Nalpha = spec.Nfreq
            for ii, freq in enumerate(spec.freqs):
                self.ml["Iter 0"][freq]         = alpha_ini
                self.std_fisher["Iter 0"][freq] = None
                self.variables                 += f'{freq}'if (ii==0 and fit=="alpha") else f', {freq}'

        self.ext_par = ext_par
        self.Nvar    = self.Nalpha + self.ext_par