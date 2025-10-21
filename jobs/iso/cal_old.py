import numpy as np
import sys
import argparse
sys.path.append('/global/homes/l/lonappan/workspace/cobi')
from cobi import mpi
from cobi.simulation import LATskyC, SATskyC
from cobi.spectra import Spectra


libdir = '/global/cfs/cdirs/sobs/cosmic_birefringence/v1'
nside = 2048
cb_model = "iso"
beta = 0.35
alpha_lat = [0.2,0.2]
alpha_lat_err = 0.2
alpha_sat_err = 0.1
nc = 'NC'

lat = LATskyC(libdir, nside, cb_model, beta, alpha=alpha_lat,alpha_err=alpha_lat_err, bandpass=True,verbose=True,nsplits=2,noise_model=nc)
sat = SATskyC(libdir, nside, cb_model, beta, alpha_err=alpha_sat_err, bandpass=False,verbose=True,nsplits=2,noise_model=nc)
latspectra = Spectra(lat,libdir,parallel=0,galcut=40,binwidth=5)
satspectra = Spectra(sat,libdir,parallel=0,galcut=40,CO=False,PS=False,binwidth=5)

start_i = 0
end_i = 100
jobs = np.arange(start_i, end_i)

for i in jobs[mpi.rank::mpi.size]:
    lat.SaveObsQUs(i)
    latspectra.obs_x_obs(i)
    sat.SaveObsQUs(i)
    satspectra.obs_x_obs(i)
mpi.barrier()


