import numpy as np
import sys
import argparse
sys.path.append('/global/homes/l/lonappan/workspace/cobi')
from cobi import mpi
from cobi.simulation import LATskyC, SATskyC
from cobi.spectra import SpectraCross


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
spec = SpectraCross(libdir, lat, sat, binwidth=5, galcut=40, aposcale=2)

#start_i = 0
#end_i = 100
#jobs = np.arange(start_i, end_i)
jobs = np.array([63, 74, 75, 77, 83, 85, 91, 96])

for i in jobs[mpi.rank::mpi.size]:
    #lat.SaveObsQUs(i)
    #sat.SaveObsQUs(i)
    spec.__spectra_matrix_core__(i, which='EB')
mpi.barrier()


