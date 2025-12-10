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
beta = 0.0
nc = 'NC'
aso = True
alpha_lat = 0
alpha_lat_err =  0.05

lat = LATskyC(libdir, nside, cb_model, beta, alpha=alpha_lat, alpha_err=alpha_lat_err, bandpass=True, nsplits=2, noise_model=nc, aso=aso, verbose=False)
spec = SpectraCross(libdir, lat, binwidth=10, galcut=40, aposcale=2)

start_i = 1
end_i = 100
jobs = np.arange(start_i, end_i)

for i in jobs[mpi.rank::mpi.size]:
    lat.SaveObsQUs(i)
    spec.__spectra_matrix_core__(i, which='EB')
mpi.barrier()


