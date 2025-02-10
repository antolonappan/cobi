import numpy as np
import sys
sys.path.append('/global/homes/l/lonappan/workspace/cobi')
from cobi import mpi
from cobi.simulation import LATsky

libdir = '/global/cfs/cdirs/sobs/cosmic_birefringence/v0'
nside = 2048
cb_model = "iso"
beta = 0.35
alpha = 0
lat = LATsky(libdir, nside, cb_model, beta, alpha=alpha, bandpass=False)

start_i = 2
end_i = 300
jobs = np.arange(start_i, end_i)
for i in jobs[mpi.rank::mpi.size]:
    lat.SaveObsQUs(i)
mpi.barrier()