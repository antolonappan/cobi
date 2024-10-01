import numpy as np
import sys
sys.path.append('/global/homes/l/lonappan/workspace/solat_cb')
from solat_cb import mpi
from solat_cb.simulation import LATsky
from solat_cb.spectra import Spectra
from solat_cb.mle import MLE

libdir ='/pscratch/sd/l/lonappan/SOLAT'
nside = 1024
cb_method = 'iso'
beta = 0.35
dust = 10
synch = 5
alpha = [-0.1,-0.1,0.2,0.2,.15,.15]
atm_noise = False
nsplits = 2
bandpass = False
fit = "As + Asd + Ad + beta + alpha"
binwidth = 20
bmin = 50
bmax = 1500

lat = LATsky(libdir,nside,cb_method,dust,synch,alpha,beta,atm_noise=atm_noise,nsplits=nsplits,bandpass=bandpass)
spec = Spectra(lat,cache=True,parallel=1)
mle = MLE(libdir,spec,fit, alpha_per_split=False,rm_same_tube=False,binwidth=binwidth,bmin=bmin,bmax=bmax)
jobs = np.arange(150)
for i in jobs[mpi.rank::mpi.size]:
    di = mle.estimate_angles(i)
mpi.barrier()