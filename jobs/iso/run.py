import numpy as np
import sys
import argparse
sys.path.append('/global/homes/l/lonappan/workspace/cobi')
from cobi import mpi
from cobi.simulation import LATsky
from cobi.spectra import Spectra
from cobi.mle import MLE

# Argument parser
parser = argparse.ArgumentParser(description="Run LATsky simulations with MPI support.")
parser.add_argument('-sim', action='store_true', help='Run the simulation loop')
parser.add_argument('-checksim', action='store_true', help='Check the simulation loop')
parser.add_argument('-specobs', action='store_true', help='Run the spectra and observation loop')
parser.add_argument('-specdust', action='store_true', help='Run the spectra and dust loop')
parser.add_argument('-specsync', action='store_true', help='Run the spectra and sync loop')
parser.add_argument('-mle', action='store_true', help='Run the MLE')
args = parser.parse_args()

# Constants
libdir = '/global/cfs/cdirs/sobs/cosmic_birefringence/v0'
nside = 2048
cb_model = "iso"
beta = 0.35

#setting 1
# alpha = 0
# alpha_err = 0
# bp = False
# nm = 'NC'

#setting 2
# alpha = 0
# alpha_err = 0.1
# bp = False
# nm = 'NC'

#setting 3
# alpha = [-0.1,-0.1,0.2,0.2,.15,.15]
# alpha_err = 0.1
# bp = False
# nm = 'NC'

#setting 4
alpha = [-0.1,-0.1,0.2,0.2,.15,.15]
#alpha = 0
alpha_err = 0.1
bp = True
nm = 'NC'


# Initialize LATsky and Spectra
lat = LATsky(libdir, nside, cb_model, beta, alpha=alpha, alpha_err=alpha_err, bandpass=bp,noise_model=nm)
spec = Spectra(lat, libdir, cache=True, parallel=0,lmax=3000)

start_i = 0
end_i = 100
jobs = np.arange(start_i, end_i)

if args.sim:
    for i in jobs[mpi.rank::mpi.size]:
        lat.SaveObsQUs(i)
    mpi.barrier()

if args.checksim:
    for i in jobs[mpi.rank::mpi.size]:
        lat.checkObsQU(i)
    mpi.barrier()

if args.specobs:
    for i in jobs[mpi.rank::mpi.size]:
        spec.obs_x_obs(i)
    mpi.barrier()

if args.specdust:
    for i in jobs[mpi.rank::mpi.size]:
        spec.dust_x_obs(i)
    mpi.barrier()

if args.specsync:
    for i in jobs[mpi.rank::mpi.size]:
        spec.sync_x_obs(i)
    mpi.barrier()

if args.mle:
    fit = "Ad + beta + alpha"
    binwidth = 10
    bmin = 50
    bmax = 3000
    mle = MLE(libdir,spec,fit, alpha_per_split=False,rm_same_tube=True,binwidth=binwidth,bmin=bmin,bmax=bmax)
    for i in jobs[mpi.rank::mpi.size]:
        mle.estimate_angles(i)
    mpi.barrier()