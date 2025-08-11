import numpy as np
import sys
import argparse
sys.path.append('/home/chervias/Cosmic_birefringence/cobi')
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
parser.add_argument('-specs_x_d', action='store_true', help='Run the spectra for s_x_d loop')
parser.add_argument('-mle', action='store_true', help='Run the MLE')
args = parser.parse_args()

# Constants
libdir = '/scratch/chervias/output_cobi/my_test_for_fore_realization/'
nside = 2048
cb_model = "iso"
beta = 0.35
mask = '/home/chervias/Cosmic_birefringence/data/Mask_fullLAT_5.0_deg_gal_with-srcs-galplane80.fits'
nhits = '/home/chervias/Cosmic_birefringence/data/LAT_ivar_galcoords.fits'
nhits_fac = 1.052
fore_realization = True
sync_model = 's5'
dust_model = 'df_baseline'

#setting 1
alpha = 0
alpha_err = 0
bp = False
nm = 'NC'

# Initialize LATsky and Spectra
lat = LATsky(libdir, nside, mask, cb_model, beta, alpha=alpha, alpha_err=alpha_err, bandpass=bp,noise_model=nm, nhits=nhits, nhits_fac=nhits_fac, fore_realization=fore_realization, sync_model=sync_model, dust_model=dust_model)
spec = Spectra(lat, libdir, cache=True, parallel=0)

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
        spec.dust_x_dust(idx=i)
        spec.dust_x_obs(i)
    mpi.barrier()

if args.specsync:
    for i in jobs[mpi.rank::mpi.size]:
        spec.sync_x_obs(i)
    mpi.barrier()

if args.specs_x_d:
    for i in jobs[mpi.rank::mpi.size]:
        spec.sync_x_dust(idx=i)
    mpi.barrier()

if args.mle:
    fit = "Ad + beta + alpha"
    #fit = "As + Asd + Ad + beta + alpha"
    binwidth = 20
    bmin = 100
    bmax = 3500
    mle = MLE(libdir,spec,fit, alpha_per_split=True,rm_same_tube=True,binwidth=binwidth,bmin=bmin,bmax=bmax)
    for i in jobs[mpi.rank::mpi.size]:
        mle.estimate_angles(i,overwrite=True)
    mpi.barrier()
