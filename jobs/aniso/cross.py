import numpy as np
import sys
import argparse
sys.path.append('/global/homes/l/lonappan/workspace/cobi')
from cobi.simulation import LATsky, Mask
from cobi import mpi
from cobi.quest import FilterEB, CrossQE


parser = argparse.ArgumentParser(description="Run LATsky simulations with MPI support.")
parser.add_argument('-sim', action='store_true', help='Run the simulation loop')
parser.add_argument('-cinv', action='store_true', help='Run the CINV filtering')
parser.add_argument('-qe', action='store_true', help='Run the QE reconstruction')
parser.add_argument('-split', type=int, default=1, help='Data split to use (default: 1)')
parser.add_argument('-mean', action='store_true', help='Compute mean field')
args = parser.parse_args()

dir = '/global/cfs/cdirs/sobs/cosmic_birefringence/v2'
alpha = [-0.1,-0.1,0.2,0.2,.15,.15]
#alpha = 0
alpha_err = 0.05
sim_config = {'set1': 400, 'reuse_last': 100}
cross_spectra = {'lens_rot':[0,100],'lens_unrot':[100,200],'unlens_unrot':[200,300]}
filt_lmax = 2048
recon_lmax = 1024
latsky = LATsky(dir, nside=1024, cb_model='aniso',Acb=1e-6,lensing=True,alpha=alpha,alpha_err=alpha_err,nsplits=4,noise_model='TOD',sim_config=sim_config,cross_spectra=cross_spectra)
#latsky = LATsky(dir, nside=1024, cb_model='aniso',Acb=1e-7,lensing=True,alpha=alpha,alpha_err=alpha_err,nsplits=4,aso=True,noise_model='NC',sim_config=sim_config,cross_spectra=cross_spectra)


start_i= 0
end_i = 100
jobs = np.arange(start_i, end_i)

if args.sim:
    for i in jobs[mpi.rank::mpi.size]:
        null = latsky.HILC_obsEB(i,split=args.split)
    mpi.barrier()

if args.cinv:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=filt_lmax, sht_backend='d')
    for i in jobs[mpi.rank::mpi.size]:
        null = filt.cinv_EB(i, split=args.split)
    mpi.barrier()

if args.qe:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=filt_lmax, sht_backend='d')
    qe = CrossQE(filt,200,lmax=filt_lmax, recon_lmax=recon_lmax, cross_spectra=cross_spectra)
    # Run on stat_index, mf_index, and null_index
    for i in jobs[mpi.rank::mpi.size]:
        null = qe.qcl_cross_only(i)
    mpi.barrier()

if args.mean:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=filt_lmax, sht_backend='d')
    qe = CrossQE(filt,200,lmax=filt_lmax, recon_lmax=recon_lmax, cross_spectra=cross_spectra)
    for i in jobs[mpi.rank::mpi.size]:
        null = qe.mean(i)
    mpi.barrier()
