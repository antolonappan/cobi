import numpy as np
import sys
import argparse
sys.path.append('/global/homes/l/lonappan/workspace/cobi')
from cobi.simulation import LATsky, Mask
from cobi import mpi
from cobi.quest import FilterEB


parser = argparse.ArgumentParser(description="Run LATsky simulations with MPI support.")
parser.add_argument('-sim', action='store_true', help='Run the simulation loop')
parser.add_argument('-cinv', action='store_true', help='Run the CINV filtering')
parser.add_argument('-qe', action='store_true', help='Run the QE reconstruction')
args = parser.parse_args()

dir = '/global/cfs/cdirs/sobs/cosmic_birefringence/v2'
#alpha = [-0.1,-0.1,0.2,0.2,.15,.15]
alpha = 0
alpha_err = 0.05
sim_config = {'set1': 400, 'reuse_last': 100}
filt_lmax = 2048
recon_lmax = 1024
latsky = LATsky(dir, nside=1024, cb_model='aniso',Acb=1e-6,lensing=True,alpha=alpha,alpha_err=alpha_err,nsplits=4,noise_model='TOD',sim_config=sim_config)

start_i= 0
end_i = 100
split = 1
jobs = np.arange(start_i, end_i)

if args.sim:
    for i in jobs[mpi.rank::mpi.size]:
        null = latsky.HILC_obsEB(i,split=split)
    mpi.barrier()

if args.cinv:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=filt_lmax, sht_backend='d')
    for i in jobs[mpi.rank::mpi.size]:
        null = filt.cinv_EB(i, split=split)
    mpi.barrier()

if args.qe:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=filt_lmax, sht_backend='d')
    qe = QE(filt,200,lmax=filt_lmax, recon_lmax=recon_lmax)
    # Run on stat_index, mf_index, and null_index
    qe_jobs = np.concatenate([qe.stat_index, qe.mf_index, qe.null_index])
    for i in qe_jobs[mpi.rank::mpi.size]:
        null = qe.qlm(i)
    mpi.barrier()
