import numpy as np
import sys
import argparse
sys.path.append('/global/homes/l/lonappan/workspace/cobi')
from cobi.simulation import LATsky, Mask
from cobi import mpi
from cobi.quest import FilterEB, QE


parser = argparse.ArgumentParser(description="Run LATsky simulations with MPI support.")
parser.add_argument('-sim', action='store_true', help='Run the simulation loop')
parser.add_argument('-cinv', action='store_true', help='Run the CINV filtering')
parser.add_argument('-qe', action='store_true', help='Run the QE reconstruction')
parser.add_argument('-rdn0', action='store_true', help='Run the RDN0 reconstruction')
parser.add_argument('-idx', type=int, default=0, help='Job index for array jobs')
args = parser.parse_args()

dir = '/global/cfs/cdirs/sobs/cosmic_birefringence/v1'
alpha = [-0.1,-0.1,0.2,0.2,.15,.15]
alpha_err = 0.1
alpha_config = {
    'alpha_vary_index': [0, 400],   
    'alpha_cons_index': [400, 500], 
    'null_alpha_index': [500, 600]  
}
latsky = LATsky(dir, nside=1024, cb_model='aniso',Acb=1e-6,lensing=True,alpha=alpha,alpha_err=alpha_err,nsplits=1,noise_model='NC',Acb_sim_config=alpha_config)

start_i= 0
end_i = 600
jobs = np.arange(start_i, end_i)

if args.sim:
    for i in jobs[mpi.rank::mpi.size]:
        null = latsky.HILC_obsEB(i)
    mpi.barrier()

if args.cinv:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=3000, sht_backend='d')
    for i in jobs[mpi.rank::mpi.size]:
        null = filt.cinv_EB(i)
    mpi.barrier()

if args.qe:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=3000, sht_backend='d')
    qe = QE(filt,200,lmax=1500, recon_lmax=1024)
    # Run on both alpha_vary_index and null_alpha_index
    qe_jobs = np.concatenate([qe.alpha_vary_index, qe.null_alpha_index])
    for i in qe_jobs[mpi.rank::mpi.size]:
        null = qe.qlm(i)
    mpi.barrier()

if args.rdn0:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=3000, sht_backend='d')
    qe = QE(filt,200,2100,2000)
    # Only run RDN0 on n0_index (excludes mean field sims)
    if args.idx in qe.n0_index:
        null = qe.RDN0_mpi(args.idx)
    mpi.barrier()

if args.n0sim:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=3000, sht_backend='d')
    qe = QE(filt,200,2100,2000)
    # Use n0_index for the jobs
    for i in qe.n0_index[mpi.rank::mpi.size]:
        null = qe.N0_sim(i, which=args.which)
    mpi.barrier()