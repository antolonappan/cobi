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
parser.add_argument('-mf', action='store_true', help='Compute mean field')
parser.add_argument('-n0sim', action='store_true', help='Run the N0 simulation')
parser.add_argument('-rdn0', action='store_true', help='Compute RDN0')
parser.add_argument('-idx', type=int, help='Simulation index to process')
args = parser.parse_args()

dir = '/global/cfs/cdirs/sobs/cosmic_birefringence/v3'
#alpha = [-0.1,-0.1,0.2,0.2,.15,.15]
alpha = 0
alpha_err = 0.05
sim_config = {'set1': 300, 'reuse_last': 100}
filt_lmax = 2048
recon_lmax = 1024
#latsky = LATsky(dir, nside=1024, cb_model='aniso',Acb=1e-6,lensing=True,alpha=alpha,alpha_err=alpha_err,nsplits=4,noise_model='TOD',sim_config=sim_config)
latsky = LATsky(dir, nside=1024, cb_model='aniso',Acb=1e-7,lensing=True,alpha=alpha,alpha_err=alpha_err,nsplits=4,aso=True,noise_model='NC',sim_config=sim_config)


start_i= 0
end_i = 500
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
    qe = CrossQE(filt,200,lmax=filt_lmax, recon_lmax=recon_lmax)
    jobs = np.concatenate([qe.stat_index, qe.null_index])
    for i in jobs[mpi.rank::mpi.size]:
        null = qe.qcl_cross_only(i)
    mpi.barrier()

if args.mf:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=filt_lmax, sht_backend='d')
    qe = CrossQE(filt,200,lmax=filt_lmax, recon_lmax=recon_lmax)
    for i in qe.mf_index[mpi.rank::mpi.size]:
        null = qe.mean_field_sim(i)
    mpi.barrier()

if args.n0sim:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=filt_lmax, sht_backend='d')
    qe = CrossQE(filt,200,lmax=filt_lmax, recon_lmax=recon_lmax)
    stat_index = qe.stat_index
    stat_list = ['stat'] * len(stat_index)
    vary_index = qe.vary_index
    vary_list = ['vary'] * len(vary_index)
    cons_index = qe.const_index
    cons_list = ['const'] * len(cons_index)
    jobs_index = np.concatenate([stat_index, vary_index, cons_index])
    jobs_list = np.concatenate([stat_list, vary_list, cons_list])
    jobs = np.arange(len(jobs_index))
    for i in jobs[mpi.rank::mpi.size]:
        null = qe.N0_sim(jobs_index[i], which=jobs_list[i])
    mpi.barrier()


if args.rdn0:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=filt_lmax, sht_backend='d')
    qe = CrossQE(filt,200,lmax=filt_lmax, recon_lmax=recon_lmax)
    null = qe.RDN0_mpi(args.idx)
    mpi.barrier()