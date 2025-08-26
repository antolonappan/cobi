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
args = parser.parse_args()

dir = '/global/cfs/cdirs/sobs/cosmic_birefringence/v1'
alpha = [-0.1,-0.1,0.2,0.2,.15,.15]
#alpha = [0,0,0,0,0,0]
alpha_err = 0.1
latsky = LATsky(dir, nside=1024, cb_model='aniso',Acb=4.0e-6, AEcb=-1.0e-3,lensing=False,alpha=alpha,alpha_err=alpha_err,nsplits=1)

start_i= 0
end_i = 300
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
    qe = QE(filt,100,3000,2048)
    for i in jobs[mpi.rank::mpi.size]:
        null = qe.qlm(i)
    mpi.barrier()

if args.rdn0:
    mask = Mask(latsky.basedir, latsky.nside,'LATxGAL', 2, gal_cut=0.8)
    filt = FilterEB(latsky, mask, lmax=3000, sht_backend='d')
    qe = QE(filt,100,3000,2048)
    for i in range(2):
        null = qe.RDN0_mpi(i)
    mpi.barrier()
