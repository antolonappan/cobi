import numpy as np
import sys
import argparse
sys.path.append('/global/homes/l/lonappan/workspace/cobi')
from cobi.simulation import LATsky
from cobi import mpi


parser = argparse.ArgumentParser(description="Run LATsky simulations with MPI support.")
parser.add_argument('-sim', action='store_true', help='Run the simulation loop')
parser.add_argument('-cinv', action='store_true', help='Run the CINV filtering')
args = parser.parse_args()

dir = '/global/cfs/cdirs/sobs/cosmic_birefringence/v1'
alpha = [-0.1,-0.1,0.2,0.2,.15,.15]
alpha_err = 0.1
latsky = LATsky(dir, nside=1024, cb_model='aniso',Acb=4.0e-6, AEcb=-1.0e-3,lensing=False,alpha=alpha,alpha_err=alpha_err,nsplits=1)

start_i = 0
end_i = 100
jobs = np.arange(start_i, end_i)

if args.sim:
    for i in jobs[mpi.rank::mpi.size]:
        null = latsky.HILC_obsEB(i)
    mpi.barrier()

if args.cinv:
