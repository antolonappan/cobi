#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --account=mp107b
#SBATCH --nodes=2
#SBATCH --ntasks=50
##SBATCH --cpus-per-task=4
#SBATCH -J SOLAT
#SBATCH -o socal.out
#SBATCH -e socal.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alonappan@ucsd.edu

module load python
conda activate cb
cd /global/homes/l/lonappan/workspace/cobi/jobs/iso
export OMP_NUM_THREADS=4

#mpirun -np $SLURM_NTASKS python cal.py
mpirun -np $SLURM_NTASKS python latonly.py