#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --account=mp107d
#SBATCH --nodes=2
#SBATCH --ntasks=100
#SBATCH -J SOLAT
#SBATCH -o so.out
#SBATCH -e so.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alonappan@ucsd.edu

module load python
conda activate cb
cd /global/homes/l/lonappan/workspace/cobi/jobs/aniso

export OMP_NUM_THREADS=8
#mpirun -np $SLURM_NTASKS python cross.py -sim
mpirun -np $SLURM_NTASKS python cross.py -cinv
#mpirun -np $SLURM_NTASKS python cross.py -qe
