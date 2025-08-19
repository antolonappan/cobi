#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --account=mp107b
#SBATCH --nodes=2
#SBATCH --ntasks=100
##SBATCH --cpus-per-task=1
#SBATCH -J SOLAT
#SBATCH -o solat.out
#SBATCH -e solat.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alonappan@ucsd.edu

module load python
conda activate cb
cd /global/homes/l/lonappan/workspace/cobi/jobs/iso

#mpirun -np $SLURM_NTASKS python run.py -sim
#mpirun -np $SLURM_NTASKS python run.py -checksim
#mpirun -np $SLURM_NTASKS python run.py -specobs
#mpirun -np $SLURM_NTASKS python run.py -specdust
#mpirun -np $SLURM_NTASKS python run.py -specsync
mpirun -np $SLURM_NTASKS python run.py -mle