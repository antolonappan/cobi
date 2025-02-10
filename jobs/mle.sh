#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --account=mp107e
#SBATCH --nodes=4
#SBATCH --ntasks=300
#SBATCH --cpus-per-task=1
#SBATCH -J SOLAT
#SBATCH -o solat.out
#SBATCH -e solat.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alonappan@ucsd.edu

module load python
conda activate cb
cd /global/homes/l/lonappan/workspace/cobi/jobs

mpirun -np $SLURM_NTASKS python run.py