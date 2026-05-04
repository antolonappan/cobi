#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --account=mp107e
#SBATCH --nodes=2
#SBATCH --ntasks=100
##SBATCH --cpus-per-task=4
#SBATCH -J CALSIM
#SBATCH -o calsim.out
#SBATCH -e calsim.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alonappan@ucsd.edu

module load python
conda activate cb
cd /global/homes/l/lonappan/workspace/cobi/jobs/iso
export OMP_NUM_THREADS=4

CONFIG=config/TQU_b0p3_baseline_C1_gc90_d1s1.yaml

mpirun -np $SLURM_NTASKS python calibration.py $CONFIG --savemap
