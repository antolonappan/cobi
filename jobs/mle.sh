#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --job-name MLE   ##name that will show up in the queue
#SBATCH --output MLE   ##filename of the output; the %j will append the jobID to the end of the name making the output files unique despite the sane job name; default is slurm-[jobID].out
#SBATCH --nodes 2  ##number of nodes to use
#SBATCH --ntasks 8  ##number of tasks (analyses) to run
#SBATCH --time 3-00:00:00  ##time for analysis (day-hour:min:sec)
#SBATCH --cpus-per-task 16  ##the number of threads the code will use
#SBATCH --partition batch  ##the partition to run in [options: batch, debug]
#SBATCH --mail-user aconcagua.chc@gmail.com  ##your email address
#SBATCH --mail-type ALL  ##slurm will email you when your job starts


export OMP_NUM_THREADS=16

cd /home/chervias/Cosmic_birefringence/cobi/jobs

srun python -u run.py -sim -checksim -specobs -specdust -mle
