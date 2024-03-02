#!/bin/sh

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=0:45:00               # limit of wall clock time 
#SBATCH --nodes=1                    # number of different nodes 
#SBATCH --ntasks=1                   # number of tasks 
#SBATCH --cpus-per-task=1            # number of cores per task
#SBATCH --mem=32GB                   # memory total
#SBATCH --output=ICAjob-%j.SLURMout  # capture output
#SBATCH --job-name ICAdecomposition  # provide a name for the job 

########## Command Lines to Run ##########

### load Conda module
module load Conda/3

### change directory to where your code is located
cd /mnt/home/chenwe81/ITM/assignment2

### run your python code
python Q4.py 

### write job information to output file
scontrol show job $SLURM_JOB_ID    