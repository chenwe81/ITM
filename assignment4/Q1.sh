#!/bin/bash

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=36:00:00              # limit of wall clock time 
#SBATCH --nodes=1                    # number of different nodes 
#SBATCH --ntasks=1                   # number of tasks 
#SBATCH --cpus-per-task=1            # number of cores per task
#SBATCH --mem=32GB                   # memory total
#SBATCH --output=DimSim-%j.SLURMout  # capture output
#SBATCH --job-name DimSim            # provide a name for the job 

########## Command Lines to Run ##########

### load Conda module
module load Conda/3

### change directory to where your code is located
cd /mnt/home/chenwe81/ITM/assignment4

### run your python code
python DimensionalitySimulation.py

### write job information to output file
scontrol show job $SLURM_JOB_ID  
js -j $SLURM_JOB_ID