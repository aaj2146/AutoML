#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=dsi # The account name for the job.
#SBATCH --job-name=KDEvUNI # The job name.
#SBATCH -c 1 # The number of cpu cores to use.
#SBATCH --time=24:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=8gb # The memory the job will use per cpu core.



module load anaconda


source activate myenv
#Command to execute Python program
python -W ignore kde_vs_uni.py
 
#End of script
