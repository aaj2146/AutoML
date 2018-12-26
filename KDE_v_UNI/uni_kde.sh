#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=dsi # The account name for the job.
#SBATCH --job-name=KDE_UNI_TASK # The job name.
#SBATCH --array 0-41
#SBATCH --cpus-per-task 2
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4gb



module load anaconda


source activate myenv2
#Command to execute Python program
python -W ignore uni_kde.py $SLURM_ARRAY_TASK_ID
 
#End of script
