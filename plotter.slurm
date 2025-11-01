#!/bin/bash

#SBATCH --cpus-per-task=36       # Number of CPU cores
# #SBATCH --mem=12G               # Memory limit
#SBATCH --mem-per-cpu=400M
#SBATCH --mail-type=END,FAIL     # Mail events
#SBATCH --mail-user=m.ulloacalzonzin@ufl.edu  # Where to send mail
#SBATCH --time=14:00:00          # Max run time
#SBATCH --output=Scan2401%j.out    # Output file, %j = job ID
#SBATCH --job-name=Scan2401 # Job name

# Print info for debugging
pwd
hostname
date

# Load necessary modules
module load python/3.12 


# Run your Python script
#python IBInterpolation.py
python Plotter.py
