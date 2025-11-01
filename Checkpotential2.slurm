#!/bin/bash

#SBATCH --cpus-per-task=1        # Number of CPU cores
#SBATCH --mem=400M               # Memory limit
# #SBATCH --mail-type=END,FAIL     # Mail events
#SBATCH --mail-user=m.ulloacalzonzin@ufl.edu  # Where to send mail
#SBATCH --time=01:00:00          # Max run time
#SBATCH --output=%j.out    # Output file, %j = job ID
#SBATCH --job-name=InterpolatorCheck # Job name

# Print info for debugging
pwd
hostname
date

# Load necessary modules
module load python/3.12 


# Run your Python script
python InterpolatorCheck.py
