#!/bin/bash

#SBATCH --cpus-per-task=9        # Number of CPU cores
#SBATCH --mem=1G              # Memory limit
#SBATCH --mail-type=END,FAIL     # Mail events
#SBATCH --mail-user=m.ulloacalzonzin@ufl.edu  # Where to send mail
#SBATCH --time=15:00:00          # Max run time
#SBATCH --output=%j.out    # Output file, %j = job ID
#SBATCH --job-name=N4F6xi5AMSB
#SBATCH --qos=rachel.houtz
##SBATCH --qos=rachel.houtz-b # Use burst QoS to borrow resources



# Print info for debugging
pwd
hostname
date

# Load necessary modules
module load python/3.12 


# Run your Python script
python Plotter.py

