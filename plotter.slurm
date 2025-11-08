#!/bin/bash

# #SBATCH --mem=150G               # Memory limit
#SBATCH --mem-per-cpu=300M
#SBATCH --mail-type=END,FAIL    
##SBATCH --mail-user=m.ulloacalzonzin@ufl.edu 
#SBATCH --time=09:00:00        
#SBATCH --output=%j.out
#SBATCH --cpus-per-task=36
#SBATCH --job-name=plotter 
#SBATCH --qos=rachel.houtz
##SBATCH --qos=rachel.houtz-b # Use burst QoS to borrow resources


# Print info for debugging
pwd
hostname
date

module load python/3.12 
python Plotter.py
