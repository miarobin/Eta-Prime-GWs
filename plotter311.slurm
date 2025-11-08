#!/bin/bash

#SBATCH --mem-per-cpu=300M
#SBATCH --time=01:00:00         
#SBATCH --output=%j.out
#SBATCH --cpus-per-task=36
#SBATCH --job-name=Numba311
#SBATCH --qos=rachel.houtz
##SBATCH --qos=rachel.houtz-b. # Use burst QoS to "borrow" more resources
##SBATCH --mem=150G               # Memory limit
##SBATCH --mail-type=END,FAIL   
##SBATCH --mail-user=m.ulloacalzonzin@ufl.edu 

# Print info for debugging
pwd
hostname
date

module load python/3.11
python Plotter.py