#!/bin/bash
#
#SBATCH --job-name=malthusian_open_borders
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res_%j.err        # File to which STDERR will be written
#SBATCH --partition=2080ti-long # Partition to submit to
#
#SBATCH --ntasks=24
#SBATCH --mem-per-cpu=4096    # Memory in MB per cpu allocated

# Activate Anaconda work environment for OpenDrift
source /home/${USER}/.bashrc
source activate ai_taxes

# we execute the job and time it
python run.py