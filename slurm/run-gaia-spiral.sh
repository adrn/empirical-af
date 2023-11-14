#!/bin/bash
#SBATCH -J run-spiral
#SBATCH -o logs/spiral.o
#SBATCH -e logs/spiral.e
#SBATCH -N 1
#SBATCH -t 18:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

source ~/.bash_profile

cd /mnt/home/apricewhelan/projects/torusimaging-paper2/scripts

date
python run-gaia-spiral.py
date
