#!/bin/bash
#SBATCH -J run-jason
#SBATCH -o logs/jason.o
#SBATCH -e logs/jason.e
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

source ~/.bash_profile

cd /mnt/home/apricewhelan/projects/torusimaging-paper2/scripts

date
python run-jason-sim.py
date
