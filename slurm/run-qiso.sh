#!/bin/bash
#SBATCH -J run-qiso
#SBATCH -o logs/qiso.o
#SBATCH -e logs/qiso.e
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

source ~/.bash_profile

cd /mnt/home/apricewhelan/projects/torusimaging-paper2/scripts

date
python run-qiso.py
date
