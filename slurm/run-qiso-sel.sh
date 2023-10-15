#!/bin/bash
#SBATCH -J run-qiso-sel
#SBATCH -o logs/qiso-sel.o
#SBATCH -e logs/qiso-sel.e
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

source ~/.bash_profile

cd /mnt/home/apricewhelan/projects/torusimaging-paper2/scripts

date
python run-qiso.py --sel
date
