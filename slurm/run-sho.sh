#!/bin/bash
#SBATCH -J run-sho
#SBATCH -o logs/sho.o
#SBATCH -e logs/sho.e
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

source ~/.bash_profile

cd /mnt/home/apricewhelan/projects/torusimaging-paper2/scripts

date
python run-harmonic-oscillator.py
date
