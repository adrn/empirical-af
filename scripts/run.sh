#!/bin/bash
#SBATCH -J actions
#SBATCH -o actions.o%j
#SBATCH -e actions.e%j
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/empirical-af/scripts

# init_conda

date

mpirun python compute_random_zvz.py

date
