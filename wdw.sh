#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t UNLIMITED
#SBATCH -p pleiades
#SBATCH -o wdw-%j.out
#SBATCH -e wdw-%j.err
#SBATCH --mem=4000

module load node

python getWaits.py
