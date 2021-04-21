#!/bin/bash
#SBATCH --chdir /scratch/izar/choung/DN_uncrowding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 8192
#SBATCH --time 1:00:00
python -u "/home/choung/DN_uncrowding/main.py"