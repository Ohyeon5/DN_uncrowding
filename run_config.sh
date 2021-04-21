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

module load gcc python cuda
source venvs/torch/bin/activate

srun python /home/choung/DN_uncrowding/main.py --norm_types ['in','bn','gn','ln']