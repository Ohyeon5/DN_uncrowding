#!/bin/bash
#SBATCH --chdir /scratch/izar/choung/DN_uncrowding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 4096
#SBATCH --account lpsy
#SBATCH --time 3:00:00

module load gcc python cuda
source venvs/torch/bin/activate

git pull https://Ohyeon5:glt55_TT@github.com/Ohyeon5/DN_uncrowding.git

python /home/choung/DN_uncrowding/main.py -i -g -t --norm_types gn bn ln