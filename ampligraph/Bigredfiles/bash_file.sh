#!/bin/bash
#SBATCH -J transE
#SBATCH -p gpu
#SBATCH --account c00772
#SBATCH -o transE_%j.txt
#SBATCH -e transE_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ardend@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00

#Load any modules that your program needs
module load python/gpu/3.10.10
pip3 install --user optuna
pip3 install --user optuna_integration


cd /N/slate/ardend/independent_study/pykeen/TransE

#Run your program
python transE.py