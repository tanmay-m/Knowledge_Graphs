#!/bin/bash

#SBATCH -J KGE
#SBATCH -p gpu
#SBATCH -A c00772
#SBATCH -o kge_out.txt
#SBATCH -e kge_err.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amerwade@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=45:00:00

#Load any modules that your program needs
module load python/gpu


#Run your program
python ./kgeCode.py
