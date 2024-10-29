#!/bin/bash

#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name="ilu0_gpu_factorsp"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --nodes=1
#SBATCH --output="ilu0_gpu_factorsp.%j.%N.out"
#SBATCH -t 00:30:00

### here where you load modules.
module load RougeEnv/2021a
module load cmake
module load rocm/4.1.0             # Load the ROCm module (use correct version)
module load rocm-clang/4.1.0
module load gcc/10.3.0


mkdir build
cd build

cmake ..

# Compile the project
make -j


./vector_add > vector_add.out
