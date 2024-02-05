#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu

export PATH=$HOME/anaconda3/bin:$PATH
export CONDA_PREFIX=$HOME/anaconda3/envs/test_gpu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
. activate base
conda activate test_gpu

python3 test2.py
