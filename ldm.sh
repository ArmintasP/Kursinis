#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu

export PATH=$HOME/anaconda3/bin:$PATH
export CONDA_PREFIX=$HOME/anaconda3/envs/ldm3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
. activate base
conda activate ldm3

# python3 feature-extract.py
python3 diffusion_decoding.py --imgidx 0 2 --gpu 0 --subject subj01 --method cvpr