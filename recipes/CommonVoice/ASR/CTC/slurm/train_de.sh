#!/bin/bash
#SBATCH -J train_de
#SBATCH -p high
#SBATCH -N 1
#SBATCH --mem=8192
#SBATCH --gres=gpu:1
#SBATCH -o slurm.%N.%J.%u.out # STDOUT
#SBATCH -e slurm.%N.%J.%u.err # STDERR

module --ignore-cache load "Python/3.8.6-GCCcore-10.2.0"
source /homedtic/gcambara/projects/ingenious/env/sb_es/bin/activate

SEED=1
python train_with_wav2vec.py hparams/train_de_with_wav2vec.yaml --seed $SEED
