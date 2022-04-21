#!/bin/bash
#SBATCH -J finetune_es
#SBATCH -p high
#SBATCH -N 1
#SBATCH --mem=8192
#SBATCH --gres=gpu:1
#SBATCH -o slurm.%N.%J.%u.out # STDOUT
#SBATCH -e slurm.%N.%J.%u.err # STDERR

module --ignore-cache load "Python/3.8.6-GCCcore-10.2.0"
source /homedtic/gcambara/projects/ingenious/env/sb_es/bin/activate

SEED=1
python finetune_noisy_data.py hparams/finetune_noisy_es_with_wav2vec.yaml --seed $SEED --skip_prep True --number_of_epochs 10
