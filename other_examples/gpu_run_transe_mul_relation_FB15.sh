#!/bin/bash
#SBATCH --job-name=run_transe
#SBATCH -p gpu
#SBATCH --time=72:00:00
#SBATCH --output=transe_FB15K_transe_no_symbol_no_description_id100_des300_n-to-1_all_change_triple_classification_20000_gama_1_gpu_out-%j.out
#SBATCH --gres gpu:1
#SBATCH --partition=k2-gpu

#default mem is 7900M
#SBATCH --mem 40000M

module add nvidia-cuda
module add apps/python3

nvidia-smi

export PYTHONPATH=$PYTHONPATH:/users/40305887/gridware/share/python/3.6.4/lib/python3.6/site-packages

echo "the job stary "

python3 train_transe_mul_relation_FB15K.py

echo "the job end "




