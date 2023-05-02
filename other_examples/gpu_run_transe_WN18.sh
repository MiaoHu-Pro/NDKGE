#!/bin/bash
#SBATCH --job-name=run_transe
#SBATCH -p gpu
#SBATCH --time=72:00:00
#SBATCH --output=transe_WN18_mention_description_id20_des50_20_gama_2_alpha_1_gpu_print_len(score)_out-%j.out
#SBATCH --gres gpu:1
#SBATCH --partition=k2-gpu

#default mem is 7900M
#SBATCH --mem 40000M

module add nvidia-cuda
module add apps/python3

nvidia-smi

export PYTHONPATH=$PYTHONPATH:/users/40305887/gridware/share/python/3.6.4/lib/python3.6/site-packages

echo "the job stary "
python3 train_transe_WN18.py
echo "the job end "




