#!/bin/bash
#SBATCH --job-name=run_transe
#SBATCH -p gpu
#SBATCH --time=72:00:00
#SBATCH --output=transe_fb15k_mention+new_des_id100_des300_10000_gama_1_alpha_1_out-%j.out
#SBATCH --gres gpu:1
#SBATCH --partition=k2-gpu

#default mem is 7900M
#SBATCH --mem 40000M

module add nvidia-cuda
module add apps/python3

nvidia-smi

export PYTHONPATH=$PYTHONPATH:/users/40305887/gridware/share/python/3.6.4/lib/python3.6/site-packages

echo "the job start "
python3 train_transe_FB15K.py
echo "the job end "




