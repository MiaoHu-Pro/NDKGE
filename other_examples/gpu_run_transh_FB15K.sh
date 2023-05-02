#!/bin/bash
#SBATCH --job-name=run_transh
#SBATCH -p gpu
#SBATCH --time=72:00:00
#SBATCH --output=./result_FB15K/transh_FB15K_mention_e20000_id50_des50_magin_1_alpha_1_out-%j.out
#SBATCH --gres gpu:1
#SBATCH --partition=k2-gpu

#default mem is 7900M
#SBATCH --mem 40000M

module add nvidia-cuda
module add apps/python3

nvidia-smi

export PYTHONPATH=$PYTHONPATH:/users/40305887/gridware/share/python/3.6.4/lib/python3.6/site-packages

echo "the job stary "
python3 train_transh_FB15K.py
echo "the job end "




