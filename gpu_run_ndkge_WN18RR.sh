#!/bin/bash
#SBATCH --job-name=run_transe
#SBATCH -p gpu
#SBATCH --time=48:00:00
#SBATCH --output=./result_WN18RR/transe_WN18RR_name_e20000_id20_des50_gama_2_alpha_1_out-%j.out
#SBATCH --gres gpu:1
#SBATCH --partition=k2-gpu

#default mem is 7900M
#SBATCH --mem 40000M

module add nvidia-cuda
module add apps/python3

nvidia-smi


export PYTHONPATH=$PYTHONPATH:/users/40305887/gridware/share/python/3.6.4/lib/python3.6/site-packages

echo "the job stary "

# python3 train_transe_ndkge.py --dataset_path ./benchmarks/FB15K237/ --setting name --id_dim 200 --word_dim 300  --nbatches 100  --margin 1 --num_epochs 20000 --learning_rate 1.0 --model_name fb15k237_name_epochs-20000_margin-1_lr-1_id_dim-200_word_dim-300

python3 train_transe_ndkge.py --dataset_path ./benchmarks/WN18RR/  --dataset WN18RR --setting name --id_dim 20 --word_dim 50  --nbatches 100  --margin 2 --num_epochs 20000 --learning_rate 1.0 --model_name wn18rr_name_epochs-20000_margin-2_lr-1_id_dim-20_word_dim-50

echo "the job end "




