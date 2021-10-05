#!/bin/bash
#SBATCH --job-name=run_transe
#SBATCH -p gpu
#SBATCH --time=48:00:00
#SBATCH --output=unseen_entity_wn18rr_id_and_des_negs-all_step-1_epochs-20000_margin-2_neg_ent-25_lr-1_id_dim-20_word_dim-300-%j.out
#SBATCH --gres gpu:1
#SBATCH --partition=k2-gpu

#default mem is 7900M
#SBATCH --mem 40000M

module add nvidia-cuda
module add apps/python3

nvidia-smi


export PYTHONPATH=$PYTHONPATH:/users/40305887/gridware/share/python/3.6.4/lib/python3.6/site-packages

echo "the job stary "

# python3 train_ndkge.py --dataset_path ./benchmarks/FB15K237/ --setting des --id_dim 200 --word_dim 300  --nbatches 100  --margin 1 --num_epochs 20000 --learning_rate 1.0 --model_name fb15k237_des_epochs-20000_margin-1_lr-1_id_dim-200_word_dim-300

python3 train_ndkge.py --dataset_path ./benchmarks/WN18RR/ --setting des --id_dim 20 --word_dim 300  --nbatches 100  --margin 2 --num_epochs 20000 --learning_rate 1.0 --model_name wn18rr_men_epochs-20000_margin-2_lr-1_id_dim-20_word_dim-300

echo "the job end "




