#!/bin/bash
#SBATCH --job-name=run_transe
#SBATCH -p gpu
#SBATCH --time=72:00:00
#SBATCH --output=./result_hadoop/hadoop_name_epochs-40000_margin-10_lr-1_id_dim-100_word_dim-300-%j.out
#SBATCH --gres gpu:1
#SBATCH --partition=k2-gpu

#default mem is 7900M
#SBATCH --mem 80000M

module add nvidia-cuda
module add apps/python3

nvidia-smi


export PYTHONPATH=$PYTHONPATH:/users/40305887/gridware/share/python/3.6.4/lib/python3.6/site-packages

echo "the job stary "

# python3 train_transe_ndkge.py --dataset_path ./benchmarks/FB15K237/ --setting des --id_dim 200 --word_dim 300  --nbatches 100  --margin 1 --num_epochs 20000 --learning_rate 1.0 --model_name fb15k237_des_epochs-20000_margin-1_lr-1_id_dim-200_word_dim-300

python3 train_transe_ndkge_hadoop.py --dataset_path ./benchmarks/hadoop_data/   --dataset hadoop_data  --setting name --id_dim 100 --word_dim 300  --nbatches 100  --margin 10 --num_epochs 40000 --learning_rate 1.0 --model_name hadoop_name_epochs-40000_margin-10_lr-1_id_dim-100_word_dim-300

echo "the job end "




