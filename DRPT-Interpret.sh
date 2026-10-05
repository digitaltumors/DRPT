#!/bin/bash

#SBATCH --account=nrnb-gpu
#SBATCH --partition=nrnb-gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=07-00:00:00

eval "$(conda shell.bash hook)"
conda activate g2pt_env

python train_drug_response_model.py \
    --onto data/ontology_ctg_av.txt \
    --gene2id data/gene2ind_ctg_av.txt \
    --cell2id data/cell2ind_av.txt \
    --genotypes mutation:data/cell2mutation_ctg_av.txt,cna:data/cell2cnamplification_ctg_av.txt,cnd:data/cell2cndeletion_ctg_av.txt \
    --mut2gene \
    --sys2cell \
    --cell2sys \
    --sys2gene \
    --gene2drug \
    --drug_embedding \
    --diff_transformer \
    --with_indices \
    --full_train \
    --jobs 16 --cuda 0 --epochs 400 --hidden_dims 128 --lr 0.0001 --wd 0.01 --subtree_order default --dropout 0.2 --l2_lambda 0.01 --batch_size 32 --z_weight 2 --val_step 10 \
    --out rsi_model.pt \
    --train data/full_model/train_dataset_rsi.txt
