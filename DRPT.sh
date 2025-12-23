#!/bin/bash

#SBATCH --account=nrnb-gpu
#SBATCH --partition=nrnb-gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=07-00:00:00

eval "$(conda shell.bash hook)"
conda activate g2pt_env

SET=$(seq 0 4)

for i in $SET
do
python train_drug_response_model.py \
	--onto data/ontology_ctg_av.txt \
	--gene2id data/gene2ind_ctg_av.txt \
	--cell2id data/cell2ind_av.txt \
	--genotypes mutation:data/cell2mutation_ctg_av.txt,cna:data/old_copynumber/cell2cnamplification_ctg_av.txt,cnd:data/old_copynumber/cell2cndeletion_ctg_av.txt \
    --mut2gene \
    --sys2cell \
    --cell2sys \
    --sys2gene \
    --drug_embedding \
    --diff_transformer \
    --with_indices \
    --jobs 16 --cuda 0 --epochs 400 --hidden_dims 128 --lr 0.0001 --wd 0.01 --subtree_order default --dropout 0.2 --l2_lambda 0.01 --batch_size 32 --z_weight 2 --val_step 10 \
	--out models/rsi_model_${i}.pt \
	--train data/train_splits/train_dataset_rsi_60_${i}.txt \
	--val data/train_splits/val_dataset_rsi_20_${i}.txt \
	--test data/train_splits/test_dataset_rsi_20_${i}.txt
done