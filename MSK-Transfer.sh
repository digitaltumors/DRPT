#!/bin/bash

#SBATCH --account=nrnb-gpu
#SBATCH --partition=nrnb-gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=07-00:00:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate g2pt_env

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

python patients/drpt_patient_transfer_msk.py \
  --drpt-bootstrap patients/drpt_bootstrap_template.py \
  --cell-line-model rsi_model.pt \
  --cell-line-response data/full_model/train_dataset_rsi.txt \
  --cell-gene2ind data/gene2ind_ctg_av.txt \
  --ontology data/ontology_ctg_av.txt \
  --patient-table data/patients/msk_patients_rsi_drugs.txt \
  --patient-mutations data/patients/cell2mutation_msk.txt \
  --patient-amplifications data/patients/cell2cnamplification_msk.txt \
  --patient-deletions data/patients/cell2cndeletion_msk.txt \
  --patient2ind data/patients/patient2ind_msk.txt \
  --embedding-transfer \
  --embedding-mlp \
  --loss-function cox_ph \
  --diff-transformer \
  --risk-score \
  --risk-threshold-method none \
  --batch-size 256 \
  --epochs 400 \
  --learning-rate 0.01 \
  --hidden-layers 128 128 \
  --transform-dropout 0.1 \
  --outer-folds 5 \
  --num-durations 10 \
  --time-bins 100 \
  --jobs 0 \
  --auc-batch-size 32 \
  --output-prefix patients/msk_rsi_drugs_embedding_transfer


