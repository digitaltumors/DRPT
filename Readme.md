# Drug Response Pathway-Informed Transformer (DRPT)

## Overview

This repo provides the code for training an interpretable Hierarchical Graph Transformer for modeling drug response prediction and enabling precision oncology. The model input genomics alteration (mutations, copy number deletions, and copy number alterations) along with some prior knowledge hierarhcy of cellular structure and function and propogates the effects of these alterations across a hiearchy  using attention. As a result, the model will optimize embedding representations for each in the hierarchy, ultimatley reflecting the 'state' of a cellular system given the context of genomic alteration profiles. Subsequently, the embeddings states of each system in the hierarchy are used to predict drug response.

This model is adapted from the sister version known as [G2PT](https://www.biorxiv.org/content/10.1101/2024.10.23.619940v2) 

## Environmental Set-Up

Use the environment.yml is provided to establish the conda environment
```
conda env create python==3.6 --name envname --file=environment.yml
```

## Usage

The following are key hyper-parameters used by the model:

1. Propagation option:
   * _--mut2gene_ : determines 
   * _--sys2cell_ : determines whether model will propogate to the root system
   * _--cell2sys_ : determines whether model will propogate back down from the root system
   * _--sys2gene_ : determines whether model will proogate back down to genes (optional)
   * _--drug_embedding_ : determines whether the model will with optimize drug emebedding from random (recommended)
   * _--diff_transformer_ : determines whether the model will use Differential Attention
3. Model parameter:
   * _--hiddens_dims_: embedding and hierarchical transformer dimension size. Recommended is 128
4. Training parameters: 
   * _--epochs_ : the number of epoch to run during the training phase. Recommended is 150-200.
   * _--val_step_: Validation step
   * _--batch_size_ : the size of each batch to process at a time. Recommended is 32.
You may increase this number to speed up the training process within the memory capacity
   * _--z_weight_ : for the continuous phenotype, with high `z_weight` will be more sampled  
   * _--dropout_: dropout option. Default is set 0.2
   * _--lr_ : Learning rate. Default is set 0.001.
   * _--wd_ : Weight decay. Default is set 0.001.
5. GPU option:
   * Single GPU option
     * _--cuda_ : the ID of GPU unit that you want to use for the model training. The default setting
     is to use GPU 0.
   * Multi GPU option (multi-node will be supported)
     * _--multiprocessing-distributed_ : determines whether model will be trained in multi-gpu distributed set-up
     * _--world_size_ : size of world, default is 1
     * _--rank_ : rank, default is 0
     * _--local_rank_ : local rank, default is 0
     * _--dist_url_ : distribute url, `tcp://127.0.0.1:2222`
     * _--dist_backend_ : distribute backend default is `nccl`
6. Model input and output:
   * _--model_: if you have trained model, put the path to the trained model.
   * _--out_: a name of directory where you want to store the trained models.

## Model Training Example (Single GPU)

Review the sample shell file, `DRPT-Integrin.sh`, to view how to run a training instance of the model.