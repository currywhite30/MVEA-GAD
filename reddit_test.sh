#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0
EXP_NAME="reddit_long_run"
DATASET="reddit"

python main.py \
       dataset=$DATASET \
       exp_name=$EXP_NAME \
       epoch=500 \
       patience=500 \
       gpu=7 \
       optimizer.lr=0.002 \
       optimizer.weight_decay=1e-4 \
       optimizer.dropout=0 \
       model.backbone="HeteroGCN" \
       model.backbone_norm="layer_norm" \
       model.hid_dim=128 \
       model.z_dim=128 \
       model.num_layers=4 \
       model.attr_loss_weight=1 \
       model.has_annealing=false \
       model.attr_reg_base=1 \
       model.attr_warmup_ratios=0.0 \
       model.clustering_weight=1.0 \
       model.n_clusters=10 \
       model.struct_loss_weight=1.0 \
       model.bce_weight=0.0 \
       model.ranking_weight=1.0 \
       model.dice_weight=0.0 \
       model.misleading_weight=1e-3 \
       model.pos_loss_weight=1 \
       model.neg_loss_weight=5 \
       model.kl_weight=1e-3 \
       model.neighbor_constrain_weight=1 \
       model.anomaly_func_type="gate" \
       model.lambda_attr_recon=0.165 \
       model.lambda_clustering_recon=0.165 \
       model.lambda_struct_recon=0.33 \
       model.lambda_neighbor_recon=0.33 \
       model.lambda_attr_unc=0.33 \
       model.lambda_clustering_unc=0.33 \
       model.lambda_struct_unc=0.33 \
       model.lambda_recon=0.4 \
       model.lambda_unc=0.6

## attr:       0.5622
## clustering: 0.5542
## adj:        0.5696
## neighbor:   0.4475      
## attr+adj: 0.5891
## adj+clustering: 0.5782
## attr+adj+clustering: 0.5662 ~ 0.59
## attr+adj+neighbor: 0.6172