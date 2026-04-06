#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0
EXP_NAME="inj_amazon_long_run"
DATASET="inj_amazon"
## inj_amazon

# model.attr_loss_weight=1.0 \
# model.has_annealing=false \
# model.attr_reg_base=1e-2\
# model.attr_warmup_ratios=0.0 \

# model.struct_loss_weight=1.0 \
# model.bce_weight=0.0 \
# model.ranking_weight=0.0 \
# model.dice_weight=1.0 \
# model.misleading_weight=1e-5 \
# model.pos_loss_weight=1.0 \
# model.neg_loss_weight=10.0 \
# model.kl_weight=1e-5 \

# python main.py \
#        dataset=$DATASET \
#        exp_name=$EXP_NAME \
#        epoch=500 \
#        gpu=5 \
#        optimizer.lr=0.002 \
#        optimizer.weight_decay=1e-4 \
#        model.hid_dim=128 \
#        model.z_dim=128 \
#        model.attr_loss_weight=1 \
#        model.has_annealing=false \
#        model.attr_reg_base=1e-2\
#        model.attr_warmup_ratios=0.0 \
#        model.clustering_weight=1.0 \
#        model.n_clusters=10 \
#        model.struct_loss_weight=5.0 \
#        model.bce_weight=0.0 \
#        model.ranking_weight=0.0 \
#        model.dice_weight=10.0 \
#        model.misleading_weight=1e-5 \
#        model.pos_loss_weight=1.0 \
#        model.neg_loss_weight=5.0 \
#        model.kl_weight=1e-5 \
#        model.neighbor_constrain_weight=0 \
#        model.anomaly_func_type="gate" \
#        model.lambda_attr_recon=0.33 \
#        model.lambda_clustering_recon=0.33 \
#        model.lambda_struct_recon=0.33 \
#        model.lambda_neighbor_recon=0.0 \
#        model.lambda_attr_unc=0.33 \
#        model.lambda_clustering_unc=0.33 \
#        model.lambda_struct_unc=0.33 \
#        model.lambda_recon=0.4 \
#        model.lambda_unc=0.6

python main.py \
       dataset=$DATASET \
       exp_name=$EXP_NAME \
       epoch=750 \
       patience=750 \
       gpu=6 \
       optimizer.lr=0.005 \
       optimizer.weight_decay=5e-4 \
       optimizer.dropout=0 \
       model.backbone="HeteroGCN" \
       model.backbone_norm="layer_norm" \
       model.hid_dim=256 \
       model.z_dim=256 \
       model.num_layers=4 \
       model.attr_loss_weight=1 \
       model.has_annealing=false \
       model.attr_reg_base=1e-2 \
       model.attr_warmup_ratios=0.0 \
       model.clustering_weight=1.0 \
       model.n_clusters=10 \
       model.struct_loss_weight=1.0 \
       model.bce_weight=0.0 \
       model.ranking_weight=1.0 \
       model.dice_weight=0.0 \
       model.misleading_weight=1e-5 \
       model.pos_loss_weight=1.0 \
       model.neg_loss_weight=5.0 \
       model.kl_weight=1e-5 \
       model.neighbor_constrain_weight=1 \
       model.anomaly_func_type="gate" \
       model.lambda_attr_recon=0.165 \
       model.lambda_clustering_recon=0.165 \
       model.lambda_struct_recon=0.33 \
       model.lambda_neighbor_recon=0.33 \
       model.lambda_attr_unc=0.33 \
       model.lambda_clustering_unc=0.33 \
       model.lambda_struct_unc=0.33 \
       model.lambda_recon=0.2 \
       model.lambda_unc=0.8 