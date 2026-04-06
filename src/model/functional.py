# -*- coding: utf-8 -*-
"""Functional Interface for PyGOD"""
# Author: Kay Liu <zliu234@uic.edu>, Yingtong Dou <ytongdou@gmail.com>
# License: BSD 2 clause

import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
import math
import logging
import numpy as np
from scipy.special import erf

logger = logging.getLogger()

def neighborhood_distribution_loss(s, s_, x, x_, eps=1e-5):
    s_float = s.float()
    edge_index = s_float.nonzero(as_tuple=False).t()
    edge_weight = s_float[edge_index[0], edge_index[1]]
    d_gt = torch.zeros(s.size(0), 1, device=s.device)
    d_gt.index_add_(0, edge_index[0], edge_weight.unsqueeze(1))
    d_gt = d_gt.clamp(min=eps) # 防止除零
    norm_weight = edge_weight / d_gt[edge_index[0]].squeeze()
    A_gt_sparse = torch.sparse_coo_tensor(edge_index, norm_weight, s.size())
    A_gt_sparse = A_gt_sparse.coalesce()
    mu_gt = torch.sparse.mm(A_gt_sparse, x)
    m2_gt = torch.sparse.mm(A_gt_sparse, x.pow(2))
    var_gt = F.relu(m2_gt - mu_gt ** 2) + eps
    s_pred = torch.sigmoid(s_)
    d_pred = s_pred.sum(dim=1, keepdim=True).clamp(min=eps)
    mu_pred = torch.mm(s_pred, x_) / d_pred
    m2_pred = torch.mm(s_pred, x_.pow(2)) / d_pred
    var_pred = F.relu(m2_pred - mu_pred ** 2) + eps
    # --- KL(GT || Pred) ---
    term_log_1 = torch.log(var_pred) - torch.log(var_gt)
    term_trace_1 = var_gt / var_pred
    term_mean_1 = (mu_gt - mu_pred).pow(2) / var_pred
    kl_1 = 0.5 * (term_log_1 + term_trace_1 + term_mean_1 - 1)
    loss_elementwise = 0.5 * (kl_1)
    loss = loss_elementwise.mean(dim=1)
    return loss

def kl_divergence_dirichlet(alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)
    beta = torch.ones_like(alpha)
    S_beta = torch.sum(beta, dim=1, keepdim=True)  # = K
    lnB = torch.lgamma(S) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S)
    dg1 = torch.digamma(alpha)
    kl = lnB + lnB_uni + torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True)
    kl = torch.sum(kl, dim=1)
    return kl

@torch.jit.script
def nig_loss_jit(gamma, nu, alpha, beta, target):
    epsilon = 1e-6
    two_beta_lambda = 2 * beta * (1 + nu)
    pi_val = np.pi
    term1 = 0.5 * torch.log(pi_val / nu + epsilon)
    term2 = alpha * torch.log(two_beta_lambda + epsilon)
    term3 = (alpha + 0.5) * torch.log(nu * (target - gamma) ** 2 + two_beta_lambda + epsilon)
    term4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    nll = term1 - term2 + term3 + term4
    error = torch.abs(target - gamma)
    return nll, error

def attr_loss_cal(total_epochs, current_epoch, gamma, nu, alpha, beta, target, alpha_cluster, q_target,\
             has_annealing=False, attr_reg_base=1e-2, attr_warmup_ratios=0.7, \
             clustering_weight=0.0):
    ## nll+reg          attr_warmup_ratios=0, attr_reg_base=1
    ## nll+退火reg      attr_warmup_ratios=0.7, attr_reg_base=1e-2
    nll, error = nig_loss_jit(gamma, nu, alpha, beta, target)
    loss_nll = torch.mean(nll, dim=1)
    loss_reg = torch.mean(error * (2 * nu + alpha), dim=1)
    if not has_annealing:
        attr_loss = loss_nll + loss_reg * attr_reg_base
    else:
        warmup_steps = int(total_epochs * attr_warmup_ratios) 
        if current_epoch < warmup_steps:
            annealing_factor = current_epoch / warmup_steps
        else:
            annealing_factor = 1.0
        dynamic_weight = attr_reg_base * annealing_factor
        attr_loss = loss_nll + dynamic_weight * loss_reg
    clustering_loss = None
    attr_total_loss = attr_loss
    if clustering_weight > 0:
        S = torch.sum(alpha_cluster, dim=1, keepdim=True)  # [N, 1]
        p = alpha_cluster / S  # [N, 10]
        clustering_loss = -torch.sum(q_target * torch.log(p + 1e-8), dim=1)  # [N]
        attr_total_loss = attr_loss + clustering_loss * clustering_weight
    return attr_total_loss, attr_loss, clustering_loss

def struct_loss_cal(alpha, beta, target_adj, 
                bce_weight=0.0,
                ranking_weight=1.0,
                dice_weight=1.0,
                misleading_weight=1.0,
                kl_weight=1e-5,
                ranking_margin=1.0,
                pos_loss_weight=0.1,
                neg_loss_weight=10.0):
    eps = 1e-8
    S = alpha + beta + eps
    p_expectation = alpha / S

    pos_mask = (target_adj == 1).float()
    neg_mask = (target_adj == 0).float()
    # 统计数量，防止除以 0
    num_pos = pos_mask.sum().clamp(min=1.0)
    num_neg = neg_mask.sum().clamp(min=1.0)

    # num_pos = target_adj.sum()
    num_total = target_adj.numel()
    pos_weight = (num_total - num_pos) / (num_pos + 1e-5)
    weights = torch.ones_like(target_adj)
    weights[target_adj == 1] = pos_weight
    struct_loss = torch.tensor(0.0, device=alpha.device)
    ## BCE 损失 
    if bce_weight > 0:
        digamma_alpha = torch.digamma(alpha + eps)
        digamma_beta = torch.digamma(beta + eps)
        digamma_S = torch.digamma(S)
        loss_ece = -(target_adj * (digamma_alpha - digamma_S) + 
                    (1 - target_adj) * (digamma_beta - digamma_S))
        struct_loss = struct_loss + torch.mean(loss_ece * weights, dim=1) * bce_weight
    if ranking_weight > 0:
        # logits = torch.log(alpha + eps) - torch.log(beta + eps)
        # pos_mask = target_adj.bool()
        # pos_logits_safe = logits.masked_fill(~pos_mask, 1e9)
        # min_pos_val, _ = torch.min(pos_logits_safe, dim=1) 
        # neg_logits_safe = logits.masked_fill(pos_mask, -1e9)
        # max_neg_val, _ = torch.max(neg_logits_safe, dim=1) 
        # has_pos_edge = (target_adj.sum(dim=1) > 0).float()
        # ranking_diff = max_neg_val - min_pos_val + ranking_margin
        # struct_loss = struct_loss + F.relu(ranking_diff) * has_pos_edge.float() * ranking_weight 
        
        logits = torch.log(alpha + eps) - torch.log(beta + eps)
        tau = 1.0
        scaled_logits = logits / tau
        log_sum_exp_all = torch.logsumexp(scaled_logits, dim=1)
        pos_mask = (target_adj == 1).float()
        pos_sum = (scaled_logits * pos_mask).sum(dim=1)
        num_pos_neighbors = pos_mask.sum(dim=1).clamp(min=1.0)
        infonce_loss_per_node = log_sum_exp_all - (pos_sum / num_pos_neighbors)
        has_edge_mask = (target_adj.sum(dim=1) > 0).float()
        struct_loss = struct_loss + (infonce_loss_per_node * has_edge_mask) * ranking_weight

    if dice_weight > 0:
        intersection = (p_expectation * target_adj).sum(dim=1)
        union = p_expectation.sum(dim=1) + target_adj.sum(dim=1)
        dice_score = (2. * intersection + 1e-5) / (union + 1e-5)
        struct_loss = struct_loss + (1.0 - dice_score) * dice_weight
    
    # kl loss
    # kl = torch.lgamma(S) - torch.lgamma(alpha) - torch.lgamma(beta) + \
    #      torch.lgamma(torch.ones_like(alpha)) + torch.lgamma(torch.ones_like(beta)) - torch.lgamma(torch.full_like(S, 2.0)) + \
    #      (alpha - 1) * (torch.digamma(alpha) - torch.digamma(S)) + \
    #      (beta - 1) * (torch.digamma(beta) - torch.digamma(S))
    # reconstruction_error = torch.abs(target_adj - p_expectation)
    # loss_kl = torch.mean(reconstruction_error * kl * weights, dim=1) * kl_weight
    num_pos_per_node = pos_mask.sum(dim=1) 
    num_neg_per_node = neg_mask.sum(dim=1)
    num_pos_safe = num_pos_per_node.clamp(min=1e-5)
    num_neg_safe = num_neg_per_node.clamp(min=1e-5)
    digamma_S = torch.digamma(S)
    digamma_alpha = torch.digamma(alpha)
    digamma_beta = torch.digamma(beta)
    kl_matrix = torch.lgamma(S) - torch.lgamma(alpha) - torch.lgamma(beta) + \
         torch.lgamma(torch.ones_like(alpha)) + torch.lgamma(torch.ones_like(beta)) - torch.lgamma(torch.full_like(S, 2.0)) + \
         (alpha - 1) * (digamma_alpha - digamma_S) + \
         (beta - 1) * (digamma_beta - digamma_S)
    reconstruction_error = torch.abs(target_adj - p_expectation)
    weighted_kl_matrix = reconstruction_error * kl_matrix
    loss_kl = torch.tensor(0.0, device=alpha.device)
    if kl_weight > 0:
        kl_sum_pos = (weighted_kl_matrix * pos_mask).sum(dim=1)
        kl_sum_neg = (weighted_kl_matrix * neg_mask).sum(dim=1)
        kl_mean_pos = kl_sum_pos / num_pos_safe
        kl_mean_neg = kl_sum_neg / num_neg_safe
        loss_kl_per_node = (kl_mean_pos * pos_loss_weight + kl_mean_neg * neg_loss_weight) / 2.0
        loss_kl = loss_kl + loss_kl_per_node * kl_weight

    # misleading loss
    if misleading_weight > 0:
        evidence_alpha = alpha - 1.0
        evidence_beta  = beta - 1.0
        # num_pos = target_adj.sum()
        # num_total = target_adj.numel()
        # pos_weight = (num_total - num_pos) / (num_pos + 1e-5) 
        # loss_misleading = pos_weight * target_adj * evidence_beta + (1 - target_adj) * evidence_alpha
        # loss_kl = loss_kl + torch.mean(loss_misleading, dim=1) * misleading_weight
        mis_sum_pos = (evidence_beta * pos_mask).sum(dim=1)
        mis_sum_neg = (evidence_alpha * neg_mask).sum(dim=1)
        mis_mean_pos = mis_sum_pos / num_pos_safe
        mis_mean_neg = mis_sum_neg / num_neg_safe
        loss_misleading_per_node = (mis_mean_pos * pos_loss_weight + mis_mean_neg * neg_loss_weight) / 2.0
        loss_kl = loss_kl + loss_misleading_per_node * misleading_weight

    # total loss    
    struct_total_loss = loss_kl + struct_loss

    # expected_mse_matrix = p_expectation  
    # mask_neighbor = (target_adj == 1).float()
    # degree = mask_neighbor.sum(dim=1).clamp(min=1e-6)
    # neighbor_score_sum = (expected_mse_matrix * mask_neighbor).sum(dim=1)
    # neighbor_score_mean = neighbor_score_sum / degree

    discriminative_score_matrix = torch.log(alpha + eps) - torch.log(beta + eps)
    mask_neighbor = (target_adj == 1).float()
    degree = mask_neighbor.sum(dim=1).clamp(min=1e-6)
    neighbor_score_sum = (discriminative_score_matrix * mask_neighbor).sum(dim=1)
    neighbor_score_mean = neighbor_score_sum / degree
    return struct_total_loss, neighbor_score_mean

def normalize_score_no_grad(t):
    min_val = t.min()
    max_val = t.max()
    normalized = (t - min_val) / (max_val - min_val + 1e-8)
    return normalized

def dominant_hega_recon_loss(
    x,
    x_,
    s,
    s_,
    nu,
    alpha_nig,
    beta,
    alpha_struct,
    beta_struct,
    alpha_cluster,
    q_target,
    current_epoch, 
    total_epochs,  
    ## 损失函数的权重参数
    attr_loss_weight=1.0,
    struct_loss_weight=0.0,
    neighbor_constrain_weight=0.0,
    ## 属性参数
    has_annealing=False,
    attr_reg_base=1e-2,
    attr_warmup_ratios=0.7,
    clustering_weight=0.0,
    ## 边结构参数
    bce_weight=0.0,
    ranking_weight=1.0,
    dice_weight=1.0,
    misleading_weight=1.0,
    pos_loss_weight=1.0,
    neg_loss_weight=1.0,
    kl_weight=1e-5,
    ranking_margin=1.0,
    ## 异常得分计算参数
    lambda_attr_recon=0.165,
    lambda_clustering_recon=0.165,
    lambda_struct_recon=0.33,
    lambda_neighbor_recon=0.33,
    lambda_attr_unc=0.33,
    lambda_clustering_unc=0.33,
    lambda_struct_unc=0.33,
    lambda_recon=0.5,
    lambda_unc=0.5,
    anomaly_func_type="add",
):
    eps = 1e-4
    device = alpha_nig.device # 获取设备

    nu = torch.clamp(nu, min=eps)
    alpha_nig = torch.clamp(alpha_nig, min=1 + eps) 
    beta = torch.clamp(beta, min=eps)
    if alpha_cluster is not None:
        alpha_cluster = torch.clamp(alpha_cluster, min=eps)

    attr_loss = None
    struct_total_loss = None
    neighbor_total_loss = None
    clustering_loss = None
    
    attr_score_norm = None
    struct_score_norm = None
    neighbot_score_norm = None
    clustering_score_norm = None

    attr_aleatoric_unc = None
    attr_epistemic_unc = None
    attr_epistemic_unc_norm = None
    
    struct_epistemic_unc = None
    struct_epistemic_unc_norm = None
    
    clustering_epistemic_unc = None
    clustering_epistemic_unc_norm = None

    avg_alpha_neighbor = None
    avg_beta_neighbor = None

    total_loss = torch.tensor(0.0, device=device)
    total_recon = torch.tensor(0.0, device=device)
    total_uncertainty = torch.tensor(0.0, device=device)

    if attr_loss_weight > 0:
        attr_total_loss, attr_loss, clustering_loss = attr_loss_cal(
            total_epochs=total_epochs, current_epoch=current_epoch, 
            gamma=x_, nu=nu, alpha=alpha_nig, beta=beta, target=x,
            alpha_cluster=alpha_cluster, q_target=q_target, has_annealing=has_annealing,
            attr_reg_base=attr_reg_base, attr_warmup_ratios=attr_warmup_ratios, 
            clustering_weight=clustering_weight
        )
        total_loss = total_loss + attr_total_loss * attr_loss_weight
        
        with torch.no_grad():
            attr_score_norm = normalize_score_no_grad(attr_loss)
            total_recon = total_recon + lambda_attr_recon * attr_score_norm
        
            alpha_minus_1 = alpha_nig - 1
            nu_safe = torch.clamp(nu, min=1e-4)
            attr_aleatoric_unc = torch.mean(beta / alpha_minus_1, dim=1)
            attr_epistemic_unc = torch.mean(beta / (nu_safe * alpha_minus_1), dim=1)
            attr_epistemic_unc_norm = normalize_score_no_grad(attr_epistemic_unc)
            total_uncertainty = total_uncertainty + lambda_attr_unc * attr_epistemic_unc_norm

        if clustering_loss is not None:
            with torch.no_grad():
                clustering_score_norm = normalize_score_no_grad(clustering_loss)
                total_recon = total_recon + lambda_clustering_recon * clustering_score_norm
                
                S = torch.sum(alpha_cluster, dim=1, keepdim=True)
                K = alpha_cluster.shape[1]
                clustering_epistemic_unc = (K / S).squeeze()
                clustering_epistemic_unc_norm = normalize_score_no_grad(clustering_epistemic_unc)
                total_uncertainty = total_uncertainty + lambda_clustering_unc * clustering_epistemic_unc_norm

    if struct_loss_weight > 0:
        struct_total_loss, struct_score = struct_loss_cal(
            alpha=alpha_struct, beta=beta_struct, target_adj=s, 
            bce_weight=bce_weight, ranking_weight=ranking_weight, 
            dice_weight=dice_weight, misleading_weight=misleading_weight,
            pos_loss_weight=pos_loss_weight, neg_loss_weight=neg_loss_weight,
            kl_weight=kl_weight, ranking_margin=ranking_margin
        )
        total_loss = total_loss + struct_total_loss * struct_loss_weight
        
        with torch.no_grad():
            struct_score_norm = normalize_score_no_grad(struct_score)
            total_recon = total_recon + lambda_struct_recon * struct_score_norm
            
            S_struct = alpha_struct + beta_struct
            u_struct_matrix = 2.0 / S_struct
            mask_neighbor = (s == 1).float()
            degree = mask_neighbor.sum(dim=1).clamp(min=1e-6)
            struct_epistemic_unc = (u_struct_matrix * mask_neighbor).sum(dim=1) / degree
            struct_epistemic_unc_norm = normalize_score_no_grad(struct_epistemic_unc)
            total_uncertainty = total_uncertainty + lambda_struct_unc * struct_epistemic_unc_norm
            
            avg_alpha_neighbor = (alpha_struct * mask_neighbor).sum(dim=1) / degree
            avg_beta_neighbor = (beta_struct * mask_neighbor).sum(dim=1) / degree

    if neighbor_constrain_weight > 0:
        neighbor_total_loss = neighborhood_distribution_loss(s, s_, x, x_)
        total_loss = total_loss + neighbor_total_loss * neighbor_constrain_weight
        with torch.no_grad():
            neighbot_score_norm = normalize_score_no_grad(neighbor_total_loss)
            total_recon = total_recon + lambda_neighbor_recon * neighbot_score_norm

    # 计算总异常得分
    if anomaly_func_type == "add":
        anomaly_score = lambda_recon * total_recon + lambda_unc * total_uncertainty
    elif anomaly_func_type == "gate":
        # final_recon_term = lambda_recon * total_recon
        # final_unc_term = lambda_unc * total_uncertainty
        # anomaly_score = final_unc_term * (1.0 + final_recon_term)

        certainty_coefficient = (2.0 * total_uncertainty - 1.0).pow(2)
        anomaly_score = lambda_recon * total_recon * certainty_coefficient + lambda_unc * total_uncertainty

    else:
        logger.error(">>>src.model.functional.py|>>>dominant_hega_recon_loss()|>>>Invalid anomaly_func_type!")
        raise RuntimeError("Invalid anomaly_func_type!")
    loss_dict = {
        "attr_loss": attr_loss,
        "adj_loss": struct_total_loss,
        "struct_loss": neighbor_total_loss,
        "clustering_loss": clustering_loss,
        "total_loss": total_loss,

        "attr_score": (lambda_attr_recon * attr_score_norm) if attr_score_norm is not None else None,
        "cluster_score": (lambda_clustering_recon * clustering_score_norm) if clustering_score_norm is not None else None,
        "adj_score": (lambda_struct_recon * struct_score_norm) if struct_score_norm is not None else None,
        "struct_score": (lambda_neighbor_recon * neighbot_score_norm) if neighbot_score_norm is not None else None,
        
        "attr_unc_data": attr_aleatoric_unc,
        
        "attr_unc_model": (lambda_attr_unc * attr_epistemic_unc_norm) if attr_epistemic_unc_norm is not None else None,
        "adj_unc_model": (lambda_struct_unc * struct_epistemic_unc_norm) if struct_epistemic_unc_norm is not None else None,
        "cluster_unc_model": (lambda_clustering_unc * clustering_epistemic_unc_norm) if clustering_epistemic_unc_norm is not None else None,
        
        "score_rec": lambda_recon * total_recon,
        "score_unc_model": lambda_unc * total_uncertainty,
        "anomaly_score": anomaly_score,
        
        "raw_epistemic_dec": attr_epistemic_unc,
        "raw_epistemic_adj": struct_epistemic_unc,
        "raw_epistemic_cluster": clustering_epistemic_unc,

        "nu": torch.mean(nu, dim=1),
        "alpha_nig": torch.mean(alpha_nig, dim=1),
        "beta": torch.mean(beta, dim=1),
        "alpha_cluster": torch.mean(alpha_cluster, dim=1) if alpha_cluster is not None else None,
        "alpha_struct": avg_alpha_neighbor,
        "beta_struct": avg_beta_neighbor,
    }
    return anomaly_score, total_loss, loss_dict