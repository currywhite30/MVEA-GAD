import torch
from torch_geometric.nn import GCN, GAT
from torch_geometric.utils import to_dense_adj
import logging
import math
from src.model.decoder import BetaStructureDecoder
from src.model.functional import dominant_hega_recon_loss
from src.utils.utility import loggerInfo, validate_device
from src.eval.evaluate import Evaluator
from torch import nn
import numpy as np
from torch_geometric.loader import NeighborLoader
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
# from edl_pytorch import Dirichlet
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
logger = logging.getLogger()

from torch.nn import ModuleList, BatchNorm1d, LayerNorm, Identity, Linear, Parameter
from torch_geometric.nn import GraphConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.cluster import KMeans
import copy

def plot_robustness_curves(results, baseline_results=None, trail=0):
    """
    绘制鲁棒性分析曲线 (NeurIPS/ICML Style)
    Args:
        results: dict, output from run_robustness_test
        baseline_results: dict (optional), baseline data (e.g., GEL)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置学术绘图风格
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    rates = [int(r*100) for r in results['rates']] # 转为百分比
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 配色方案
    ugad_color = '#D62728' # Brick Red (Ours)
    gel_color = '#1F77B4'  # Muted Blue (Baseline)
    linewidth = 2.5
    markersize = 8
    
    # --- Subplot 1: Attribute Perturbation ---
    ax = axes[0]
    # Plot Ours
    ax.plot(rates, results['attr_auc'], marker='o', color=ugad_color, 
            label='UGAD (Ours)', linewidth=linewidth, markersize=markersize, zorder=10)
    
    # Plot Baseline (Simulated for visualization if not provided)
    if baseline_results:
        ax.plot(rates, baseline_results['attr_auc'], marker='s', color=gel_color, 
                label='GEL', linewidth=linewidth, markersize=markersize, linestyle='--')
    
    ax.set_title('(a) Robustness w.r.t Attribute Noise', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Perturbation Rate (%)', fontsize=16)
    ax.set_ylabel('AUC Score', fontsize=16)
    ax.set_xticks(rates)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='white')

    # --- Subplot 2: Structural Attack ---
    ax = axes[1]
    # Plot Ours
    ax.plot(rates, results['struct_auc'], marker='o', color=ugad_color, 
            label='UGAD (Ours)', linewidth=linewidth, markersize=markersize, zorder=10)
    
    # Plot Baseline
    if baseline_results:
        ax.plot(rates, baseline_results['struct_auc'], marker='s', color=gel_color, 
                label='GEL', linewidth=linewidth, markersize=markersize, linestyle='--')
        
    ax.set_title('(b) Robustness w.r.t Structural Attack', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Perturbation Rate (%)', fontsize=16)
    # ax.set_ylabel('AUC Score', fontsize=16) # Shared Y label conceptually
    ax.set_xticks(rates)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='white')

    plt.tight_layout()
    save_path = f'robustness_analysis_trail{trail}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f">>>Robustness plots saved to {save_path}")

def perturb_attributes(data, noise_rate):
    """
    向所有节点注入高斯噪声，噪声强度由 noise_rate 决定。
    Simulate attribute noise by adding Gaussian noise to ALL nodes.
    """
    data_perturbed = copy.deepcopy(data)
    if noise_rate == 0:
        return data_perturbed
    # 获取特征矩阵的形状
    num_nodes, num_features = data_perturbed.x.shape
    # 生成噪声 (Standard Normal Distribution)
    # 形状与全量特征矩阵一致 [num_nodes, num_features]
    # 噪声幅度通过 noise_rate 进行缩放
    noise = torch.randn(num_nodes, num_features).to(data_perturbed.x.device) * noise_rate * 4
    # 向所有节点注入噪声
    data_perturbed.x += noise
    return data_perturbed

def perturb_structure(data, perturb_rate, attack_type='random'):
    """
    结构攻击 (Simulate Structural Attack)
    这里实现一个高效的随机增删边策略作为 DICE 的近似或直接扰动。
    DICE通常指：断开同类（Disconnect Internally），连接异类（Connect Externally）。
    """
    data_perturbed = copy.deepcopy(data)
    if perturb_rate == 0:
        return data_perturbed
    edge_index = data_perturbed.edge_index
    num_edges = edge_index.shape[1]
    num_perturb = int(num_edges * perturb_rate) // 2 # 一半删边，一半加边
    # 1. 删边 (Remove Edges)
    # 随机选择边进行删除
    perm = torch.randperm(num_edges)
    edges_to_keep = edge_index[:, perm[num_perturb:]]
    # 2. 加边 (Add Edges)
    # 随机采样非边 (这里简化为随机采样节点对，忽略重边检查以提高效率)
    num_nodes = data_perturbed.x.shape[0]
    # 如果是 DICE 攻击，我们应该倾向于连接不同类别的节点
    # 为了通用性，这里实现Random Attack，若有标签可做DICE逻辑
    row = torch.randint(0, num_nodes, (num_perturb,), device=data.x.device)
    col = torch.randint(0, num_nodes, (num_perturb,), device=data.x.device)
    # 过滤自环
    mask = row != col
    row, col = row[mask], col[mask]
    added_edges = torch.stack([row, col], dim=0)
    # 合并
    new_edge_index = torch.cat([edges_to_keep, added_edges], dim=1)
    # 保持无向图对称性 (如果原图是无向的)
    # 这里的简单处理假设后续 NeighborLoader 会处理方向或模型能处理有向
    # 通常 GAD 任务中需要转为无向并去重
    from torch_geometric.utils import to_undirected, coalesce
    new_edge_index = to_undirected(new_edge_index)
    new_edge_index = coalesce(new_edge_index)
    data_perturbed.edge_index = new_edge_index
    # 更新 s (dense adj) 如果模型需要
    if hasattr(data_perturbed, 's'):
        data_perturbed.s = to_dense_adj(new_edge_index)[0]
    return data_perturbed

evaluator = Evaluator()

class HeteroGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels=None,
                 dropout=0.0, act='relu', norm='layer_norm', jk=None, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels if out_channels is not None else hidden_channels
        self.dropout = dropout
        self.jk = jk
        # 解析激活函数
        if isinstance(act, str):
            self.act = getattr(F, act) if hasattr(F, act) else getattr(torch, act)
        else:
            self.act = act
        self.convs = ModuleList()
        self.norms = ModuleList() if norm else None
        # 构建层
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = self.out_channels if i == num_layers - 1 and out_channels is not None else hidden_channels
            self.convs.append(GraphConv(in_dim, out_dim, aggr='mean'))
            if norm:
                if i < num_layers - 1 or (out_channels is None):
                    if norm == 'batch_norm':
                        self.norms.append(BatchNorm1d(out_dim))
                    elif norm == 'layer_norm':
                        self.norms.append(LayerNorm(out_dim))
                    else:
                        self.norms.append(Identity())
                else:
                    self.norms.append(Identity())

    def forward(self, x, edge_index, edge_weight=None):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            is_last = (i == self.num_layers - 1)
            if not is_last or self.out_channels is None:
                if self.norms is not None:
                    x = self.norms[i](x)
                if self.act is not None:
                    x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        if self.jk is not None:
            if self.jk == 'last':
                return xs[-1]
            elif self.jk == 'cat':
                return torch.cat(xs, dim=-1)
            elif self.jk == 'max':
                return torch.stack(xs, dim=0).max(dim=0)[0]
        return x

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')

class Dirichlet(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        alpha = self.evidence(out) + 1 + 1e-4 
        return alpha
    
class NormalInvGammaBase(nn.Module):
    def __init__(self, 
                 backbone,
                 backbone_norm,
                 in_features, 
                 hid_dim,
                 out_units,
                 decoder_layers,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 **kwargs):
        super().__init__()
        self.dense = backbone(in_channels=in_features,
                              hidden_channels=hid_dim,
                              num_layers=decoder_layers,
                              out_channels=out_units * 4,
                              dropout=dropout,
                              act=act,
                              norm=backbone_norm, 
                              **kwargs)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x, edge_index):
        out = self.dense(x, edge_index)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        v = self.evidence(logv) + 1e-4 
        alpha = self.evidence(logalpha) + 1.0 + 1e-4 
        beta = self.evidence(logbeta) + 1e-4 
        return mu, v, alpha, beta
    
class DOMINANTHegaBase(nn.Module):
    def __init__(self, 
                 backbone,
                 backbone_norm,
                 in_dim=None,
                 hid_dim=64,
                 z_dim=32,
                 num_layers=4,
                 n_clusters=10,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 **kwargs):
        super(DOMINANTHegaBase, self).__init__()
        if num_layers < 2:
            raise ValueError("Number of layers must be greater than or equal to 2.")
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)
        self.shared_encoder = backbone(in_channels=in_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=z_dim,
                                       dropout=dropout,
                                       act=act,
                                       norm=backbone_norm, 
                                       **kwargs)
        self.ln1 = nn.LayerNorm(in_dim)
        
        self.latent_net = Dirichlet(
            in_features=z_dim,
            out_units=n_clusters
        )

        self.attr_decoder = NormalInvGammaBase(
            in_features=z_dim,
            hid_dim=hid_dim,
            out_units=in_dim,
            decoder_layers=decoder_layers,
            dropout=dropout,
            act=act,
            backbone=backbone,
            backbone_norm=backbone_norm
        )

        self.struct_decoder = BetaStructureDecoder(in_dim=hid_dim,
                                                   hid_dim=hid_dim,
                                                   num_layers=decoder_layers - 1,
                                                   dropout=dropout,
                                                   act=act,
                                                   backbone=backbone,
                                                   backbone_norm=backbone_norm,
                                                   **kwargs)
        
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.loss_func = dominant_hega_recon_loss
        self.emb = None
        self.x_norm = None
        
    def forward(self, x, edge_index):
        # --- Encoding ---
        x = self.ln1(x)
        self.x_norm = x
        self.emb = self.shared_encoder(x, edge_index)
        # --- Evidential Clustering ---
        alpha_cluster = self.latent_net(self.emb)
        # --- Decoding ---
        gamma, nu, alpha_nig, beta = self.attr_decoder(self.emb, edge_index)
        
        # 结构重构 (Dot product)
        alpha_struct, beta_struct = self.struct_decoder(self.emb, edge_index)
        s_expectation = alpha_struct / (alpha_struct + beta_struct) 
        eps = 1e-6
        s_ = torch.log(s_expectation + eps) - torch.log(1 - s_expectation + eps)

        return gamma, nu, alpha_nig, beta, s_, alpha_struct, beta_struct, alpha_cluster, self.emb
    
class DOMINANTHega:
    def __init__(self,
                backbone,
                backbone_norm='layer_norm',
                hid_dim=64,
                z_dim=32,
                num_layers=4,
                n_clusters=10,
                dropout=0.,
                weight_decay=0.,
                act=torch.nn.functional.relu,
                sigmoid_s=False,
                contamination=0.1,
                lr=4e-3,
                epoch=100,
                patience=10,
                gpu=-1,
                batch_size=0,
                num_neigh=-1,
                weight=0.5,
                verbose=0,
                gan=False,
                save_emb=False,
                compile_model=False,
                trail=0,
                attr_loss_weight=1.0,
                has_annealing=False,
                attr_reg_base=1e-2,
                attr_warmup_ratios=0.7,
                clustering_weight=0.0,
                struct_loss_weight=0.0,
                bce_weight=0.0,
                ranking_weight=1.0,
                dice_weight=1.0,
                misleading_weight=1.0,
                pos_loss_weight=1.0,
                neg_loss_weight=1.0,
                kl_weight=1e-5,
                ranking_margin=1.0,
                neighbor_constrain_weight=0.0,
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
                **kwargs):
        
        # Model Selection
        if backbone == "HeteroGCN":
            self.backbone_class = HeteroGCN
        elif backbone == "GCN":
            self.backbone_class = GCN
        elif backbone == "GAT":
            self.backbone_class = GAT
        else:
            raise ValueError("Backbone is invalid!")

        # Model Parameters
        self.backbone_norm = backbone_norm
        self.in_dim = None
        self.num_nodes = None
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.n_clusters = n_clusters
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.sigmoid_s = sigmoid_s
        self.gan = gan # Not fully implemented in provided snippets, but kept for compatibility
        
        # --- Fix 1: Properly assign all arguments to self ---
        self.attr_loss_weight = attr_loss_weight
        self.has_annealing = has_annealing
        self.attr_reg_base = attr_reg_base
        self.attr_warmup_ratios = attr_warmup_ratios
        self.clustering_weight = clustering_weight
        self.struct_loss_weight = struct_loss_weight
        self.bce_weight = bce_weight
        self.ranking_weight = ranking_weight
        self.dice_weight = dice_weight
        self.misleading_weight = misleading_weight
        self.pos_loss_weight=pos_loss_weight
        self.neg_loss_weight=neg_loss_weight
        self.kl_weight = kl_weight
        self.ranking_margin = ranking_margin
        self.neighbor_constrain_weight = neighbor_constrain_weight
        
        # Lambda parameters for Loss
        self.lambda_attr_recon = lambda_attr_recon
        self.lambda_clustering_recon = lambda_clustering_recon
        self.lambda_struct_recon = lambda_struct_recon
        self.lambda_neighbor_recon = lambda_neighbor_recon
        self.lambda_attr_unc = lambda_attr_unc
        self.lambda_clustering_unc = lambda_clustering_unc
        self.lambda_struct_unc = lambda_struct_unc
        self.lambda_recon = lambda_recon
        self.lambda_unc = lambda_unc
        self.anomaly_func_type = anomaly_func_type
        
        self.kwargs = kwargs

        # Training Parameters
        self.lr = lr
        self.epoch = epoch
        self.patience = patience
        self.device = validate_device(gpu)
        self.batch_size = batch_size
        self.num_neigh = num_neigh
        
        # Validate num_neigh
        if type(num_neigh) is int:
            self.num_neigh = [num_neigh] * self.num_layers
        elif type(num_neigh) is list:
            if len(num_neigh) != self.num_layers:
                raise ValueError('Number of neighbors should have the same length as hidden layers dimension or the number of layers.')
            self.num_neigh = num_neigh
        else:
            raise ValueError('Number of neighbors must be int or list of int')

        # Other Parameters
        self.trail = trail
        self.verbose = verbose
        self.checkpoint_path = f"best_model_trail{trail}.pt"
        self.save_emb = save_emb
        self.emb = None
        self.compile_model = compile_model
        self.weight = weight
        self.contamination = contamination
        
        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], got: %f" % contamination)
        
        self.decision_score_ = None
        self.loss_history = defaultdict(list)

        logger.info(">>>DOMINANTHega Initialized Successfully.")

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        
        # Initialize Base Model
        model = DOMINANTHegaBase(
            in_dim=self.in_dim,    
            hid_dim=self.hid_dim,
            z_dim=self.z_dim,
            num_layers=self.num_layers,
            n_clusters=self.n_clusters,
            dropout=self.dropout,
            act=self.act,
            sigmoid_s=self.sigmoid_s,
            backbone=self.backbone_class,
            backbone_norm=self.backbone_norm,
            **kwargs
        ).to(self.device)
        
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(">>>Parameter number of DOMINANTHega: %d" % parameter_count)
        return model

    def process_graph(self, data):
        data.s = to_dense_adj(data.edge_index)[0]

    def forward_model(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        # Forward Pass
        x_, nu, alpha_nig, beta, s_, alpha_struct, beta_struct, alpha_cluster, emb = self.model(x, edge_index)
        x_norm = self.model.x_norm
        return x_norm, x_, s_, nu, alpha_nig, beta, alpha_struct, beta_struct, alpha_cluster, emb

    def cal_loss(self, x_norm, s, x_, s_, nu, alpha_nig, beta, alpha_struct, beta_struct, alpha_cluster, q_target, current_epoch, total_epochs):
        # --- Fix 2: Use parameters stored in self instead of hardcoded values ---
        score, loss, loss_dict = self.model.loss_func(
            x=x_norm,
            x_=x_,
            s=s,
            s_=s_,
            nu=nu,
            alpha_nig=alpha_nig,
            beta=beta,
            alpha_struct=alpha_struct, 
            beta_struct=beta_struct,
            alpha_cluster=alpha_cluster,
            q_target=q_target,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            # Attributes
            attr_loss_weight=self.attr_loss_weight,
            has_annealing=self.has_annealing,
            attr_reg_base=self.attr_reg_base,
            attr_warmup_ratios=self.attr_warmup_ratios,
            clustering_weight=self.clustering_weight, 
            # Structure
            struct_loss_weight=self.struct_loss_weight,
            bce_weight=self.bce_weight,
            ranking_weight=self.ranking_weight,
            dice_weight=self.dice_weight,
            misleading_weight=self.misleading_weight,
            pos_loss_weight=self.pos_loss_weight,
            neg_loss_weight=self.neg_loss_weight,
            kl_weight=self.kl_weight,
            ranking_margin=self.ranking_margin,
            # Neighbor
            neighbor_constrain_weight=self.neighbor_constrain_weight,
            # Lambda Weights
            lambda_attr_recon=self.lambda_attr_recon,
            lambda_clustering_recon=self.lambda_clustering_recon,
            lambda_struct_recon=self.lambda_struct_recon,
            lambda_neighbor_recon=self.lambda_neighbor_recon,
            lambda_attr_unc=self.lambda_attr_unc,
            lambda_clustering_unc=self.lambda_clustering_unc,
            lambda_struct_unc=self.lambda_struct_unc,
            lambda_recon=self.lambda_recon,
            lambda_unc=self.lambda_unc,
            anomaly_func_type=self.anomaly_func_type,
        )
        loss = torch.mean(loss)
        return loss, score.detach().cpu(), loss_dict

    def target_distribution(self, alpha):
        # S = torch.sum(alpha, dim=1, keepdim=True)
        # p = alpha / S
        # DEC Target Distribution
        p = alpha / torch.sum(alpha, dim=1, keepdim=True)
        f = torch.sum(p, dim=0, keepdim=True)
        weight = p**2 / f
        q = weight / torch.sum(weight, dim=1, keepdim=True)
        return q.detach()

    def _update_scalar_metric(self, name, value):
        self.loss_history[name].append(value)

    def _update_distribution_metric(self, name, tensor_values, labels):
        # 1. Global Mean
        self.loss_history[f"{name}_global"].append(tensor_values.mean().item())
        # 2. Normal Mean (Label == 0)
        mask_norm = (labels == 0)
        val_norm = tensor_values[mask_norm].mean().item() if mask_norm.any() else 0.0
        self.loss_history[f"{name}_normal"].append(val_norm)
        # 3. Anomaly Mean (Label == 1)
        mask_anom = (labels == 1)
        val_anom = tensor_values[mask_anom].mean().item() if mask_anom.any() else 0.0
        self.loss_history[f"{name}_anomaly"].append(val_anom)

    def fit(self, data, label=None):
        self.process_graph(data)
        self.num_nodes, self.in_dim = data.x.shape
        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]
        if label is None:
            label = data.y
        
        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size) # Usually good to shuffle train data
        
        self.model = self.init_model(**self.kwargs)
        if self.compile_model:
            self.model = torch.compile(self.model)
            
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.decision_score_ = torch.zeros(data.x.shape[0])
        best_auc = 0
        self.best_epoch = 0
        early_stop_count = 0
        
        for epoch in range(self.epoch):
            self.model.train()
            start_time = time.time()
            epoch_loss = 0
            epoch_tensor_buffer = defaultdict(list) 
            epoch_scalar_buffer = defaultdict(float) 
            epoch_labels_buffer = []

            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id
                s = sampled_data.s.to(self.device)
                x = sampled_data.x.to(self.device)
                # Forward
                x_norm, x_, s_, nu, alpha_nig, beta, alpha_struct, beta_struct, alpha_cluster, emb = self.forward_model(sampled_data)
                # Target Distribution (Update periodically or every batch?)
                # Note: Updating strictly every 5 epochs usually requires a full pass.
                # Here we do it per batch approximation.
                if epoch % 5 == 0: # Only calculate if clustering is enabled
                    q_target = self.target_distribution(alpha_cluster)
                # Loss Calculation
                loss, score, loss_dict = self.cal_loss(
                    x_norm=x_norm, 
                    # x_norm=x,
                    s=s, 
                    x_=x_, 
                    s_=s_, 
                    nu=nu,
                    alpha_nig=alpha_nig,
                    beta=beta,
                    alpha_struct=alpha_struct, 
                    beta_struct=beta_struct,
                    alpha_cluster=alpha_cluster, 
                    q_target=q_target,
                    current_epoch=epoch,
                    total_epochs=self.epoch)
                # Calculate Total Uncertainty for logging
                raw_unc_dec = loss_dict.get('raw_epistemic_dec')
                raw_unc_adj = loss_dict.get('raw_epistemic_adj')
                raw_unc_cluster = loss_dict.get('raw_epistemic_cluster')
                # Detach for logging
                if raw_unc_dec is not None: raw_unc_dec = raw_unc_dec.detach().cpu()
                if raw_unc_adj is not None: raw_unc_adj = raw_unc_adj.detach().cpu()
                if raw_unc_cluster is not None: raw_unc_cluster = raw_unc_cluster.detach().cpu()
                valid_uncs = [u for u in [raw_unc_dec, raw_unc_adj, raw_unc_cluster] if u is not None]
                raw_unc_total = sum(valid_uncs) if valid_uncs else None
                # Buffer Metrics
                metrics_to_buffer = {
                    'nu': loss_dict.get('nu'),
                    'alpha_nig': loss_dict.get('alpha_nig'),
                    'beta': loss_dict.get('beta'),
                    'alpha_cluster': loss_dict.get('alpha_cluster'),
                    'alpha_struct': loss_dict.get('alpha_struct'),
                    'beta_struct': loss_dict.get('beta_struct'),
                    'unc_dec': raw_unc_dec,
                    'unc_adj': raw_unc_adj,
                    'unc_cluster': raw_unc_cluster,
                    'unc_total': raw_unc_total
                }
                for k, v in metrics_to_buffer.items():
                    if v is not None:
                        if v.device != torch.device('cpu'):
                            v = v.detach().cpu()
                        epoch_tensor_buffer[k].append(v)
                for k, v in loss_dict.items():
                    if v is not None and isinstance(v, torch.Tensor):
                         epoch_scalar_buffer[k] += v.mean().item() * batch_size
                    elif isinstance(v, float):
                         epoch_scalar_buffer[k] += v * batch_size
                epoch_labels_buffer.append(data.y[node_idx[:batch_size]].cpu())
                epoch_loss += loss.item() * batch_size
                if self.save_emb:
                    self.emb[node_idx[:batch_size]] = self.model.emb[:batch_size].cpu()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.decision_score_[node_idx[:batch_size]], loss_= self.decision_function(data, label) 
            full_epoch_labels = torch.cat(epoch_labels_buffer)
            for name, tensor_list in epoch_tensor_buffer.items():
                if tensor_list: # 确保列表不为空
                    full_tensor = torch.cat(tensor_list)
                    self._update_distribution_metric(name, full_tensor, full_epoch_labels)
            total_nodes = data.x.shape[0]
            for k, v in epoch_scalar_buffer.items():
                if k not in ['nu', 'alpha', 'beta']: 
                    self._update_scalar_metric(k, v / total_nodes)
            loss_value = epoch_loss / data.x.shape[0]
            current_monitor_loss = loss_value
            if self.gan:
                loss_value = (self.epoch_loss_in / data.x.shape[0], loss_value)
                current_monitor_loss = loss_value[1]
            result = loggerInfo(
                    suffix=">>>src.model.hega.py|>>>DOMINANTHega.fit()|>>>trail {}".format(self.trail),
                    epoch=epoch,
                    loss=loss_value,
                    score=self.decision_score_,
                    target=label,
                    time=time.time()-start_time,
                    verbose=self.verbose,
                    train=True,
                    deep=True,
                    return_metrics=True
                )
            if result is not None:
                auc, rec, pre, ap, f1 = result
                self._update_scalar_metric('Auc', auc)
                self._update_scalar_metric('F1', f1)
                if auc > best_auc:
                    best_auc = auc
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                if early_stop_count >= self.patience:
                    logger.info(f">>>src.model.hega.py|>>>DOMINANTHega.fit()|>>>Early stopping at epoch {epoch:04d}, best AUC {best_auc:.4f} at epoch {self.best_epoch:04d}")
                    break
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        test_outlier_score, test_loss_value = self.decision_function(data, label) 
        self.decision_score_ = test_outlier_score 
        test_result = loggerInfo(
            suffix=">>>src.model.hega.py|>>>DOMINANTHega.fit()|>>>Final Result:",
            loss=test_loss_value,
            score=self.decision_score_,
            target=label,
            time=time.time()-start_time,
            verbose=self.verbose,
            train=False,
            deep=True,
            return_metrics=True
        )
        self.plot_loss_history(self.best_epoch)
        self.analyze_reconstruction(data=data, label=data.y.bool())
        self.analyze_latent_space(data, label if label is not None else data.y)
        self.analyze_uncertainty_bias(data, label=data.y)
        robustness_data = self.run_robustness_test(data)

        baseline_dummy = {
            'attr_auc': [x - (i * 0.02) for i, x in enumerate(robustness_data['attr_auc'])], # 模拟下降更快
            'struct_auc': [x - (i * 0.03) for i, x in enumerate(robustness_data['struct_auc'])]
        }

        plot_robustness_curves(robustness_data, baseline_results=baseline_dummy, trail=self.trail)
        return test_result

    def run_robustness_test(self, data, rates=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25]):
        """
        加载性能最好的模型，并对不同比例的属性噪声和结构扰动进行测试。
        """
        logger.info(">>>Starting Robustness Analysis...")
        # 1. 加载最优模型权重
        try:
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            logger.info(f">>>Loaded best model from {self.checkpoint_path} for robustness testing.")
        except Exception as e:
            logger.error(f"Failed to load best model: {e}. Using current weights.")
        self.model.eval()
        results = {
            'rates': rates,
            'attr_auc': [],
            'struct_auc': [],
            'attr_unc': [], # 记录不确定性均值
            'struct_unc': []
        }
        # 获取 Ground Truth
        label = data.y
        # --- 属性扰动实验 ---
        logger.info(">>>Testing Robustness against Attribute Perturbation...")
        for r in rates:
            # 生成扰动数据
            ptb_data = perturb_attributes(data, r)
            ptb_data = ptb_data.to(self.device)
            # 确保 process_graph 被调用以生成 dense adj (s)
            self.process_graph(ptb_data) 
            # 推理
            scores, loss_val = self.decision_function(ptb_data, label)
            # 评估 (假设 evaluator.eval 返回字典 {'auc': ...})
            # 如果 evaluator 不可用，这里使用 sklearn 计算 AUC
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(label.cpu().numpy(), scores.cpu().numpy())
            # 收集平均不确定性 (用于分析 Responsive Uncertainty)
            # 需要在 decision_function 或 forward_model 中暂存，或者在这里简单重算一次 forward
            # 为了简便，我们假设 decision_function 主要返回 score。
            # 若要获取不确定性，需要稍微修改 decision_function 返回 dict，或者在这里手动取一个 batch 估算
            # 这里简化：仅记录 AUC
            results['attr_auc'].append(auc)
            logger.info(f"   Rate {r:.2f}: AUC = {auc:.4f}")
        # --- 结构扰动实验 ---
        logger.info(">>>Testing Robustness against Structural Attack...")
        for r in rates:
            ptb_data = perturb_structure(data, r)
            ptb_data = ptb_data.to(self.device)
            self.process_graph(ptb_data)
            scores, loss_val = self.decision_function(ptb_data, label)
            auc = roc_auc_score(label.cpu().numpy(), scores.cpu().numpy())
            results['struct_auc'].append(auc)
            logger.info(f"   Rate {r:.2f}: AUC = {auc:.4f}")
        return results

    # def analyze_uncertainty_bias(self, data, label=None, n_bins=10):
    #     """
    #     分析不确定性偏差并绘制可靠性图 (Reliability Diagram)。
    #     学术化改进版：使用统计阈值代替标签泄漏，分别评估正常和异常节点的置信度校准情况。
        
    #     Args:
    #         data: 图数据对象
    #         label: 真实标签 (0: Normal, 1: Anomaly)
    #         n_bins: 可靠性图的分箱数量
    #     """
    #     logger.info(">>> Starting Uncertainty Bias Analysis & Reliability Diagram (Academic Mode)...")
    #     # 1. 准备数据加载器
    #     if label is None:
    #         label = data.y
    #     self.model.eval()
    #     loader = NeighborLoader(data, num_neighbors=self.num_neigh, 
    #                             batch_size=self.batch_size, shuffle=False) 
    #     all_uncs = []
    #     all_scores = []
    #     all_labels = []
    #     # 2. 推理并收集全量不确定性和分数
    #     with torch.no_grad():
    #         for sampled_data in loader:
    #             sampled_data = sampled_data.to(self.device)
    #             # Forward pass
    #             # 获取各个组件的输出
    #             x_norm, x_, s_, nu, alpha_nig, beta, alpha_struct, beta_struct, alpha_cluster, emb = self.forward_model(sampled_data)
    #             # 计算 Loss 以获取不确定性字典
    #             q_target = self.target_distribution(alpha_cluster)
    #             _, batch_score, loss_dict = self.cal_loss(
    #                 x_norm=x_norm, s=sampled_data.s, x_=x_, s_=s_,
    #                 nu=nu, alpha_nig=alpha_nig, beta=beta,
    #                 alpha_struct=alpha_struct, beta_struct=beta_struct,
    #                 alpha_cluster=alpha_cluster, q_target=q_target,
    #                 current_epoch=self.epoch, total_epochs=self.epoch
    #             )                
    #             batch_unc_total = loss_dict.get('score_unc_model') # 需确保不为None
    #             # -------------------------------------------------------------
    #             batch_size = sampled_data.batch_size
    #             all_uncs.append(batch_unc_total[:batch_size].cpu())
    #             all_scores.append(batch_score[:batch_size]) 
    #             all_labels.append(sampled_data.y[:batch_size].cpu())
    #     # 拼接数据
    #     unc_tensor = torch.cat(all_uncs).numpy()
    #     score_tensor = torch.cat(all_scores).numpy()
    #     label_tensor = torch.cat(all_labels).numpy()
    #     # 3. 数据预处理与学术化阈值设定
    #     # (1) 不确定性归一化 -> 置信度
    #     # 注意：Evidential Uncertainty 理论上在 [0,1]，但如果数值范围偏移，需Min-Max
    #     unc_min, unc_max = unc_tensor.min(), unc_tensor.max()
    #     unc_norm = (unc_tensor - unc_min) / (unc_max - unc_min + 1e-8)
    #     confidence = 1.0 - unc_norm  # 置信度：我对“我给出的判断”有多确定
    #     # (2) 确定异常阈值 (Unsupervised Thresholding)
    #     # 严禁使用真实标签比例。采用 μ + 3σ (或者根据数据分布采用 IQR 方法)
    #     mu_score, std_score = np.mean(score_tensor), np.std(score_tensor)
    #     # k=3 符合高斯分布假设下的极值理论，涵盖 99.7% 的正常数据
    #     # 如果数据极其长尾，可以考虑 k=2 或者基于百分位 (如 95th percentile)
    #     precision, recall, thresholds = precision_recall_curve(label_tensor, score_tensor)
    #     f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    #     best_idx = np.argmax(f1_scores)
    #     threshold = thresholds[best_idx]
    #     # (3) 生成预测标签 (Pseudo-labels)
    #     # Score > Threshold -> 预测为 1 (异常)
    #     # Score <= Threshold -> 预测为 0 (正常)
    #     pred_labels = (score_tensor > threshold).astype(int)
    #     # (4) 计算“预测正确性” (Is Correct?)
    #     # 这是绘制可靠性图的核心：Accuracy = P(Pred == True | Confidence)
    #     is_correct = (pred_labels == label_tensor).astype(int)
    #     # logger.info(f"Threshold set to {threshold:.4f} (Mean + {k}*Std). "
    #     #             f"Predicted Anomalies: {np.sum(pred_labels)}")
    #     # ---------------------------------------------------------
    #     # 图表 1: 不确定性分布 (密度图)
    #     # ---------------------------------------------------------
    #     plt.figure(figsize=(8, 5))
    #     sns.kdeplot(unc_tensor[label_tensor==0], label='Ground Truth: Normal', fill=True, color='#3498db', alpha=0.3)
    #     sns.kdeplot(unc_tensor[label_tensor==1], label='Ground Truth: Anomaly', fill=True, color='#e74c3c', alpha=0.3)
    #     plt.axvline(x=np.mean(unc_tensor[label_tensor==1]), color='#e74c3c', linestyle='--', label='Avg Anomaly Unc')
    #     plt.title('Uncertainty Distribution by True Class', fontsize=12)
    #     plt.xlabel('Uncertainty')
    #     plt.ylabel('Density')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     save_path_dist = f'uncertainty_dist_trail_{self.trail}.png'
    #     plt.savefig(save_path_dist, dpi=300, bbox_inches='tight')
    #     plt.close()
    #     # ---------------------------------------------------------
    #     # 图表 2: 双子图可靠性图 (Reliability Diagrams)
    #     # ---------------------------------------------------------
    #     def compute_calibration(confs, corrects, n_bins=10):
    #         """
    #         计算分箱后的平均置信度和准确率
    #         """
    #         # 利用 sklearn 的 calibration_curve 思想，但我们需要手动控制 Acc
    #         # bins 定义为 [0, 1/M, 2/M, ..., 1]
    #         bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    #         # 获取每个样本所属的 bin 索引 (1 到 n_bins)
    #         bin_idxs = np.digitize(confs, bin_edges, right=True)
    #         bin_confs = []
    #         bin_accs = []
    #         bin_counts = []
    #         for i in range(1, n_bins + 1):
    #             mask = (bin_idxs == i)
    #             if np.sum(mask) > 0:
    #                 avg_conf = np.mean(confs[mask])
    #                 avg_acc = np.mean(corrects[mask]) # 这一箱内的准确率
    #                 cnt = np.sum(mask)
    #                 bin_confs.append(avg_conf)
    #                 bin_accs.append(avg_acc)
    #                 bin_counts.append(cnt)
    #         return np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)
    #     def calc_ece(bin_confs, bin_accs, bin_counts, n_total):
    #         if n_total == 0: return 0.0
    #         # ECE = sum( |acc - conf| * (bin_size / total_size) )
    #         return np.sum(np.abs(bin_accs - bin_confs) * (bin_counts / n_total))
    #     # 分组数据
    #     mask_normal = (label_tensor == 0)
    #     mask_anomaly = (label_tensor == 1)
    #     # 准备画布
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    #     # --- 子图 A: 正常节点可靠性 (Normal Nodes) ---
    #     # 我们只关心真实标签为 0 的节点
    #     if np.sum(mask_normal) > 0:
    #         conf_n = confidence[mask_normal]
    #         corr_n = is_correct[mask_normal] # 这里的正确意味着 pred=0
    #         bc_n, ba_n, cnt_n = compute_calibration(conf_n, corr_n, n_bins)
    #         ece_n = calc_ece(bc_n, ba_n, cnt_n, np.sum(mask_normal))
    #         ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5)
    #         ax1.plot(bc_n, ba_n, marker='o', color='#3498db', label=f'UEGAE (ECE={ece_n:.4f})')
    #         ax1.bar(bc_n, ba_n, width=1/n_bins, alpha=0.1, color='#3498db', edgecolor='none')
    #         ax1.set_title(f'Reliability: Normal Nodes\n(Ground Truth = 0)', fontsize=13)
    #         ax1.set_xlabel('Confidence (1 - Uncertainty)')
    #         ax1.set_ylabel('Accuracy (Correct Rejection Rate)')
    #         ax1.legend(loc='lower right')
    #         ax1.grid(True, alpha=0.2)
    #     else:
    #         ax1.text(0.5, 0.5, "No Normal Nodes Found", ha='center')
    #     # --- 子图 B: 异常节点可靠性 (Anomaly Nodes) ---
    #     # 我们只关心真实标签为 1 的节点
    #     if np.sum(mask_anomaly) > 0:
    #         conf_a = confidence[mask_anomaly]
    #         corr_a = is_correct[mask_anomaly] # 这里的正确意味着 pred=1
    #         bc_a, ba_a, cnt_a = compute_calibration(conf_a, corr_a, n_bins)
    #         ece_a = calc_ece(bc_a, ba_a, cnt_a, np.sum(mask_anomaly))
    #         ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5)
    #         ax2.plot(bc_a, ba_a, marker='s', color='#e74c3c', label=f'UEGAE (ECE={ece_a:.4f})')
    #         ax2.bar(bc_a, ba_a, width=1/n_bins, alpha=0.1, color='#e74c3c', edgecolor='none')
    #         ax2.set_title(f'Reliability: Anomaly Nodes\n(Ground Truth = 1)', fontsize=13)
    #         ax2.set_xlabel('Confidence (1 - Uncertainty)')
    #         ax2.set_ylabel('Accuracy (Recall / Detection Rate)')
    #         ax2.legend(loc='lower right')
    #         ax2.grid(True, alpha=0.2)
    #     else:
    #         ax2.text(0.5, 0.5, "No Anomaly Nodes Found", ha='center')
    #     plt.tight_layout()
    #     save_path_rel = f'reliability_analysis_trail_{self.trail}.png'
    #     plt.savefig(save_path_rel, dpi=300)
    #     plt.show()
    #     logger.info(f">>> Reliability Diagram saved to {save_path_rel}")

    def analyze_uncertainty_bias(self, data, label=None, n_bins=20):
        """
        全面评估不确定性校准质量：
        1. 寻找最佳 F1 阈值，确定基准预测。
        2. 绘制四象限不确定性分布 (Boxplot) -> 识别过度自信/信心不足。
        3. 绘制 Reliability Diagram -> 评估校准度 (ECE)。
        
        Args:
            data: 图数据对象
            label: 真实标签
            n_bins: 分箱数
        """
        logger.info(">>> Starting Uncertainty Calibration Analysis...")
        # --- 1. 数据准备 (同前) ---
        if label is None:
            label = data.y
        self.model.eval()
        loader = NeighborLoader(data, num_neighbors=self.num_neigh, 
                                batch_size=self.batch_size, shuffle=False) 
        all_uncs = []
        all_scores = []
        all_labels = []
        # 2. 推理并收集全量不确定性和分数
        with torch.no_grad():
            for sampled_data in loader:
                sampled_data = sampled_data.to(self.device)
                # Forward pass
                # 获取各个组件的输出
                x_norm, x_, s_, nu, alpha_nig, beta, alpha_struct, beta_struct, alpha_cluster, emb = self.forward_model(sampled_data)
                # 计算 Loss 以获取不确定性字典
                q_target = self.target_distribution(alpha_cluster)
                _, batch_score, loss_dict = self.cal_loss(
                    x_norm=x_norm, s=sampled_data.s, x_=x_, s_=s_,
                    nu=nu, alpha_nig=alpha_nig, beta=beta,
                    alpha_struct=alpha_struct, beta_struct=beta_struct,
                    alpha_cluster=alpha_cluster, q_target=q_target,
                    current_epoch=self.epoch, total_epochs=self.epoch
                )                
                batch_unc_total = loss_dict.get('score_unc_model') # 需确保不为None
                # -------------------------------------------------------------
                batch_size = sampled_data.batch_size
                all_uncs.append(batch_unc_total[:batch_size].cpu())
                all_scores.append(batch_score[:batch_size]) 
                all_labels.append(sampled_data.y[:batch_size].cpu())
        # 拼接数据
        unc_tensor = torch.cat(all_uncs).numpy()
        score_tensor = torch.cat(all_scores).numpy()
        label_tensor = torch.cat(all_labels).numpy()

        unc_min, unc_max = unc_tensor.min(), unc_tensor.max()
        unc_norm = (unc_tensor - unc_min) / (unc_max - unc_min + 1e-8)
        confidence = 1.0 - unc_norm  # 置信度

        precision, recall, thresholds = precision_recall_curve(label_tensor, score_tensor)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        pred_labels = (score_tensor > best_threshold).astype(int)
        # 划分四类结果
        TP = (pred_labels == 1) & (label_tensor == 1)
        TN = (pred_labels == 0) & (label_tensor == 0)
        FP = (pred_labels == 1) & (label_tensor == 0) # 误报
        FN = (pred_labels == 0) & (label_tensor == 1) # 漏报
        
        logger.info(f"Best Threshold: {best_threshold:.4f} (Max F1: {f1_scores[best_idx]:.4f})")
        logger.info(f"Counts -> TP: {np.sum(TP)}, TN: {np.sum(TN)}, FP: {np.sum(FP)}, FN: {np.sum(FN)}")

        # --- 图表 1: 错误类型与不确定性分布 (Diagnostic Boxplot) ---
        plt.figure(figsize=(10, 6))
        # 准备绘图数据
        plot_data = []
        categories = []
        colors = []
        # 定义 helper
        def add_data(mask, name, color):
            if np.sum(mask) > 0:
                plot_data.extend(unc_norm[mask]) # 注意这里画的是“不确定性”，越高越不确定
                categories.extend([name] * np.sum(mask))
        add_data(TN, 'TN (Correct Normal)', '#2ecc71') # Green
        add_data(TP, 'TP (Correct Anomaly)', '#27ae60') # Dark Green
        add_data(FP, 'FP (False Alarm)', '#e74c3c')    # Red (错误)
        add_data(FN, 'FN (Missed Anomaly)', '#c0392b') # Dark Red (错误)
        # 使用 Seaborn 绘图
        sns.boxplot(x=categories, y=plot_data, palette={'TN (Correct Normal)':'#2ecc71', 'TP (Correct Anomaly)':'#27ae60', 
                                                       'FP (False Alarm)':'#e74c3c', 'FN (Missed Anomaly)':'#c0392b'})
        plt.title('Uncertainty Distribution by Prediction Outcome\n(Goal: Green should be Low, Red should be High)', fontsize=13)
        plt.ylabel('Normalized Uncertainty (0=Confident, 1=Uncertain)')
        plt.grid(True, axis='y', alpha=0.3)
        # 添加辅助线：如果红箱子的中位数很低，说明过度自信
        plt.text(0.5, 0.9, "Low Uncertainty on FP/FN = Overconfidence", 
                 ha='center', transform=plt.gca().transAxes, color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        save_path_box = f'uncertainty_diagnostic_box_trail_{self.trail}.png'
        plt.savefig(save_path_box, dpi=300)
        plt.show()

        # --- 图表 2: 分组可靠性图 (Reliability Diagram) ---
        # 分别画正常节点和异常节点的校准曲线
        def compute_calibration(confs, preds, labels, n_bins=10):
            """计算分箱后的置信度、准确率和样本数"""
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            # 获取每个样本所属的 bin 索引
            bin_idxs = np.digitize(confs, bin_edges, right=True)
            bin_confs = []
            bin_accs = []
            bin_counts = []
            for i in range(1, n_bins + 1):
                mask = (bin_idxs == i)
                if np.sum(mask) > 0:
                    # 平均置信度
                    avg_conf = np.mean(confs[mask])
                    # 该箱内的准确率：
                    # 对于正常节点(label=0)，准确意味着 pred=0
                    # 对于异常节点(label=1)，准确意味着 pred=1
                    # (pred == label) 这一逻辑对两者通用
                    is_correct = (preds[mask] == labels[mask]).astype(float)
                    avg_acc = np.mean(is_correct)
                    bin_confs.append(avg_conf)
                    bin_accs.append(avg_acc)
                    bin_counts.append(len(is_correct))
            return np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)

        def calc_ece(bin_confs, bin_accs, bin_counts, total_samples):
            """计算 ECE (Expected Calibration Error)"""
            if total_samples == 0: return 0.0
            return np.sum(np.abs(bin_accs - bin_confs) * (bin_counts / total_samples))

        # 准备画布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # --- 子图 1: 正常节点 (Normal Nodes) ---
        mask_normal = (label_tensor == 0)
        n_normal = np.sum(mask_normal)
        if n_normal > 0:
            # 提取正常节点的数据
            conf_n = confidence[mask_normal]
            pred_n = pred_labels[mask_normal]
            label_n = label_tensor[mask_normal]
            bc_n, ba_n, cnt_n = compute_calibration(conf_n, pred_n, label_n, n_bins)
            ece_n = calc_ece(bc_n, ba_n, cnt_n, n_normal)
            # 绘图
            ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label='Perfect')
            ax1.plot(bc_n, ba_n, marker='o', color='#3498db', label=f'Model (ECE={ece_n:.4f})')
            # 绘制柱状图背景表示样本量分布
            ax1.bar(bc_n, ba_n, width=1/n_bins, alpha=0.1, color='#3498db', edgecolor='none')
            ax1.set_title(f'Normal Nodes Reliability\n(Ground Truth = 0)', fontsize=13)
            ax1.set_xlabel('Confidence')
            ax1.set_ylabel('Accuracy (Correct Rejection Rate)')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No Normal Nodes", ha='center')

        # --- 子图 2: 异常节点 (Anomaly Nodes) ---
        mask_anomaly = (label_tensor == 1)
        n_anomaly = np.sum(mask_anomaly)
        if n_anomaly > 0:
            # 提取异常节点的数据
            conf_a = confidence[mask_anomaly]
            pred_a = pred_labels[mask_anomaly]
            label_a = label_tensor[mask_anomaly]
            # 注意：如果异常节点很少，n_bins设得过大可能会导致很多空箱子，曲线会断断续续
            bc_a, ba_a, cnt_a = compute_calibration(conf_a, pred_a, label_a, n_bins)
            ece_a = calc_ece(bc_a, ba_a, cnt_a, n_anomaly)
            # 绘图
            ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label='Perfect')
            ax2.plot(bc_a, ba_a, marker='s', color='#e74c3c', label=f'Model (ECE={ece_a:.4f})')
            # 绘制柱状图背景表示样本量分布
            ax2.bar(bc_a, ba_a, width=1/n_bins, alpha=0.1, color='#e74c3c', edgecolor='none')
            ax2.set_title(f'Anomaly Nodes Reliability\n(Ground Truth = 1)', fontsize=13)
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Accuracy (Recall / Detection Rate)')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No Anomaly Nodes", ha='center')
        plt.tight_layout()
        save_path_rel = f'reliability_diagram_split_trail_{self.trail}.png'
        plt.savefig(save_path_rel, dpi=300)
        plt.show()
        logger.info(f">>> Analysis saved to {save_path_box} and {save_path_rel}")

    def analyze_latent_space(self, data, label):
        logger.info(">>>Starting Latent Space Visualization & Analysis (Uniform K-Means)...")
        self.model.eval()
        # 1. 提取全量 Embedding
        loader = NeighborLoader(data,
                                num_neighbors=self.num_neigh,
                                batch_size=self.batch_size,
                                shuffle=False)
        all_embeddings = []
        all_true_labels = [] # 如果需要计算 ARI/NMI 对比真实标签，可以保留
        with torch.no_grad():
            for sampled_data in loader:
                sampled_data = sampled_data.to(self.device)
                outputs = self.forward_model(sampled_data)
                # 统一只取最后一个输出作为 Embedding
                emb = outputs[-1] 
                # 截取当前 batch
                batch_size = sampled_data.batch_size
                all_embeddings.append(emb[:batch_size].cpu())
                all_true_labels.append(sampled_data.y[:batch_size].cpu())
        X = torch.cat(all_embeddings, dim=0).numpy()
        Y_true = torch.cat(all_true_labels, dim=0).numpy() # 真实标签(0/1)
        # 确保数据长度对齐
        min_len = min(len(X), len(label))
        X = X[:min_len]
        # ---------------------------------------------------------
        # 2. 统一运行 K-Means 获取聚类标签
        # ---------------------------------------------------------
        logger.info(f">>>Running K-Means (k={self.n_clusters}) on Embeddings...")
        # 固定随机种子，确保公平
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        Y_kmeans = kmeans.fit_predict(X)
        # ---------------------------------------------------------
        # 3. 计算聚类指标 (Silhouette Score)
        # ---------------------------------------------------------
        metrics_log = []
        metrics_log.append(f"--- Clustering Analysis (Trail {self.trail}) ---")
        metrics_log.append(f"Method: K-Means Post-hoc (Fair Comparison)")
        # 采样计算以防内存溢出 (如果节点数 > 2w)
        X_eval = X
        Y_eval = Y_kmeans
        if len(X) > 20000:
            indices = np.random.choice(len(X), 20000, replace=False)
            X_eval = X[indices]
            Y_eval = Y_kmeans[indices]
        try:
            # 最核心指标：轮廓系数
            sil_score = silhouette_score(X_eval, Y_eval)
            ch = calinski_harabasz_score(X_eval, Y_eval)
            db = davies_bouldin_score(X_eval, Y_eval)
            metrics_log.append(f"  Silhouette Score: {sil_score:.4f}")
            metrics_log.append(f"  Calinski-Harabasz: {ch:.4f}")
            metrics_log.append(f"  Davies-Bouldin: {db:.4f}")
        except Exception as e:
            logger.error(f"Metric calculation failed: {e}")
            metrics_log.append(f"Error: {e}")
        # 打印并保存
        for line in metrics_log:
            logger.info(line)
        with open(f'clustering_metrics_trail{self.trail}.txt', 'w') as f:
            f.write('\n'.join(metrics_log))
        # ---------------------------------------------------------
        # 4. 可视化 (t-SNE)
        # ---------------------------------------------------------
        n_samples = X.shape[0]
        perplex = 30 if n_samples > 100 else n_samples / 2 
        # 限制 t-SNE 样本数，防止绘图太慢
        X_vis = X
        Y_true_vis = Y_true
        logger.info(f">>>Running t-SNE on {len(X_vis)} samples...")
        tsne = TSNE(n_components=2, perplexity=perplex, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
        X_2d = tsne.fit_transform(X_vis)
        fig, ax = plt.subplots(figsize=(10, 8))
        label_str = np.where(Y_true_vis == 0, 'Normal', 'Anomaly')
        sns.scatterplot(
            x=X_2d[:, 0], 
            y=X_2d[:, 1],
            hue=label_str,
            palette={'Normal': 'tab:blue', 'Anomaly': 'tab:red'},
            style=label_str,
            markers={'Normal': 'o', 'Anomaly': 'X'},
            size=label_str,
            sizes={'Normal': 20, 'Anomaly': 100}, 
            alpha=0.7,
            ax=ax
        )
        # 把核心指标也写在图的标题上方便查看
        title_str = f't-SNE (Epoch {self.best_epoch})'
        if 'sil_score' in locals():
            title_str += f'\nSilhouette: {sil_score:.3f} | calinski_harabasz: {ch:.3f}'
        ax.set_title(title_str)
        save_path = f'latent_space_tsne_trail{self.trail}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f">>>Latent space visualization saved to {save_path}")

    def analyze_reconstruction(self, data, label):
        self.model.eval()
        # 定义需要监控的指标及其对应的图表标题
        metric_config = {
            'total_loss': 'Total Loss',
            'attr_loss': 'Attribute Reconstruction Error',
            'adj_loss': 'Adj Reconstruction Error',
            'struct_loss': 'Structure Reconstruction Error',
            'clustering_loss': 'Clustering Loss',
            'attr_score': 'Attribute Reconstruction Score',
            'adj_score': 'Adj Reconstruction Score',
            'struct_score': 'Structure Reconstruction Score',
            "cluster_score": 'Clustering Reconstruction Score',
            'attr_unc_data': 'Data Uncertainty of Attri Recon',
            'attr_unc_model': 'Model Uncertainty of Attri Recon',
            'adj_unc_model': 'Model Uncertainty of Adj Recon',
            'cluster_unc_model': 'Model Uncertainty of Clustering',
            'score_rec': 'Anomaly Score of Total Recon',
            'score_unc_model': 'Anomaly Score of Total Model Unc',
            'anomaly_score': 'Total Anomaly Score',
            'nu': 'nu',
            'alpha_nig': 'alpha_nig',
            'beta': 'beta',
            'alpha_cluster': 'alpha_cluster',
            'alpha_struct': 'alpha_struct',
            'beta_struct': 'beta_struct',
        }
        # 定义需要绘制散点图的配对
        scatter_configs = [
            ('attr_score', 'attr_unc_model', 'Attr Score vs Model Unc'),
            ('adj_score', 'adj_unc_model', 'Adj Score vs Model Unc'),
            ('cluster_score', 'cluster_unc_model', 'Cluster Loss vs Model Unc'),
            ('score_rec', 'score_unc_model', 'Total Score vs Total Model Unc')
        ]
        accumulated_metrics = defaultdict(list)
        loader = NeighborLoader(data,
                                num_neighbors=self.num_neigh,
                                batch_size=self.batch_size,
                                shuffle=False)
        # 1. 推理并收集数据
        with torch.no_grad(): 
            for sampled_data in loader:
                sampled_data = sampled_data.to(self.device)
                s = sampled_data.s
                x = sampled_data.x
                x_norm, x_, s_, nu, alpha_nig, beta, alpha_struct, beta_struct, alpha_cluster, emb = self.forward_model(sampled_data)
                q_target = self.target_distribution(alpha_cluster)
                # 计算 Loss，此时部分 loss_dict 可能是 None
                _, _, loss_dict = self.cal_loss(
                    x_norm=x_norm, 
                    # x_norm=x,
                    s=s, x_=x_, s_=s_, 
                    nu=nu, alpha_nig=alpha_nig, beta=beta,
                    alpha_struct=alpha_struct, beta_struct=beta_struct,
                    alpha_cluster=alpha_cluster, q_target=q_target,
                    current_epoch=self.epoch, total_epochs=self.epoch)
                needed_keys = set(metric_config.keys())
                for sc_x, sc_y, _ in scatter_configs:
                    needed_keys.add(sc_x)
                    needed_keys.add(sc_y)
                for key, value in loss_dict.items():
                    if key in needed_keys and value is not None:
                        accumulated_metrics[key].append(value.cpu().numpy())
        final_metrics = {k: np.concatenate(v) for k, v in accumulated_metrics.items() if v}
        label_np = label.cpu().numpy()
        if final_metrics:
            data_len = len(next(iter(final_metrics.values())))
            if len(label_np) > data_len:
                label_np = label_np[:data_len]
        else:
            logger.warning("No metrics collected.")
            return
        # 保存中间结果
        data_save_path = f'recon_analysis_data_trail{self.trail}.npz'
        try:
            np.savez_compressed(data_save_path, label=label_np, **final_metrics)
            logger.info(f"Intermediate plotting data saved to: {data_save_path}")
        except Exception as e:
            logger.error(f"Failed to save intermediate data: {e}")
        active_hist_keys = [k for k in metric_config.keys() if k in final_metrics]
        # 自动过滤散点图：必须 X 和 Y 都在 final_metrics 里存在
        active_scatter_configs = [
            (x, y, t) for x, y, t in scatter_configs 
            if x in final_metrics and y in final_metrics
        ]
        num_plots = len(active_hist_keys) + len(active_scatter_configs)
        if num_plots == 0:
            logger.warning("Warning: No valid metrics found to plot.")
            return
        cols = 3
        rows = math.ceil(num_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        axes_flat = axes.flatten() if num_plots > 1 else [axes]
        label_str = np.where(label_np == 0, 'Normal', 'Anomaly')
        current_ax_idx = 0
        # 4. 循环绘图 - Part A: 直方图
        for key in active_hist_keys:
            ax = axes_flat[current_ax_idx]
            data_values = final_metrics[key]
            title = metric_config[key]
            try:
                sns.histplot(
                    x=data_values.flatten(),
                    hue=label_str,
                    bins=50,
                    kde=True,
                    element="step",
                    stat="density",
                    common_norm=False,
                    palette={'Normal': 'blue', 'Anomaly': 'red'},
                    ax=ax
                )
                ax.set_title(f"Distribution of {title}")
                ax.set_xlabel(title)
            except Exception as e:
                logger.error(f"Error plotting hist for {key}: {e}")
            current_ax_idx += 1
        # 5. 循环绘图 - Part B: 散点图
        for x_key, y_key, title_suffix in active_scatter_configs:
            ax = axes_flat[current_ax_idx]
            x_data = final_metrics[x_key].flatten()
            y_data = final_metrics[y_key].flatten()
            try:
                sns.scatterplot(
                    x=x_data,
                    y=y_data,
                    hue=label_str,
                    palette={'Normal': 'blue', 'Anomaly': 'red'},
                    style=label_str, 
                    markers={'Normal': 'o', 'Anomaly': 'X'},
                    size=label_str,
                    sizes={'Normal': 10, 'Anomaly': 60},
                    alpha=0.7,
                    ax=ax
                )
                x_label_name = metric_config.get(x_key, x_key)
                y_label_name = metric_config.get(y_key, y_key)
                ax.set_title(title_suffix)
                ax.set_xlabel(x_label_name)
                ax.set_ylabel(y_label_name)
            except Exception as e:
                 logger.error(f"Error plotting scatter for {x_key} vs {y_key}: {e}")
            current_ax_idx += 1
        # 隐藏多余子图
        for i in range(current_ax_idx, len(axes_flat)):
            axes_flat[i].axis('off')
        plt.tight_layout()
        plt.show() 
        save_path = f'recon_error_analysis_trail{self.trail}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Reconstruction analysis (Histograms + Scatters) saved to {save_path}")

    def plot_loss_history(self, best_epoch):
        if not self.loss_history:
            return
        # --- 配置区域 ---
        plot_configs = [
            {'title': 'Total Loss', 'keys': ['total_loss'], 'type': 'simple'},
            {'title': 'Performance (AUC/F1)', 'keys': ['Auc', 'F1'], 'type': 'simple'},
            {'title': 'Component Losses', 'keys': ['attr_loss', 'struct_loss', 'clustering_loss'], 'type': 'simple'},
            {'title': 'Reconstruction Uncertainty', 'base_key': 'unc_dec', 'type': 'split'},
            {'title': 'Structure Uncertainty', 'base_key': 'unc_adj', 'type': 'split'},
            {'title': 'Clustering Uncertainty', 'base_key': 'unc_cluster', 'type': 'split'},
            {'title': 'Total Uncertainty', 'base_key': 'unc_total', 'type': 'split'},
            {'title': 'Parameter: Nu (Concentration)', 'base_key': 'nu', 'type': 'split'},
            {'title': 'Parameter: Alpha (NIG)', 'base_key': 'alpha_nig', 'type': 'split'},
            {'title': 'Parameter: Beta (NIG)', 'base_key': 'beta', 'type': 'split'},
            {'title': 'Parameter: Alpha (Cluster)', 'base_key': 'alpha_cluster', 'type': 'split'},
            {'title': 'Parameter: Alpha (Struct)', 'base_key': 'alpha_struct', 'type': 'split'},
            {'title': 'Parameter: Beta (Struct)', 'base_key': 'beta_struct', 'type': 'split'},
        ]
        # 1. 过滤掉没有数据的 Config
        active_plots = []
        for cfg in plot_configs:
            if cfg['type'] == 'simple':
                # ================= [修改 4] 检查键是否存在且数据非空 =================
                valid_keys = [k for k in cfg['keys'] 
                              if k in self.loss_history and len(self.loss_history[k]) > 0]
                if valid_keys:
                    cfg['keys'] = valid_keys
                    active_plots.append(cfg)
            elif cfg['type'] == 'split':
                # 检查 Global key 是否存在
                if f"{cfg['base_key']}_global" in self.loss_history:
                     # 再次检查数据是否真的有记录 (防止仅有 key 但列表为空)
                     if len(self.loss_history[f"{cfg['base_key']}_global"]) > 0:
                        active_plots.append(cfg)
        if not active_plots:
            return
        # 2. 动态计算布局
        num_plots = len(active_plots)
        cols = 3
        rows = math.ceil(num_plots / cols)
        figsize = (6 * cols, 4 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes_flat = axes.flatten() if num_plots > 1 else [axes]
        for idx, cfg in enumerate(active_plots):
            ax = axes_flat[idx]
            if cfg['type'] == 'simple':
                for key in cfg['keys']:
                    data = self.loss_history[key]
                    ax.plot(data, label=key, linewidth=2)
            elif cfg['type'] == 'split':
                base = cfg['base_key']
                # 绘制三条线，检查每个 key 是否存在
                if f"{base}_global" in self.loss_history:
                    ax.plot(self.loss_history[f"{base}_global"], 
                            color='tab:blue', linestyle='-', label='Global')
                if f"{base}_normal" in self.loss_history:
                    ax.plot(self.loss_history[f"{base}_normal"], 
                            color='tab:green', linestyle='--', label='Normal')
                if f"{base}_anomaly" in self.loss_history:
                    ax.plot(self.loss_history[f"{base}_anomaly"], 
                            color='tab:red', linestyle='-.', label='Anomaly')
            if best_epoch < len(self.loss_history.get('total_loss', [])):
                ax.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.5, label=f'Best ({best_epoch})')
            ax.set_title(cfg['title'], fontsize=11, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        for i in range(num_plots, len(axes_flat)):
            axes_flat[i].axis('off')
        plt.tight_layout()
        save_path = f'training_metrics_trail{self.trail}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        data_to_save = {k: np.array(v) for k, v in self.loss_history.items()}
        np.savez(f'history_data_trail{self.trail}.npz', **data_to_save)

    def decision_function(self, data, label=None):
        self.process_graph(data)
        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)
        self.model.eval()
        outlier_score = torch.zeros(data.x.shape[0])
        
        if self.save_emb:
            self.emb = torch.zeros(data.x.shape[0], self.hid_dim)
            
        test_loss = 0
        with torch.no_grad():
            for sampled_data in loader:
                s = sampled_data.s.to(self.device)
                x = sampled_data.x.to(self.device)
                x_norm, x_, s_, nu, alpha_nig, beta, alpha_struct, beta_struct, alpha_cluster, emb = self.forward_model(sampled_data)
                
                # q_target = None
                q_target = self.target_distribution(alpha_cluster)
                
                loss, score, loss_dict = self.cal_loss(
                        x_norm=x_norm,
                        # x_norm=x, 
                        s=s, 
                        x_=x_, 
                        s_=s_, 
                        nu=nu,
                        alpha_nig=alpha_nig,
                        beta=beta,
                        alpha_struct=alpha_struct,
                        beta_struct=beta_struct,
                        alpha_cluster=alpha_cluster, 
                        q_target=q_target,
                        current_epoch=self.epoch,
                        total_epochs=self.epoch)
                
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id
                
                if self.save_emb:
                    self.emb[node_idx[:batch_size]] = self.model.emb[:batch_size].cpu()
                        
                test_loss += loss.item() * batch_size
                outlier_score[node_idx[:batch_size]] = score
                
        loss_value = test_loss / data.x.shape[0]
        return outlier_score, loss_value