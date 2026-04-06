import torch
from torch_geometric.nn import GCN, GAT
from torch_geometric.utils import to_dense_adj
import logging
import math
import os
from src.model.decoder import BetaStructureDecoder
from src.model.functional import dominant_hega_recon_loss
from src.utils.utility import loggerInfo, validate_device
from src.eval.evaluate import Evaluator
from torch import nn
import numpy as np
from torch_geometric.loader import NeighborLoader
import time
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.manifold import TSNE
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


# ═════════════════════════════════════════════════════════════
#  论文绘图风格模板（博士论文 / 期刊投稿标准）
# ═════════════════════════════════════════════════════════════

# 中文宋体风格（Noto Serif CJK SC）
_CN_FONT_PATH = '/data/baisongran/content-2/fonts/NotoSerifCJKsc-Regular.otf'
_CN_BOLD_PATH = '/data/baisongran/content-2/fonts/NotoSerifCJKsc-Bold.otf'

# 英文 Times New Roman 风格（Latin Modern Roman，LaTeX 默认）
_EN_FONT_PATH = '/data/baisongran/content-2/fonts/lmroman10-regular.otf'
_EN_BOLD_PATH = '/data/baisongran/content-2/fonts/lmroman10-bold.otf'

# 希腊字母兜底（Latin Modern Roman 不含 λ 等希腊字形）
_GREEK_FONT_PATH = '/data/baisongran/content-2/fonts/FreeSerif.ttf'

# 各类备选路径（当主路径不存在时依次尝试）
_CN_FALLBACKS = [
    '/data/baisongran/content-2/fonts/NotoSerifCJK-Light.ttc',
    '/data/baisongran/content-2/fonts/NotoSansCJK-Regular.ttc',
    '/data/baisongran/content-2/fonts/wqy-zenhei.ttc',
]
_EN_FALLBACKS = [
    '/data/baisongran/content-2/fonts/LatinModernRoman.ttf',
    '/data/baisongran/content-2/fonts/LiberationSerif-Regular.ttf',
    '/data/baisongran/content-2/fonts/FreeSerif.ttf',
]
_EN_BOLD_FALLBACKS = [
    '/data/baisongran/content-2/fonts/LatinModernRoman-Bold.ttf',
    '/data/baisongran/content-2/fonts/LiberationSerif-Bold.ttf',
    '/data/baisongran/content-2/fonts/FreeSerifBold.ttf',
]


def _resolve_font(primary, fallbacks):
    """返回第一个实际存在的字体路径，否则返回 None。"""
    for p in [primary] + (fallbacks or []):
        if p and os.path.exists(p):
            return p
    return None


# ─────────────────────────────────────────────────────────────
#  全局字体 & 样式初始化（每次绑图前调用一次即可）
# ─────────────────────────────────────────────────────────────

def _setup_thesis_style():
    """
    配置博士论文风格的全局 matplotlib 参数。

    策略：
      1. 用文件路径直接注册字体，彻底避免名称匹配失败导致乱码。
      2. font.sans-serif 优先列表：宋体（中文）> Latin Modern Roman（英文）> 兜底。
      3. 字号整体放大，打印时清晰可读。
      4. pdf.fonttype=42：字体嵌入模式，符合学位论文/期刊投稿标准。
    """
    cn_reg  = _resolve_font(_CN_FONT_PATH,    _CN_FALLBACKS)
    cn_bold = _resolve_font(_CN_BOLD_PATH,    _CN_FALLBACKS)
    en_reg  = _resolve_font(_EN_FONT_PATH,    _EN_FALLBACKS)
    en_bold = _resolve_font(_EN_BOLD_PATH,    _EN_BOLD_FALLBACKS)
    greek   = _resolve_font(_GREEK_FONT_PATH, [])

    for path in [cn_reg, cn_bold, en_reg, en_bold, greek]:
        if path:
            fm.fontManager.addfont(path)

    # 收集注册后的字体名称（用于 rcParams 列表）
    registered_names = {}
    for path_key, path_val in [('cn', cn_reg), ('en', en_reg)]:
        if path_val:
            for f in fm.fontManager.ttflist:
                if f.fname == path_val and path_key not in registered_names:
                    registered_names[path_key] = f.name
                    break

    sans_list = []
    if 'cn' in registered_names:
        sans_list.append(registered_names['cn'])
    if 'en' in registered_names:
        sans_list.append(registered_names['en'])
    sans_list += ['Latin Modern Roman', 'Liberation Serif',
                  'FreeSerif', 'DejaVu Serif', 'DejaVu Sans']

    plt.rcParams.update({
        # ── 字体 ──────────────────────────────────────────────
        'font.family':        'sans-serif',
        'font.sans-serif':    sans_list,
        'axes.unicode_minus': False,

        # ── 字号（比默认大，论文印刷清晰） ───────────────────
        'font.size':          14,
        'axes.titlesize':     17,
        'axes.labelsize':     14,
        'xtick.labelsize':    13,
        'ytick.labelsize':    13,
        'legend.fontsize':    13,
        'figure.titlesize':   16,

        # ── 线条 ──────────────────────────────────────────────
        'axes.linewidth':     1.2,
        'grid.linewidth':     0.7,
        'lines.linewidth':    2.0,

        # ── 网格 ──────────────────────────────────────────────
        'axes.grid':          True,
        'grid.alpha':         0.35,
        'grid.linestyle':     '--',

        # ── 背景 ──────────────────────────────────────────────
        'figure.facecolor':   'white',
        'axes.facecolor':     '#f7f7f7',

        # ── PDF 字体嵌入（学位论文标准） ──────────────────────
        'pdf.fonttype':       42,
        'ps.fonttype':        42,

        # ── 保存 ──────────────────────────────────────────────
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.facecolor':  'white',
    })


# ─────────────────────────────────────────────────────────────
#  FontProperties 快捷工厂
# ─────────────────────────────────────────────────────────────

def _fp_cn(size=14, bold=False):
    """宋体风格 FontProperties，直接绑定文件路径，确保中文字符不乱码。"""
    path = _resolve_font(
        _CN_BOLD_PATH if bold else _CN_FONT_PATH,
        _CN_FALLBACKS
    )
    if path:
        return fm.FontProperties(
            fname=path, size=size,
            weight='bold' if bold else 'normal'
        )
    return fm.FontProperties(size=size, weight='bold' if bold else 'normal')


def _fp_en(size=14, bold=False):
    """Latin Modern Roman FontProperties（Times New Roman 风格）。"""
    path = _resolve_font(
        _EN_BOLD_PATH if bold else _EN_FONT_PATH,
        _EN_BOLD_FALLBACKS if bold else _EN_FALLBACKS
    )
    if path:
        return fm.FontProperties(
            fname=path, size=size,
            weight='bold' if bold else 'normal'
        )
    return fm.FontProperties(size=size, weight='bold' if bold else 'normal')


# ─────────────────────────────────────────────────────────────
#  论文配色方案
# ─────────────────────────────────────────────────────────────

THESIS_COLORS = {
    'black':      '#1a1a1a',
    'blue':       '#2166ac',
    'red':        '#d6604d',
    'green':      '#4dac26',
    'orange':     '#f4a742',
    'purple':     '#7b3294',
    'teal':       '#1b7837',
    'gray':       '#636363',
    'light_blue': '#74add1',
    'light_red':  '#f4a582',
}

# 多条曲线循环配色
CURVE_COLORS = [
    '#2166ac',   # 蓝
    '#d6604d',   # 红
    '#4dac26',   # 绿
    '#7b3294',   # 紫
    '#f4a742',   # 橙
    '#1b7837',   # 深绿
    '#b2182b',   # 深红
    '#4575b4',   # 钢蓝
]


# ─────────────────────────────────────────────────────────────
#  辅助绑图函数
# ─────────────────────────────────────────────────────────────

def _label_subplot(ax, label, size=14):
    """在子图左上角外侧添加 (a)(b)(c) 编号，使用英文衬线字体。"""
    ax.text(-0.10, 1.06, label,
            transform=ax.transAxes,
            fontproperties=_fp_en(size=size, bold=True),
            va='bottom', ha='right')


def _set_labels(ax, title=None, xlabel=None, ylabel=None,
                title_size=15, label_size=14):
    """用宋体 FontProperties 设置 title / xlabel / ylabel，避免乱码。"""
    if title:
        ax.set_title(title, fontproperties=_fp_cn(size=title_size, bold=True))
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=_fp_cn(size=label_size))
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=_fp_cn(size=label_size))


def _legend(ax, **kwargs):
    """统一设置图例字体为宋体。"""
    return ax.legend(prop=_fp_cn(size=13), **kwargs)


def _plot_split_curve(ax, loss_history, base_key, label_tag, best_epoch=None):
    """
    在 ax 上绘制 split 类型曲线（全局 / 正常 / 异常 三条线）。
    base_key: 如 'unc_dec'，会自动拼接 _global / _normal / _anomaly。
    """
    C = THESIS_COLORS
    key_global  = f'{base_key}_global'
    key_normal  = f'{base_key}_normal'
    key_anomaly = f'{base_key}_anomaly'

    if key_global in loss_history and len(loss_history[key_global]) > 0:
        ax.plot(loss_history[key_global],
                color=C['blue'], linestyle='-', linewidth=2.0, label='全局均值')
    if key_normal in loss_history and len(loss_history[key_normal]) > 0:
        ax.plot(loss_history[key_normal],
                color=C['green'], linestyle='--', linewidth=2.0, label='正常节点')
    if key_anomaly in loss_history and len(loss_history[key_anomaly]) > 0:
        ax.plot(loss_history[key_anomaly],
                color=C['red'], linestyle='-.', linewidth=2.0, label='异常节点')

    if best_epoch is not None:
        max_len = max(
            len(loss_history.get(key_global, [])),
            len(loss_history.get(key_normal, [])),
            len(loss_history.get(key_anomaly, [])),
        )
        # if best_epoch < max_len:
        #     ax.axvline(x=best_epoch, color=C['gray'], linestyle=':',
        #                alpha=0.6, linewidth=1.0,
        #                label=f'最优轮次 ({best_epoch})')


# ═════════════════════════════════════════════════════════════
#  鲁棒性测试辅助函数
# ═════════════════════════════════════════════════════════════

def perturb_attributes(data, noise_rate):
    """向所有节点注入高斯噪声，噪声强度由 noise_rate 决定。"""
    data_perturbed = copy.deepcopy(data)
    if noise_rate == 0:
        return data_perturbed
    num_nodes, num_features = data_perturbed.x.shape
    noise = torch.randn(num_nodes, num_features).to(data_perturbed.x.device) * noise_rate * 4
    data_perturbed.x += noise
    return data_perturbed


def perturb_structure(data, perturb_rate, attack_type='random'):
    """结构攻击：随机增删边策略。"""
    data_perturbed = copy.deepcopy(data)
    if perturb_rate == 0:
        return data_perturbed
    edge_index = data_perturbed.edge_index
    num_edges = edge_index.shape[1]
    num_perturb = int(num_edges * perturb_rate) // 2
    perm = torch.randperm(num_edges)
    edges_to_keep = edge_index[:, perm[num_perturb:]]
    num_nodes = data_perturbed.x.shape[0]
    row = torch.randint(0, num_nodes, (num_perturb,), device=data.x.device)
    col = torch.randint(0, num_nodes, (num_perturb,), device=data.x.device)
    mask = row != col
    row, col = row[mask], col[mask]
    added_edges = torch.stack([row, col], dim=0)
    new_edge_index = torch.cat([edges_to_keep, added_edges], dim=1)
    from torch_geometric.utils import to_undirected, coalesce
    new_edge_index = to_undirected(new_edge_index)
    new_edge_index = coalesce(new_edge_index)
    data_perturbed.edge_index = new_edge_index
    if hasattr(data_perturbed, 's'):
        data_perturbed.s = to_dense_adj(new_edge_index)[0]
    return data_perturbed


evaluator = Evaluator()


# ═════════════════════════════════════════════════════════════
#  模型组件
# ═════════════════════════════════════════════════════════════

from torch_geometric.nn import MessagePassing


class EnhancedGraphConv(MessagePassing):
    """
    带有特征级门控机制（Feature-wise Gating）的 EnhancedGraphConv。

    三路分解计算：
      1. W_self(x_i)                 — 自身语义保留
      2. W_nbr(mean_j x_j)           — 邻域平滑
      3. W_inter(x_i ⊙ gate)         — 门控二阶交互

    其中 gate = sigmoid(W_gate(mean_j x_j))。
    W_inter 零初始化，训练初期等价于原版带有自环的 GraphConv。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.W_self  = nn.Linear(in_channels, out_channels, bias=False)
        self.W_nbr   = nn.Linear(in_channels, out_channels, bias=False)
        self.W_gate  = nn.Linear(in_channels, in_channels)
        self.W_inter = nn.Linear(in_channels, out_channels, bias=False)
        self.bias    = nn.Parameter(torch.zeros(out_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.xavier_uniform_(self.W_nbr.weight)
        nn.init.xavier_uniform_(self.W_gate.weight)
        nn.init.zeros_(self.W_gate.bias)
        nn.init.zeros_(self.W_inter.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        h_agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        gate = torch.sigmoid(self.W_gate(h_agg))
        h_inter = x * gate
        out = self.W_self(x) + self.W_nbr(h_agg) + self.W_inter(h_inter) + self.bias
        return out

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'

class HeteroGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels=None,
                 dropout=0.0, act='relu', norm='layer_norm', jk=None, **kwargs):
        super().__init__()
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers      = num_layers
        self.out_channels    = out_channels if out_channels is not None else hidden_channels
        self.dropout         = dropout
        self.jk              = jk
 
        if isinstance(act, str):
            self.act = getattr(F, act) if hasattr(F, act) else getattr(torch, act)
        else:
            self.act = act
 
        self.convs = ModuleList()
        self.norms = ModuleList() if norm else None
 
        for i in range(num_layers):
            in_dim  = in_channels if i == 0 else hidden_channels
            out_dim = (self.out_channels
                       if (i == num_layers - 1 and out_channels is not None)
                       else hidden_channels)
            self.convs.append(EnhancedGraphConv(in_dim, out_dim))
 
            if norm:
                add_norm = (i < num_layers - 1) or (out_channels is None)
                if add_norm:
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
        x = self.ln1(x)
        self.x_norm = x
        self.emb = self.shared_encoder(x, edge_index)
        if self.n_clusters == 0:
            alpha_cluster = None
        else:
            alpha_cluster = self.latent_net(self.emb)
        gamma, nu, alpha_nig, beta = self.attr_decoder(self.emb, edge_index)
        alpha_struct, beta_struct = self.struct_decoder(self.emb, edge_index)
        s_expectation = alpha_struct / (alpha_struct + beta_struct)
        eps = 1e-6
        s_ = torch.log(s_expectation + eps) - torch.log(1 - s_expectation + eps)
        return gamma, nu, alpha_nig, beta, s_, alpha_struct, beta_struct, alpha_cluster, self.emb


# ═════════════════════════════════════════════════════════════
#  主模型类
# ═════════════════════════════════════════════════════════════

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

        if backbone == "HeteroGCN":
            self.backbone_class = HeteroGCN
        elif backbone == "GCN":
            self.backbone_class = GCN
        elif backbone == "GAT":
            self.backbone_class = GAT
        else:
            raise ValueError("Backbone is invalid!")

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
        self.gan = gan

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
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = neg_loss_weight
        self.kl_weight = kl_weight
        self.ranking_margin = ranking_margin
        self.neighbor_constrain_weight = neighbor_constrain_weight

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

        self.lr = lr
        self.epoch = epoch
        self.patience = patience
        self.device = validate_device(gpu)
        self.batch_size = batch_size
        self.num_neigh = num_neigh

        if type(num_neigh) is int:
            self.num_neigh = [num_neigh] * self.num_layers
        elif type(num_neigh) is list:
            if len(num_neigh) != self.num_layers:
                raise ValueError('Number of neighbors should have the same '
                                 'length as the number of layers.')
            self.num_neigh = num_neigh
        else:
            raise ValueError('Number of neighbors must be int or list of int')

        self.trail = trail
        self.verbose = verbose
        self.checkpoint_path = f"best_model_trail{trail}.pt"
        self.save_emb = save_emb
        self.emb = None
        self.compile_model = compile_model
        self.weight = weight
        self.contamination = contamination

        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % contamination)

        self.decision_score_ = None
        self.loss_history = defaultdict(list)

        logger.info(">>>DOMINANTHega Initialized Successfully.")

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)

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

        parameter_count = sum(p.numel() for p in model.parameters()
                              if p.requires_grad)
        logger.info(">>>Parameter number of DOMINANTHega: %d" % parameter_count)
        return model

    def process_graph(self, data):
        data.s = to_dense_adj(data.edge_index)[0]

    def forward_model(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        x_, nu, alpha_nig, beta, s_, alpha_struct, beta_struct, \
            alpha_cluster, emb = self.model(x, edge_index)
        x_norm = self.model.x_norm
        return (x_norm, x_, s_, nu, alpha_nig, beta,
                alpha_struct, beta_struct, alpha_cluster, emb)

    def cal_loss(self, x_norm, s, x_, s_, nu, alpha_nig, beta,
                 alpha_struct, beta_struct, alpha_cluster,
                 q_target, current_epoch, total_epochs):
        score, loss, loss_dict = self.model.loss_func(
            x=x_norm, x_=x_, s=s, s_=s_,
            nu=nu, alpha_nig=alpha_nig, beta=beta,
            alpha_struct=alpha_struct,
            beta_struct=beta_struct,
            alpha_cluster=alpha_cluster,
            q_target=q_target,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            attr_loss_weight=self.attr_loss_weight,
            has_annealing=self.has_annealing,
            attr_reg_base=self.attr_reg_base,
            attr_warmup_ratios=self.attr_warmup_ratios,
            clustering_weight=self.clustering_weight,
            struct_loss_weight=self.struct_loss_weight,
            bce_weight=self.bce_weight,
            ranking_weight=self.ranking_weight,
            dice_weight=self.dice_weight,
            misleading_weight=self.misleading_weight,
            pos_loss_weight=self.pos_loss_weight,
            neg_loss_weight=self.neg_loss_weight,
            kl_weight=self.kl_weight,
            ranking_margin=self.ranking_margin,
            neighbor_constrain_weight=self.neighbor_constrain_weight,
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
        p = alpha / torch.sum(alpha, dim=1, keepdim=True)
        f = torch.sum(p, dim=0, keepdim=True)
        weight = p**2 / f
        q = weight / torch.sum(weight, dim=1, keepdim=True)
        return q.detach()

    def _update_scalar_metric(self, name, value):
        self.loss_history[name].append(value)

    def _update_distribution_metric(self, name, tensor_values, labels):
        self.loss_history[f"{name}_global"].append(tensor_values.mean().item())
        mask_norm = (labels == 0)
        val_norm = tensor_values[mask_norm].mean().item() if mask_norm.any() else 0.0
        self.loss_history[f"{name}_normal"].append(val_norm)
        mask_anom = (labels == 1)
        val_anom = tensor_values[mask_anom].mean().item() if mask_anom.any() else 0.0
        self.loss_history[f"{name}_anomaly"].append(val_anom)

    # ═════════════════════════════════════════════════════════
    #  训练主循环
    # ═════════════════════════════════════════════════════════

    def fit(self, data, label=None):
        self.process_graph(data)
        self.num_nodes, self.in_dim = data.x.shape
        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]
        if label is None:
            label = data.y

        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)

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

                x_norm, x_, s_, nu, alpha_nig, beta, alpha_struct, \
                    beta_struct, alpha_cluster, emb = self.forward_model(
                        sampled_data)

                if alpha_cluster is not None:
                    if epoch % 5 == 0:
                        q_target = self.target_distribution(alpha_cluster)
                else:
                    q_target = None

                loss, score, loss_dict = self.cal_loss(
                    x_norm=x_norm, s=s, x_=x_, s_=s_,
                    nu=nu, alpha_nig=alpha_nig, beta=beta,
                    alpha_struct=alpha_struct,
                    beta_struct=beta_struct,
                    alpha_cluster=alpha_cluster,
                    q_target=q_target,
                    current_epoch=epoch,
                    total_epochs=self.epoch)

                raw_unc_dec = loss_dict.get('raw_epistemic_dec')
                raw_unc_adj = loss_dict.get('raw_epistemic_adj')
                raw_unc_cluster = loss_dict.get('raw_epistemic_cluster')
                if raw_unc_dec is not None:
                    raw_unc_dec = raw_unc_dec.detach().cpu()
                if raw_unc_adj is not None:
                    raw_unc_adj = raw_unc_adj.detach().cpu()
                if raw_unc_cluster is not None:
                    raw_unc_cluster = raw_unc_cluster.detach().cpu()
                valid_uncs = [u for u in [raw_unc_dec, raw_unc_adj,
                                          raw_unc_cluster] if u is not None]
                raw_unc_total = sum(valid_uncs) if valid_uncs else None

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
                    self.emb[node_idx[:batch_size]] = \
                        self.model.emb[:batch_size].cpu()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.decision_score_[node_idx[:batch_size]], loss_ = \
                    self.decision_function(data, label)

            full_epoch_labels = torch.cat(epoch_labels_buffer)
            for name, tensor_list in epoch_tensor_buffer.items():
                if tensor_list:
                    full_tensor = torch.cat(tensor_list)
                    self._update_distribution_metric(
                        name, full_tensor, full_epoch_labels)
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
                suffix=">>>src.model.hega.py|>>>DOMINANTHega.fit()"
                       "|>>>trail {}".format(self.trail),
                epoch=epoch,
                loss=loss_value,
                score=self.decision_score_,
                target=label,
                time=time.time() - start_time,
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
                    torch.save(self.model.state_dict(),
                               self.checkpoint_path)
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                if early_stop_count >= self.patience:
                    logger.info(
                        f">>>Early stopping at epoch {epoch:04d}, "
                        f"best AUC {best_auc:.4f} at epoch "
                        f"{self.best_epoch:04d}")
                    break

        self.model.load_state_dict(torch.load(self.checkpoint_path))
        test_outlier_score, test_loss_value = self.decision_function(
            data, label)
        self.decision_score_ = test_outlier_score

        test_result = loggerInfo(
            suffix=">>>src.model.hega.py|>>>DOMINANTHega.fit()"
                   "|>>>Final Result:",
            loss=test_loss_value,
            score=self.decision_score_,
            target=label,
            time=time.time() - start_time,
            verbose=self.verbose,
            train=False,
            deep=True,
            return_metrics=True
        )

        # ── 绑制四种论文图 ──────────────────────────────────
        self._plot_figure1_loss_uncertainty(self.best_epoch)
        self._plot_figure2_evidence_params(self.best_epoch)
        self._plot_figure3_test_distributions(data, data.y)
        self._plot_figure4_test_scatter(data, data.y)
        robustness_data = self.run_robustness_test(data)
        self._plot_figure5_robustness(robustness_data)
        # ── 保存训练历史数据 ─────────────────────────────────
        self._save_history_excel()

        return test_result

    # ═════════════════════════════════════════════════════════
    #  图 1：训练动态——损失与不确定性变化
    #  子图 (a) 损失曲线各成分
    #  子图 (b) 属性不确定性变化
    #  子图 (c) 聚类不确定性变化
    #  子图 (d) 结构不确定性变化
    # ═════════════════════════════════════════════════════════

    def _plot_figure1_loss_uncertainty(self, best_epoch):
        """图 1：训练过程中损失与不确定性的变化趋势。"""
        _setup_thesis_style()
        C = THESIS_COLORS

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle('训练动态——损失与不确定性变化',
                     fontproperties=_fp_cn(size=16, bold=True), y=1.01)

        # ── (a) 损失曲线各成分 ─────────────────────────────
        ax = axes[0, 0]
        _label_subplot(ax, '(a)')
        loss_keys_map = {
            # 'total_loss':      ('总损失',     C['black'],  '-'),
            'attr_loss':       ('属性损失',   C['blue'],   '--'),
            'struct_loss':     ('结构损失',   C['red'],    '-.'),
            'clustering_loss': ('聚类损失',   C['purple'], ':'),
        }
        for key, (label_cn, color, ls) in loss_keys_map.items():
            if key in self.loss_history and len(self.loss_history[key]) > 0:
                ax.plot(self.loss_history[key], color=color,
                        linestyle=ls, linewidth=2.0, label=label_cn)
        # if best_epoch < len(self.loss_history.get('total_loss', [])):
        #     ax.axvline(x=best_epoch, color=C['gray'], linestyle=':',
        #                alpha=0.6, linewidth=1.0,
        #                label=f'最优轮次 ({best_epoch})')
        _set_labels(ax, title='损失曲线各成分',
                    xlabel='训练轮次', ylabel='损失值')
        _legend(ax, loc='best')

        # ── (b) 属性不确定性变化 ────────────────────────────
        ax = axes[0, 1]
        _label_subplot(ax, '(b)')
        _plot_split_curve(ax, self.loss_history, 'unc_dec',
                          '属性不确定性', best_epoch)
        _set_labels(ax, title='属性重构不确定性变化',
                    xlabel='训练轮次', ylabel='不确定性')
        _legend(ax, loc='best')

        # ── (c) 聚类不确定性变化 ────────────────────────────
        ax = axes[1, 0]
        _label_subplot(ax, '(c)')
        _plot_split_curve(ax, self.loss_history, 'unc_cluster',
                          '聚类不确定性', best_epoch)
        _set_labels(ax, title='聚类不确定性变化',
                    xlabel='训练轮次', ylabel='不确定性')
        _legend(ax, loc='best')

        # ── (d) 结构不确定性变化 ────────────────────────────
        ax = axes[1, 1]
        _label_subplot(ax, '(d)')
        _plot_split_curve(ax, self.loss_history, 'unc_adj',
                          '结构不确定性', best_epoch)
        _set_labels(ax, title='结构不确定性变化',
                    xlabel='训练轮次', ylabel='不确定性')
        _legend(ax, loc='best')

        plt.tight_layout()
        save_path = f'fig1_loss_uncertainty_trail{self.trail}.pdf'
        fig.savefig(save_path, format='pdf')
        plt.close(fig)
        logger.info(f'[VIS] 已保存图 1（损失与不确定性变化） → {save_path}')

    # ═════════════════════════════════════════════════════════
    #  图 2：训练动态——证据参数变化
    #  子图 (a) 属性证据参数 nu
    #  子图 (b) 结构证据参数 alpha
    #  子图 (c) 结构证据参数 beta
    #  子图 (d) 聚类参数 alpha
    # ═════════════════════════════════════════════════════════

    def _plot_figure2_evidence_params(self, best_epoch):
        """图 2：训练过程中证据参数的变化趋势。"""
        _setup_thesis_style()
        C = THESIS_COLORS

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle('训练动态——证据参数变化',
                     fontproperties=_fp_cn(size=16, bold=True), y=1.01)

        param_configs = [
            ('(a)', axes[0, 0], 'nu',            '属性证据参数 nu'),
            ('(b)', axes[0, 1], 'alpha_struct',   '结构证据参数 alpha'),
            ('(c)', axes[1, 0], 'beta_struct',    '结构证据参数 beta'),
            ('(d)', axes[1, 1], 'alpha_cluster',  '聚类参数 alpha'),
        ]

        for label_tag, ax, base_key, title_cn in param_configs:
            _label_subplot(ax, label_tag)
            _plot_split_curve(ax, self.loss_history, base_key,
                              title_cn, best_epoch)
            _set_labels(ax, title=title_cn,
                        xlabel='训练轮次', ylabel='参数值')
            _legend(ax, loc='best')

        plt.tight_layout()
        save_path = f'fig2_evidence_params_trail{self.trail}.pdf'
        fig.savefig(save_path, format='pdf')
        plt.close(fig)
        logger.info(f'[VIS] 已保存图 2（证据参数变化） → {save_path}')

    # ═════════════════════════════════════════════════════════
    #  图 3：测试阶段——概率分布图（6 个子图）
    #  (a) 属性重构得分   (b) 邻接重构得分
    #  (c) 邻域重构得分   (d) 属性模型不确定性
    #  (e) 聚类模型不确定性 (f) 邻接模型不确定性
    # ═════════════════════════════════════════════════════════

    def _plot_figure3_test_distributions(self, data, label):
        """图 3：训练好的模型在测试时各指标的概率分布（平滑 KDE 曲线 + 阴影）。"""
        from scipy.stats import gaussian_kde

        _setup_thesis_style()
        C = THESIS_COLORS

        # ── 收集测试指标 ───────────────────────────────────
        final_metrics, label_np = self._collect_test_metrics(data, label)
        if not final_metrics:
            logger.warning('图 3：未收集到任何指标，跳过。')
            return

        fig, axes = plt.subplots(2, 3, figsize=(21, 11))
        fig.suptitle('测试阶段——各指标概率分布',
                     fontproperties=_fp_cn(size=16, bold=True), y=1.01)

        dist_configs = [
            ('(a)', axes[0, 0], 'attr_score',        '属性重构得分分布'),
            ('(b)', axes[0, 1], 'adj_score',         '结构重构得分分布'),
            ('(c)', axes[0, 2], 'struct_score',      '邻域重构得分分布'),
            ('(d)', axes[1, 0], 'attr_unc_model',    '属性模型不确定性分布'),
            ('(e)', axes[1, 1], 'cluster_unc_model', '聚类模型不确定性分布'),
            ('(f)', axes[1, 2], 'adj_unc_model',     '结构模型不确定性分布'),
        ]

        # KDE 采样点数（越多曲线越平滑）
        N_GRID = 500

        for label_tag, ax, key, title_cn in dist_configs:
            _label_subplot(ax, label_tag)
            if key in final_metrics:
                vals = final_metrics[key].flatten()
                vals_normal  = vals[label_np == 0]
                vals_anomaly = vals[label_np == 1]

                # 确定全局 x 轴范围（稍微外扩 5%）
                v_min = vals.min()
                v_max = vals.max()
                margin = (v_max - v_min) * 0.05 + 1e-8
                x_grid = np.linspace(v_min - margin, v_max + margin, N_GRID)

                # ── 正常节点 KDE 曲线 + 阴影 ──────────────
                if len(vals_normal) > 1 and np.std(vals_normal) > 1e-10:
                    kde_n = gaussian_kde(vals_normal, bw_method='scott')
                    y_normal = kde_n(x_grid)
                    ax.plot(x_grid, y_normal,
                            color=C['blue'], linewidth=2.2,
                            label='正常节点')
                    ax.fill_between(x_grid, 0, y_normal,
                                    color=C['blue'], alpha=0.18)

                # ── 异常节点 KDE 曲线 + 阴影 ──────────────
                if len(vals_anomaly) > 1 and np.std(vals_anomaly) > 1e-10:
                    kde_a = gaussian_kde(vals_anomaly, bw_method='scott')
                    y_anomaly = kde_a(x_grid)
                    ax.plot(x_grid, y_anomaly,
                            color=C['red'], linewidth=2.2,
                            label='异常节点')
                    ax.fill_between(x_grid, 0, y_anomaly,
                                    color=C['red'], alpha=0.18)

                ax.set_ylim(bottom=0)
                _set_labels(ax, title=title_cn,
                            xlabel='数值', ylabel='概率密度')
                _legend(ax, loc='best')
            else:
                ax.text(0.5, 0.5, '数据不可用',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontproperties=_fp_cn(size=13),
                        color=C['gray'])
                _set_labels(ax, title=f'{title_cn}（不适用）')

        plt.tight_layout()
        save_path = f'fig3_test_distributions_trail{self.trail}.pdf'
        fig.savefig(save_path, format='pdf')
        plt.close(fig)
        logger.info(f'[VIS] 已保存图 3（测试概率分布） → {save_path}')

    # ═════════════════════════════════════════════════════════
    #  图 4：测试阶段——散点图（4 个子图）
    #  (a) 属性得分 vs 属性不确定性
    #  (b) 结构得分 vs 结构不确定性
    #  (c) 聚类得分 vs 聚类不确定性
    #  (d) 总得分 vs 总不确定性
    # ═════════════════════════════════════════════════════════

    def _plot_figure4_test_scatter(self, data, label):
        """图 4：训练好的模型在测试时得分与不确定性的关系散点图。"""
        _setup_thesis_style()
        C = THESIS_COLORS

        final_metrics, label_np = self._collect_test_metrics(data, label)
        if not final_metrics:
            logger.warning('图 4：未收集到任何指标，跳过。')
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle('测试阶段——得分与不确定性散点图',
                     fontproperties=_fp_cn(size=16, bold=True), y=1.01)

        scatter_configs = [
            ('(a)', axes[0, 0], 'attr_score',    'attr_unc_model',
             '属性得分', '属性模型不确定性',
             '属性得分 vs 属性不确定性'),
            ('(b)', axes[0, 1], 'adj_score',     'adj_unc_model',
             '结构得分', '结构模型不确定性',
             '结构得分 vs 结构不确定性'),
            ('(c)', axes[1, 0], 'cluster_score', 'cluster_unc_model',
             '聚类得分', '聚类模型不确定性',
             '聚类得分 vs 聚类不确定性'),
            ('(d)', axes[1, 1], 'score_rec',     'score_unc_model',
             '总重构得分', '总模型不确定性',
             '总得分 vs 总不确定性'),
        ]

        for label_tag, ax, x_key, y_key, x_cn, y_cn, title_cn in scatter_configs:
            _label_subplot(ax, label_tag)
            if x_key in final_metrics and y_key in final_metrics:
                x_data = final_metrics[x_key].flatten()
                y_data = final_metrics[y_key].flatten()
                mask_normal  = (label_np == 0)
                mask_anomaly = (label_np == 1)
                # 正常节点
                ax.scatter(x_data[mask_normal], y_data[mask_normal],
                           c=C['blue'], marker='o', s=12,
                           alpha=0.50, label='正常节点', zorder=2)
                # 异常节点
                ax.scatter(x_data[mask_anomaly], y_data[mask_anomaly],
                           c=C['red'], marker='X', s=50,
                           alpha=0.70, label='异常节点', zorder=3)
                _set_labels(ax, title=title_cn,
                            xlabel=x_cn, ylabel=y_cn)
                _legend(ax, loc='best')
            else:
                ax.text(0.5, 0.5, '数据不可用',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontproperties=_fp_cn(size=13),
                        color=C['gray'])
                _set_labels(ax, title=f'{title_cn}（不适用）')

        plt.tight_layout()
        save_path = f'fig4_test_scatter_trail{self.trail}.pdf'
        fig.savefig(save_path, format='pdf')
        plt.close(fig)
        logger.info(f'[VIS] 已保存图 4（测试散点图） → {save_path}')

    def _plot_figure5_robustness(self, results):
        """
        图 5：鲁棒性分析——属性噪声与结构攻击下的性能变化。
        子图 (a) 属性扰动下 AUC 变化
        子图 (b) 结构攻击下 AUC 变化
        """
        _setup_thesis_style()
        C = THESIS_COLORS

        rates = [int(r * 100) for r in results['rates']]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('鲁棒性分析——扰动下的检测性能变化',
                     fontproperties=_fp_cn(size=16, bold=True), y=1.02)

        # ── (a) 属性扰动 ──────────────────────────────────
        ax = axes[0]
        _label_subplot(ax, '(a)')
        ax.plot(rates, results['attr_auc'],
                marker='o', color=C['blue'], linewidth=2.2,
                markersize=8, label='属性扰动 AUC', zorder=3)
        ax.fill_between(rates, results['attr_auc'],
                        alpha=0.15, color=C['blue'])
        ax.set_ylim(0.2, 1.0)
        ax.set_xticks(rates)
        _set_labels(ax, title='属性噪声扰动',
                    xlabel='扰动比例 (%)', ylabel='AUC')
        _legend(ax, loc='lower left')

        # ── (b) 结构攻击 ──────────────────────────────────
        ax = axes[1]
        _label_subplot(ax, '(b)')
        ax.plot(rates, results['struct_auc'],
                marker='s', color=C['red'], linewidth=2.2,
                markersize=8, label='结构攻击 AUC', zorder=3)
        ax.fill_between(rates, results['struct_auc'],
                        alpha=0.15, color=C['red'])
        ax.set_ylim(0.2, 1.0)
        ax.set_xticks(rates)
        _set_labels(ax, title='结构攻击扰动',
                    xlabel='扰动比例 (%)', ylabel='AUC')
        _legend(ax, loc='lower left')

        plt.tight_layout()
        save_path = f'fig5_robustness_trail{self.trail}.pdf'
        fig.savefig(save_path, format='pdf')
        plt.close(fig)
        logger.info(f'[VIS] 已保存图 5（鲁棒性分析） → {save_path}')

    # ═════════════════════════════════════════════════════════
    #  测试指标收集（图 3、图 4 共用）
    # ═════════════════════════════════════════════════════════

    def _collect_test_metrics(self, data, label):
        """
        在测试模式下前向推理，收集各节点的逐项指标。
        返回: (final_metrics: dict[str, np.ndarray], label_np: np.ndarray)
        """
        self.model.eval()
        loader = NeighborLoader(data,
                                num_neighbors=self.num_neigh,
                                batch_size=self.batch_size,
                                shuffle=False)

        needed_keys = {
            'attr_score', 'adj_score', 'struct_score', 'cluster_score',
            'attr_unc_model', 'adj_unc_model', 'cluster_unc_model',
            'score_rec', 'score_unc_model', 'anomaly_score',
        }

        accumulated = defaultdict(list)

        with torch.no_grad():
            for sampled_data in loader:
                sampled_data = sampled_data.to(self.device)
                x_norm, x_, s_, nu, alpha_nig, beta, alpha_struct, \
                    beta_struct, alpha_cluster, emb = self.forward_model(
                        sampled_data)
                if alpha_cluster is not None:
                    q_target = self.target_distribution(alpha_cluster)
                else:
                    q_target = None

                _, _, loss_dict = self.cal_loss(
                    x_norm=x_norm, s=sampled_data.s,
                    x_=x_, s_=s_,
                    nu=nu, alpha_nig=alpha_nig, beta=beta,
                    alpha_struct=alpha_struct,
                    beta_struct=beta_struct,
                    alpha_cluster=alpha_cluster,
                    q_target=q_target,
                    current_epoch=self.epoch,
                    total_epochs=self.epoch)

                batch_size = sampled_data.batch_size
                for key, value in loss_dict.items():
                    if key in needed_keys and value is not None:
                        if isinstance(value, torch.Tensor):
                            accumulated[key].append(
                                value[:batch_size].cpu().numpy())
                        else:
                            accumulated[key].append(
                                np.array(value)[:batch_size])

        final_metrics = {}
        for k, v in accumulated.items():
            if v:
                final_metrics[k] = np.concatenate(v)

        label_np = label.cpu().numpy()
        if final_metrics:
            data_len = len(next(iter(final_metrics.values())))
            if len(label_np) > data_len:
                label_np = label_np[:data_len]

        return final_metrics, label_np

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
    
    # ═════════════════════════════════════════════════════════
    #  训练历史 Excel 保存
    # ═════════════════════════════════════════════════════════

    def _save_history_excel(self):
        """将训练历史指标保存为 Excel（含 npz 备份）。"""
        import pandas as pd

        # npz 备份
        data_to_save = {k: np.array(v) for k, v in self.loss_history.items()}
        np.savez(f'history_data_trail{self.trail}.npz', **data_to_save)

        MAX_EPOCHS = 500
        split_bases = [
            'unc_dec', 'unc_adj', 'unc_cluster', 'unc_total',
            'nu', 'alpha_nig', 'beta',
            'alpha_cluster', 'alpha_struct', 'beta_struct'
        ]
        simple_keys = ['total_loss', 'Auc', 'F1',
                       'attr_loss', 'struct_loss', 'clustering_loss']

        row_dict = {}
        for k in simple_keys:
            if k in self.loss_history and len(self.loss_history[k]) > 0:
                row_dict[k] = np.array(
                    self.loss_history[k])[:MAX_EPOCHS]
        for base in split_bases:
            for suffix in ('global', 'normal', 'anomaly'):
                col = f'{base}_{suffix}'
                if col in self.loss_history and \
                        len(self.loss_history[col]) > 0:
                    row_dict[col] = np.array(
                        self.loss_history[col])[:MAX_EPOCHS]

        if row_dict:
            max_len = min(max(len(v) for v in row_dict.values()),
                          MAX_EPOCHS)
            aligned = {}
            for col, arr in row_dict.items():
                if len(arr) < max_len:
                    pad = np.full(max_len - len(arr), np.nan)
                    aligned[col] = np.concatenate([arr, pad])
                else:
                    aligned[col] = arr[:max_len]
            df_history = pd.DataFrame(aligned)
            df_history.index.name = 'epoch'
            excel_path = f'training_history_trail{self.trail}.xlsx'
            df_history.to_excel(excel_path)
            logger.info(f">>>训练历史 Excel 已保存: {excel_path}  "
                        f"shape={df_history.shape}")

    # ═════════════════════════════════════════════════════════
    #  推理 / 决策函数
    # ═════════════════════════════════════════════════════════

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
                x_norm, x_, s_, nu, alpha_nig, beta, alpha_struct, \
                    beta_struct, alpha_cluster, emb = self.forward_model(
                        sampled_data)

                if alpha_cluster is not None:
                    q_target = self.target_distribution(alpha_cluster)
                else:
                    q_target = None

                loss, score, loss_dict = self.cal_loss(
                    x_norm=x_norm, s=s, x_=x_, s_=s_,
                    nu=nu, alpha_nig=alpha_nig, beta=beta,
                    alpha_struct=alpha_struct,
                    beta_struct=beta_struct,
                    alpha_cluster=alpha_cluster,
                    q_target=q_target,
                    current_epoch=self.epoch,
                    total_epochs=self.epoch)

                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id

                if self.save_emb:
                    self.emb[node_idx[:batch_size]] = \
                        self.model.emb[:batch_size].cpu()

                test_loss += loss.item() * batch_size
                outlier_score[node_idx[:batch_size]] = score

        loss_value = test_loss / data.x.shape[0]
        return outlier_score, loss_value