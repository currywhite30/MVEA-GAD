#!/bin/bash

# ==========================================
# 0. 基础配置
# ==========================================
# 自动生成本次批次目录名称
BATCH_DIR="logs/batch_$(date +%Y%m%d_%H%M%S)"

echo "========================================================"
echo "🚀 开始并行超参数搜索"
echo "📂 本次实验日志将隔离存储于: $BATCH_DIR"
echo "========================================================"

# 创建目录
mkdir -p "$BATCH_DIR"

# ==========================================
# 2. 定义启动函数 (已修复)
# ==========================================
run_exp() {
    # 获取前三个固定参数
    local gpu=$1
    local name=$2
    local dataset=$3
    
    # 移除前三个参数，剩下的全部作为 hydra 的 override 参数
    shift 3
    local args="$@"

    echo "Runtime info: 启动 $name (GPU $gpu)..."
    
    # 打印到控制台以便调试
    # echo "  Params: $args"

    # 关键修改：
    # 1. 使用 hydra.run.dir 隔离日志
    # 2. 重定向标准输出和错误到日志文件，避免前台混乱
    nohup python main.py \
        dataset=$dataset \
        exp_name="$name" \
        model.hid_dim=32 \
        model.z_dim=32 \
        hydra.run.dir="${BATCH_DIR}/${name}" \
        gpu=$gpu \
        $args \
        > "${BATCH_DIR}/${name}.log" 2>&1 & 
}

# ==========================================
# 3. 启动 7 组实验 (已修复换行符)
# ==========================================

# ⚠️ 注意：请确保你有足够的显卡。
# 如果显卡较少，需要修改下方的 gpu 编号 (例如用 % 运算循环使用 gpu)

# (1) attr + 退火 reg
run_exp 0 "exp1_attr_only" "books" \
    model.attr_loss_weight=1.0 model.has_annealing=true \
    model.clustering_weight=0.0 model.struct_loss_weight=0.0 \
    model.neighbor_constrain_weight=0.0 model.anomaly_func_type="add"

# (2) attr + 退火 reg + clustering
run_exp 1 "exp2_attr_cluster" "books" \
    model.attr_loss_weight=1.0 model.has_annealing=true \
    model.clustering_weight=1.0 model.struct_loss_weight=0.0 \
    model.neighbor_constrain_weight=0.0 model.anomaly_func_type="add"

# (3) struct + reg (BCE)
run_exp 2 "exp3_struct_bce" "books" \
    model.attr_loss_weight=0.0 model.struct_loss_weight=1.0 \
    model.bce_weight=1.0 model.ranking_weight=0.0 \
    model.dice_weight=0.0 model.neighbor_constrain_weight=0.0 \
    model.anomaly_func_type="add"

# (4) ranking-dice + reg
run_exp 3 "exp4_struct_rankdice" "books" \
    model.attr_loss_weight=0.0 model.struct_loss_weight=1.0 \
    model.bce_weight=0.0 model.ranking_weight=1.0 \
    model.dice_weight=1.0 model.neighbor_constrain_weight=0.0 \
    model.anomaly_func_type="add"

# (5) attr + struct (无邻域约束)
run_exp 4 "exp5_combined_no_neighbor" "books" \
    model.attr_loss_weight=1.0 model.has_annealing=true \
    model.clustering_weight=1.0 model.struct_loss_weight=1.0 \
    model.bce_weight=0.0 model.ranking_weight=1.0 \
    model.dice_weight=1.0 model.neighbor_constrain_weight=0.0 \
    model.anomaly_func_type="add"

# (6) attr + struct + 邻域约束 (Weight 0.1)
run_exp 5 "exp6_combined_neighbor" "books" \
    model.attr_loss_weight=1.0 model.has_annealing=true \
    model.clustering_weight=1.0 model.struct_loss_weight=1.0 \
    model.bce_weight=0.0 model.ranking_weight=1.0 \
    model.dice_weight=1.0 model.neighbor_constrain_weight=0.1 \
    model.anomaly_func_type="add"

# (7) attr + struct + 邻域约束 + Gate
run_exp 6 "exp7_combined_gate" "books" \
    model.attr_loss_weight=1.0 model.has_annealing=true \
    model.clustering_weight=1.0 model.struct_loss_weight=1.0 \
    model.bce_weight=0.0 model.ranking_weight=1.0 \
    model.dice_weight=1.0 model.neighbor_constrain_weight=0.1 \
    model.anomaly_func_type="gate"

echo "✅ 所有任务已后台启动。请使用 'jobs' 或 'ps -ef | grep python' 查看状态。"