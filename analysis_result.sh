BATCH_DIR="logs/batch_20260127_201602"

echo "⏳ 所有任务已后台启动，正在运行中..."
echo "⏳ 等待实验结束 (这可能需要一段时间)..."
wait
echo "✅ 所有实验已结束。"
echo "🔎 正在分析目录 [$BATCH_DIR] 中的结果..."

# 嵌入 Python 脚本，并传入 BATCH_DIR 作为参数
python3 -c "
import os
import sys
import re
import glob

# 获取 Shell 传入的本次批次路径
target_dir = sys.argv[1]

# 假设 main.py 输出格式为: Final Result AUC: 0.1234
pattern = re.compile(r'Final Result AUC:\s*([0-9\.]+)')

best_auc = -1.0
best_exp_name = None
best_log_file = None

# 只扫描 target_dir 下的 .log 文件
search_path = os.path.join(target_dir, '**', '*.log')
# recursive=True 确保能找到子文件夹里的日志
log_files = glob.glob(search_path, recursive=True)

if not log_files:
    print(f'❌ 警告: 在 {target_dir} 下没有找到任何 .log 文件。')
    sys.exit(1)

for log_file in log_files:
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            match = pattern.search(content)
            if match:
                auc = float(match.group(1))
                
                # 提取实验名 (即 batch_xxx 下一级的文件夹名)
                rel_path = os.path.relpath(log_file, target_dir)
                exp_name = rel_path.split(os.sep)[0]
                
                if auc > best_auc:
                    best_auc = auc
                    best_exp_name = exp_name
                    best_log_file = log_file
    except Exception as e:
        print(f'读取错误: {log_file} - {e}')

print('\n' + '='*60)
if best_exp_name:
    print(f'🏆 本次批次最佳实验: 【 {best_exp_name} 】')
    print(f'📈 最高 AUC:        {best_auc}')
    print(f'📂 对应日志:        {best_log_file}')
    print('='*60 + '\n')
    
    # 打印对应配置提示
    print('💡 配置回顾:')
    if 'exp1' in best_exp_name: print('   配置: (1) 只用 attr + 退火 reg')
    elif 'exp2' in best_exp_name: print('   配置: (2) attr + 退火 reg + clustering')
    elif 'exp3' in best_exp_name: print('   配置: (3) 只用 struct + reg (BCE)')
    elif 'exp4' in best_exp_name: print('   配置: (4) 只用 ranking-dice + reg')
    elif 'exp5' in best_exp_name: print('   配置: (5) attr + struct (无邻域约束)')
    elif 'exp6' in best_exp_name: print('   配置: (6) attr + struct + 邻域约束 (Weight 0.1)')
    elif 'exp7' in best_exp_name: print('   配置: (7) attr + struct + 邻域约束 + Gate')
else:
    print('❌ 未在本次批次的日志中提取到 AUC 结果，请检查 main.py 是否正确输出了 \"Final Result AUC: ...\"')
print('='*60)
" "$BATCH_DIR"