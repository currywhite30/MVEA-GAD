import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def calculate_anomaly_score(error, uncertainty, w_rec=0.5, w_unc=0.5):
    """
    计算异常得分的函数
    Score = w_rec * Error * (2 * Uncertainty - 1)^2 + w_unc * Uncertainty
    """
    # 确定性系数 kappa: U=0或1时为1, U=0.5时为0
    # 这决定了 Error 项的权重是否生效
    certainty_coefficient = (2 * uncertainty - 1) ** 2
    
    # 组合得分
    score = w_rec * error * certainty_coefficient + w_unc * uncertainty
    return score

def plot_anomaly_heatmap():
    # --------------------------
    # 1. 参数设置
    # --------------------------
    resolution = 100
    e = np.linspace(0, 1, resolution) # Error (现在是 X轴)
    u = np.linspace(0, 1, resolution) # Uncertainty (现在是 Y轴)
    
    # 生成网格
    # meshgrid(x, y) -> 返回的 E 对应 x轴的变化, U 对应 y轴的变化
    E, U = np.meshgrid(e, u)
    
    # 权重设置
    W_REC = 0.5
    W_UNC = 0.5
    
    # --------------------------
    # 2. 计算得分矩阵
    # --------------------------
    Z = calculate_anomaly_score(E, U, w_rec=W_REC, w_unc=W_UNC)
    
    # --------------------------
    # 3. 定义关键点
    # --------------------------
    # (Error, Uncertainty, Label) - 数据定义不变
    points = [
        (1.0, 0.0, "(1) High E, Confident"),      # 右下
        (1.0, 1.0, "(2) High E, Unsure"),         # 右上
        (1.0, 0.5, "(3) High E, Mid-Conf"),       # 右中
        (0.0, 0.0, "(4) Low E, Confident"),       # 左下
        (0.0, 1.0, "(5) Low E, Unsure"),          # 左上
        (0.0, 0.5, "(6) Low E, Mid-Conf")         # 左中
    ]
    
    # --------------------------
    # 4. 绘图
    # --------------------------
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    # 注意：这里 x=E, y=U
    contour = plt.contourf(E, U, Z, levels=20, cmap='RdYlBu_r', alpha=0.9)
    cbar = plt.colorbar(contour)
    cbar.set_label('Anomaly Score', rotation=270, labelpad=15, fontsize=12)
    
    # 打印数值表格
    print(f"{'Scenario':<25} | {'Error (X)':<9} | {'Unc (Y)':<7} | {'Score':<5}")
    print("-" * 55)
    
    for e_val, u_val, label in points:
        score = calculate_anomaly_score(e_val, u_val, W_REC, W_UNC)
        print(f"{label:<25} | {e_val:<9.1f} | {u_val:<7.1f} | {score:<5.3f}")
        
        # 在图上画点 (注意 plt.scatter 的参数是 x, y)
        # 所以这里是 e_val, u_val
        plt.scatter(e_val, u_val, color='white', edgecolor='black', s=100, zorder=10)
        
        # 添加标签文本
        # 根据 Y轴 (Uncertainty) 的位置决定文字是在点上方还是下方，防止出界
        offset_y = -0.06 if u_val > 0.85 else 0.06
        
        plt.text(e_val, u_val + offset_y, 
                 f"{label}\nS={score:.2f}", 
                 color='black', fontsize=9, ha='center', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # 装饰图表
    plt.title(f'Anomaly Score Heatmap\nX=Error, Y=Uncertainty', fontsize=14)
    plt.xlabel('Reconstruction Error (E)', fontsize=12)
    plt.ylabel('Uncertainty (U)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 坐标轴范围
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig("anomaly_score_heatmap_transposed.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_anomaly_heatmap()