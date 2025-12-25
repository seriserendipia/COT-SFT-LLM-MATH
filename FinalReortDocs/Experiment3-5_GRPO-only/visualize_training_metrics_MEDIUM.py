"""
训练指标可视化脚本 - 实验3-5 (GRPO-only) - MEDIUM版本 ⭐推荐
字体配置：中号（适合论文双栏、PPT）

字体大小配置：
- 主图标题: 22pt
- 子图标题: 17pt
- XY轴标签: 15pt
- 刻度数字: 13pt
- 图例: 13pt
- 底部说明: 11pt
- 字体: Arial
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 13

# 配置
CSV_FILE = "training_metrics.csv"
OUTPUT_PNG = "training_metrics_visualization_MEDIUM.svg"
OUTPUT_DIR = "../../output_archive"

def plot_with_smooth(ax, x, y, color, label, window=50, alpha_raw=0.15):
    """绘制原始数据和平滑曲线"""
    # 原始数据（半透明）
    ax.plot(x, y, color=color, alpha=alpha_raw, linewidth=0.8)
    
    # 平滑数据（滚动平均）
    y_smooth = pd.Series(y).rolling(window=window, min_periods=1, center=False).mean()
    ax.plot(x, y_smooth, color=color, linewidth=2.5, label=label)
    
    return y_smooth

def main():
    print("="*70)
    print("📊 实验3-5 (GRPO-only) 训练可视化 - MEDIUM版本 ⭐推荐")
    print("="*70)
    
    # 读取CSV
    if not Path(CSV_FILE).exists():
        print(f"❌ 错误: 找不到 {CSV_FILE}")
        print("   请先运行 extract_training_metrics.py")
        return
    
    df = pd.read_csv(CSV_FILE)
    print(f"\n✅ 读取数据: {len(df)} 条记录")
    
    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
    
    # 颜色方案
    colors = sns.color_palette("husl", 5)
    
    # ==========================================================================
    # 主图：准确率 (Accuracy)
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    reward_col = 'rewards/reward_func/mean'
    if reward_col in df.columns:
        accuracy = df[reward_col] * 100
        plot_with_smooth(ax1, df['step'], accuracy, colors[0], 'Smoothed trend (rolling avg.)', window=50)
        
        ax1.set_xlabel('Training Step', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('(a) Model Performance During GRPO Training (GRPO-only, no SFT)', 
                     fontsize=22, fontweight='bold', pad=10)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # 去除上边框和右边框
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    
    # ==========================================================================
    # 子图 (b): 训练损失 (Training Loss)
    # ==========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    if 'loss' in df.columns:
        plot_with_smooth(ax2, df['step'], df['loss'], colors[1], 'Smoothed', window=50)
        ax2.set_xlabel('Training Step', fontweight='bold')
        ax2.set_ylabel('Loss', fontweight='bold')
        ax2.set_title('(b) Training Loss', fontsize=17, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    # ==========================================================================
    # 子图 (c): KL散度 (KL Divergence)
    # ==========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    if 'kl' in df.columns:
        plot_with_smooth(ax3, df['step'], df['kl'], colors[2], 'Smoothed', window=50)
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.6, linewidth=1.5, 
                   label='Warning threshold (0.05)')
        ax3.set_xlabel('Training Step', fontweight='bold')
        ax3.set_ylabel('KL Divergence', fontweight='bold')
        ax3.set_title('(c) KL Divergence from Base Model', fontsize=17, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
    
    # ==========================================================================
    # 子图 (d): 策略熵 (Policy Entropy)
    # ==========================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    if 'entropy' in df.columns:
        plot_with_smooth(ax4, df['step'], df['entropy'], colors[3], 'Smoothed', window=50)
        ax4.set_xlabel('Training Step', fontweight='bold')
        ax4.set_ylabel('Policy Entropy', fontweight='bold')
        ax4.set_title('(d) Policy Entropy', fontsize=17, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
    
    # ==========================================================================
    # 子图 (e): 梯度范数 (Gradient Norm)
    # ==========================================================================
    ax5 = fig.add_subplot(gs[1, 3])
    if 'grad_norm' in df.columns:
        plot_with_smooth(ax5, df['step'], df['grad_norm'], colors[4], 'Smoothed', window=50)
        ax5.axhline(y=1.0, color='orange', linestyle='--', alpha=0.6, linewidth=1.5,
                   label='High value warning (1.0)')
        ax5.set_xlabel('Training Step', fontweight='bold')
        ax5.set_ylabel('Gradient Norm', fontweight='bold')
        ax5.set_title('(e) Gradient Norm', fontsize=17, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
    
    # 添加全局说明
    fig.text(0.5, 0.02, 
             'Bold lines show smoothed trends (rolling average, window=50 steps); '
             'light shaded areas show raw per-batch measurements.',
             ha='center', fontsize=11, style='italic', color='gray')
    
    # 保存图表
    plt.savefig(OUTPUT_PNG, format='svg', bbox_inches='tight')
    print(f"\n✅ 图表已保存到: {OUTPUT_PNG}")
    print(f"   字体配置: MEDIUM ⭐ (主标题22pt, 子标题17pt, 轴标签15pt)")
    
    # 同时保存到output_archive
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    archive_path = Path(OUTPUT_DIR) / OUTPUT_PNG
    plt.savefig(archive_path, format='svg', bbox_inches='tight')
    print(f"📦 备份已保存到: {archive_path}")
    
    # 打印统计摘要
    print("\n" + "="*70)
    print("📈 GRPO-only Training Performance Summary")
    print("="*70)
    
    if reward_col in df.columns:
        initial_acc = df[reward_col].iloc[0] * 100
        final_acc = df[reward_col].iloc[-1] * 100
        peak_acc = df[reward_col].max() * 100
        mean_acc = df[reward_col].mean() * 100
        
        print(f"\n   Initial Accuracy:    {initial_acc:.1f}%")
        print(f"   Final Accuracy:      {final_acc:.1f}%")
        print(f"   Peak Accuracy:       {peak_acc:.1f}%")
        print(f"   Improvement:         +{final_acc-initial_acc:.1f}%")
    
    print("="*70)

if __name__ == "__main__":
    main()
