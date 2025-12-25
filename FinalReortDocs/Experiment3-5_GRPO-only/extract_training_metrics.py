"""
训练指标提取脚本 - 实验3-5 (GRPO-only)
从SLURM日志文件中提取训练指标并保存为CSV格式

日志文件: slurm-4871595-3-5-grpo-train-full-set.out
实验配置: GRPO-only (无SFT预训练)
"""

import re
import json
import pandas as pd
from pathlib import Path

# 配置
LOG_FILE = "../../output_archive/slurm-4871595-3-5-grpo-train-full-set.out"
OUTPUT_CSV = "training_metrics.csv"

def extract_metrics_from_log(log_file_path):
    """
    从日志文件中提取训练指标
    
    日志格式示例:
    {'loss': -0.0626, 'grad_norm': 0.4672642946243286, 
     'learning_rate': 1.3368983957219251e-07, ...}
    """
    metrics_list = []
    
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 匹配包含指标的字典行
    pattern = r"\{'loss':.*?\}"
    matches = re.findall(pattern, content)
    
    print(f"找到 {len(matches)} 条训练记录")
    
    for match in matches:
        try:
            # 将字符串转换为字典
            # 替换单引号为双引号以符合JSON格式
            json_str = match.replace("'", '"')
            metrics = json.loads(json_str)
            metrics_list.append(metrics)
        except json.JSONDecodeError as e:
            # 如果JSON解析失败，尝试用eval
            try:
                metrics = eval(match)
                metrics_list.append(metrics)
            except Exception as e:
                print(f"跳过无效记录: {e}")
                continue
    
    return metrics_list

def main():
    print("="*70)
    print("🔍 实验3-5 (GRPO-only) 训练指标提取")
    print("="*70)
    
    # 检查日志文件是否存在
    log_path = Path(LOG_FILE)
    if not log_path.exists():
        print(f"❌ 错误: 找不到日志文件 {LOG_FILE}")
        return
    
    print(f"\n📂 读取日志文件: {LOG_FILE}")
    
    # 提取指标
    metrics_list = extract_metrics_from_log(log_path)
    
    if not metrics_list:
        print("❌ 未找到训练指标数据")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(metrics_list)
    
    # 添加step列（如果不存在）
    if 'step' not in df.columns:
        df.insert(0, 'step', range(len(df)))
    
    print(f"\n📊 提取的指标数量: {len(df)} 条记录")
    print(f"📋 指标列数: {len(df.columns)} 列")
    
    # 显示前几行
    print("\n预览前5行数据:")
    print(df.head())
    
    # 保存为CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 数据已保存到: {OUTPUT_CSV}")
    
    # 打印统计摘要
    print("\n" + "="*70)
    print("📈 训练摘要统计")
    print("="*70)
    
    if 'loss' in df.columns:
        print(f"Loss: {df['loss'].iloc[0]:.4f} → {df['loss'].iloc[-1]:.4f}")
    
    if 'learning_rate' in df.columns:
        print(f"Learning Rate: {df['learning_rate'].iloc[0]:.2e} → {df['learning_rate'].iloc[-1]:.2e}")
    
    if 'rewards/reward_func/mean' in df.columns:
        initial_acc = df['rewards/reward_func/mean'].iloc[0] * 100
        final_acc = df['rewards/reward_func/mean'].iloc[-1] * 100
        peak_acc = df['rewards/reward_func/mean'].max() * 100
        mean_acc = df['rewards/reward_func/mean'].mean() * 100
        print(f"Accuracy (reward mean):")
        print(f"  Initial: {initial_acc:.1f}%")
        print(f"  Final: {final_acc:.1f}%")
        print(f"  Peak: {peak_acc:.1f}%")
        print(f"  Mean: {mean_acc:.1f}%")
    
    if 'kl' in df.columns:
        print(f"KL Divergence: {df['kl'].iloc[0]:.6f} → {df['kl'].iloc[-1]:.6f}")
        print(f"  Max KL: {df['kl'].max():.6f}")
    
    if 'entropy' in df.columns:
        print(f"Entropy: {df['entropy'].iloc[0]:.4f} → {df['entropy'].iloc[-1]:.4f}")
    
    if 'grad_norm' in df.columns:
        print(f"Gradient Norm: Mean={df['grad_norm'].mean():.4f}, Max={df['grad_norm'].max():.4f}")
    
    print("="*70)

if __name__ == "__main__":
    main()
