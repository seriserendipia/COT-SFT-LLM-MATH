"""
GRPO训练奖励统计分析脚本

功能：
1. 加载训练过程中记录的奖励统计JSON日志
2. 生成可视化图表分析训练趋势
3. 输出关键统计指标

用法：
    # 基础用法（自动查找日志文件）
    python utilities_snippet/analyze_reward_logs.py
    
    # 指定日志文件路径
    python utilities_snippet/analyze_reward_logs.py --log_file qwen_grpo_lora_multigpu/reward_statistics.json
    
    # 保存图表到指定路径
    python utilities_snippet/analyze_reward_logs.py --output reward_analysis.png
    
    # 只显示统计信息，不绘图
    python utilities_snippet/analyze_reward_logs.py --no_plot

输出：
    - 终端：打印训练统计摘要
    - 图表：mean_reward, accuracy, std_reward 随训练步数的变化曲线
    - 文件：默认保存为 reward_analysis.png

依赖：
    pip install matplotlib numpy
"""

import json
import argparse
import os
from pathlib import Path

def load_reward_logs(log_file):
    """加载奖励统计JSON日志"""
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"日志文件不存在: {log_file}")
    
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    if not logs:
        raise ValueError("日志文件为空！")
    
    print(f"✅ 加载日志成功: {log_file}")
    print(f"📊 日志条目数: {len(logs)}")
    return logs


def print_summary_statistics(logs):
    """打印训练统计摘要"""
    import numpy as np
    
    # 提取数据
    mean_rewards = [log["mean_reward"] for log in logs]
    accuracies = [log["accuracy"] for log in logs]
    std_rewards = [log["std_reward"] for log in logs]
    
    # 计算整体统计
    print("\n" + "="*70)
    print("📈 训练统计摘要")
    print("="*70)
    
    print(f"\n🔢 奖励函数调用次数:")
    print(f"   总调用: {logs[-1]['call_count']}")
    print(f"   记录条目: {len(logs)}")
    
    print(f"\n📊 平均奖励 (Mean Reward):")
    print(f"   初始值: {mean_rewards[0]:.4f}")
    print(f"   最终值: {mean_rewards[-1]:.4f}")
    print(f"   最大值: {max(mean_rewards):.4f} (第{mean_rewards.index(max(mean_rewards))+1}次记录)")
    print(f"   最小值: {min(mean_rewards):.4f} (第{mean_rewards.index(min(mean_rewards))+1}次记录)")
    print(f"   变化: {mean_rewards[-1] - mean_rewards[0]:+.4f}")
    
    print(f"\n🎯 准确率 (Accuracy):")
    print(f"   初始: {accuracies[0]:.2%}")
    print(f"   最终: {accuracies[-1]:.2%}")
    print(f"   最高: {max(accuracies):.2%} (第{accuracies.index(max(accuracies))+1}次记录)")
    print(f"   最低: {min(accuracies):.2%} (第{accuracies.index(min(accuracies))+1}次记录)")
    print(f"   变化: {(accuracies[-1] - accuracies[0])*100:+.2f}%")
    
    print(f"\n📉 标准差 (Std Reward):")
    print(f"   初始: {std_rewards[0]:.4f}")
    print(f"   最终: {std_rewards[-1]:.4f}")
    print(f"   平均: {np.mean(std_rewards):.4f}")
    
    # 健康度检查
    print(f"\n🏥 训练健康度检查:")
    
    # 检查1: 准确率是否提升
    if accuracies[-1] > accuracies[0]:
        print("   ✅ 准确率提升")
    else:
        print("   ❌ 准确率下降或持平")
    
    # 检查2: 平均奖励是否提升
    if mean_rewards[-1] > mean_rewards[0]:
        print("   ✅ 平均奖励提升")
    else:
        print("   ❌ 平均奖励下降或持平")
    
    # 检查3: 是否存在梯度信号（标准差不为0）
    if np.mean(std_rewards) > 0.1:
        print("   ✅ 有效梯度信号 (std > 0.1)")
    else:
        print("   ⚠️  梯度信号弱 (std < 0.1)")
    
    # 检查4: 奖励分布是否合理（不是全0或全1）
    avg_acc = np.mean(accuracies)
    if 0.1 < avg_acc < 0.9:
        print("   ✅ 奖励分布合理 (10%-90%)")
    elif avg_acc < 0.1:
        print("   ❌ 奖励过低 (几乎全错)")
    else:
        print("   ⚠️  奖励过高 (几乎全对，可能过拟合)")
    
    print("="*70 + "\n")


def plot_reward_trends(logs, output_file="reward_analysis.png"):
    """绘制奖励统计趋势图"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("❌ 缺少matplotlib库，无法绘图")
        print("   请安装: pip install matplotlib")
        return
    
    # 提取数据
    calls = [log["call_count"] for log in logs]
    mean_rewards = [log["mean_reward"] for log in logs]
    accuracies = [log["accuracy"] for log in logs]
    std_rewards = [log["std_reward"] for log in logs]
    
    # 创建图表（3个子图）
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 子图1: 平均奖励
    axes[0].plot(calls, mean_rewards, linewidth=2, color='blue', marker='o', markersize=3)
    axes[0].set_title("Mean Reward over Training", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Reward Function Calls")
    axes[0].set_ylabel("Mean Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Neutral (0.5)')
    axes[0].legend()
    
    # 子图2: 准确率
    axes[1].plot(calls, [acc * 100 for acc in accuracies], linewidth=2, color='green', marker='s', markersize=3)
    axes[1].set_title("Accuracy (% Correct Answers) over Training", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Reward Function Calls")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])
    
    # 子图3: 标准差
    axes[2].plot(calls, std_rewards, linewidth=2, color='orange', marker='^', markersize=3)
    axes[2].set_title("Standard Deviation of Rewards over Training", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Reward Function Calls")
    axes[2].set_ylabel("Std Reward")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Weak Signal (0.1)')
    axes[2].legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 图表已保存: {output_file}")
    
    # 显示图表
    try:
        plt.show()
    except:
        print("⚠️  无法显示图表（可能是SSH环境），但已保存到文件")


def main():
    parser = argparse.ArgumentParser(
        description="分析GRPO训练过程中的奖励统计日志",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 默认用法（自动查找日志）
  python utilities_snippet/analyze_reward_logs.py
  
  # 指定日志文件
  python utilities_snippet/analyze_reward_logs.py --log_file path/to/reward_statistics.json
  
  # 只查看统计，不绘图
  python utilities_snippet/analyze_reward_logs.py --no_plot
  
  # 自定义输出图表路径
  python utilities_snippet/analyze_reward_logs.py --output my_analysis.png
        """
    )
    
    parser.add_argument(
        "--log_file", 
        type=str, 
        default="qwen_grpo_lora_multigpu/reward_statistics.json",
        help="奖励统计JSON日志文件路径 (默认: qwen_grpo_lora_multigpu/reward_statistics.json)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="reward_analysis.png",
        help="输出图表文件路径 (默认: reward_analysis.png)"
    )
    
    parser.add_argument(
        "--no_plot", 
        action="store_true",
        help="不生成图表，仅显示统计信息"
    )
    
    args = parser.parse_args()
    
    # 加载日志
    try:
        logs = load_reward_logs(args.log_file)
    except Exception as e:
        print(f"❌ 加载日志失败: {e}")
        return
    
    # 打印统计摘要
    print_summary_statistics(logs)
    
    # 绘制趋势图（如果需要）
    if not args.no_plot:
        plot_reward_trends(logs, args.output)


if __name__ == "__main__":
    main()
