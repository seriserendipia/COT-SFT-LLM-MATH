"""
Visualize training metrics from extracted CSV data (Academic Paper Style)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_with_smooth(ax, x, y, color, label, window=50):
    """
    Plot data with smoothing: raw data in light color + smoothed trend in bold
    
    Args:
        ax: matplotlib axis
        x: x-axis data (steps)
        y: y-axis data (metric values)
        color: line color
        label: legend label
        window: rolling window size for smoothing
    """
    # 1. Plot raw data (light color, high transparency)
    ax.plot(x, y, color=color, alpha=0.15, linewidth=1, label='_nolegend_')
    
    # 2. Calculate and plot smoothed data (bold line)
    y_smooth = pd.Series(y).rolling(window=window, min_periods=1).mean()
    ax.plot(x, y_smooth, color=color, linewidth=2.5, label=label)

def plot_training_metrics(csv_path):
    """
    Create academic-style visualization plots for training metrics
    """
    # Read data
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} training steps")
    
    # ==========================================
    # Set academic paper style
    # ==========================================
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 17
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['axes.linewidth'] = 1.2
    
    # Determine reward column
    if 'rewards/reward_func/mean' in df.columns:
        reward_col = 'rewards/reward_func/mean'
    else:
        reward_col = 'reward'
    
    # Calculate smoothing window (2% of total steps)
    smooth_window = max(10, int(len(df) * 0.02))
    
    # ==========================================
    # Create custom layout: Top row for main metric, bottom row for diagnostics
    # ==========================================
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1], hspace=0.35, wspace=0.45)
    
    # --- Plot 1: Training Reward/Accuracy (Main Focus) - Spans entire top row ---
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot raw data first (light background)
    ax1.plot(df['step'], df[reward_col] * 100, 
             color='#2ca02c', alpha=0.15, linewidth=1, 
             label='Raw per-batch accuracy')
    
    # Plot smoothed data (bold foreground)
    y_smooth = pd.Series(df[reward_col] * 100).rolling(window=smooth_window, min_periods=1).mean()
    ax1.plot(df['step'], y_smooth, 
             color='#2ca02c', linewidth=2.5, 
             label='Smoothed trend (rolling avg.)')
    
    # Optional: Add meaningful baseline (e.g., SFT-only performance or target accuracy)
    # Uncomment and adjust the following line if you have a meaningful baseline:
    # ax1.axhline(y=58.0, color='#ff7f0e', linestyle='--', alpha=0.7, linewidth=1.5, 
    #             label='SFT baseline (before GRPO)')
    
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=17)
    ax1.set_xlabel('Training Steps', fontweight='bold', fontsize=17)
    ax1.set_title('(a) Model Performance During GRPO (SFT-based) Training', 
                  fontweight='bold', fontsize=26, pad=15)
    
    # Improved legend with clear explanation
    legend = ax1.legend(loc='lower right', frameon=True, fontsize=15,
                       facecolor='white', edgecolor='gray', framealpha=0.9)
    legend.get_frame().set_linewidth(1.0)
    
    ax1.grid(axis='y', linestyle='--', alpha=0.2, zorder=0)
    
    # Remove top and right spines (despine)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Plot 2: Loss ---
    ax2 = fig.add_subplot(gs[1, 0])
    plot_with_smooth(ax2, df['step'], df['loss'], 
                     color='#1f77b4', label='_nolegend_', window=smooth_window)
    ax2.set_ylabel('Loss', fontsize=17, fontweight='bold')
    ax2.set_xlabel('Training Steps', fontsize=15)
    ax2.set_title('(b) Training Loss', fontsize=18, fontweight='bold', pad=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', linestyle='--', alpha=0.2, zorder=0)
    
    # --- Plot 3: KL Divergence ---
    ax3 = fig.add_subplot(gs[1, 1])
    plot_with_smooth(ax3, df['step'], df['kl'], 
                     color='#d62728', label='_nolegend_', window=smooth_window)
    ax3.set_ylabel('KL Divergence', fontsize=17, fontweight='bold')
    ax3.set_xlabel('Training Steps', fontsize=15)
    ax3.set_title('(c) KL Divergence', fontsize=18, fontweight='bold', pad=12)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', linestyle='--', alpha=0.2, zorder=0)
    
    # --- Plot 4: Entropy ---
    ax4 = fig.add_subplot(gs[1, 2])
    plot_with_smooth(ax4, df['step'], df['entropy'], 
                     color='#9467bd', label='_nolegend_', window=smooth_window)
    ax4.set_ylabel('Policy Entropy', fontsize=17, fontweight='bold')
    ax4.set_xlabel('Training Steps', fontsize=15)
    ax4.set_title('(d) Policy Entropy', fontsize=18, fontweight='bold', pad=12)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='y', linestyle='--', alpha=0.2, zorder=0)
    
    # --- Plot 5: Gradient Norm ---
    ax5 = fig.add_subplot(gs[1, 3])
    plot_with_smooth(ax5, df['step'], df['grad_norm'], 
                     color='#ff7f0e', label='_nolegend_', window=smooth_window)
    ax5.set_ylabel('Gradient Norm', fontsize=17, fontweight='bold')
    ax5.set_xlabel('Training Steps', fontsize=15)
    ax5.set_title('(e) Gradient Norm', fontsize=18, fontweight='bold', pad=12)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(axis='y', linestyle='--', alpha=0.2, zorder=0)
    
    plt.tight_layout()
    
    # # Add figure caption/note at the bottom
    # fig.text(0.5, 0.01, 
    #          'Note: Bold lines show smoothed trends (rolling average); light shaded areas show raw per-batch measurements.',
    #          ha='center', fontsize=9, style='italic', color='gray',
    #          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.8))
    
    # Adjust layout to make room for caption
    plt.subplots_adjust(bottom=0.05)
    
    # Save figure as SVG
    output_path = csv_path.parent / 'training_metrics_visualization.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0.2)
    print(f"\n📊 Visualization saved to: {output_path}")
    print(f"   字体配置: LARGE (主标题26pt, 子标题18pt, 轴标签17pt, 图表尺寸18x11)")
    
    # Show plot
    plt.show()
    
    # Print additional insights
    print("\n" + "="*80)
    print("📈 GRPO Training Performance Summary")
    print("="*80)
    print("\n⚠️  IMPORTANT NOTE:")
    print("   The 'accuracy' shown is computed IN-TRAINING on the training set,")
    print("   NOT on an independent validation/test set.")
    print("   Each batch generates 8 completions and compares with ground truth.")
    print("-"*80)
    
    initial_acc = df[reward_col].iloc[0] * 100
    final_acc = df[reward_col].iloc[-1] * 100
    max_acc = df[reward_col].max() * 100
    mean_acc = df[reward_col].mean() * 100
    
    print(f"\n📊 Training Set Performance (Online Evaluation):")
    print(f"   Initial Accuracy:    {initial_acc:.1f}%")
    print(f"   Final Accuracy:      {final_acc:.1f}%")
    print(f"   Peak Accuracy:       {max_acc:.1f}%")
    print(f"   Mean Accuracy:       {mean_acc:.1f}%")
    print(f"   Improvement:         +{final_acc-initial_acc:.1f}% ({((final_acc/initial_acc)-1)*100:.1f}% relative)")
    
    print(f"\n🔧 Training Diagnostics:")
    print(f"   Final Loss:          {df['loss'].iloc[-1]:.4f}")
    print(f"   Final KL Divergence: {df['kl'].iloc[-1]:.4f} (safe < 0.05)")
    print(f"   Max Gradient Norm:   {df['grad_norm'].max():.4f}")
    print(f"   Training Stability:  {'✅ Stable' if df['grad_norm'].max() < 1.0 else '⚠️ Some instability detected'}")
    
    print(f"\n💡 For paper reporting:")
    print(f"   ✓ Use this for 'training dynamics' and 'convergence analysis'")
    print(f"   ✗ Do NOT report as 'validation accuracy' or 'test accuracy'")
    print(f"   → Need separate evaluation on test set for final performance")
    print("="*80)

def main():
    csv_path = Path("training_metrics.csv")
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print("Please run extract_training_metrics.py first!")
        return
    
    plot_training_metrics(csv_path)

if __name__ == "__main__":
    main()
