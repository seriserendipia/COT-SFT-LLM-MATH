"""
GPU vs TPU 对比可视化
读取 results_tpu.json + results_gpu.json，生成 comparison.png

图 1（左）: 编译开销 vs 稳态 step time（对数轴）—— 重点故事
图 2（中）: 吞吐量 tokens/sec + MFU%
图 3（右）: Baseline / Fine-tuned 准确率对比

附图 4: per-step 时间曲线（第一步 spike 的教学展示）

用法:
    python make_comparison.py
    python make_comparison.py --tpu results_tpu.json --gpu results_gpu.json --out comparison.png
"""
import json, sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--tpu', default=os.path.join(os.path.dirname(__file__), 'results_tpu.json'))
parser.add_argument('--gpu', default=os.path.join(os.path.dirname(__file__), 'results_gpu.json'))
parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'comparison.png'))
args = parser.parse_args()

# ── Load data ────────────────────────────────────────────────────────────────
def load(path, label):
    if not os.path.exists(path):
        print(f"[WARN] {label} result file not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)

tpu_data = load(args.tpu, 'TPU')
gpu_data = load(args.gpu, 'GPU')

if tpu_data is None and gpu_data is None:
    print("No result files found. Run tpu_sft_pipeline.py and gpu_sft_pipeline.py first.")
    sys.exit(1)

def p(data, *keys, default=None, aliases=None):
    """安全取 data['perf'][key]，支持 aliases 兼容旧字段名"""
    if data is None: return default
    perf = data.get('perf', {})
    # 先按主 key 查找
    val = perf
    for k in keys:
        if isinstance(val, dict): val = val.get(k)
        else: val = None
        if val is None: break
    if val is not None:
        return val
    # 兜底：尝试 aliases
    if aliases:
        for alt in aliases:
            v = perf.get(alt)
            if v is not None: return v
    return default

# ── Colors & labels ──────────────────────────────────────────────────────────
TPU_COLOR = '#4C9BE8'   # 蓝
GPU_COLOR = '#F28B30'   # 橙
ALPHA = 0.85

tpu_label = f"TPU ({p(tpu_data, 'device_name', default='TPU')})" if tpu_data else None
gpu_label = f"GPU ({p(gpu_data, 'device_name', default='GPU')})" if gpu_data else None

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle('TPU vs GPU: Tunix/JAX SFT on GSM8K-CoT (Qwen2.5-Coder-1.5B LoRA)',
             fontsize=14, fontweight='bold', y=0.98)

# 2 rows × 2 cols; bottom-right spans 2 cols for step-time curve
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)
ax1 = fig.add_subplot(gs[0, 0])   # Compilation + Steady step
ax2 = fig.add_subplot(gs[0, 1])   # Throughput + MFU
ax3 = fig.add_subplot(gs[0, 2])   # Accuracy
ax4 = fig.add_subplot(gs[1, :])   # Step-time curve (full width)


# ── Helper: grouped bar ───────────────────────────────────────────────────────
def grouped_bar(ax, categories, tpu_vals, gpu_vals, ylabel='', title='', log=False, fmt='.1f'):
    x = np.arange(len(categories))
    w = 0.35
    bars_present = []
    if tpu_data is not None:
        b = ax.bar(x - w/2, tpu_vals, w, label=tpu_label, color=TPU_COLOR, alpha=ALPHA)
        bars_present.append(b)
    if gpu_data is not None:
        b = ax.bar(x + w/2 if tpu_data else x, gpu_vals, w, label=gpu_label, color=GPU_COLOR, alpha=ALPHA)
        bars_present.append(b)
    if log:
        ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 标注数值
    for bars in bars_present:
        for bar in bars:
            h = bar.get_height()
            if h is None or np.isnan(h): continue
            label_str = f'{h:{fmt}}' if isinstance(fmt, str) else fmt(h)
            ax.annotate(label_str, xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7.5)


# ── Plot 1: Compilation overhead vs steady step (log scale) ──────────────────
cats1 = ['Compile\nOverhead (s)', 'Steady Step\nTime (s)', 'First Step\nTime (s)']
tpu1 = [
    p(tpu_data, 'compilation_overhead_sec', default=0) or 0,
    p(tpu_data, 'training_steady_step_sec', default=0) or 0,
    p(tpu_data, 'training_first_step_sec', default=0) or 0,
]
gpu1 = [
    p(gpu_data, 'compilation_overhead_sec', default=0) or 0,
    p(gpu_data, 'training_steady_step_sec', default=0) or 0,
    p(gpu_data, 'training_first_step_sec', default=0) or 0,
]
grouped_bar(ax1, cats1, tpu1, gpu1,
            ylabel='Seconds (log scale)', title='XLA Compilation vs Steady State',
            log=True, fmt='.1f')
ax1.set_ylim(bottom=0.01)


# ── Plot 2: Throughput + MFU ──────────────────────────────────────────────────
# 两个 y 轴：左边 throughput，右边 MFU%
ax2b = ax2.twinx()
x = np.arange(2)
w = 0.35

tpu_thru = p(tpu_data, 'throughput_tokens_per_sec', default=0) or 0
gpu_thru = p(gpu_data, 'throughput_tokens_per_sec', default=0) or 0
tpu_mfu = p(tpu_data, 'training_mfu_percent', default=0) or 0
gpu_mfu = p(gpu_data, 'training_mfu_percent', default=0) or 0

if tpu_data:
    b1 = ax2.bar(0 - w/2, tpu_thru, w, color=TPU_COLOR, alpha=ALPHA, label=tpu_label)
    b3 = ax2b.bar(1 - w/2, tpu_mfu, w, color=TPU_COLOR, alpha=0.6, hatch='//')
if gpu_data:
    offset = w/2 if tpu_data else 0
    b2 = ax2.bar(0 + offset, gpu_thru, w, color=GPU_COLOR, alpha=ALPHA, label=gpu_label)
    b4 = ax2b.bar(1 + offset, gpu_mfu, w, color=GPU_COLOR, alpha=0.6, hatch='//')

ax2.set_ylabel('Throughput (tokens/sec)', fontsize=9)
ax2b.set_ylabel('MFU %', fontsize=9)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Throughput\n(tokens/sec)', 'MFU (%)'], fontsize=9)
ax2.set_title('Training Efficiency', fontsize=10, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 标注
for val, xpos, ax_ in [(tpu_thru, -w/2, ax2), (gpu_thru, w/2 if tpu_data else 0, ax2),
                        (tpu_mfu, 1-w/2, ax2b), (gpu_mfu, 1+w/2 if tpu_data else 1, ax2b)]:
    if val:
        ax_.annotate(f'{val:.1f}', xy=(xpos, val), xytext=(0, 3),
                     textcoords='offset points', ha='center', va='bottom', fontsize=7.5)

patches = []
if tpu_data: patches.append(mpatches.Patch(color=TPU_COLOR, label=tpu_label))
if gpu_data: patches.append(mpatches.Patch(color=GPU_COLOR, label=gpu_label))
ax2.legend(handles=patches, fontsize=8)


# ── Plot 3: Accuracy comparison ───────────────────────────────────────────────
cats3 = ['Baseline', 'Fine-tuned', 'Improvement']
tpu3 = [
    (p(tpu_data, 'baseline_accuracy', default=0) or 0) * 100,
    (p(tpu_data, 'finetuned_accuracy', default=0) or 0) * 100,
    (p(tpu_data, 'accuracy_improvement', default=0) or 0) * 100,
]
gpu3 = [
    (p(gpu_data, 'baseline_accuracy', default=0) or 0) * 100,
    (p(gpu_data, 'finetuned_accuracy', default=0) or 0) * 100,
    (p(gpu_data, 'accuracy_improvement', default=0) or 0) * 100,
]
grouped_bar(ax3, cats3, tpu3, gpu3,
            ylabel='Accuracy (%)', title='GSM8K-CoT Accuracy', fmt='.1f')

# McNemar p-value 注释
notes = []
if tpu_data:
    pv = p(tpu_data, 'mcnemar_p_value', default=None)
    if pv is not None: notes.append(f"TPU p={pv:.4f}")
if gpu_data:
    pv = p(gpu_data, 'mcnemar_p_value', default=None)
    if pv is not None: notes.append(f"GPU p={pv:.4f}")
if notes:
    ax3.text(0.98, 0.02, ' | '.join(notes), transform=ax3.transAxes,
             ha='right', va='bottom', fontsize=7.5, color='gray')


# ── Plot 4: Per-step time curve ───────────────────────────────────────────────
ax4.set_title('Per-Step Training Time (first step spike = XLA JIT compilation)',
              fontsize=10, fontweight='bold')
ax4.set_xlabel('Training Step', fontsize=9)
ax4.set_ylabel('Step Time (seconds)', fontsize=9)

has_curve = False
if tpu_data:
    steps = p(tpu_data, 'step_times', default=[])
    if steps:
        ax4.plot(range(1, len(steps)+1), steps, color=TPU_COLOR, linewidth=1.5,
                 label=tpu_label, alpha=0.9)
        # 标注第一步
        ax4.annotate(f'1st: {steps[0]:.0f}s',
                     xy=(1, steps[0]), xytext=(len(steps)*0.05+2, steps[0]*0.95),
                     arrowprops=dict(arrowstyle='->', color=TPU_COLOR),
                     fontsize=8, color=TPU_COLOR)
        has_curve = True

if gpu_data:
    steps = p(gpu_data, 'step_times', default=[])
    if steps:
        ax4.plot(range(1, len(steps)+1), steps, color=GPU_COLOR, linewidth=1.5,
                 label=gpu_label, alpha=0.9)
        ax4.annotate(f'1st: {steps[0]:.0f}s',
                     xy=(1, steps[0]), xytext=(len(steps)*0.05+2, steps[0]*0.85),
                     arrowprops=dict(arrowstyle='->', color=GPU_COLOR),
                     fontsize=8, color=GPU_COLOR)
        has_curve = True

if has_curve:
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.set_yscale('log')
else:
    ax4.text(0.5, 0.5, 'Per-step data not yet available\n(run pipeline to generate step_times)',
             ha='center', va='center', transform=ax4.transAxes,
             fontsize=11, color='gray', style='italic')
    ax4.set_visible(True)


# ── Text box: summary stats ────────────────────────────────────────────────────
def summary_line(data, label):
    if data is None: return ''
    perf = data.get('perf', {})
    acc_b = perf.get('baseline_accuracy', 0)
    acc_ft = perf.get('finetuned_accuracy', 0)
    thru = perf.get('throughput_tokens_per_sec', 0)
    wall = perf.get('total_wall_sec', 0)
    sig = perf.get('significant_at_005', False)
    return (f"{label}: baseline={acc_b:.1%} → ft={acc_ft:.1%} "
            f"(+{(acc_ft-acc_b):.1%}, sig={'✓' if sig else '✗'})  "
            f"thru={thru:.0f} tok/s  wall={wall/60:.0f}min")

lines = [l for l in [summary_line(tpu_data, 'TPU'), summary_line(gpu_data, 'GPU')] if l]
fig.text(0.5, 0.01, '\n'.join(lines), ha='center', va='bottom',
         fontsize=8.5, color='#333333',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#cccccc'))

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(args.out, dpi=150, bbox_inches='tight')
print(f"Saved: {args.out}")

# 同时打印 ASCII 对比表
print("\n" + "="*65)
print(f"  {'Metric':<30}  {'TPU':>14}  {'GPU':>14}")
print("="*65)
rows = [
    ("Baseline accuracy",       f"{(p(tpu_data,'baseline_accuracy') or 0)*100:.1f}%", f"{(p(gpu_data,'baseline_accuracy') or 0)*100:.1f}%"),
    ("Fine-tuned accuracy",     f"{(p(tpu_data,'finetuned_accuracy') or 0)*100:.1f}%", f"{(p(gpu_data,'finetuned_accuracy') or 0)*100:.1f}%"),
    ("Accuracy improvement",    f"{(p(tpu_data,'accuracy_improvement') or 0)*100:+.1f}%", f"{(p(gpu_data,'accuracy_improvement') or 0)*100:+.1f}%"),
    ("Compile overhead (s)",    f"{p(tpu_data,'compilation_overhead_sec') or 0:.1f}", f"{p(gpu_data,'compilation_overhead_sec') or 0:.1f}"),
    ("Steady step (s)",         f"{p(tpu_data,'training_steady_step_sec') or 0:.3f}", f"{p(gpu_data,'training_steady_step_sec') or 0:.3f}"),
    ("Throughput (tok/s)",      f"{p(tpu_data,'throughput_tokens_per_sec') or 0:.0f}", f"{p(gpu_data,'throughput_tokens_per_sec') or 0:.0f}"),
    ("MFU (%)",                 f"{p(tpu_data,'training_mfu_percent') or 0:.1f}", f"{p(gpu_data,'training_mfu_percent') or 0:.1f}"),
    ("Infer steady (s/sample)", f"{p(tpu_data,'finetuned_inference_steady_per_sample',aliases=['finetuned_inference_steady_sec']) or 0:.3f}", f"{p(gpu_data,'finetuned_inference_steady_per_sample',aliases=['finetuned_inference_steady_sec']) or 0:.3f}"),
    ("Total wall time (min)",   f"{(p(tpu_data,'total_wall_sec') or 0)/60:.0f}", f"{(p(gpu_data,'total_wall_sec') or 0)/60:.0f}"),
]
for name, tv, gv in rows:
    tv = tv if tpu_data else 'N/A'
    gv = gv if gpu_data else 'N/A'
    print(f"  {name:<30}  {tv:>14}  {gv:>14}")
print("="*65)
