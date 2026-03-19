"""Generate TPU SFT experiment visualizations for demo.
Usage: python3 generate_figures.py
Output: fig1~fig7 + table1~table4 PNG files in TPU-project/figures/
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'figures')
os.makedirs(OUT, exist_ok=True)

# ── 数据 ─────────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'results_tpu.json')) as f:
    raw = json.load(f)
step_times = np.array(raw['perf']['step_times'])  # 1866 steps

# 全量结果 (7465 train / 1316 eval, printed output)
FULL = dict(
    baseline_acc=57.8, finetuned_acc=66.0, improvement=8.1,
    mcnemar_chi2=30.78, mcnemar_p=0.0000,
    disc_base=129, disc_ft=236,
    train_samples=7465, eval_samples=1316, train_steps=1866,
    mfu=83.4, tflops=37.53, peak=45,
    steady_step=0.084, throughput=24314,
    compile_sec=173, model_load=16.4, final_loss=0.341,
)
# 3000 条结果
SMALL = dict(
    baseline_acc=61.5, finetuned_acc=67.5, improvement=6.0,
    train_samples=3000, eval_samples=200, mcnemar_p=0.105,
)
# Batch 推理
BATCH = dict(b1=0.700, b16=0.358)


# ── 图1: 准确率 + McNemar ────────────────────────────────────────────
def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 准确率
    bars = ax1.bar(['Baseline\n(before SFT)', 'Fine-tuned\n(after SFT)'],
                   [FULL['baseline_acc'], FULL['finetuned_acc']],
                   color=['#4A90D9', '#E8744F'], width=0.5, edgecolor='white', linewidth=1.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('GSM8K-CoT Accuracy (1316 eval samples)')
    for b in bars:
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
                 f'{b.get_height():.1f}%', ha='center', fontweight='bold', fontsize=13)
    ax1.annotate(f'+{FULL["improvement"]:.1f}%\np < 0.0001',
                 xy=(1, FULL['finetuned_acc']), xytext=(1.35, 74),
                 fontsize=11, fontweight='bold', color='#2E7D32',
                 arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # McNemar 不一致对
    vals = [FULL['disc_base'], FULL['disc_ft']]
    bars = ax2.bar(['Base correct\nSFT wrong', 'Base wrong\nSFT correct'],
                   vals, color=['#E57373', '#81C784'], width=0.5, edgecolor='white', linewidth=1.5)
    for b in bars:
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+3,
                 str(int(b.get_height())), ha='center', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Questions')
    ax2.set_title('McNemar Discordant Pairs')
    ax2.text(0.5, 0.92, f'Net gain: +{vals[1]-vals[0]} questions',
             transform=ax2.transAxes, ha='center', fontsize=12, fontweight='bold', color='#2E7D32',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32'))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig1_accuracy.png'); plt.close()
    print('  fig1_accuracy.png')


# ── 图2: 训练动态 ───────────────────────────────────────────────────
def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Step time (log)
    ax1.semilogy(range(len(step_times)), step_times, color='#4A90D9', lw=0.5, alpha=0.7)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Step Time (s, log)')
    ax1.set_title('Per-Step Training Time')
    ax1.annotate(f'JIT compile\n{step_times[0]:.0f}s + {step_times[1]:.0f}s',
                 xy=(1, step_times[0]), xytext=(250, step_times[0]*0.5),
                 fontsize=10, color='#C62828',
                 arrowprops=dict(arrowstyle='->', color='#C62828'))
    steady = np.mean(step_times[len(step_times)//2:])
    ax1.axhline(steady, color='#E8744F', ls='--', lw=1.5, alpha=0.8)
    ax1.text(len(step_times)*0.55, steady*1.8,
             f'Steady {steady*1000:.1f}ms/step\n({1/steady:.0f} steps/sec)',
             fontsize=10, color='#E8744F', fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Throughput (moving avg)
    window = 50
    throughput = 4 * 512 / step_times[2:]  # skip JIT
    smoothed = np.convolve(throughput, np.ones(window)/window, mode='valid')
    ax2.plot(np.arange(len(smoothed)) + 2 + window//2, smoothed / 1000,
             color='#4A90D9', lw=1.5)
    ax2.axhline(FULL['throughput']/1000, color='#E8744F', ls='--', lw=1.5, alpha=0.8)
    ax2.text(len(smoothed)*0.55, FULL['throughput']/1000 + 0.5,
             f'Steady {FULL["throughput"]/1000:.1f}K tok/s', fontsize=10,
             color='#E8744F', fontweight='bold')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Throughput (K tokens/sec)')
    ax2.set_title('Training Throughput (50-step moving avg)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_training.png'); plt.close()
    print('  fig2_training.png')


# ── 图3: 时间拆解 ───────────────────────────────────────────────────
def fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    jit_total = 173 + 43 + 47  # train + 2x infer first (batch=1)
    jit_total_new = 173 + 51 + 51  # train + 2x infer first (batch=16)

    configs = [
        ('batch=1 (old)  ~41 min', {
            'Model\nLoad': 16.4, 'JIT\nCompile': jit_total,
            'SFT\nTrain': 157, 'Inference\n(1316x2)': 1316*2*0.7,
        }),
        ('batch=16 (new)  ~24 min', {
            'Model\nLoad': 16.4, 'JIT\nCompile': jit_total_new,
            'SFT\nTrain': 157, 'Inference\n(1316x2)': 1316*2*0.358,
        }),
    ]
    colors = ['#90CAF9', '#FFB74D', '#81C784', '#E57373']

    for ax, (title, d) in zip([ax1, ax2], configs):
        names = list(d.keys())
        vals = [v / 60 for v in d.values()]
        bars = ax.barh(names, vals, color=colors, edgecolor='white', linewidth=1.5, height=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 0.3, b.get_y()+b.get_height()/2,
                    f'{v:.1f} min', va='center', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (min)')
        ax.set_title(title)
        ax.set_xlim(0, max(v/60 for v in configs[0][1].values())*1.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_time_breakdown.png'); plt.close()
    print('  fig3_time_breakdown.png')


# ── 图4: MFU + 硬件 ─────────────────────────────────────────────────
def fig4():
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.2])

    # MFU bar
    ax = fig.add_subplot(gs[0])
    ax.barh(['MFU'], [FULL['mfu']], color='#4CAF50', height=0.4, edgecolor='white')
    ax.barh(['MFU'], [100-FULL['mfu']], left=[FULL['mfu']], color='#E0E0E0', height=0.4)
    ax.text(FULL['mfu']/2, 0, f'{FULL["mfu"]}%', ha='center', va='center',
            fontsize=18, fontweight='bold', color='white')
    ax.set_xlim(0, 100); ax.set_xlabel('% of Peak')
    ax.set_title('Model FLOPs Utilization')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # TFLOP/s
    ax = fig.add_subplot(gs[1])
    bars = ax.bar(['Actual', 'Peak'], [FULL['tflops'], FULL['peak']],
                  color=['#4A90D9', '#B0BEC5'], width=0.5, edgecolor='white', linewidth=1.5)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                f'{b.get_height():.1f}', ha='center', fontweight='bold', fontsize=13)
    ax.set_ylabel('TFLOP/s per chip')
    ax.set_title('TPU v2 Utilization')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Config card
    ax = fig.add_subplot(gs[2]); ax.axis('off')
    info = [
        ('Model', 'Qwen2.5-Coder-1.5B'),
        ('LoRA Params', '9.2M / 1.54B (0.6%)'),
        ('Hardware', 'TPU v2-8 (4 chips)'),
        ('Batch x SeqLen', '4 x 512'),
        ('Throughput', f'{FULL["throughput"]:,} tok/s'),
        ('MFU', f'{FULL["mfu"]}%'),
        ('JIT Compile', f'{FULL["compile_sec"]:.0f}s'),
        ('Steady Step', f'{FULL["steady_step"]*1000:.1f}ms'),
        ('Train (1 epoch)', f'{FULL["train_steps"]} steps'),
    ]
    y = 0.95
    for k, v in info:
        ax.text(0.05, y, f'{k}:', fontsize=10, fontweight='bold', transform=ax.transAxes, va='top')
        ax.text(0.55, y, v, fontsize=10, transform=ax.transAxes, va='top')
        y -= 0.105
    ax.set_title('Experiment Config', pad=10)
    ax.add_patch(plt.Rectangle((0, 0.02), 1, 0.96, fill=False, edgecolor='#BDBDBD',
                                lw=1.5, transform=ax.transAxes, clip_on=False))

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_hardware.png'); plt.close()
    print('  fig4_hardware.png')


# ── 图5: 数据量对比 ─────────────────────────────────────────────────
def fig5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Bar chart
    x = np.arange(2); w = 0.3
    ax1.bar(x-w/2, [SMALL['baseline_acc'], FULL['baseline_acc']], w,
            label='Baseline', color='#4A90D9', edgecolor='white', linewidth=1.5)
    ax1.bar(x+w/2, [SMALL['finetuned_acc'], FULL['finetuned_acc']], w,
            label='Fine-tuned', color='#E8744F', edgecolor='white', linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['3000 train\n200 eval', '7465 train\n1316 eval'])
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.set_title('SFT Effect vs Dataset Size')
    ax1.legend(loc='upper left')
    for i, (imp, p_val) in enumerate([
        (SMALL['improvement'], SMALL['mcnemar_p']),
        (FULL['improvement'], FULL['mcnemar_p']),
    ]):
        sig = 'p < 0.001' if p_val < 0.001 else f'p = {p_val:.3f}'
        c = '#2E7D32' if p_val < 0.05 else '#B71C1C'
        top = max(SMALL['finetuned_acc'] if i == 0 else FULL['finetuned_acc'],
                  SMALL['baseline_acc'] if i == 0 else FULL['baseline_acc'])
        ax1.text(i, top + 3, f'+{imp}%\n{sig}', ha='center', fontsize=10, fontweight='bold', color=c)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Summary table
    ax2.axis('off')
    tbl = [
        ['', '3000 samples', '7465 (full)'],
        ['Improvement', '+6.0%', '+8.1%'],
        ['McNemar p', '0.105', '< 0.0001'],
        ['Significant?', 'NO', 'YES'],
        ['Train time', '~1 min', '~2.6 min'],
        ['Eval samples', '200', '1316'],
    ]
    table = ax2.table(cellText=tbl, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.6)
    for j in range(3):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')
    table[3, 1].set_facecolor('#FFCDD2')
    table[3, 1].set_text_props(fontweight='bold', color='#B71C1C')
    table[3, 2].set_facecolor('#C8E6C9')
    table[3, 2].set_text_props(fontweight='bold', color='#2E7D32')
    ax2.set_title('Data Scaling Summary', pad=20)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig5_data_scaling.png'); plt.close()
    print('  fig5_data_scaling.png')


# ── 图6: 推理优化 ───────────────────────────────────────────────────
def fig6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Latency
    bars = ax1.bar(['batch=1\n(sequential)', 'batch=16\n(batched)'],
                   [BATCH['b1']*1000, BATCH['b16']*1000],
                   color=['#E57373', '#81C784'], width=0.45, edgecolor='white', linewidth=1.5)
    for b in bars:
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+10,
                 f'{b.get_height():.0f}ms', ha='center', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Per-sample Latency (ms)')
    ax1.set_title('Inference Latency Comparison')
    ax1.text(0.5, 0.85, '2x faster', transform=ax1.transAxes, ha='center',
             fontsize=14, fontweight='bold', color='#2E7D32',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32'))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Pipeline total time
    ns = [200, 500, 1316]
    base_overhead = (16.4 + 275 + 157) / 60  # min (load + jit + train)
    old = [base_overhead + n*2*0.7/60 for n in ns]
    new = [base_overhead + n*2*0.358/60 for n in ns]
    x = np.arange(len(ns)); w = 0.3
    b1 = ax2.bar(x-w/2, old, w, label='batch=1', color='#E57373', edgecolor='white')
    b2 = ax2.bar(x+w/2, new, w, label='batch=16', color='#81C784', edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{n}' for n in ns])
    ax2.set_xlabel('Eval samples (x2 rounds)')
    ax2.set_ylabel('Total Pipeline Time (min)')
    ax2.set_title('Pipeline Time vs Eval Count')
    ax2.legend()
    for bars in [b1, b2]:
        for b in bars:
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                     f'{b.get_height():.0f}m', ha='center', fontsize=9, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig6_inference_opt.png'); plt.close()
    print('  fig6_inference_opt.png')


# ── 图7: 一页总览 Dashboard ──────────────────────────────────────────
def fig7():
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('TPU SFT Experiment Overview  |  Qwen2.5-Coder-1.5B + LoRA  |  TPU v2-8 (4 chips)',
                 fontsize=15, fontweight='bold', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (0,0) Accuracy
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(['Baseline', 'SFT'], [FULL['baseline_acc'], FULL['finetuned_acc']],
                  color=['#4A90D9', '#E8744F'], width=0.5)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{b.get_height():.1f}%',
                ha='center', fontweight='bold', fontsize=11)
    ax.set_ylim(0, 85)
    ax.set_title(f'Accuracy (+{FULL["improvement"]}%, p<0.0001)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (0,1) MFU donut
    ax = fig.add_subplot(gs[0, 1])
    ax.pie([FULL['mfu'], 100-FULL['mfu']], colors=['#4CAF50', '#E0E0E0'],
           startangle=90, wedgeprops=dict(width=0.35))
    ax.text(0, 0, f'{FULL["mfu"]}%', ha='center', va='center',
            fontsize=20, fontweight='bold', color='#2E7D32')
    ax.set_title('MFU (Model FLOPs Util.)')

    # (0,2) Key numbers
    ax = fig.add_subplot(gs[0, 2]); ax.axis('off')
    for i, (big, small) in enumerate([
        (f'{FULL["throughput"]:,}', 'tokens/sec'),
        (f'{FULL["tflops"]:.1f}', 'TFLOP/s per chip'),
        (f'{FULL["steady_step"]*1000:.1f}ms', 'per step'),
        (f'{FULL["compile_sec"]:.0f}s', 'JIT compile'),
        (f'{FULL["train_steps"]}', 'steps (1 epoch)'),
    ]):
        y = 0.9 - i * 0.18
        ax.text(0.1, y, big, fontsize=15, fontweight='bold', transform=ax.transAxes, va='top')
        ax.text(0.58, y, small, fontsize=10, color='#666', transform=ax.transAxes, va='top')
    ax.set_title('Performance')

    # (1,0) Step time
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(range(len(step_times)), step_times, color='#4A90D9', lw=0.5, alpha=0.7)
    ax.axhline(FULL['steady_step'], color='#E8744F', ls='--', lw=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Step Time (JIT {step_times[0]:.0f}s -> {FULL["steady_step"]*1000:.0f}ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (1,1) Time pie
    ax = fig.add_subplot(gs[1, 1])
    sizes = [275, 157, 1316*2*0.358, 77]
    labels = ['JIT\nCompile', 'SFT\nTrain', 'Inference', 'Other']
    colors = ['#FFB74D', '#81C784', '#E57373', '#90CAF9']
    total = sum(sizes)
    ax.pie(sizes, labels=labels, colors=colors,
           autopct=lambda p: f'{p:.0f}%' if p > 5 else '', startangle=140,
           pctdistance=0.75, wedgeprops=dict(linewidth=1.5, edgecolor='white'))
    ax.set_title(f'Time Breakdown (~{total/60:.0f} min)')

    # (1,2) McNemar
    ax = fig.add_subplot(gs[1, 2])
    bars = ax.bar(['Regress\n(base OK, sft wrong)', 'Fixed\n(base wrong, sft OK)'],
                  [FULL['disc_base'], FULL['disc_ft']],
                  color=['#E57373', '#81C784'], width=0.5)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+3,
                str(int(b.get_height())), ha='center', fontweight='bold', fontsize=13)
    ax.set_title(f'McNemar: chi2={FULL["mcnemar_chi2"]}, p<0.0001')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(f'{OUT}/fig7_dashboard.png'); plt.close()
    print('  fig7_dashboard.png')


# ── 表1: 主要实验结果 ────────────────────────────────────────────────
def table1():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_title('Table 1: Main Experiment Results (Full Dataset)', fontsize=14,
                 fontweight='bold', pad=20)

    cols = ['Metric', 'Value']
    rows = [
        ['Model', 'Qwen2.5-Coder-1.5B + LoRA (9.2M params)'],
        ['Hardware', 'TPU v2-8 (4 chips, bf16)'],
        ['Dataset', 'GSM8K-CoT (7465 train / 1316 eval)'],
        ['Batch x SeqLen', '4 x 512'],
        ['Training', '1866 steps, 1 epoch, ~2.6 min'],
        ['Baseline Accuracy', '57.8% (761/1316)'],
        ['Fine-tuned Accuracy', '66.0% (868/1316)'],
        ['Improvement', '+8.1% (+107 questions)'],
        ['McNemar chi2', '30.78'],
        ['McNemar p-value', '< 0.0001 (Significant)'],
        ['Final Loss', '0.341'],
    ]
    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.7)
    # Header styling
    for j in range(2):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight improvement row
    table[8, 0].set_facecolor('#E8F5E9')
    table[8, 1].set_facecolor('#E8F5E9')
    table[8, 1].set_text_props(fontweight='bold', color='#2E7D32')
    # Highlight significance
    table[10, 1].set_facecolor('#E8F5E9')
    table[10, 1].set_text_props(fontweight='bold', color='#2E7D32')

    plt.tight_layout()
    plt.savefig(f'{OUT}/table1_results.png'); plt.close()
    print('  table1_results.png')


# ── 表2: 性能指标 ────────────────────────────────────────────────────
def table2():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('off')
    ax.set_title('Table 2: TPU Training Performance', fontsize=14,
                 fontweight='bold', pad=20)

    cols = ['Metric', 'Value', 'Notes']
    rows = [
        ['MFU', '83.4%', '37.5 / 45 TFLOP/s per chip'],
        ['Throughput', '24,314 tok/s', 'Batch=4, SeqLen=512'],
        ['Steady Step Time', '84.0 ms', '~12 steps/sec'],
        ['First Step (JIT)', '85 s', 'XLA compilation overhead'],
        ['Total JIT Compile', '173 s', 'Training graph compilation'],
        ['Model Load Time', '16.4 s', 'HuggingFace -> Tunix/JAX'],
        ['Training Time', '157 s (2.6 min)', '1866 steps, 1 epoch'],
        ['Total Pipeline', '~24 min', 'With batch=16 inference'],
    ]
    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.7)
    for j in range(3):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight MFU
    table[1, 1].set_facecolor('#E8F5E9')
    table[1, 1].set_text_props(fontweight='bold', color='#2E7D32')

    plt.tight_layout()
    plt.savefig(f'{OUT}/table2_performance.png'); plt.close()
    print('  table2_performance.png')


# ── 表3: 数据量对比 ──────────────────────────────────────────────────
def table3():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis('off')
    ax.set_title('Table 3: Data Scaling Comparison (3000 vs Full Dataset)', fontsize=14,
                 fontweight='bold', pad=20)

    cols = ['Metric', '3000 Samples', '7465 (Full)']
    rows = [
        ['Train Samples', '3000', '7465'],
        ['Eval Samples', '200', '1316'],
        ['Training Steps', '750 (1 epoch)', '1866 (1 epoch)'],
        ['Training Time', '~1 min', '~2.6 min'],
        ['Baseline Accuracy', '61.5%', '57.8%'],
        ['Fine-tuned Accuracy', '67.5%', '66.0%'],
        ['Improvement', '+6.0%', '+8.1%'],
        ['McNemar p-value', '0.105', '< 0.0001'],
        ['Statistically Significant?', 'NO', 'YES'],
    ]
    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.7)
    for j in range(3):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Significance row highlighting
    table[9, 1].set_facecolor('#FFCDD2')
    table[9, 1].set_text_props(fontweight='bold', color='#B71C1C')
    table[9, 2].set_facecolor('#C8E6C9')
    table[9, 2].set_text_props(fontweight='bold', color='#2E7D32')
    # p-value row
    table[8, 1].set_text_props(color='#B71C1C')
    table[8, 2].set_facecolor('#E8F5E9')
    table[8, 2].set_text_props(fontweight='bold', color='#2E7D32')

    plt.tight_layout()
    plt.savefig(f'{OUT}/table3_data_scaling.png'); plt.close()
    print('  table3_data_scaling.png')


# ── 表4: 推理优化 + 时间估算 ─────────────────────────────────────────
def table4():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # 上半：推理优化对比
    ax1.axis('off')
    ax1.set_title('Table 4a: Inference Optimization (batch=1 vs batch=16)', fontsize=13,
                  fontweight='bold', pad=15)
    cols1 = ['Metric', 'batch=1 (old)', 'batch=16 (new)', 'Speedup']
    rows1 = [
        ['Per-sample Latency', '700 ms', '358 ms', '~2x'],
        ['First Batch (JIT)', '43 s (1 sample)', '51 s (16 samples)', '-'],
        ['Full Eval (1316x2)', '30.7 min', '15.7 min', '~2x'],
        ['Total Pipeline', '~41 min', '~24 min', '1.7x'],
    ]
    t1 = ax1.table(cellText=rows1, colLabels=cols1, loc='center', cellLoc='center')
    t1.auto_set_font_size(False)
    t1.set_fontsize(10.5)
    t1.scale(1.1, 1.7)
    for j in range(4):
        t1[0, j].set_facecolor('#37474F')
        t1[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, 5):
        t1[i, 2].set_facecolor('#E8F5E9')
        t1[i, 3].set_facecolor('#E8F5E9')
        t1[i, 3].set_text_props(fontweight='bold', color='#2E7D32')

    # 下半：三种方案时间估算
    ax2.axis('off')
    ax2.set_title('Table 4b: Pipeline Time Estimates (batch=16)', fontsize=13,
                  fontweight='bold', pad=15)
    cols2 = ['Strategy', 'Eval Samples', 'Estimated Time', 'Use Case']
    rows2 = [
        ['A: Quick Demo', '200', '~10 min', 'In-class demo'],
        ['B: Balanced', '~500', '~14 min', 'Homework / lab'],
        ['C: Full Experiment', '1316', '~24 min', 'Report / paper'],
    ]
    t2 = ax2.table(cellText=rows2, colLabels=cols2, loc='center', cellLoc='center')
    t2.auto_set_font_size(False)
    t2.set_fontsize(10.5)
    t2.scale(1.1, 1.7)
    for j in range(4):
        t2[0, j].set_facecolor('#37474F')
        t2[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight recommended
    t2[3, 0].set_facecolor('#E3F2FD')
    t2[3, 0].set_text_props(fontweight='bold')

    plt.tight_layout(h_pad=3)
    plt.savefig(f'{OUT}/table4_inference_opt.png'); plt.close()
    print('  table4_inference_opt.png')


if __name__ == '__main__':
    print('Generating figures...\n')
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print()
    table1(); table2(); table3(); table4()
    print('\nDone! 7 figures + 4 tables saved.')
