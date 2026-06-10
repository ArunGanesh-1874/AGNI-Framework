import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('/workspace/comixattack/results/plots', exist_ok=True)

# ─── Data ───
methods = ['AGNI-Full', 'AGNI-Strict', 'DeepWordBug', 'TextFooler', 'Random']
colors  = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#A8DADC']

# Hinglish
hinglish_asr = [0.7654, 0.2920, 0.7614, 0.8251, 0.2513]
hinglish_sim = [0.9291, 0.9697, 0.9336, 0.7100, 0.8975]
hinglish_aqi = [0.4461, 0.2305, 0.3631, 0.0888, 0.1898]
hinglish_avgq= [39.3,   18.7,   50.0,   85.6,   15.3]

# Tamil
tamil_asr = [0.6663, 0.2920, 0.5712, 0.4442, 0.1472]
tamil_sim = [0.9291, 0.9697, 0.9336, 0.7100, 0.8975]
tamil_aqi = [0.2243, 0.1596, 0.1504, 0.0316, 0.1120]
tamil_avgq= [27.1,   18.7,   30.5,   38.4,   6.5]

x = np.arange(len(methods))
width = 0.35

# ─── Figure 1: AQI Comparison (Both Languages) ───
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('AGNI Framework — Attack Quality Index (AQI) Comparison',
             fontsize=14, fontweight='bold', y=1.02)

for ax, aqi, title in zip(axes,
                           [hinglish_aqi, tamil_aqi],
                           ['Hinglish (Hindi-English)', 'Tamil-English']):
    bars = ax.bar(methods, aqi, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('AQI Score', fontsize=11)
    ax.set_ylim(0, 0.55)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add value labels
    for bar, val in zip(bars, aqi):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    # Highlight AGNI-Full
    bars[0].set_edgecolor('#E63946')
    bars[0].set_linewidth(2.5)

plt.tight_layout()
plt.savefig('/workspace/comixattack/results/plots/aqi_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: aqi_comparison.png')

# ─── Figure 2: ASR vs AQI scatter ───
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('AGNI Framework — ASR vs AQI Trade-off',
             fontsize=14, fontweight='bold', y=1.02)

for ax, asr, aqi, title in zip(axes,
                                [hinglish_asr, tamil_asr],
                                [hinglish_aqi, tamil_aqi],
                                ['Hinglish', 'Tamil-English']):
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax.scatter(asr[i], aqi[i], c=color, s=200,
                   edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(method, (asr[i], aqi[i]),
                    textcoords='offset points', xytext=(8, 4),
                    fontsize=9, fontweight='bold')
    ax.set_xlabel('Attack Success Rate (ASR)', fontsize=11)
    ax.set_ylabel('Attack Quality Index (AQI)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/workspace/comixattack/results/plots/asr_vs_aqi.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: asr_vs_aqi.png')

# ─── Figure 3: Full comparison table heatmap ───
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('AGNI Framework — Complete Metric Comparison',
             fontsize=14, fontweight='bold')

metrics      = ['ASR', 'AvgSim', 'AQI']
metric_label = ['ASR ↑', 'AvgSim ↑', 'AQI ↑']

for ax, data, title in zip(
    axes,
    [
        np.array([hinglish_asr, hinglish_sim, hinglish_aqi]),
        np.array([tamil_asr,    tamil_sim,    tamil_aqi])
    ],
    ['Hinglish', 'Tamil-English']
):
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
    ax.set_yticks(range(len(metric_label)))
    ax.set_yticklabels(metric_label, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')

    for i in range(len(metrics)):
        for j in range(len(methods)):
            val = data[i, j]
            color = 'white' if val < 0.3 or val > 0.7 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig('/workspace/comixattack/results/plots/metric_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: metric_heatmap.png')

# ─── Figure 4: Query efficiency comparison ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('AGNI Framework — Query Efficiency Comparison',
             fontsize=14, fontweight='bold')

for ax, avgq, asr, title in zip(
    axes,
    [hinglish_avgq, tamil_avgq],
    [hinglish_asr,  tamil_asr],
    ['Hinglish', 'Tamil-English']
):
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax.scatter(avgq[i], asr[i], c=color, s=200,
                   edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(method, (avgq[i], asr[i]),
                    textcoords='offset points', xytext=(5, 4),
                    fontsize=9, fontweight='bold')
    ax.set_xlabel('Avg Query Count (lower = better)', fontsize=11)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Ideal region annotation
    ax.annotate('Ideal region\n(high ASR, low queries)',
                xy=(15, 0.85), fontsize=8, color='green',
                style='italic')

plt.tight_layout()
plt.savefig('/workspace/comixattack/results/plots/query_efficiency.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: query_efficiency.png')

print('\nAll plots saved to: /workspace/comixattack/results/plots/')
