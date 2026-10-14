import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('/workspace/comixattack/results/plots', exist_ok=True)

# ─── Data — final verified numbers, six methods, both languages ───
methods = ['AGNI-Full', 'AGNI-Strict', 'DeepWordBug', 'TextFooler', 'Random', 'AdvCodeMix']
colors  = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#A8DADC', '#9B5DE5']

# Hinglish (average ASR/AvgQ across XLM-R/MuRIL/mBERT, sentiment+hate combined)
hinglish_asr  = [0.7654, 0.2982, 0.7612, 0.8253, 0.2512, 0.7769]
hinglish_sim  = [0.9291, 0.9754, 0.9336, 0.7100, 0.8975, 0.9070]
hinglish_aqi  = [0.4297, 0.2414, 0.3530, 0.0809, 0.1907, 0.3041]
hinglish_avgq = [39.30,  16.89,  49.97,  85.59,  15.31,  56.45]

# Tamil-English (average ASR/AvgQ across XLM-R/MuRIL/mBERT, sentiment+offensive combined)
tamil_asr  = [0.6663, 0.1797, 0.5712, 0.4442, 0.1472, 0.6080]
tamil_sim  = [0.9291, 0.9541, 0.9336, 0.7100, 0.8975, 0.9072]
tamil_aqi  = [0.4501, 0.1526, 0.3697, 0.1935, 0.1235, 0.3497]
tamil_avgq = [27.10,  10.93,  30.46,  38.39,  6.47,   36.34]

AVG_Q_MAX = 99.3  # TextFooler Hinglish hate XLM-R, highest observed AvgQ

x = np.arange(len(methods))

# ─── Figure 1: AQI Comparison (Both Languages) ───
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('AGNI Framework — Attack Quality Index (AQI) Comparison',
             fontsize=14, fontweight='bold', y=1.02)

for ax, aqi, title in zip(axes, [hinglish_aqi, tamil_aqi],
                           ['Hinglish (Hindi-English)', 'Tamil-English']):
    bars = ax.bar(methods, aqi, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('AQI Score', fontsize=11)
    ax.set_ylim(0, 0.55)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, aqi):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    bars[0].set_edgecolor('#E63946')
    bars[0].set_linewidth(2.5)

plt.tight_layout()
plt.savefig('/workspace/comixattack/results/plots/aqi_comparison_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: aqi_comparison_v2.png')

# ─── Figure 2: Query Efficiency (AvgQ vs ASR) ───
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('AGNI Framework — Query Efficiency Comparison', fontsize=14, fontweight='bold')

for ax, avgq, asr, title in zip(axes, [hinglish_avgq, tamil_avgq],
                                 [hinglish_asr, tamil_asr], ['Hinglish', 'Tamil-English']):
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax.scatter(avgq[i], asr[i], c=color, s=200, edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(method, (avgq[i], asr[i]), textcoords='offset points',
                    xytext=(6, 5), fontsize=8, fontweight='bold')
    ax.set_xlabel('Avg Query Count (lower = better)', fontsize=11)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/workspace/comixattack/results/plots/query_efficiency_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: query_efficiency_v2.png')

# ─── Figure 3: ASR vs AQI Trade-off ───
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('AGNI Framework — ASR vs AQI Trade-off', fontsize=14, fontweight='bold', y=1.02)

for ax, asr, aqi, title in zip(axes, [hinglish_asr, tamil_asr],
                                [hinglish_aqi, tamil_aqi], ['Hinglish', 'Tamil-English']):
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax.scatter(asr[i], aqi[i], c=color, s=200, edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(method, (asr[i], aqi[i]), textcoords='offset points',
                    xytext=(6, 5), fontsize=8, fontweight='bold')
    ax.set_xlabel('Attack Success Rate (ASR)', fontsize=11)
    ax.set_ylabel('Attack Quality Index (AQI)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/workspace/comixattack/results/plots/asr_vs_aqi_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: asr_vs_aqi_v2.png')

# ─── Figure 4: Metric Heatmap ───
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('AGNI Framework — Complete Metric Comparison', fontsize=14, fontweight='bold')

metric_label = ['ASR ↑', 'AvgSim ↑', 'AQI ↑']
for ax, data, title in zip(axes,
    [np.array([hinglish_asr, hinglish_sim, hinglish_aqi]),
     np.array([tamil_asr, tamil_sim, tamil_aqi])],
    ['Hinglish', 'Tamil-English']):
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
    ax.set_yticks(range(len(metric_label)))
    ax.set_yticklabels(metric_label, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    for i in range(len(metric_label)):
        for j in range(len(methods)):
            val = data[i, j]
            color = 'white' if val < 0.3 or val > 0.7 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig('/workspace/comixattack/results/plots/metric_heatmap_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: metric_heatmap_v2.png')

print('\nAll plots saved to: /workspace/comixattack/results/plots/')
