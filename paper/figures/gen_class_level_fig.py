import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

data = json.load(open('/tmp/work/prj_01M0NK9FC2P84ZZJ7D2HVQ9DM1/ICLR2027_KD/class_level_analysis_v2.json'))
rd = data['raw_data']
complexity = np.array(rd['complexity_2factor'])
gain = np.array(rd['accuracy_gain'])
separability = np.array(rd['separabilities'])
lambda_2f = np.array(rd['lambda_2factor'])

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Panel A: Complexity vs Gain scatter
ax = axes[0, 0]
colors = ['#2196F3' if g >= 0 else '#F44336' for g in gain]
ax.scatter(complexity, gain * 100, c=colors, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
r, p = stats.pearsonr(complexity, gain)
slope, intercept, _, _, _ = stats.linregress(complexity, gain)
x_fit = np.linspace(complexity.min(), complexity.max(), 100)
ax.plot(x_fit, (slope * x_fit + intercept) * 100, 'k--', alpha=0.5, linewidth=1.5)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Class Complexity Score', fontsize=11)
ax.set_ylabel('Accuracy Gain (%)', fontsize=11)
ax.set_title(f'(a) Complexity vs. Gain (r={r:.3f}, p={p:.3f})', fontsize=10)
ax.tick_params(labelsize=9)

# Panel B: Gain distribution pie chart
ax = axes[0, 1]
gd = data['gain_distribution']
sizes = [gd['n_positive'], gd['n_negative'], gd['n_zero']]
labels = [f'Positive\n({gd["n_positive"]})', f'Negative\n({gd["n_negative"]})', f'Zero\n({gd["n_zero"]})']
colors_pie = ['#4CAF50', '#F44336', '#9E9E9E']
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
                                   startangle=90, textprops={'fontsize': 9})
ax.set_title(f'(b) Gain Distribution (mean={gd["mean"]*100:.1f}%)', fontsize=10)

# Panel C: Quartile bar chart
ax = axes[1, 0]
quartiles = data['quartile_analysis']
q_labels = ['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)']
q_gains = [q['mean_gain'] * 100 for q in quartiles]
q_errors = [q['std_gain'] * 100 for q in quartiles]
colors_q = ['#BBDEFB', '#64B5F6', '#1976D2', '#0D47A1']
bars = ax.bar(q_labels, q_gains, yerr=q_errors, color=colors_q, capsize=5, edgecolor='white', linewidth=0.8)
for bar, gain_val, q in zip(bars, q_gains, quartiles):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'+{gain_val:.1f}%\n({q["n_positive"]}/{q["n_negative"]}/{q["n_zero"]})',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_ylabel('Mean Accuracy Gain (%)', fontsize=11)
ax.set_title('(c) Gain by Complexity Quartile', fontsize=10)
ax.tick_params(labelsize=9)
ax.set_ylim(-1, 10)

# Panel D: Lambda distribution comparison (using summary stats from JSON)
ax = axes[1, 1]
lc = data['lambda_comparison']
labels_d = ['2-factor', '4-factor']
means = [lc['twofactor']['mean'], lc['fourfactor']['mean']]
mins = [lc['twofactor']['min'], lc['fourfactor']['min']]
maxs = [lc['twofactor']['max'], lc['fourfactor']['max']]
stds = [lc['twofactor']['std'], lc['fourfactor']['std']]
colors_d = ['#2196F3', '#FF9800']
x_pos = np.arange(2)
bars_d = ax.bar(x_pos, means, yerr=stds, color=colors_d, capsize=8, width=0.5, alpha=0.8, edgecolor='white')
# Add range as error bars (min-max)
for i in range(2):
    ax.plot([x_pos[i], x_pos[i]], [mins[i], maxs[i]], 'k-', alpha=0.3, linewidth=2)
    ax.plot(x_pos[i], maxs[i], 'k^', alpha=0.5, markersize=6)
    ax.plot(x_pos[i], mins[i], 'kv', alpha=0.5, markersize=6)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_d, fontsize=9)
ax.set_ylabel(r'Per-class $\lambda_c$', fontsize=11)
ax.set_title(f'(d) $\\lambda$ Distribution (std ratio={lc["std_ratio"]:.1f}$\\times$)', fontsize=10)
ax.tick_params(labelsize=9)
# Add std annotation
ax.text(0, maxs[0] + 0.001, f'std={stds[0]:.4f}', ha='center', fontsize=8, color='#2196F3')
ax.text(1, maxs[1] + 0.001, f'std={stds[1]:.4f}', ha='center', fontsize=8, color='#FF9800')

plt.tight_layout()
plt.savefig('/tmp/work/prj_01M0NK9FC2P84ZZJ7D2HVQ9DM1/ICLR2027_KD/paper/figures/class_level_mechanism.pdf',
            bbox_inches='tight', dpi=300)
print("4-panel figure saved!")
