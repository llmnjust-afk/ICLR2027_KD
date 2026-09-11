#!/usr/bin/env python3
"""Generate Theorem 1 quantitative fitting figure.

Panel (a): K distribution (mode count) for IN-10, IN-100, IN-200, IN-1K
Panel (b): Mode structure heterogeneity (frac K>2) vs actual accuracy gain
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Data from compute_theorem1_fit.py
datasets = {
    'IN-10':    {'C': 10,   'gain': 0.00, 'frac_K_gt2': 0.000, 'K_dist': {2: 10}},
    'Nette':    {'C': 10,   'gain': 5.28, 'frac_K_gt2': 0.000, 'K_dist': {2: 10}},
    'Woof':     {'C': 10,   'gain': 1.12, 'frac_K_gt2': 0.000, 'K_dist': {2: 10}},
    'IN-100':   {'C': 100,  'gain': 2.65, 'frac_K_gt2': 0.050, 'K_dist': {2: 95, 3: 2, 4: 2, 5: 1}},
    'IN-200':   {'C': 200,  'gain': 3.30, 'frac_K_gt2': 0.285, 'K_dist': {2: 143, 3: 9, 4: 4, 5: 4, 6: 6, 7: 5, 8: 4, 9: 5, 10: 1, 11: 2, 13: 3, 14: 1, 15: 3, 16: 3, 17: 1, 18: 1, 19: 2, 20: 3}},
    'IN-1K':    {'C': 1000, 'gain': 3.58, 'frac_K_gt2': 0.600, 'K_dist': {2: 400, 3: 80, 4: 41, 5: 43, 6: 44, 7: 40, 8: 42, 9: 35, 10: 23, 11: 36, 12: 35, 13: 28, 14: 26, 15: 23, 16: 26, 17: 17, 18: 13, 19: 20, 20: 28}},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel (a): K distribution stacked bar chart
in_datasets = ['IN-10', 'IN-100', 'IN-200', 'IN-1K']
x_pos = np.arange(len(in_datasets))
bar_width = 0.6

# Group K values into bins
K_bins = [(2, 'K=2'), (3, 'K=3'), (4, 'K=4-5'), (6, 'K=6-10'), (11, 'K=11-20')]

colors = ['#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#b07aa1']

for i, (K_max, label) in enumerate(K_bins):
    fracs = []
    for ds in in_datasets:
        dist = datasets[ds]['K_dist']
        total = sum(dist.values())
        if i == 0:
            frac = dist.get(2, 0) / total
        elif i == 1:
            frac = dist.get(3, 0) / total
        elif i == 2:
            frac = sum(dist.get(k, 0) for k in [4, 5]) / total
        elif i == 3:
            frac = sum(dist.get(k, 0) for k in range(6, 11)) / total
        else:
            frac = sum(dist.get(k, 0) for k in range(11, 21)) / total
        fracs.append(frac)
    
    bottoms = np.zeros(len(in_datasets))
    for j in range(i):
        prev_K_max, _ = K_bins[j]
        prev_fracs = []
        for ds in in_datasets:
            dist = datasets[ds]['K_dist']
            total = sum(dist.values())
            if j == 0:
                frac = dist.get(2, 0) / total
            elif j == 1:
                frac = dist.get(3, 0) / total
            elif j == 2:
                frac = sum(dist.get(k, 0) for k in [4, 5]) / total
            elif j == 3:
                frac = sum(dist.get(k, 0) for k in range(6, 11)) / total
            prev_fracs.append(frac)
        bottoms += np.array(prev_fracs)
    
    ax1.bar(x_pos, fracs, bar_width, bottom=bottoms, label=label, color=colors[i], edgecolor='white', linewidth=0.5)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(['10', '100', '200', '1000'])
ax1.set_xlabel('Number of classes (C)', fontsize=12)
ax1.set_ylabel('Fraction of classes', fontsize=12)
ax1.set_title('(a) Mode count distribution', fontsize=13)
ax1.legend(fontsize=9, loc='upper right', title='Mode count', title_fontsize=9)
ax1.set_ylim(0, 1.05)

# Panel (b): frac_K_gt2 vs gain scatter
all_ds = ['IN-10', 'Nette', 'Woof', 'IN-100', 'IN-200', 'IN-1K']
for ds in all_ds:
    d = datasets[ds]
    if ds.startswith('IN-') and ds != 'IN-10':
        color = '#4e79a7'
        marker = 'o'
    elif ds == 'IN-10':
        color = '#4e79a7'
        marker = 'o'
    elif ds == 'Nette':
        color = '#59a14f'
        marker = 's'
    else:  # Woof
        color = '#e15759'
        marker = 's'
    
    ax2.scatter(d['frac_K_gt2'], d['gain'], c=color, marker=marker, s=100, zorder=3, edgecolors='white', linewidth=0.5)
    
    # Offset labels to avoid overlap
    offset = (0.02, 0.15)
    if ds == 'IN-10':
        offset = (0.02, -0.35)
    elif ds == 'Nette':
        offset = (0.02, 0.15)
    elif ds == 'Woof':
        offset = (0.02, -0.35)
    elif ds == 'IN-100':
        offset = (-0.02, 0.15)
    elif ds == 'IN-1K':
        offset = (-0.02, 0.15)
    
    ha = 'left' if offset[0] >= 0 else 'right'
    ax2.annotate(ds, (d['frac_K_gt2'], d['gain']), xytext=(d['frac_K_gt2']+offset[0], d['gain']+offset[1]),
                fontsize=9, ha=ha)

# Add trend line for IN datasets only (excluding Nette/Woof)
in_only = ['IN-10', 'IN-100', 'IN-200', 'IN-1K']
x_in = np.array([datasets[ds]['frac_K_gt2'] for ds in in_only])
y_in = np.array([datasets[ds]['gain'] for ds in in_only])
z = np.polyfit(x_in, y_in, 1)
p = np.poly1d(z)
x_line = np.linspace(-0.02, 0.65, 100)
ax2.plot(x_line, p(x_line), '--', color='#4e79a7', alpha=0.5, linewidth=1.5, label='IN trend')

r = np.corrcoef(x_in, y_in)[0, 1]
ax2.text(0.35, 1.0, f'IN datasets: r={r:.2f}', fontsize=9, color='#4e79a7', style='italic')

ax2.set_xlabel('Fraction of classes with $K > 2$ modes', fontsize=12)
ax2.set_ylabel('CAGS gain over unguided (%)', fontsize=12)
ax2.set_title('(b) Mode heterogeneity vs.\\ accuracy gain', fontsize=13)
ax2.legend(fontsize=9, loc='lower right')
ax2.set_xlim(-0.05, 0.7)
ax2.set_ylim(-0.5, 6.5)
ax2.grid(True, alpha=0.2)

# Legend for markers
circle_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4e79a7', markersize=8, label='ImageNet subsets')
square_patch = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#59a14f', markersize=8, label='10-class variants')
ax2.legend(handles=[circle_patch, square_patch, plt.Line2D([0], [0], linestyle='--', color='#4e79a7', alpha=0.5, label='IN trend')], fontsize=9, loc='lower right')

plt.tight_layout()
output_path = '/tmp/work/prj_01M0NK9FC2P84ZZJ7D2HVQ9DM1/ICLR2027_KD/paper/figures/theorem1_fit.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved to {output_path}")
plt.close()
