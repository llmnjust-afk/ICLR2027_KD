#!/usr/bin/env python3
"""Generate D4M sensitivity curve figure locally."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# D4M results (t_start: (mean, std, n_seeds))
results = {
    5: (15.06, 0.0, 1),
    10: (16.08, 0.0, 1),
    15: (16.38, 0.0, 1),
    20: (18.38, 0.0, 1),
    25: (20.21, 0.39, 3),
    30: (22.58, 0.0, 1),
    35: (22.80, 0.0, 1),
    40: (23.07, 0.59, 3),
    45: (22.42, 0.0, 1),
}

t_values = sorted(results.keys())
accs = [results[t][0] for t in t_values]
stds = [results[t][1] for t in t_values]

unguided = 22.79
unguided_std = 0.18
cags = 25.70
cags_std = 0.07

fig, ax = plt.subplots(figsize=(6, 4))

ax.errorbar(t_values, accs, yerr=stds, fmt='o-', color='#d62728', markersize=7,
            linewidth=2, capsize=4, label='D4M (real-image init)', zorder=3)

ax.axhline(y=unguided, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.8,
           label=f'Unguided ({unguided}%)')
ax.axhline(y=cags, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.8,
           label=f'CAGS optimal ({cags}%)')

ax.fill_between(t_values, cags - cags_std, cags + cags_std, color='#1f77b4', alpha=0.1)
ax.fill_between(t_values, unguided - unguided_std, unguided + unguided_std, color='#2ca02c', alpha=0.1)

ax.set_xlabel(r'Denoising start timestep ($t_{\mathrm{start}}$)', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
ax.set_title('D4M Sensitivity to Denoising Start Timestep', fontsize=13)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 50)
ax.set_ylim(14, 27)

plt.tight_layout()
plt.savefig('/tmp/work/prj_01M0NK9FC2P84ZZJ7D2HVQ9DM1/ICLR2027_KD/paper/figures/d4m_sensitivity.pdf', dpi=300, bbox_inches='tight')
print("Saved d4m_sensitivity.pdf")
plt.close()
