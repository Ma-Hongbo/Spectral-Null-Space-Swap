#!/usr/bin/env python3
"""Appendix A.2 figure: attention entropy bar chart, all 8 datasets (L32-35 average).
Adapted from the original scratchpad plot_entropy_multidataset.py; reads from data/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data")
OUT = os.path.join(DIR, "figA2_entropy_8datasets.png")

data = json.load(open(f"{DATA}/entropy_all_datasets.json"))

models = ["perp", "base", "full", "sub"]
labels = ["Null", "Base", "Full", "Sub"]
colors = ["#2196F3", "#78909C", "#9C27B0", "#F44336"]

datasets = ["aime24", "aime25", "amc23", "cmimc25", "hmmt25", "math500", "gsm8k", "ALL"]
ds_labels = ["AIME'24", "AIME'25", "AMC'23", "CMIMC'25", "HMMT'25", "MATH-500", "GSM-8K", "All"]

vals = {}
for m in models:
    vals[m] = []
    for ds in datasets:
        h = np.mean([data[f"{m}_{ds}_{l}"]["H_mean"] for l in [32, 33, 34, 35]])
        vals[m].append(h)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.linewidth': 0.6})
fig, ax = plt.subplots(figsize=(12, 4.2), dpi=250)

x = np.arange(len(datasets))
n = len(models)
width = 0.18
offsets = np.arange(n) - (n - 1) / 2

for i, (m, label, color) in enumerate(zip(models, labels, colors)):
    ax.bar(x + offsets[i] * width, vals[m], width,
           label=label, color=color, edgecolor='white', linewidth=0.8, alpha=0.88)

ax.set_xticks(x)
ax.set_xticklabels(ds_labels, fontsize=11)
ax.set_ylabel('Mean Attention Entropy (L32–35)', fontsize=12)
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

all_v = [v for m in models for v in vals[m]]
ax.set_ylim(min(all_v) - 0.02, max(all_v) + 0.02)
ax.axhline(y=vals["base"][-1], color='#78909C', linewidth=0.6, linestyle='--', alpha=0.4)

plt.tight_layout()
fig.savefig(OUT, dpi=250, bbox_inches='tight', facecolor='white')
fig.savefig(OUT.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print("wrote", OUT, "and", OUT.replace('.png', '.pdf'))
for ds in datasets:
    row = [vals[m][datasets.index(ds)] for m in models]
    print(ds, " ".join(f"{v:.4f}" for v in row))
