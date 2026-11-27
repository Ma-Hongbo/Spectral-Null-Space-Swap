#!/usr/bin/env python3
"""Appendix A.1 figure: layer-selection justification (Qwen3-4B, 36 layers).
(a) layer-wise mean attention entropy for the four variants (avg over math/wiki/alpaca/AIME)
(b) per-layer consistency of the ordering H_perp < H_base < H_sub across the 4 datasets.
Style matches fig5_combined.py."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data")
OUT = os.path.join(DIR, "figA1_layerwise.png")

eu = json.load(open(f"{DATA}/entropy_univ.json"))
ai = json.load(open(f"{DATA}/entropy_aime.json"))
MODELS = ['base', 'sub', 'perp', 'full']
C = {'perp': "#2196F3", 'base': "#78909C", 'full': "#9C27B0", 'sub': "#F44336"}
MK = {'perp': 'D', 'base': '^', 'full': 's', 'sub': 'o'}
LB = {'perp': 'Null', 'base': 'Base', 'full': 'Full', 'sub': 'Sub'}

def per_layer(d, ds):
    return {m: {int(l): float(np.mean(v)) for l, v in d[ds][m].items()} for m in MODELS}

DS = [('math', per_layer(eu, 'math')), ('wiki', per_layer(eu, 'wiki')),
      ('alpaca', per_layer(eu, 'alpaca')),
      ('aime', {m: {int(l): float(np.mean(v)) for l, v in ai[m].items()} for m in MODELS})]

LAYERS = list(range(36))
avg = {m: {l: float(np.mean([pm[m][l] for _, pm in DS])) for l in LAYERS} for m in MODELS}
cons = {l: sum(1 for _, pm in DS if pm['perp'][l] < pm['base'][l] < pm['sub'][l]) for l in LAYERS}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.linewidth': 0.6, 'axes.labelsize': 12.5,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.3), dpi=250,
                               gridspec_kw={'width_ratios': [1.35, 1], 'wspace': 0.22})

# ---- panel (a): layer-wise entropy profiles ----
for m in MODELS:
    ax1.plot(LAYERS, [avg[m][l] for l in LAYERS], marker=MK[m], ms=4, lw=1.4,
             color=C[m], label=LB[m], markeredgewidth=0.5)
ax1.axvspan(31.5, 35.5, color="#2196F3", alpha=0.07, lw=0)
ax1.axvline(31.5, color="#2196F3", ls="--", lw=0.9, alpha=0.6)
ax1.axvline(35.5, color="#2196F3", ls="--", lw=0.9, alpha=0.6)
ax1.text(33.5, 2.05, "layers 32--35", ha='center', fontsize=9, color="#1976D2")
ax1.set_xlabel("Layer")
ax1.set_ylabel("Mean attention entropy $H_t$ (nats)")
ax1.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 35])
ax1.set_ylim(0.5, 2.25)
ax1.legend(loc="upper right", fontsize=9, frameon=False, ncol=2)
ax1.spines[['top', 'right']].set_visible(False)
ax1.set_title("(a) Layer-wise attention entropy", fontsize=12, weight='bold', pad=6)

# ---- panel (b): per-layer ordering consistency ----
cols = ["#2196F3" if 31.5 <= l <= 35.5 else "#B0BEC5" for l in LAYERS]
ax2.bar(LAYERS, [cons[l] for l in LAYERS], color=cols, width=0.75, edgecolor='none')
ax2.axhline(4, color='0.55', lw=0.7, ls=':')
ax2.annotate("4/4 datasets", xy=(35, 4), xytext=(22.5, 3.95), fontsize=9,
             color="#1565C0", arrowprops=dict(arrowstyle="-", lw=0.7, color="#1565C0"))
ax2.annotate("3/4", xy=(32, 3), xytext=(32.6, 3.28), fontsize=9, color="#1565C0")
ax2.set_xlabel("Layer")
ax2.set_ylabel("# datasets with $H_{\\mathrm{Null}}<H_{\\mathrm{Base}}<H_{\\mathrm{Sub}}$")
ax2.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 35])
ax2.set_ylim(0, 4.5)
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_title("(b) Ordering consistency per layer", fontsize=12, weight='bold', pad=6)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace('.png', '.pdf'), bbox_inches="tight")
print("wrote", OUT, "and", OUT.replace('.png', '.pdf'))
print("L32-35 avg per model:", {m: round(np.mean([avg[m][l] for l in range(32, 36)]), 3) for m in MODELS})
