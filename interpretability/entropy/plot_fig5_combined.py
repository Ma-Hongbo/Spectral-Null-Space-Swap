#!/usr/bin/env python3
"""Figure 5: Mechanistic Analysis — 3-panel combined figure.
(a) Attention entropy bar chart (4 datasets, L32–35 avg)
(b) Top-8 attention heatmap (L32–35, per-row normalized, 2×2, colorbar)
(c) Per-problem scatter: H_base vs H_model (L35, MATH-500, perp-stretched)
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data")
OUT = os.path.join(DIR, "fig5_combined.png")

ent_data = json.load(open(f"{DATA}/entropy_all_datasets.json"))
hmap = json.load(open(f"{DATA}/topk_heatmap.json"))
R_univ = json.load(open(f"{DATA}/entropy_univ.json"))
R_aime = json.load(open(f"{DATA}/entropy_aime.json"))

# ── Style ──
C_NULL = "#2196F3"
C_BASE = "#78909C"
C_FULL = "#9C27B0"
C_SUB  = "#F44336"

FS_TITLE = 14
FS_LABEL = 12.5
FS_TICK  = 11
FS_LEGEND = 11.5
BW = 'bold'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.linewidth': 0.6, 'axes.labelsize': FS_LABEL,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
    'mathtext.default': 'regular',
})

# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 5.5))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.05, 1], wspace=0.28)

# ── (a) Entropy Bar Chart — 4 datasets ──
ax_a = fig.add_subplot(gs[0])

models = ["perp", "base", "full", "sub"]
model_labels = ["Null", "Base", "Full", "Sub"]
model_colors = [C_NULL, C_BASE, C_FULL, C_SUB]

datasets = ["cmimc25", "aime25", "math500", "aime24"]
ds_labels = ["CMIMC'25", "AIME'25", "MATH-500", "AIME'24"]

vals = {}
for m in models:
    vals[m] = []
    for ds in datasets:
        h = np.mean([ent_data[f"{m}_{ds}_{l}"]["H_mean"] for l in [32, 33, 34, 35]])
        vals[m].append(h)

x = np.arange(len(datasets))
n = len(models)
width = 0.18
offsets = np.arange(n) - (n - 1) / 2

for i, (m, label, color) in enumerate(zip(models, model_labels, model_colors)):
    ax_a.bar(x + offsets[i] * width, vals[m], width,
             label=label, color=color, edgecolor='white', linewidth=0.8, alpha=0.88)

ax_a.set_xticks(x)
ax_a.set_xticklabels(ds_labels, fontsize=FS_TICK, fontweight=BW)
ax_a.set_ylabel('Mean Attention Entropy (L32–35)', fontweight=BW)
ax_a.legend(fontsize=FS_LEGEND, loc='upper left', framealpha=0.9)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

all_v = [v for m in models for v in vals[m]]
ymin = min(all_v) - 0.015
ymax = max(all_v) + 0.02
ax_a.set_ylim(ymin, ymax)
ax_a.axhline(y=np.mean(vals["base"]), color=C_BASE, linewidth=0.6, linestyle='--', alpha=0.3)

# ── (b) Top-8 Heatmap — L32–35, 2×2, band-normalized with numbers ──
gs_b = GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[1],
                                  hspace=0.22, wspace=0.06,
                                  height_ratios=[0.12, 1, 1])

K = 8
LAYERS = [32, 33, 34, 35]
hmap_models = ["perp", "base", "full", "sub"]
hmap_labels = ["Null", "Base", "Full", "Sub"]

PAD = 0.008
bands = []
for k in range(K):
    hi = 1.0 - k * (1.0 / K) - PAD
    lo = 1.0 - (k + 1) * (1.0 / K) + PAD
    bands.append((lo, hi))

grid_pos = [(1, 0), (1, 1), (2, 0), (2, 1)]
for idx, L in enumerate(LAYERS):
    ax_b = fig.add_subplot(gs_b[grid_pos[idx]])
    d = hmap[str(L)]
    raw = np.array([d[m][:K] for m in hmap_models]).T  # (K, 4)

    display = np.zeros_like(raw)
    for i in range(K):
        row = raw[i]
        lo_band, hi_band = bands[i]
        rmin, rmax = row.min(), row.max()
        if rmax - rmin > 1e-12:
            t = (row - rmin) / (rmax - rmin)
        else:
            t = np.full_like(row, 0.5)
        display[i] = lo_band + t * (hi_band - lo_band)

    ax_b.imshow(display, cmap='Blues', vmin=0, vmax=1,
                aspect='auto', interpolation='nearest')

    for i in range(K):
        for j in range(len(hmap_models)):
            v = raw[i, j]
            n_val = display[i, j]
            tc = 'white' if n_val > 0.55 else '#1a1a1a'
            fmt = f'{v:.3f}' if v >= 0.01 else f'{v:.4f}'
            ax_b.text(j, i, fmt, ha='center', va='center', fontsize=7.5, color=tc)

    ax_b.set_xticks(range(len(hmap_models)))
    if idx >= 2:
        ax_b.set_xticklabels(hmap_labels, fontsize=10.5, fontweight=BW)
    else:
        ax_b.set_xticklabels([])

    ax_b.set_yticks(range(K))
    if idx % 2 == 0:
        ax_b.set_yticklabels([str(k+1) for k in range(K)], fontsize=9.5)
        ax_b.set_ylabel('Rank', fontsize=11, fontweight=BW)
    else:
        ax_b.set_yticklabels([])

    ax_b.set_title(f'L{L}', fontsize=12, fontweight=BW, pad=3)
    ax_b.tick_params(length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    for i in range(K + 1):
        ax_b.axhline(i - 0.5, color='white', linewidth=1.2)
    for j in range(len(hmap_models) + 1):
        ax_b.axvline(j - 0.5, color='white', linewidth=1.2)

# ── Aligned titles via fig.text ──
TITLE_Y = 0.90
bbox_a = gs[0].get_position(fig)
bbox_b = gs[1].get_position(fig)
fig.text((bbox_a.x0 + bbox_a.x1) / 2, TITLE_Y,
         '(a) Attention Entropy Across Datasets',
         fontsize=FS_TITLE, fontweight=BW, ha='center', va='top')
fig.text((bbox_b.x0 + bbox_b.x1) / 2, TITLE_Y,
         '(b) Top-8 Attention Weight',
         fontsize=FS_TITLE, fontweight=BW, ha='center', va='top')

# ── (c) Scatter: H_base vs H_model (AIME, L35, perp-stretched) ──
ax_c = fig.add_subplot(gs[2])
ls = '35'

R_c = R_aime                       # AIME per-problem data (top keys = models)
hb = np.array(R_c['base'][ls])

STRETCH = 3.0

def perp_stretch(xo, yo, k):
    d_par = (xo + yo) / 2.0
    d_perp = (yo - xo) / 2.0
    d_perp *= k
    return d_par - d_perp, d_par + d_perp

C_BASE_DARK = "#37474F"  # prominent Base (replaces faint gray triangle)
series = [
    ("sub",  "Sub",  C_SUB,       "o", 46, 0.60),
    ("perp", "Null", C_NULL,      "D", 44, 0.70),
    ("base", "Base", C_BASE_DARK, "s", 44, 0.75),
]

for tag, label, color, marker, size, alpha in series:
    hm = np.array(R_c[tag][ls])
    xs, ys = perp_stretch(hb, hm, STRETCH)
    ax_c.scatter(xs, ys, s=size, alpha=alpha, color=color, marker=marker,
                 edgecolors='white', linewidths=0.3, label=label, zorder=3)

lo = hb.min() - 0.05
hi = hb.max() + 0.05
ax_c.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.3, zorder=1)

all_xs, all_ys = [], []
for tag, *_ in series:
    hm = np.array(R_c[tag][ls])
    xs, ys = perp_stretch(hb, hm, STRETCH)
    all_xs.extend(xs); all_ys.extend(ys)
pad = 0.03
vmin = min(min(all_xs), min(all_ys)) - pad
vmax = max(max(all_xs), max(all_ys)) + pad
ax_c.set_xlim(vmin, vmax)
ax_c.set_ylim(vmin, vmax)
ax_c.set_aspect('equal')

ax_c.set_xlabel(r'$H_{\mathrm{Base}}$', fontsize=FS_LABEL, fontweight=BW)
ax_c.set_ylabel(r'$H_{\mathrm{model}}$', fontsize=FS_LABEL, fontweight=BW)
ax_c.grid(True, alpha=0.15, zorder=0)
ax_c.legend(fontsize=13, loc='upper left', framealpha=0.95, markerscale=1.8,
            handletextpad=0.4, borderpad=0.6)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

bbox_c = gs[2].get_position(fig)
fig.text((bbox_c.x0 + bbox_c.x1) / 2, TITLE_Y,
         '(c) Per-Problem Entropy (AIME, L35)',
         fontsize=FS_TITLE, fontweight=BW, ha='center', va='top')

p_perp = np.array(R_c['perp'][ls]); p_sub = np.array(R_c['sub'][ls])
print(f"AIME scatter: n={len(hb)}, Null<Base={100*(p_perp<hb).mean():.0f}%, "
      f"Sub>Base={100*(p_sub>hb).mean():.0f}%, full ordering={100*((p_perp<hb)&(p_sub>hb)).mean():.0f}%")

# ── Save ──
fig.savefig(OUT, dpi=250, bbox_inches='tight', facecolor='white')
fig.savefig(OUT.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved {OUT} and {OUT.replace('.png', '.pdf')}")
