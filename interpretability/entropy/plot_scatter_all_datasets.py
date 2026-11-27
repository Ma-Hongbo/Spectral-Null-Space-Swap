#!/usr/bin/env python3
"""Per-problem scatter H_base vs H_model at L35 (perp-stretched), for EVERY dataset
with per-problem data: math, wiki, alpaca (entropy_univ.json) + aime (entropy_aime.json).
Prominent marker style (no gray triangle): Sub=red circle, Null=blue diamond, Base=dark square.
Outputs figA_scatter_{ds}.png per dataset + figA_scatter_all.png (2x2 grid)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data")

eu = json.load(open(f"{DATA}/entropy_univ.json"))
ai = json.load(open(f"{DATA}/entropy_aime.json"))

def perprob(ds):
    if ds == 'aime':
        return {m: ai[m]['35'] for m in ['base', 'sub', 'perp', 'full']}
    return {m: eu[ds][m]['35'] for m in ['base', 'sub', 'perp', 'full']}

DATASETS = [('math', 'MATH-500'), ('wiki', 'Wiki'), ('alpaca', 'Alpaca'), ('aime', 'AIME')]

C = {'perp': '#2196F3', 'base': '#37474F', 'sub': '#F44336'}
MK = {'perp': 'D', 'base': 's', 'sub': 'o'}
SZ = {'perp': 42, 'base': 40, 'sub': 44}
AL = {'perp': 0.70, 'base': 0.75, 'sub': 0.60}
LB = {'perp': 'Null', 'base': 'Base', 'sub': 'Sub'}
ORDER = ['sub', 'perp', 'base']
STRETCH = 3.0

def perp_stretch(xo, yo, k):
    d_par = (xo + yo) / 2.0
    d_perp = (yo - xo) / 2.0
    return d_par - d_perp * k, d_par + d_perp * k

def draw(ax, pp):
    hb = np.array(pp['base'], dtype=float)
    xs_all, ys_all = [], []
    for tag in ORDER:
        hm = np.array(pp[tag], dtype=float)
        xs, ys = perp_stretch(hb, hm, STRETCH)
        xs_all.extend(xs); ys_all.extend(ys)
        ax.scatter(xs, ys, s=SZ[tag], alpha=AL[tag], color=C[tag], marker=MK[tag],
                   edgecolors='white', linewidths=0.4, label=LB[tag], zorder=3)
    lo, hi = hb.min() - 0.05, hb.max() + 0.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.3, zorder=1)
    pad = 0.03
    vmin = min(min(xs_all), min(ys_all)) - pad
    vmax = max(max(xs_all), max(ys_all)) + pad
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$H_{\mathrm{Base}}$', fontsize=12)
    ax.set_ylabel(r'$H_{\mathrm{model}}$', fontsize=12)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95, markerscale=1.6,
              handletextpad=0.4, borderpad=0.6)
    ax.grid(True, alpha=0.15, zorder=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # ordering stats (stretch preserves sign of H_model - H_base)
    stats = {}
    stats['n'] = len(hb)
    stats['perp_below'] = float(np.mean(np.array(pp['perp']) < hb))
    stats['sub_above'] = float(np.mean(np.array(pp['sub']) > hb))
    return stats

# ---- per-dataset files ----
for key, title in DATASETS:
    pp = perprob(key)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=250)
    st = draw(ax, pp)
    ax.set_title(f'{title} (L35)', fontsize=12, weight='bold', pad=8)
    fig.savefig(f"{DIR}/figA_scatter_{key}.png", bbox_inches='tight', facecolor='white')
    fig.savefig(f"{DIR}/figA_scatter_{key}.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"{key}: n={st['n']}  Null<Base: {st['perp_below']*100:.1f}%  Sub>Base: {st['sub_above']*100:.1f}%")

# ---- 2x2 grid ----
fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.6), dpi=250)
for (key, title), ax in zip(DATASETS, axes.flat):
    st = draw(ax, perprob(key))
    ax.set_title(f'{title} (L35)', fontsize=12, weight='bold', pad=6)
fig.tight_layout()
fig.savefig(f"{DIR}/figA_scatter_all.png", bbox_inches='tight', facecolor='white')
fig.savefig(f"{DIR}/figA_scatter_all.pdf", bbox_inches='tight', facecolor='white')
print("wrote figA_scatter_all.png + .pdf")
