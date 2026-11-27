#!/usr/bin/env python3
"""Figure 6: Norm Decomposition — 3-panel combined figure.
(a) Energy partition bar chart (param. vs func. space)
(b) Parameter-space layerwise norms
(c) Function-space layerwise norms on AIME (broken y-axis)
"""
import os
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──
# weight_norms.json and functional_norms.json ship in ./data next to this script
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = "fig_norm_combined.png"

wt_data = json.load(open(f"{DATA_DIR}/weight_norms.json"))
fn_data = json.load(open(f"{DATA_DIR}/functional_norms.json"))
summary = wt_data["summary"]

layers = list(range(36))
fn_layers = list(range(1, 37))  # functional layer 1..36 → transformer layer 0..35
tl = np.array(layers)

# ── Colors ──
C_FULL = "#9C27B0"       # purple — Full (ΔW)
C_SUB  = "#F44336"       # red — Sub (ΔW∥)
C_PERP = "#2196F3"       # blue — Perp (ΔW⊥)

C_SUB_PARAM  = "#F4C2C2"  # light pink
C_SUB_FUNC   = "#8B1A2B"  # dark crimson
C_PERP_PARAM = "#B8D4F0"  # light blue
C_PERP_FUNC  = "#1E5AA8"  # dark blue

# ── Style ──
LW_FULL = 2.2; LW_COMP = 1.8; MS = 2.5
FS_TITLE = 12; FS_LABEL = 11; FS_TICK = 9.5; FS_LEGEND = 9
BW = 'bold'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.linewidth': 0.6, 'axes.labelsize': FS_LABEL,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
    'mathtext.default': 'regular',
})

# ── Data ──
norm_full_wt = np.array([summary[str(l)]["norm_full"] for l in layers])
norm_sub_wt  = np.array([summary[str(l)]["norm_sub"]  for l in layers])
norm_perp_wt = np.array([summary[str(l)]["norm_perp"] for l in layers])

norm_full_fn = np.array([fn_data[str(l)]["dh_full_norm2_avg"]**0.5 for l in fn_layers])
norm_perp_fn = np.array([fn_data[str(l)]["dh_perp_norm2_avg"]**0.5 for l in fn_layers])
norm_sub_fn  = np.array([fn_data[str(l)]["dh_sub_norm2_avg"]**0.5  for l in fn_layers])

# Bar chart — parameter-space: simple mean; function-space: energy-weighted
wt_frac_sub  = np.mean([summary[str(l)]["frac_sub"]  for l in layers])
wt_frac_perp = np.mean([summary[str(l)]["frac_perp"] for l in layers])

total_perp = sum(fn_data[str(l)]["dh_perp_norm2_avg"] * fn_data[str(l)]["count"] for l in fn_layers)
total_sub  = sum(fn_data[str(l)]["dh_sub_norm2_avg"]  * fn_data[str(l)]["count"] for l in fn_layers)
total_full = sum(fn_data[str(l)]["dh_full_norm2_avg"] * fn_data[str(l)]["count"] for l in fn_layers)
fn_frac_perp = total_perp / total_full
fn_frac_sub  = total_sub / total_full


def y2d(y):
    """Piecewise linear transform for broken y-axis on panel (c).
    [0, 15] → [0, 55]       low-value region (normal scale)
    [15, 220] → [55, 65]    compressed gap (zigzag break)
    [220, 530] → [65, 135]  high-value region (expanded scale)
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    out[y <= 15] = y[y <= 15] * (55.0 / 15.0)
    m = (y > 15) & (y <= 220)
    out[m] = 55 + (y[m] - 15) * (10.0 / 205.0)
    out[y > 220] = 65 + (y[y > 220] - 220) * (70.0 / 310.0)
    return out


# ══════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 4.5))

# ── (a) Energy Partition ──────────────────────────────────────
bar_w = 0.32; gap = 0.08
pos     = [0 - bar_w/2 - gap/2,  0 + bar_w/2 + gap/2,
           1.0 - bar_w/2 - gap/2, 1.0 + bar_w/2 + gap/2]
heights = [wt_frac_sub*100, fn_frac_sub*100, wt_frac_perp*100, fn_frac_perp*100]
colors  = [C_SUB_PARAM, C_SUB_FUNC, C_PERP_PARAM, C_PERP_FUNC]
txt_colors  = ['#666666', 'k', '#666666', 'k']
txt_weights = ['normal', BW, 'normal', BW]
# Paper notation: energy fraction f∥ = ‖ΔW∥‖²_F/‖ΔW‖²_F (param.),
# functional fraction f̃_* = ‖Δh_*‖²/‖Δh_full‖² (func.)
labels = [
    r"$f_\parallel$ (param.)",
    r"$\tilde{f}_\parallel$ (func.)",
    r"$f_\perp$ (param.)",
    r"$\tilde{f}_\perp$ (func.)",
]

for i, (p, h, c, lb) in enumerate(zip(pos, heights, colors, labels)):
    ax_a.bar(p, h, width=bar_w, color=c, edgecolor=c, linewidth=0.8, label=lb)
    ax_a.text(p, h + 0.8, f'{h:.0f}%', ha='center', va='bottom',
              fontsize=9.5, fontweight=txt_weights[i], color=txt_colors[i])

ax_a.set_xticks([0, 1.0])
ax_a.set_xticklabels(['Sub', 'Null'], fontsize=11, fontweight=BW)
ax_a.set_ylabel('Fraction (%)', fontweight=BW)
ax_a.set_title('(a) Energy Partition', fontsize=FS_TITLE, fontweight=BW)
ax_a.set_xlim(-0.45, 1.45); ax_a.set_ylim(0, 82)
ax_a.legend(fontsize=FS_LEGEND, loc='upper left', framealpha=0.9,
            handlelength=1.2, ncol=2, columnspacing=0.8)
ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)

# ── (b) Parameter-Space Norms ─────────────────────────────────
ax_b.plot(tl, norm_full_wt, color=C_FULL, lw=LW_FULL, marker='o', ms=MS,
          label=r'$\Vert\Delta W^{(\ell)}\Vert_F$')
ax_b.plot(tl, norm_perp_wt, color=C_PERP, lw=LW_COMP, marker='s', ms=MS,
          label=r'$\Vert\Delta W_\perp^{(\ell)}\Vert_F$')
ax_b.plot(tl, norm_sub_wt, color=C_SUB, lw=LW_COMP, marker='^', ms=MS,
          label=r'$\Vert\Delta W_\parallel^{(\ell)}\Vert_F$')

ax_b.set_xlabel(r'Layer $\ell$', fontweight=BW)
ax_b.set_ylabel('Frobenius Norm', fontweight=BW)
ax_b.set_title('(b) Parameter-Space Norms', fontsize=FS_TITLE, fontweight=BW)
ax_b.legend(fontsize=FS_LEGEND, loc='upper left')
ax_b.set_xlim(-0.5, 35.5); ax_b.set_ylim(7, 25)
ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
ax_b.grid(True, alpha=0.12)

# ── (c) Function-Space Norms on AIME (broken axis) ───────────
ax_c.plot(tl, y2d(norm_full_fn), color=C_FULL, lw=LW_FULL, marker='o', ms=MS,
          label=r'$\Vert\Delta h_{\mathrm{full}}^{(\ell)}\Vert$', zorder=3)
ax_c.plot(tl, y2d(norm_perp_fn), color=C_PERP, lw=LW_COMP, marker='s', ms=MS,
          label=r'$\Vert\Delta h_\perp^{(\ell)}\Vert$', zorder=3)
ax_c.plot(tl, y2d(norm_sub_fn), color=C_SUB, lw=LW_COMP, marker='^', ms=MS,
          label=r'$\Vert\Delta h_\parallel^{(\ell)}\Vert$', zorder=3)

ax_c.set_ylim(-3, 140); ax_c.set_xlim(-0.5, 35.5)

real_ticks = [0, 5, 10, 250, 350, 450, 500]
ax_c.set_yticks(y2d(np.array(real_ticks, dtype=float)))
ax_c.set_yticklabels([str(int(v)) for v in real_ticks])

# Break zone + zigzag
ax_c.axhspan(55, 65, color='#F5F5F5', zorder=0)
ax_c.fill_between([-2, 0.8], 54, 66, color='white', zorder=4, clip_on=False)
zz_xs = np.array([-0.7, 0.2, -0.7, 0.2, -0.7])
zz_ys = np.linspace(53, 67, 5)
ax_c.plot(zz_xs, zz_ys, color='k', lw=0.8, zorder=5, clip_on=False)

ax_c.set_xlabel(r'Layer $\ell$', fontweight=BW)
ax_c.set_ylabel('Hidden-State Norm (RMS)', fontweight=BW)
ax_c.set_title('(c) Function-Space Norms on AIME', fontsize=FS_TITLE, fontweight=BW)
ax_c.legend(fontsize=FS_LEGEND, loc='lower right')
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
ax_c.grid(True, alpha=0.12, zorder=0)

# ── Save ──
plt.tight_layout()
fig.savefig(OUT, dpi=250, bbox_inches='tight', facecolor='white')
fig.savefig(OUT.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved {OUT} and {OUT.replace('.png', '.pdf')}")
