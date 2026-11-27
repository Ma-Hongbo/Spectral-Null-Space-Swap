#!/usr/bin/env python3
"""Dual-axis figure: inner product and angle between DeltaW_perp and G_perp, per dataset.

One figure, two lines, two y-axes:
  left  (blue)  inner product <DeltaW_perp, G_perp>  -- broken axis, see below
  right (red)   how far the angle sits from orthogonal, in arcseconds

The left axis is BROKEN: gsm8k's inner product (-0.976) is ~4x the next largest, so on a
single linear axis it flattens every other dataset against zero. The top panel carries the
six comparable datasets, the bottom panel carries gsm8k alone. The line is drawn in both
panels and clipped by each panel's ylim, which is the standard broken-axis construction.

The raw angle is 90.0009-90.0026 deg, so it is reported as 90 - angle in arcseconds -- an
angle still, just re-referenced to orthogonal, because the raw value would be a flat line.

  python plot_cos_ip.py                       # ALL scope (all 252 weight matrices)
  python plot_cos_ip.py --scope late          # L32-35 window only
  python plot_cos_ip.py --angle-mode ratio    # |cos| / (1/sqrt(D)) instead of arcsec
  python plot_cos_ip.py --angle-mode deg      # raw angle (will be flat)

Writes fig_cos_ip.pdf (vector, for the paper) and fig_cos_ip.png (for eyeballing).
"""
import json, os, math, argparse
import matplotlib

# Machine-specific locations come from the environment (see README):
#   S3_ROOT      directory holding paths2507.env, data/ and models/   (default: .)
#   S3_HF_HOME   HuggingFace cache, if not the usual ~/.cache/huggingface
ROOT = os.environ.get("S3_ROOT", ".")
if os.environ.get("S3_HF_HOME"):
    os.environ.setdefault("HF_HOME", os.environ["S3_HF_HOME"])
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser()
ap.add_argument('--scope', default='all', choices=['all', 'late'])
ap.add_argument('--angle-mode', default='dev', choices=['deg', 'dev', 'ratio'])
ap.add_argument('--src', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "grad_cos_7datasets"))
ap.add_argument('--out', default="fig_cos_ip")
# break bounds for the left axis: top panel = everything comparable, bottom = gsm8k
ap.add_argument('--top-lim', default='-0.29,0.03')
ap.add_argument('--bot-lim', default='-1.02,-0.94')
a = ap.parse_args()

DATASETS = ['aime24', 'aime25', 'amc23', 'cmimc25', 'hmmt25', 'math500', 'gsm8k']
LABELS = ["AIME'24", "AIME'25", "AMC'23", "CMIMC'25", "HMMT'25", "MATH-500", "GSM-8K"]

xs, ip, ang, used_lab = [], [], [], []
for i, ds in enumerate(DATASETS):
    p = f'{a.src}/{ds}.json'
    if not os.path.exists(p):
        print(f'skip {ds}: not computed')
        continue
    d = json.load(open(p))
    key_cos = 'catcos_all' if a.scope == 'all' else 'catcos_late'
    key_ip = 'sum_ip_all' if a.scope == 'all' else 'sum_ip_late'
    if key_cos not in d:
        print(f'skip {ds}: no {key_cos} (rerun with the patched script)')
        continue
    c = d[key_cos]
    deg = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    D = d['dim_total'] if a.scope == 'all' else d['dim_late']
    if a.angle_mode == 'dev':
        v = (90.0 - deg) * 3600.0
        ang_lab = 'Angle'
        leg2 = r'$90^\circ-\angle(\Delta W_\perp, G_\perp)$'
        ref = 0.0
    elif a.angle_mode == 'ratio':
        v = abs(c) / (1.0 / math.sqrt(D))
        ang_lab = 'Angle'
        leg2 = r'$|\cos|\,/\,(1/\sqrt{D})$'
        ref = 1.0
    else:
        v = deg
        ang_lab = 'Angle'
        leg2 = r'$\angle(\Delta W_\perp, G_\perp)$'
        ref = 90.0
    xs.append(i); ip.append(d[key_ip]); ang.append(v); used_lab.append(LABELS[i])

if not xs:
    raise SystemExit('no datasets had the cosine fields; rerun the patched job first')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif', 'font.size': 10,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False,
})
C_IP, C_ANG = '#1f4e9c', '#b2182b'
top_lim = tuple(float(t) for t in a.top_lim.split(','))
bot_lim = tuple(float(t) for t in a.bot_lim.split(','))

fig, (axT, axB) = plt.subplots(
    2, 1, sharex=True, figsize=(5.8, 3.6),
    gridspec_kw=dict(height_ratios=[3.1, 1.0], hspace=0.10))

# ---- inner product, drawn on both panels and clipped by each ylim -------------
l1, = axT.plot(xs, ip, color=C_IP, marker='o', ms=4.5, lw=1.6, mfc='white',
               mew=1.4, label=r'Inner product $\langle\Delta W_\perp, G_\perp\rangle$')
axB.plot(xs, ip, color=C_IP, marker='o', ms=4.5, lw=1.6, mfc='white', mew=1.4)
for ax in (axT, axB):
    ax.set_ylabel(r'$\langle\Delta W_\perp, G_\perp\rangle$' if ax is axT else '',
                  color=C_IP, labelpad=6)
    ax.tick_params(axis='y', colors=C_IP)
    ax.grid(axis='y', color='0.9', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
axT.axhline(0.0, color='0.55', lw=0.8, ls='--', zorder=0)
axT.set_ylim(*top_lim)
axB.set_ylim(*bot_lim)

# ---- angle, on the top panel only (its range needs no breaking) --------------
axR = axT.twinx()
l2, = axR.plot(xs, ang, color=C_ANG, marker='s', ms=4.2, lw=1.6, ls='--', mfc='white',
               mew=1.4, label=leg2)
axR.set_ylabel(ang_lab, color=C_ANG, labelpad=6)
axR.tick_params(axis='y', colors=C_ANG)
axR.axhline(ref, color=C_ANG, lw=0.7, ls=':', alpha=0.5, zorder=0)
lo, hi = min(ang), max(ang)
if hi - lo < 1.0:
    pad = max(0.05 * max(abs(lo), 1.0), (hi - lo) * 0.35)
    axR.set_ylim(lo - pad, hi + pad)

# ---- break marks -------------------------------------------------------------
d = 0.011
for ax, ys in ((axT, (-d, +d)), (axB, (1 - d, 1 + d))):
    for xf in (-d, 1 - d):
        ax.plot((xf, xf + 2 * d), ys, transform=ax.transAxes, color='k',
                clip_on=False, lw=0.9)
axT.spines['bottom'].set_visible(False)
axB.spines['top'].set_visible(False)
axR.spines['bottom'].set_visible(False)
axT.tick_params(bottom=False)

axB.set_xticks(xs)
axB.set_xticklabels(used_lab, fontsize=9, rotation=28, ha='right', rotation_mode='anchor')
axB.set_xlim(-0.45, len(xs) - 0.55)

axT.legend(handles=[l1, l2], loc='lower left', fontsize=8.5,
           handlelength=2.4, borderaxespad=0.6)
fig.subplots_adjust(bottom=0.20)
fig.savefig(a.out + '.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(a.out + '.png', dpi=300, bbox_inches='tight', pad_inches=0.02)

print(f'scope={a.scope} angle_mode={a.angle_mode} n={len(xs)}')
for lab, v, t in zip(used_lab, ip, ang):
    print(f'  {lab:<10} ip={v:+.5f}   right-axis value={t:+.3f}')
print(f'left axis: top {top_lim} bottom {bot_lim}   right axis: {axR.get_ylim()}')
print(f'wrote {a.out}.pdf and .png')
