#!/usr/bin/env python3
"""Bar version of fig_cos_ip: inner product (left axis) and angle (right axis) per dataset.

Same data and styling as plot_cos_ip.py, but drawn as bars on a SINGLE, COMPLETE y-axis
(no broken panels). Because both quantities are negative, the bars hang downward from
zero.

gsm8k's inner product (-0.976) is ~4x the next largest, so its bar is drawn as an
INTERRUPTED bar: two segments with a gap and diagonal break marks, the usual convention
for a bar that dwarfs its neighbours without pretending the value is smaller than it is.
The bar still ends at its true value; only the middle is elided.

  python plot_cos_ip_bar.py
  python plot_cos_ip_bar.py --angle-mode ratio
  python plot_cos_ip_bar.py --no-break          # draw gsm8k as a plain bar
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
ap.add_argument('--out', default="fig_cos_ip_bar")
# dest is needed: `a.break` is not valid Python, `break` being a keyword
ap.add_argument('--break', dest='break_bar', action='store_true',
                help='interrupt the largest bar (default: draw it whole)')
ap.add_argument('--gap', default='-0.28,-0.42', help='elided band, only with --break')
a = ap.parse_args()

DATASETS = ['aime24', 'aime25', 'amc23', 'cmimc25', 'hmmt25', 'math500', 'gsm8k']
LABELS = ["AIME'24", "AIME'25", "AMC'23", "CMIMC'25", "HMMT'25", "MATH-500", "GSM-8K"]

xs, ip, ang, used_lab = [], [], [], []
for i, ds in enumerate(DATASETS):
    p = f'{a.src}/{ds}.json'
    if not os.path.exists(p):
        print(f'skip {ds}: not computed'); continue
    d = json.load(open(p))
    key_cos = 'catcos_all' if a.scope == 'all' else 'catcos_late'
    key_ip = 'sum_ip_all' if a.scope == 'all' else 'sum_ip_late'
    if key_cos not in d:
        print(f'skip {ds}: no {key_cos}'); continue
    c = d[key_cos]
    deg = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    D = d['dim_total'] if a.scope == 'all' else d['dim_late']
    if a.angle_mode == 'dev':
        v, ref = (90.0 - deg) * 3600.0, 0.0
        leg2 = r'$90^\circ-\angle(\Delta W_\perp, G_\perp)$'
    elif a.angle_mode == 'ratio':
        v, ref = abs(c) / (1.0 / math.sqrt(D)), 1.0
        leg2 = r'$|\cos|\,/\,(1/\sqrt{D})$'
    else:
        v, ref = deg, 90.0
        leg2 = r'$\angle(\Delta W_\perp, G_\perp)$'
    xs.append(i); ip.append(d[key_ip]); ang.append(v); used_lab.append(LABELS[i])

if not xs:
    raise SystemExit('no data')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif', 'font.size': 10,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False,
})
C_IP, C_ANG = '#1f4e9c', '#b2182b'
YLIM_BOT = -0.3          # bars past this are drawn to the edge and annotated
W = 0.34
gap_hi, gap_lo = (float(t) for t in a.gap.split(','))
# the bar to interrupt, if asked for: the one far beyond all the others
brk = max(range(len(ip)), key=lambda k: abs(ip[k])) if a.break_bar else None

fig, ax1 = plt.subplots(figsize=(6.0, 3.4))
ax2 = ax1.twinx()

for k in range(len(xs)):
    # bar CENTRES, placed either side of the tick so the two series never overlap
    xl, xr = xs[k] - W / 2 - 0.035, xs[k] + W / 2 + 0.035
    # ---- left axis: inner product (bars hang down from zero) ----
    if k == brk and ip[k] < gap_lo:
        # upper stub (0 -> gap_hi) and lower segment (gap_lo -> true value)
        ax1.bar(xl, gap_hi, width=W, bottom=0.0, color=C_IP,
                edgecolor='black', lw=0.5, zorder=3)
        ax1.bar(xl, ip[k] - gap_lo, width=W, bottom=gap_lo, color=C_IP,
                edgecolor='black', lw=0.5, zorder=3)
        for yb in (gap_hi, gap_lo):          # diagonal break marks across the bar
            ax1.plot([xl, xl + W], [yb + 0.030, yb - 0.030], color='white',
                     lw=2.2, solid_capstyle='butt', zorder=4, clip_on=False)
            ax1.plot([xl, xl + W], [yb + 0.030, yb - 0.030], color='black',
                     lw=0.9, solid_capstyle='butt', zorder=5, clip_on=False)
        ax1.text(xl - W / 2 - 0.05, ip[k], f'{ip[k]:.3f}', ha='right', va='center',
                 fontsize=8, color=C_IP)
    elif ip[k] < YLIM_BOT:
        # value runs past the bottom of the axis: draw the bar to the edge, mark the cut
        # with a downward arrowhead, and print the true value on the bar
        ax1.bar(xl, YLIM_BOT, width=W, color=C_IP, edgecolor='black', lw=0.5, zorder=3)
        ax1.plot([xl], [YLIM_BOT], marker='v', ms=7, color=C_IP, clip_on=False, zorder=6)
        ax1.text(xl - W / 2 - 0.045, YLIM_BOT + 0.016, f'{ip[k]:.3f}', ha='right',
                 va='bottom', fontsize=9, color=C_IP, fontweight='bold', zorder=7)
    else:
        ax1.bar(xl, ip[k], width=W, color=C_IP, edgecolor='black',
                lw=0.5, zorder=3)
    # ---- right axis: angle ----
    ax2.bar(xr, ang[k], width=W, color=C_ANG, edgecolor='black',
            lw=0.5, alpha=0.85, zorder=3)

ax1.axhline(0.0, color='0.4', lw=0.9, zorder=2)
ax1.set_ylabel(r'$\langle\Delta W_\perp,\,G_\perp\rangle$', color=C_IP, labelpad=6)
ax2.set_ylabel('Angle', color=C_ANG, labelpad=6)
ax1.tick_params(axis='y', colors=C_IP)
ax2.tick_params(axis='y', colors=C_ANG)
# both limits END AT ZERO: with a positive top the right axis's zero line sits lower
# than the left's, and the red bars visibly hang from a different height than the blue
ax1.set_ylim(YLIM_BOT, 0.0)
ax2.set_ylim(-29.0, 0.0)
ax1.set_xticks(xs)
ax1.set_xticklabels(used_lab, fontsize=9, rotation=28, ha='right', rotation_mode='anchor')
ax1.set_xlim(-0.6, len(xs) - 0.4)
ax1.grid(axis='y', color='0.9', lw=0.6, zorder=0)
ax1.set_axisbelow(True)

h1 = plt.Rectangle((0, 0), 1, 1, fc=C_IP, ec='black', lw=0.5)
h2 = plt.Rectangle((0, 0), 1, 1, fc=C_ANG, ec='black', lw=0.5, alpha=0.85)
ax1.legend([h1, h2],
           [r'Inner product $\langle\Delta W_\perp, G_\perp\rangle$', 'Angle'],
           loc='lower left', fontsize=8.5, handlelength=1.6, borderaxespad=0.6)

fig.subplots_adjust(left=0.12, right=0.88, top=0.97, bottom=0.22)
fig.savefig(a.out + '.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(a.out + '.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f'scope={a.scope} mode={a.angle_mode}  broken bar = {used_lab[brk] if brk is not None else "none"}')
for lab, v, t in zip(used_lab, ip, ang):
    print(f'  {lab:<10} ip={v:+.5f}   angle={t:+.3f}')
print(f'wrote {a.out}.pdf and .png')
