#!/usr/bin/env python3
"""Appendix A.3 figure: attention entropy vs. pretraining step (Pythia-1.4B).
Data rescued from session 15acc6dc run log (ent_cot.py); see data/saturation_entropy.json.
Style matches fig5_combined.py."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data", "saturation_entropy.json")
OUT = os.path.join(DIR, "figA4_saturation.png")

d = json.load(open(DATA))
steps = np.array(d["steps"], dtype=float)
H = np.array(d["H_mean"])
HL = np.array(d["H_last4"])

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.linewidth': 0.6, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
})

fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=250)
C1, C2 = "#1565C0", "#F44336"

# final-level guides for convergence reading
ax.axhline(2.3719, color=C1, ls="--", lw=1.0, alpha=0.6)
ax.axhline(0.9759, color=C2, ls="--", lw=1.0, alpha=0.6)

ax.plot(steps, H, "o-", color=C1, lw=1.6, ms=5, label="mean $H_t$ (all layers)")
ax.plot(steps, HL, "s-", color=C2, lw=1.6, ms=5, label="mean $H_t$, last four layers")

ax.set_xlabel("Pretraining step")
ax.set_ylabel("Attention entropy (nats)")
ax.set_xlim(-4000, 148000)
ax.set_ylim(0.5, 6.3)
ax.legend(loc="upper right", fontsize=9.5, frameon=False)
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace('.png', '.pdf'), bbox_inches="tight")
print("wrote", OUT, "and", OUT.replace('.png', '.pdf'))
