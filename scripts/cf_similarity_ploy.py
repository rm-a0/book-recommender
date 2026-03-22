import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.sparse import load_npz

os.makedirs("graphs", exist_ok=True)

# ---------- colour palette ----------
BG      = "#111111"
SURFACE = "#1a1a1a"
BORDER  = "#2c2c2c"
BLUE    = "#6b9ee0"
AMBER   = "#c9a96e"
TEXT    = "#ececec"
MUTED   = "#999999"
GRID    = "#1e1e1e"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   SURFACE,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  MUTED,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "font.family":      "sans-serif",
    "font.size":        11,
})

# ---------- load sparse matrix ----------
sim = load_npz("artifacts/item_similarity.npz")
data = sim.data

shape   = sim.shape
nnz     = sim.nnz
total   = shape[0] * shape[1]
fill_pct = 100.0 * nnz / total
median_val = float(np.median(data))
mean_val   = float(np.mean(data))
min_val    = float(np.min(data))
max_val    = float(np.max(data))

print(f"Matrix shape:  {shape}")
print(f"NNZ:           {nnz:,}")
print(f"Sparsity:      {100.0 - fill_pct:.4f}%  (fill rate {fill_pct:.4f}%)")
print(f"Median:        {median_val:.4f}")
print(f"Mean:          {mean_val:.4f}")
print(f"Min:           {min_val:.4f}")
print(f"Max:           {max_val:.4f}")

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_facecolor(SURFACE)

ax.hist(data, bins=80, color=BLUE, alpha=0.85, zorder=3)

# median line
ax.axvline(
    x=median_val,
    color=AMBER,
    linestyle="--",
    linewidth=1.5,
    zorder=5,
    label=f"median = {median_val:.3f}",
)

# grid
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

# y thousands formatter
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else str(int(x)))
)

ax.set_xlabel("Adjusted cosine similarity", labelpad=8)
ax.set_ylabel("Pairs", labelpad=8)

# annotation box top-right
info_lines = [
    f"non-zero pairs: {nnz:,}",
    f"matrix: {shape[0]:,} x {shape[1]:,}",
    f"fill rate: {fill_pct:.3f}%",
]
info_text = "\n".join(info_lines)
ax.text(
    0.97, 0.95,
    info_text,
    transform=ax.transAxes,
    color=MUTED,
    fontsize=9,
    va="top",
    ha="right",
    linespacing=1.6,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor=SURFACE,
        edgecolor=BORDER,
        alpha=0.8,
    ),
)

legend = ax.legend(
    frameon=True,
    framealpha=0.15,
    edgecolor=BORDER,
    labelcolor=MUTED,
    fontsize=10,
)
legend.get_frame().set_facecolor(SURFACE)

fig.tight_layout()
out = "graphs/graph3_cf_similarity.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")