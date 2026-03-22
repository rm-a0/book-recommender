import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.makedirs("graphs", exist_ok=True)

# ---------- colour palette ----------
BG      = "#111111"
SURFACE = "#1a1a1a"
BORDER  = "#2c2c2c"
RAW     = "#6b9ee0"   # blue - visible on dark bg (original #3a3a3a was invisible)
PROC    = "#c9a96e"
RED     = "#e07070"
BLUE    = "#6b9ee0"
GREEN   = "#6abf7b"
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

# ---------- load raw ratings ----------
raw = pd.read_csv(
    "data/raw/Ratings.csv",
    sep=",",
    encoding="ISO-8859-1",
    on_bad_lines="skip",
)
raw.columns = [c.encode("ascii", "ignore").decode().strip().strip('"').strip() for c in raw.columns]
print(f"[debug] raw columns: {list(raw.columns)}")
rating_col = next(c for c in raw.columns if "rating" in c.lower())
print(f"[debug] rating_col={rating_col!r}")
# cast rating to int (CSV stores everything as string)
raw[rating_col] = pd.to_numeric(raw[rating_col], errors="coerce")
raw = raw.dropna(subset=[rating_col])
raw[rating_col] = raw[rating_col].astype(int)
print(f"[debug] rows after cast: {len(raw):,}")

# ---------- load processed ratings ----------
proc = pd.read_parquet("data/processed/ratings.parquet")

# ---------- compute counts ----------
all_ratings = list(range(0, 11))

# reindex guarantees integer labels 0-10 are always present with fill_value=0
raw_counts  = raw[rating_col].value_counts().reindex(all_ratings, fill_value=0)
proc_counts = proc["Book-Rating"].value_counts().reindex(all_ratings, fill_value=0)

raw_vals  = [int(raw_counts[r])  for r in all_ratings]
proc_vals = [int(proc_counts[r]) for r in all_ratings]

implicit_zeros = raw_vals[0]
raw_total      = int(raw[rating_col].notna().sum())
explicit_count = raw_total - implicit_zeros
proc_total     = int(len(proc))

print(f"Raw total:       {raw_total:,}")
print(f"Implicit zeros:  {implicit_zeros:,}")
print(f"Explicit count:  {explicit_count:,}")
print(f"Processed total: {proc_total:,}")

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_facecolor(SURFACE)

w = 0.38
xs = np.arange(len(all_ratings))

bars_raw  = ax.bar(xs - w / 2, raw_vals,  width=w, color=RAW,  label="raw",       zorder=3)
bars_proc = ax.bar(xs + w / 2, proc_vals, width=w, color=PROC, label="processed", zorder=3)

# grid
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

# y-axis thousands formatter
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

ax.set_xticks(xs)
ax.set_xticklabels([str(r) for r in all_ratings], color=MUTED)
ax.set_xlabel("Book Rating", labelpad=8)
ax.set_ylabel("Count", labelpad=8)

# annotate implicit-zero bar
zero_bar = bars_raw[0]
bar_x = zero_bar.get_x() + zero_bar.get_width() / 2
bar_y = zero_bar.get_height()
label_text = f"{implicit_zeros // 1000}k implicit zeros dropped"

ax.annotate(
    label_text,
    xy=(bar_x, bar_y),
    xytext=(bar_x + 1.6, bar_y * 0.88),
    color=RED,
    fontsize=9.5,
    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2),
    ha="left",
    va="center",
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
out = "graphs/graph1_rating_dist.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")