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
RAW     = "#6b9ee0"   # blue - visible on dark bg
PROC    = "#c9a96e"
RED     = "#e07070"
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

# ---------- load data ----------
raw = pd.read_csv(
    "data/raw/Ratings.csv",
    sep=None,
    engine="python",
    encoding="ISO-8859-1",
    on_bad_lines="skip",
)
raw.columns = [c.encode("ascii", "ignore").decode().strip().strip('"').strip() for c in raw.columns]
print(f"[debug] raw columns: {list(raw.columns)}")

# find columns by case-insensitive substring match
rating_col = next(c for c in raw.columns if "rating" in c.lower())
isbn_col   = next(c for c in raw.columns if "isbn"   in c.lower())
print(f"[debug] rating_col={rating_col!r}  isbn_col={isbn_col!r}")

# cast rating to int (CSV stores everything as string)
raw[rating_col] = pd.to_numeric(raw[rating_col], errors="coerce")
raw = raw.dropna(subset=[rating_col])
raw[rating_col] = raw[rating_col].astype(int)
print(f"[debug] rows after cast: {len(raw):,}")

# explicit only from raw (rating > 0)
raw_explicit = raw[raw[rating_col] > 0]
print(f"[debug] explicit rows: {len(raw_explicit):,}")
raw_per_book = raw_explicit.groupby(isbn_col).size()

proc = pd.read_parquet("data/processed/ratings.parquet")
proc_per_book = proc.groupby("ISBN").size()

raw_book_count  = len(raw_per_book)
proc_book_count = len(proc_per_book)
books_dropped   = raw_book_count - proc_book_count
pct_dropped     = (100.0 * books_dropped / raw_book_count) if raw_book_count > 0 else 0.0

print(f"Raw book count:       {raw_book_count:,}")
print(f"Processed book count: {proc_book_count:,}")
print(f"Books dropped:        {books_dropped:,} ({pct_dropped:.1f}%)")

# ---------- log bins ----------
max_val = max(raw_per_book.max(), proc_per_book.max())
bins = np.logspace(0, np.log10(max_val), 70)

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_facecolor(SURFACE)

ax.hist(
    raw_per_book,
    bins=bins,
    color=RAW,
    alpha=0.85,
    label=f"raw ({raw_book_count:,} books)",
    zorder=3,
)
ax.hist(
    proc_per_book,
    bins=bins,
    color=PROC,
    alpha=0.75,
    label=f"processed ({proc_book_count:,} books)",
    zorder=4,
)

ax.set_xscale("log")
ax.set_yscale("log")

# cutoff at 5
ax.axvline(x=5, color=RED, linestyle="--", linewidth=1.4, zorder=5)
ax.axvspan(0.8, 5, color=RED, alpha=0.07, zorder=2)

# "dropped" label above shaded region
ylims = ax.get_ylim()
ax.text(
    2.2,
    ylims[1] * 0.55,
    "dropped",
    color=RED,
    fontsize=9.5,
    ha="center",
    va="bottom",
    zorder=6,
)

# grid
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

ax.set_xlabel("Ratings per book", labelpad=8)
ax.set_ylabel("Number of books", labelpad=8)

legend = ax.legend(
    frameon=True,
    framealpha=0.15,
    edgecolor=BORDER,
    labelcolor=MUTED,
    fontsize=10,
)
legend.get_frame().set_facecolor(SURFACE)

fig.tight_layout()
out = "graphs/graph2_long_tail.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")