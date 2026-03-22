import os
import sys
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("graphs", exist_ok=True)

sys.path.insert(0, ".")
from src.recommender.service import RecommenderService  # noqa: E402

# ---------- colour palette ----------
BG      = "#111111"
SURFACE = "#1a1a1a"
BORDER  = "#2c2c2c"
BLUE    = "#6b9ee0"
GREEN   = "#6abf7b"
AMBER   = "#c9a96e"
TEXT    = "#ececec"
MUTED   = "#999999"

STRATEGY_COLORS = {
    "readers_also":    BLUE,
    "hidden_gems":     BLUE,
    "similar_themes":  GREEN,
    "same_author":     GREEN,
    "age_group":       GREEN,
    "top_picks":       AMBER,
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  MUTED,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "font.family":      "sans-serif",
    "font.size":        11,
})

# ---------- fetch recommendations ----------
svc = RecommenderService()

results_raw = svc.search_books("the fellowship of the ring")
seed = results_raw[0]
seed_label = f'Seed: "{seed.title}" - {seed.author}'
print(f"Seed: {seed.isbn} | {seed.title} | {seed.author}")

all_recs = svc.recommend_all(seed.isbn, top_k=3)

# build row data: list of (strategy_label, strategy_name, [rec1, rec2, rec3])
rows = []
for result in all_recs:
    padded = list(result.recommendations) + [None, None, None]
    rows.append((result.strategy_label, result.strategy_name, padded[:3]))

# ---------- layout constants ----------
N_ROWS    = len(rows)
N_COLS    = 4  # strategy | #1 | #2 | #3
COL_W     = [0.20, 0.265, 0.265, 0.265]  # fractional widths
ROW_H     = 0.115   # axes fraction per data row
HEADER_H  = 0.07
ACCENT_W  = 0.006   # width of left accent bar (axes frac)
PAD       = 0.012
SEED_Y    = 0.97

fig_h = 3.2 + N_ROWS * 0.72
fig, ax = plt.subplots(figsize=(13, fig_h))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(BG)

def trunc(s, n):
    if s is None:
        return ""
    return s[:n] + ("..." if len(s) > n else "")

# compute x positions of column left edges
col_lefts = []
x = 0.0
for w in COL_W:
    col_lefts.append(x)
    x += w

# total table height in axes coords
table_top    = SEED_Y - 0.06
total_height = HEADER_H + N_ROWS * ROW_H
table_bottom = table_top - total_height

# ---------- seed label ----------
ax.text(
    col_lefts[0], SEED_Y,
    seed_label,
    color=MUTED,
    fontsize=9,
    fontstyle="italic",
    va="top",
    ha="left",
)

# ---------- header row ----------
header_y = table_top
header_labels = ["Strategy", "#1", "#2", "#3"]
for ci, label in enumerate(header_labels):
    x0 = col_lefts[ci]
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, header_y - HEADER_H),
        COL_W[ci], HEADER_H,
        boxstyle="square,pad=0",
        facecolor=SURFACE,
        edgecolor="none",
        zorder=1,
    ))
    ax.text(
        x0 + COL_W[ci] / 2,
        header_y - HEADER_H / 2,
        label,
        color=MUTED,
        fontsize=9.5,
        fontfamily="monospace",
        ha="center",
        va="center",
        zorder=2,
    )

# separator under header
ax.plot([0, 1], [header_y - HEADER_H, header_y - HEADER_H], color=BORDER, lw=0.8, zorder=3)

# ---------- data rows ----------
for ri, (strategy_label, strategy_name, recs) in enumerate(rows):
    row_top = table_top - HEADER_H - ri * ROW_H
    row_bot = row_top - ROW_H
    bg_col  = SURFACE if ri % 2 == 0 else BG
    accent_col = STRATEGY_COLORS.get(strategy_name, MUTED)

    # row background
    for ci in range(N_COLS):
        ax.add_patch(mpatches.FancyBboxPatch(
            (col_lefts[ci], row_bot),
            COL_W[ci], ROW_H,
            boxstyle="square,pad=0",
            facecolor=bg_col,
            edgecolor="none",
            zorder=1,
        ))

    # accent bar on strategy column left edge
    ax.add_patch(mpatches.FancyBboxPatch(
        (col_lefts[0], row_bot + PAD / 2),
        ACCENT_W, ROW_H - PAD,
        boxstyle="square,pad=0",
        facecolor=accent_col,
        edgecolor="none",
        zorder=3,
    ))

    # strategy name
    ax.text(
        col_lefts[0] + ACCENT_W + PAD * 2,
        (row_top + row_bot) / 2,
        strategy_label,
        color=TEXT,
        fontsize=9.5,
        va="center",
        ha="left",
        zorder=4,
    )

    # recommendation cells
    for ci, rec in enumerate(recs, start=1):
        cx = col_lefts[ci]
        cw = COL_W[ci]
        cy = (row_top + row_bot) / 2

        if rec is None:
            ax.text(cx + PAD, cy, "—", color=MUTED, fontsize=9, va="center", ha="left", zorder=4)
        else:
            title_str  = trunc(rec.title.title(), 38)
            author_str = trunc(rec.author.title(), 28)

            ax.text(
                cx + PAD,
                cy + ROW_H * 0.14,
                title_str,
                color=TEXT,
                fontsize=8.5,
                va="center",
                ha="left",
                zorder=4,
            )
            ax.text(
                cx + PAD,
                cy - ROW_H * 0.20,
                author_str,
                color=MUTED,
                fontsize=7.8,
                fontstyle="italic",
                va="center",
                ha="left",
                zorder=4,
            )

    # separator line below row
    ax.plot([0, 1], [row_bot, row_bot], color=BORDER, lw=0.5, zorder=3)

# outer border lines
ax.plot([0, 1], [table_top, table_top],                   color=BORDER, lw=0.8, zorder=3)
ax.plot([0, 1], [table_top - total_height, table_top - total_height], color=BORDER, lw=0.8, zorder=3)

fig.tight_layout(pad=0.3)
out = "graphs/graph5_strategy_comparison.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")