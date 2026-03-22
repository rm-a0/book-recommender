import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.makedirs("graphs", exist_ok=True)

# ---------- colour palette ----------
BG      = "#111111"
SURFACE = "#1a1a1a"
BORDER  = "#2c2c2c"
TEXT    = "#ececec"
MUTED   = "#999999"

GENRE_COLORS = {
    "Fantasy":  "#6b9ee0",
    "Mystery":  "#e07070",
    "Romance":  "#c9a96e",
    "Sci-Fi":   "#a29bfe",
    "Horror":   "#fd79a8",
    "History":  "#6abf7b",
    "Other":    "#2c2c2c",
}

GENRE_KEYWORDS = {
    "Fantasy":  ["fantasy", "magic", "dragon", "wizard", "tolkien"],
    "Mystery":  ["mystery", "detective", "crime", "thriller", "murder"],
    "Romance":  ["romance", "love", "historical romance"],
    "Sci-Fi":   ["science fiction", "space", "dystopia", "futur"],
    "Horror":   ["horror", "supernatural", "ghost", "vampire"],
    "History":  ["history", "historical", "biography", "memoir"],
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

# ---------- load embeddings ----------
embeddings = np.load("artifacts/book_embeddings.npy")
with open("artifacts/embedding_isbn_map.json") as f:
    isbn_list = json.load(f)

meta = pd.read_parquet("data/processed/book_metadata_enriched.parquet")
meta = meta.set_index("ISBN")

# ---------- assign genre ----------
def assign_genre(subjects_str):
    if not isinstance(subjects_str, str) or subjects_str.strip() == "":
        return "Other"
    s = subjects_str.lower()
    for genre, keywords in GENRE_KEYWORDS.items():
        for kw in keywords:
            if kw in s:
                return genre
    return "Other"

genres = []
for isbn in isbn_list:
    if isbn in meta.index:
        subj = meta.at[isbn, "subjects"]
        genres.append(assign_genre(subj))
    else:
        genres.append("Other")

genres = np.array(genres)

# ---------- sample 6000 ----------
rng = np.random.default_rng(42)
n_total = len(embeddings)
sample_idx = rng.choice(n_total, size=min(6000, n_total), replace=False)
sample_idx = np.sort(sample_idx)

emb_sample    = embeddings[sample_idx]
genres_sample = genres[sample_idx]
n_sampled     = len(sample_idx)

print(f"Total embeddings: {n_total:,}")
print(f"Sampled:          {n_sampled:,}")

# ---------- UMAP ----------
import umap  # noqa: E402  (import after heavy numpy work)

reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=15,
    min_dist=0.08,
    metric="cosine",
)
proj = reducer.fit_transform(emb_sample)

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_facecolor(BG)

# Other first
mask_other = genres_sample == "Other"
ax.scatter(
    proj[mask_other, 0],
    proj[mask_other, 1],
    s=3,
    alpha=0.2,
    color=GENRE_COLORS["Other"],
    linewidths=0,
    zorder=1,
)

# named genres on top
for genre, color in GENRE_COLORS.items():
    if genre == "Other":
        continue
    mask = genres_sample == genre
    ax.scatter(
        proj[mask, 0],
        proj[mask, 1],
        s=5,
        alpha=0.65,
        color=color,
        linewidths=0,
        label=genre,
        zorder=2,
    )

# no ticks or spines
ax.set_xticks([])
ax.set_yticks([])
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

# legend (exclude Other)
handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=GENRE_COLORS[g],
           markersize=7, label=g)
    for g in GENRE_COLORS if g != "Other"
]
legend = ax.legend(
    handles=handles,
    frameon=True,
    framealpha=0.2,
    edgecolor=BORDER,
    labelcolor=MUTED,
    fontsize=9.5,
    loc="upper right",
)
legend.get_frame().set_facecolor(SURFACE)

# bottom-left annotation
ax.text(
    0.01, 0.01,
    f"UMAP - {n_sampled:,} books sampled",
    transform=ax.transAxes,
    color=MUTED,
    fontsize=8.5,
    va="bottom",
    ha="left",
)

fig.tight_layout(pad=0.4)
out = "graphs/graph4_umap.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")