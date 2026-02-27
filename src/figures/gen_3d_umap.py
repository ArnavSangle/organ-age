"""
gen_3d_umap.py
==============
Generate Figures A, B, C: 3D UMAP latent-space scatters for the ISEF poster.

  A – Unaligned embeddings,  colored by modality  (shows clustering)
  B – Aligned embeddings,    colored by age        (shows age gradient)
  C – Aligned embeddings,    colored by modality   (shows mixing)

Saves interactive HTML + static PNG to figures/poster/.
"""
import os, sys, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap as umap_lib

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTDIR   = os.path.join(ROOT, "figures", "poster")
HTMLDIR  = os.path.join(OUTDIR, "html")
CACHEDIR = os.path.join(OUTDIR, "cache")
os.makedirs(OUTDIR,   exist_ok=True)
os.makedirs(HTMLDIR,  exist_ok=True)
os.makedirs(CACHEDIR, exist_ok=True)

# ── Dark-theme layout template ────────────────────────────────────────────────
BG   = "#0f0f1a"
FONT = "Georgia, 'Times New Roman', serif"
TEXT_SCALE = 2.2
AXIS_TEXT_SCALE = TEXT_SCALE / 2.0

def fs(px):
    """Scale font sizes for poster readability without changing figure geometry."""
    return int(round(px * TEXT_SCALE))

def fas(px):
    """Axis text/ticks should be smaller than titles/legends."""
    return int(round(px * AXIS_TEXT_SCALE))

def _split_title_lines(text, max_chars=80):
    if len(text) <= max_chars:
        return [text]
    if " — " in text:
        left, right = text.split(" — ", 1)
        if max(len(left), len(right)) <= (max_chars - 8):
            return [left, right]
    mid = len(text) // 2
    cut = text.rfind(" ", 0, mid)
    if cut == -1:
        cut = text.find(" ", mid)
    if cut == -1:
        return [text]
    return [text[:cut], text[cut + 1:]]

def _fit_plotly_title(text, base_px=52, min_px=18):
    lines = _split_title_lines(text)
    longest = max(len(line) for line in lines)
    line_count = len(lines)
    max_by_width = int(2100 / max(0.56 * longest, 1.0))
    max_by_margin = int(115 / (1.22 * line_count))
    size = min(fs(base_px), max_by_width, max_by_margin)
    size = max(size, fs(min_px))
    return dict(
        text="<br>".join(lines),
        font=dict(family=FONT, color="white", size=size),
        x=0.5,
        y=0.975,
        xanchor="center",
        yanchor="top",
        automargin=True,
    )

MOD_COLORS = {"rna": "#B45309", "xray": "#3B82F6", "mri": "#0F766E"}
MOD_NAMES  = {"rna": "RNA (GTEx)", "xray": "X-ray (CheXpert)", "mri": "MRI (IXI)"}

def _axis_base(rng=None):
    """Base axis style dict; optionally fixed range to prevent auto-snap."""
    d = dict(
        backgroundcolor=BG,
        gridcolor="#2a2a5a",
        showgrid=True,
        zeroline=True,
        zerolinecolor="#6868cc",
        zerolinewidth=5,
        showline=True,
        linecolor="#5a5aaa",
        linewidth=4,
        showticklabels=False,   # UMAP axes have no meaningful units
    )
    if rng is not None:
        d["range"] = rng
    return d

def dark_scene(xr=None, yr=None, zr=None):
    """3D scene dict with optional fixed ranges per axis."""
    title_font = dict(family=FONT, color="rgba(160,160,220,0.8)", size=fas(28))
    return dict(
        xaxis=dict(**_axis_base(xr), title=dict(text="UMAP 1", font=title_font)),
        yaxis=dict(**_axis_base(yr), title=dict(text="UMAP 2", font=title_font)),
        zaxis=dict(**_axis_base(zr), title=dict(text="UMAP 3", font=title_font)),
        bgcolor=BG,
        aspectmode="cube",
        camera=dict(eye=dict(x=1.4, y=1.4, z=0.7)),
    )

def base_layout(title, xr=None, yr=None, zr=None):
    return go.Layout(
        scene=dark_scene(xr, yr, zr),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT),
        legend=dict(
            font=dict(family=FONT, color="white", size=fs(38)),
            bgcolor="rgba(15,15,30,0.7)",
            bordercolor="rgba(100,100,180,0.4)",
            borderwidth=2,
            itemsizing="constant",
            tracegroupgap=12,
        ),
        title=_fit_plotly_title(title),
        margin=dict(l=0, r=0, t=130, b=0),
        width=2400, height=2400,
    )

def save(fig, stem):
    html = os.path.join(HTMLDIR, f"{stem}.html")
    png  = os.path.join(OUTDIR,  f"{stem}.png")
    fig.write_html(html)
    fig.write_image(png, scale=1)
    print(f"  Saved {stem}.png  |  html/{stem}.html")

# ── Load parquets ─────────────────────────────────────────────────────────────
print("Loading parquets …")
t0 = time.time()

BASE_PATH    = os.path.join(ROOT, "data/processed/aligned/v3_aligned_base.parquet")
CONTRA_PATH  = os.path.join(ROOT, "data/processed/aligned/v3_aligned_contrastive.parquet")

# Identify embedding column groups
meta_cols = {'sample_id','subject_id','age','sex','organ','modality','source','features','embedding'}

df_base   = pd.read_parquet(BASE_PATH).reset_index(drop=True)
df_contra = pd.read_parquet(CONTRA_PATH).reset_index(drop=True)

emb_cols = [c for c in df_base.columns   if c not in meta_cols]   # 192 emb_* cols
z_cols   = [c for c in df_contra.columns if c.startswith("z_")]   # 768 z_* cols
print(f"  base emb cols: {len(emb_cols)}, aligned z cols: {len(z_cols)}  ({time.time()-t0:.1f}s)")

# ── Stratified subsample ──────────────────────────────────────────────────────
np.random.seed(42)
N_XRAY = 5000
N_RNA  = 5000

rna_pos  = df_base.index[df_base.modality == "rna"].tolist()
mri_pos  = df_base.index[df_base.modality == "mri"].tolist()
xray_pos = df_base.index[df_base.modality == "xray"].tolist()
xray_pos = np.random.choice(xray_pos, min(N_XRAY, len(xray_pos)), replace=False).tolist()
rna_pos  = np.random.choice(rna_pos,  min(N_RNA,  len(rna_pos)),  replace=False).tolist()

sample_pos = xray_pos + rna_pos + mri_pos
print(f"  Subsample: xray={len(xray_pos)}, rna={len(rna_pos)}, mri={len(mri_pos)}, total={len(sample_pos)}")

sub_base   = df_base.iloc[sample_pos]
sub_contra = df_contra.iloc[sample_pos]

modalities = sub_base["modality"].values
ages       = sub_base["age"].values

# ── Build embedding matrices ───────────────────────────────────────────────────
# Unaligned: fill NaN with 0 (each sample has its own modality's dims filled)
X_unaligned = sub_base[emb_cols].fillna(0.0).values.astype(np.float32)

# Aligned: contrastive projections (no NaNs per-modality)
X_aligned = sub_contra[z_cols].fillna(0.0).values.astype(np.float32)

print(f"  X_unaligned shape: {X_unaligned.shape}")
print(f"  X_aligned   shape: {X_aligned.shape}")

# ── 3D UMAP ───────────────────────────────────────────────────────────────────
print("Running UMAP (unaligned) …")
t1 = time.time()
reducer_u = umap_lib.UMAP(n_components=3, random_state=42,
                           n_neighbors=20, min_dist=0.15,
                           metric="euclidean", low_memory=True)
coords_u = reducer_u.fit_transform(X_unaligned)
print(f"  Done ({time.time()-t1:.1f}s)")

print("Running UMAP (aligned) …")
t2 = time.time()
reducer_a = umap_lib.UMAP(n_components=3, random_state=42,
                           n_neighbors=20, min_dist=0.15,
                           metric="euclidean", low_memory=True)
coords_a = reducer_a.fit_transform(X_aligned)
print(f"  Done ({time.time()-t2:.1f}s)")

# ── Compute fixed axis ranges (5% padding) ────────────────────────────────────
def _fixed_ranges(coords, pad=0.05):
    """Return [xr, yr, zr] with symmetric padding for fixed axis display."""
    ranges = []
    for i in range(3):
        lo, hi = coords[:, i].min(), coords[:, i].max()
        margin = (hi - lo) * pad
        ranges.append([lo - margin, hi + margin])
    return ranges

u_ranges = _fixed_ranges(coords_u)
a_ranges = _fixed_ranges(coords_a)

# ── Figure A: Unaligned, colored by modality ──────────────────────────────────
print("Building Figure A …")
traces_A = []
for mod in ["rna", "xray", "mri"]:
    mask = modalities == mod
    traces_A.append(go.Scatter3d(
        x=coords_u[mask, 0], y=coords_u[mask, 1], z=coords_u[mask, 2],
        mode="markers",
        marker=dict(size=1.8, color=MOD_COLORS[mod], opacity=0.75),
        name=MOD_NAMES[mod],
    ))

fig_A = go.Figure(data=traces_A, layout=base_layout(
    "Unaligned Latent Space (v3) — Colored by Modality",
    xr=u_ranges[0], yr=u_ranges[1], zr=u_ranges[2],
))
save(fig_A, "fig_A_umap_unaligned_modality")

# ── Figure B: Aligned, colored by age ────────────────────────────────────────
print("Building Figure B …")
fig_B = go.Figure(data=[go.Scatter3d(
    x=coords_a[:, 0], y=coords_a[:, 1], z=coords_a[:, 2],
    mode="markers",
    marker=dict(
        size=1.8, color=ages, colorscale="Plasma", opacity=0.75,
        colorbar=dict(
            title=dict(text="Age (years)", font=dict(family=FONT, color="white", size=fs(30))),
            tickfont=dict(family=FONT, color="white", size=fs(24)),
            thickness=30, len=0.6,
        ),
    ),
    showlegend=False,
)], layout=base_layout(
    "Aligned Latent Space (v3.5) — Age Gradient",
    xr=a_ranges[0], yr=a_ranges[1], zr=a_ranges[2],
))
save(fig_B, "fig_B_umap_aligned_age")

# ── Figure C: Aligned, colored by modality ────────────────────────────────────
print("Building Figure C …")
traces_C = []
for mod in ["rna", "xray", "mri"]:
    mask = modalities == mod
    traces_C.append(go.Scatter3d(
        x=coords_a[mask, 0], y=coords_a[mask, 1], z=coords_a[mask, 2],
        mode="markers",
        marker=dict(size=1.8, color=MOD_COLORS[mod], opacity=0.75),
        name=MOD_NAMES[mod],
    ))

fig_C = go.Figure(data=traces_C, layout=base_layout(
    "Aligned Latent Space (v3.5) — Colored by Modality",
    xr=a_ranges[0], yr=a_ranges[1], zr=a_ranges[2],
))
save(fig_C, "fig_C_umap_aligned_modality")

# ── Save UMAP coords for Figure E ─────────────────────────────────────────────
np.save(os.path.join(CACHEDIR, "_coords_aligned.npy"), coords_a)
np.save(os.path.join(CACHEDIR, "_ages.npy"), ages)
print("Saved aligned coords + ages for Figure E.")
print(f"\nAll UMAP figures done in {time.time()-t0:.0f}s total.")
