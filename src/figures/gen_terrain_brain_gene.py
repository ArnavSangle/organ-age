"""
gen_terrain_brain_gene.py
=========================
Generate Figures D, E, G for the ISEF poster:

  D – 3D Brain MRI isosurface render (PyVista)
  E – Aging manifold terrain (Plotly surface)
  G – Gene attribution 3D landscape (Plotly surface/bar)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTDIR = os.path.join(ROOT, "figures", "poster")
os.makedirs(OUTDIR, exist_ok=True)

BG   = "#0f0f1a"
FONT = "Georgia, 'Times New Roman', serif"

AXIS_STYLE = dict(
    backgroundcolor=BG,
    gridcolor="#2a2a5a",
    showgrid=True,
    zeroline=True,
    zerolinecolor="#6868cc",
    zerolinewidth=5,
    showline=True,
    linecolor="#5a5aaa",
    linewidth=4,
    # showticklabels is set per-axis below
)

def _title_font(size=28):
    return dict(family=FONT, color="rgba(160,160,220,0.8)", size=size)

def save(fig, stem):
    html = os.path.join(OUTDIR, f"{stem}.html")
    png  = os.path.join(OUTDIR, f"{stem}.png")
    fig.write_html(html)
    fig.write_image(png, scale=1)
    print(f"  Saved {stem}.html + .png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE E — Aging Manifold Terrain
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Figure E: Aging Terrain ===")

coords_path = os.path.join(OUTDIR, "_coords_aligned.npy")
ages_path   = os.path.join(OUTDIR, "_ages.npy")

if os.path.exists(coords_path) and os.path.exists(ages_path):
    coords_a = np.load(coords_path)
    ages     = np.load(ages_path)
    print(f"  Loaded coords {coords_a.shape}, ages {ages.shape}")

    # Use x/y of aligned 3D UMAP, age as both height and color
    x, y, z = coords_a[:, 0], coords_a[:, 1], ages.astype(float)

    # Filter out extreme age outliers (CheXpert has age 0-90, some noise at extremes)
    valid = (z > 5) & (z < 90)
    x, y, z = x[valid], y[valid], z[valid]

    # Build interpolated surface grid
    xi = np.linspace(x.min(), x.max(), 250)
    yi = np.linspace(y.min(), y.max(), 250)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi, yi), method="linear")

    # Fill remaining NaN with nearest-neighbour then smooth
    zi_nn = griddata((x, y), z, (xi, yi), method="nearest")
    nan_mask = np.isnan(zi)
    zi[nan_mask] = zi_nn[nan_mask]
    zi = gaussian_filter(zi, sigma=3)

    # Fixed axis ranges with small padding
    pad = 0.05
    xlo, xhi = x.min(), x.max(); xm = (xhi-xlo)*pad
    ylo, yhi = y.min(), y.max(); ym = (yhi-ylo)*pad
    zlo, zhi = 20.0, 85.0

    fig_E = go.Figure(data=[
        go.Surface(
            x=xi, y=yi, z=zi,
            colorscale="Plasma",
            cmin=20, cmax=85,
            showscale=True,
            colorbar=dict(
                title=dict(text="Age (years)", font=dict(family=FONT, color="white", size=30)),
                tickfont=dict(family=FONT, color="white", size=24),
                thickness=30, len=0.6,
            ),
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3,
                          roughness=0.5, fresnel=0.2),
            lightposition=dict(x=1, y=1, z=2),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white",
                       project_z=False, width=1),
            ),
            opacity=0.92,
        ),
    ])

    fig_E.update_layout(
        font=dict(family=FONT),
        scene=dict(
            xaxis=dict(**AXIS_STYLE, showticklabels=False, range=[xlo-xm, xhi+xm],
                       title=dict(text="UMAP 1",      font=_title_font())),
            yaxis=dict(**AXIS_STYLE, showticklabels=False, range=[ylo-ym, yhi+ym],
                       title=dict(text="UMAP 2",      font=_title_font())),
            zaxis=dict(**AXIS_STYLE, showticklabels=True, nticks=4,
                       range=[zlo, zhi],
                       tickfont=dict(family=FONT, color="rgba(160,160,220,0.65)", size=20),
                       title=dict(text="Age (years)", font=_title_font())),
            bgcolor=BG,
            aspectmode="cube",
            camera=dict(eye=dict(x=1.3, y=1.3, z=0.9)),
        ),
        paper_bgcolor=BG,
        title=dict(text="Aging Manifold — Latent Space Terrain",
                   font=dict(family=FONT, color="white", size=52), x=0.5, y=0.97),
        margin=dict(l=0, r=0, t=130, b=0),
        width=2400, height=2400,
    )
    save(fig_E, "fig_E_manifold_terrain")
else:
    print("  SKIP: run gen_3d_umap.py first to produce _coords_aligned.npy")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — Brain MRI 3D Render (PyVista)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Figure D: Brain MRI ===")
try:
    import pyvista as pv
    import nibabel as nib

    ixi_dir = os.path.join(ROOT, "data/raw/ixi/T1")
    nii_files = sorted(f for f in os.listdir(ixi_dir) if f.endswith(".nii.gz"))
    nii_path  = os.path.join(ixi_dir, nii_files[0])
    print(f"  Loading {nii_files[0]} …")

    img  = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)
    print(f"  Volume shape: {data.shape}, range: [{data.min():.1f}, {data.max():.1f}]")

    # Build PyVista grid from the NIfTI volume
    grid = pv.ImageData(dimensions=data.shape)
    grid.spacing = img.header.get_zooms()[:3]
    grid.point_data["intensity"] = data.flatten(order="F")

    # Isosurface at ~30th percentile of nonzero voxels — good for brain cortex
    vox = data[data > data.max() * 0.05]
    iso_val = float(np.percentile(vox, 25))
    print(f"  Isosurface value: {iso_val:.1f}")
    brain_surface = grid.contour(isosurfaces=[iso_val], scalars="intensity")

    pv.start_xvfb()   # start virtual display if on headless server
except Exception:
    pass  # no Xvfb on Windows, continue without it

try:
    plotter = pv.Plotter(off_screen=True, window_size=[2400, 2400])
    plotter.set_background(BG)

    plotter.add_mesh(
        brain_surface,
        color="#0F766E",
        opacity=0.88,
        smooth_shading=True,
        specular=0.6,
        specular_power=20,
    )

    # Key light (bright white) + warm amber fill
    key_light = pv.Light(position=(300, 300, 400), intensity=0.85)
    key_light.positional = True
    fill_light = pv.Light(position=(-200, -200, 150), intensity=0.35,
                          color="#B45309")
    fill_light.positional = True
    plotter.add_light(key_light)
    plotter.add_light(fill_light)

    plotter.camera_position = "xz"
    plotter.camera.zoom(1.3)

    out_path = os.path.join(OUTDIR, "fig_D_brain_render.png")
    plotter.screenshot(out_path, transparent_background=False)
    plotter.close()
    print(f"  Saved fig_D_brain_render.png")

except Exception as e:
    print(f"  PyVista render failed: {e}")
    print("  Falling back to matplotlib 3D slice render …")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 16), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG)

    # Show three orthogonal slices
    s = data.shape
    cx, cy, cz = s[0]//2, s[1]//2, s[2]//2

    # Sagittal slice (x = mid)
    Y, Z = np.meshgrid(np.arange(s[1]), np.arange(s[2]))
    X = np.full_like(Y, cx)
    ax.plot_surface(X, Y, Z,
                    facecolors=plt.cm.gray(data[cx, :, :].T / data.max()),
                    alpha=0.6, shade=False)

    # Coronal slice (y = mid)
    X2, Z2 = np.meshgrid(np.arange(s[0]), np.arange(s[2]))
    Y2 = np.full_like(X2, cy)
    ax.plot_surface(X2, Y2, Z2,
                    facecolors=plt.cm.gray(data[:, cy, :].T / data.max()),
                    alpha=0.6, shade=False)

    # Axial slice (z = mid)
    X3, Y3 = np.meshgrid(np.arange(s[0]), np.arange(s[1]))
    Z3 = np.full_like(X3, cz)
    ax.plot_surface(X3, Y3, Z3,
                    facecolors=plt.cm.gray(data[:, :, cz].T / data.max()),
                    alpha=0.6, shade=False)

    ax.set_axis_off()
    ax.set_title("Brain MRI — Orthogonal Slices", color="white", fontsize=20, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_D_brain_render.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved fig_D_brain_render.png (matplotlib fallback)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE G — Gene Attribution 3D Landscape
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Figure G: Gene Attribution ===")

gene_dir = os.path.join(ROOT, "analysis/v4_5_gene_importance")
organs_avail = []
for fname in sorted(os.listdir(gene_dir)):
    if fname.endswith(".csv"):
        organ = fname.replace("gene_importance_", "").replace(".csv", "")
        organs_avail.append(organ)

print(f"  Organs: {organs_avail}")
TOP_N = 20

all_dfs = []
for organ in organs_avail:
    df = pd.read_csv(os.path.join(gene_dir, f"gene_importance_{organ}.csv"))
    df = df.nlargest(TOP_N, "score_abs")[["gene", "score_signed"]].copy()
    df["organ"] = organ
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)

# Build gene × organ matrix (pivot)
pivot = combined.pivot_table(index="gene", columns="organ",
                              values="score_signed", fill_value=0.0)

# Keep top-20 genes by mean absolute score across organs
top_genes = pivot.abs().mean(axis=1).nlargest(TOP_N).index
pivot = pivot.loc[top_genes]

genes   = list(pivot.index)
organs  = list(pivot.columns)
Z_mat   = pivot.values  # shape (n_genes, n_organs)

# Smooth for surface aesthetics
Z_smooth = gaussian_filter(Z_mat.astype(float), sigma=0.8)

X_idx, Y_idx = np.meshgrid(np.arange(len(organs)), np.arange(len(genes)))

z_abs_max = float(np.abs(Z_mat).max())

fig_G = go.Figure(data=[
    go.Surface(
        x=X_idx, y=Y_idx, z=Z_smooth,
        colorscale=[
            [0.0, "#0F766E"],   # teal (negative — younger)
            [0.5, "#0f0f1a"],   # dark neutral
            [1.0, "#B45309"],   # amber (positive — older)
        ],
        cmin=-z_abs_max, cmax=z_abs_max,
        showscale=True,
        colorbar=dict(
            title=dict(text="Attribution Score", font=dict(family=FONT, color="white", size=30)),
            tickfont=dict(family=FONT, color="white", size=24),
            thickness=30, len=0.6,
        ),
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.4,
                      roughness=0.4, fresnel=0.3),
        lightposition=dict(x=2, y=2, z=3),
        opacity=0.92,
    )
])

fig_G.update_layout(
    font=dict(family=FONT),
    scene=dict(
        xaxis=dict(**AXIS_STYLE,
                   showticklabels=True,
                   ticktext=organs, tickvals=list(range(len(organs))),
                   range=[-0.5, len(organs) - 0.5],
                   tickfont=dict(family=FONT, color="rgba(200,200,255,0.85)", size=22),
                   title=dict(text="Organ", font=_title_font())),
        yaxis=dict(**AXIS_STYLE,
                   showticklabels=True,
                   ticktext=genes[::-1], tickvals=list(range(len(genes))),
                   range=[-0.5, len(genes) - 0.5],
                   tickfont=dict(family=FONT, color="rgba(200,200,255,0.85)", size=16),
                   title=dict(text="Gene", font=_title_font())),
        zaxis=dict(**AXIS_STYLE,
                   showticklabels=True,
                   nticks=4,
                   range=[-z_abs_max * 1.05, z_abs_max * 1.05],
                   tickfont=dict(family=FONT, color="rgba(200,200,255,0.85)", size=20),
                   title=dict(text="Score", font=_title_font())),
        bgcolor=BG,
        aspectmode="cube",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
    ),
    paper_bgcolor=BG,
    title=dict(text="Gene Attribution Landscape (v4.5)",
               font=dict(family=FONT, color="white", size=52), x=0.5, y=0.97),
    margin=dict(l=0, r=0, t=130, b=0),
    width=2400, height=2400,
)
save(fig_G, "fig_G_gene_attribution")

print("\nAll done.")
