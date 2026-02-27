"""
prep_isef_figures.py
====================
Assembles all figures needed for the ISEF trifold poster into figures/isef/.

Steps
-----
1. Generates pipeline_diagram.png via figlib.fig_conceptual_overview.
2. Copies the seven poster figures with canonical ISEF filenames.

Run from the project root:
    python src/figures/prep_isef_figures.py
"""
import sys
import os
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from figlib import fig_conceptual_overview

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "figures", "poster")
os.makedirs(OUT, exist_ok=True)

# ── 1. Pipeline diagram (generated) ──────────────────────────────────────────
pipeline_out = os.path.join(OUT, "01_pipeline_diagram.png")
fig_conceptual_overview(pipeline_out)
print(f"[OK] {pipeline_out}")

# ── 2. UMAP before alignment ──────────────────────────────────────────────────
src = os.path.join(ROOT, "figures", "organon_multimodal", "umap_multimodal",
                   "umap_multimodal_modality_v3.png")
dst = os.path.join(OUT, "02_umap_unaligned.png")
shutil.copy2(src, dst)
print(f"[OK] {dst}")

# ── 3. UMAP after alignment ───────────────────────────────────────────────────
src = os.path.join(ROOT, "figures", "synapse_aligned", "umap",
                   "multimodal_v3_5_aligned_modality.png")
dst = os.path.join(OUT, "03_umap_aligned.png")
shutil.copy2(src, dst)
print(f"[OK] {dst}")

# ── 4. Population pred vs chrono (all modalities) ────────────────────────────
src = os.path.join(ROOT, "figures", "organ_age",
                   "normative_pred_vs_chrono_all.png")
dst = os.path.join(OUT, "04_pred_vs_chrono_all.png")
shutil.copy2(src, dst)
print(f"[OK] {dst}")

# ── 5. Calibrated overlay (key organs) ───────────────────────────────────────
src = os.path.join(ROOT, "figures", "organ_age",
                   "calibrated_overlay_summary_key_organs.png")
dst = os.path.join(OUT, "05_calibrated_overlay_key_organs.png")
shutil.copy2(src, dst)
print(f"[OK] {dst}")

# ── 6. Hero subject bar panel ────────────────────────────────────────────────
src = os.path.join(ROOT, "figures", "vitalis_v4", "1117F_panel.png")
dst = os.path.join(OUT, "06_hero_panel_1117F.png")
shutil.copy2(src, dst)
print(f"[OK] {dst}")

# ── 7. Hero subject radar ────────────────────────────────────────────────────
src = os.path.join(ROOT, "figures", "vitalis_v4", "GTEX-1117F", "radar.png")
dst = os.path.join(OUT, "07_hero_radar_1117F.png")
shutil.copy2(src, dst)
print(f"[OK] {dst}")

# ── 8. Hero CI plot (for paper fig 10 / backup poster reference) ─────────────
src = os.path.join(ROOT, "figures", "vitalis_v4", "GTEX-1117F", "ci_plot.png")
dst = os.path.join(OUT, "08_hero_ci_plot_1117F.png")
shutil.copy2(src, dst)
print(f"[OK] {dst}")

print(f"\n[DONE] All ISEF figures written to: {OUT}")
print("Filenames:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
