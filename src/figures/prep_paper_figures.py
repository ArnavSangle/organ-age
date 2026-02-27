"""
prep_paper_figures.py
=====================
Copy all figures referenced by Paper.tex into figures/paper/ (flat layout).
After running this, Paper.tex graphicspath can point to just {figures/paper/}.

Run from project root:
    python src/figures/prep_paper_figures.py
"""
import os, shutil

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT    = os.path.join(ROOT, "figures", "paper")
os.makedirs(OUT, exist_ok=True)

# (source_relative_to_ROOT, filename_in_paper)
FIGURES = [
    # organ_age
    ("figures/organ_age/normative_pred_vs_chrono_all.png",          "normative_pred_vs_chrono_all.png"),
    ("figures/organ_age/normative_gap_vs_chrono_all.png",           "normative_gap_vs_chrono_all.png"),
    ("figures/organ_age/normative_gap_hist_all.png",                "normative_gap_hist_all.png"),
    ("figures/organ_age/calibrated_pred_vs_chrono_with_ci.png",     "calibrated_pred_vs_chrono_with_ci.png"),
    ("figures/organ_age/calibrated_overlay_summary_key_organs.png", "calibrated_overlay_summary_key_organs.png"),
    # vitalis_v4
    ("figures/vitalis_v4/1117F_panel.png",                          "1117F_panel.png"),
    ("figures/vitalis_v4/GTEX-1117F/ci_plot.png",                   "ci_plot.png"),
    ("figures/vitalis_v4/GTEX-1117F/radar.png",                     "radar.png"),
    # v4_5
    ("figures/v4_5/top_genes/top_20_genes_rank_overlay.png",        "top_20_genes_rank_overlay.png"),
    ("figures/v4_5/ig_three_overlay_rna.png",                       "ig_three_overlay_rna.png"),
    ("figures/v4_5/ig_liver_rna_top20.png",                         "ig_liver_rna_top20.png"),
]

for src_rel, dst_name in FIGURES:
    src = os.path.join(ROOT, src_rel)
    dst = os.path.join(OUT, dst_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"[OK] {dst_name}")
    else:
        print(f"[MISSING] {src_rel}")

print(f"\n[DONE] Paper figures → {OUT}")
print("Files:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
