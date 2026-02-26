"""Tasks 6, 7, 8 — gene attribution, latent attribution, placeholder map."""
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

PYROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ANA = os.path.join(PYROOT, "data", "analysis")
ANA      = os.path.join(PYROOT, "analysis")
RESULTS  = os.path.join(PYROOT, "results")

# Load parquets for Pearson-r computation on unimodal rows
norm = pd.read_parquet(os.path.join(DATA_ANA, "organ_age_normative.parquet"))
cal  = pd.read_parquet(os.path.join(DATA_ANA, "organ_age_calibrated.parquet"))
organ_full = pd.read_csv(os.path.join(RESULTS, "all_organ_metrics.csv")).set_index("organ")

# Add Pearson r to ablation table (unimodal = filter normative parquet by modality)
print("[Pearson r for unimodal] ...")
abl = pd.read_csv(os.path.join(RESULTS, "ablation_metrics.csv"))
r_rna,_  = pearsonr(norm.loc[norm.modality=="rna","age_chrono"], norm.loc[norm.modality=="rna","age_pred"])
r_xray,_ = pearsonr(norm.loc[norm.modality=="xray","age_chrono"], norm.loc[norm.modality=="xray","age_pred"])
r_mri,_  = pearsonr(norm.loc[norm.modality=="mri","age_chrono"], norm.loc[norm.modality=="mri","age_pred"])
r_v35,_  = pearsonr(norm["age_chrono"], norm["age_pred"])
r_map = {"RNA only": round(r_rna,4), "X-ray only": round(r_xray,4),
         "MRI only": round(r_mri,4), "Aligned fusion (v3.5)": round(r_v35,4)}
print("  Pearson r by config:", r_map)
for idx, row in abl.iterrows():
    cfg = row["config"]
    if cfg in r_map:
        abl.at[idx,"r"] = r_map[cfg]
abl.to_csv(os.path.join(RESULTS, "ablation_metrics.csv"), index=False)

# Extras: best unimodal MAE + relative improvement
best_uni_mae = float(abl[abl.config.str.contains("only")]["MAE"].min())
v35_mae      = float(abl[abl.config=="Aligned fusion (v3.5)"]["MAE"].iloc[0])
v3_mae       = float(abl[abl.config=="Naive fusion (v3)"]["MAE"].iloc[0]) if any(abl.config=="Naive fusion (v3)") else float(abl[abl.config.str.contains("Naive|v3")]["MAE"].iloc[0])
rel_uni  = 100*(best_uni_mae - v35_mae) / best_uni_mae
rel_v3   = 100*(v3_mae  - v35_mae) / v3_mae
print(f"  Best unimodal MAE: {best_uni_mae:.3f}  Aligned: {v35_mae:.3f}  Rel reduction from uni: {rel_uni:.1f}%")
print(f"  Naive MAE: {v3_mae:.3f}  Rel reduction v3->v3.5: {rel_v3:.1f}%")

# ================================================================== TASK 6
print("\n[Task 6] Gene attribution ...")
GENE_DIR = os.path.join(ANA, "v4_5_gene_importance")
key_organs_genes = ["liver", "kidney", "brain_cortex", "heart"]
gene_results = {"top5_concentration_pct": {}, "shared_in_all_4_organs": [],
                "n_shared_in_all": 0, "n_unique_per_organ": {}}
top20_sets = {}
for organ in key_organs_genes:
    fpath = os.path.join(GENE_DIR, f"gene_importance_{organ}.csv")
    df_g  = pd.read_csv(fpath)
    top20 = df_g.nlargest(20,"score_abs")["gene"].tolist()
    top5  = df_g.nlargest(5,"score_abs")["gene"].tolist()
    top20_sets[organ] = set(top20)
    total_mass = df_g.nlargest(20,"score_abs")["score_abs"].sum()
    top5_mass  = df_g.nlargest(5,"score_abs")["score_abs"].sum()
    conc = 100*top5_mass/total_mass
    gene_results["top5_concentration_pct"][organ] = round(float(conc),2)
    print(f"  {organ}: top5/top20 concentration={conc:.1f}%  top5={top5}")

shared = set.intersection(*top20_sets.values())
gene_results["shared_in_all_4_organs"] = sorted(list(shared))
gene_results["n_shared_in_all"]        = len(shared)
gene_results["n_unique_per_organ"]     = {
    o: len(top20_sets[o] - set.union(*[top20_sets[oo] for oo in key_organs_genes if oo!=o]))
    for o in key_organs_genes
}
print(f"  Shared in all 4: {len(shared)} genes = {sorted(shared)}")
print(f"  Organ-unique: {gene_results['n_unique_per_organ']}")

with open(os.path.join(RESULTS, "gene_attribution_metrics.json"), "w") as f:
    json.dump(gene_results, f, indent=2)
print("  Saved gene_attribution_metrics.json")

# ================================================================== TASK 7
print("\n[Task 7] Latent attribution ...")
IG_DIR = os.path.join(ANA, "v4_5_ig_latent")
key_organs_ig = ["liver", "kidney", "brain_cortex"]
latent_results = {"top3_fraction_pct":{}, "top5_fraction_pct":{}, "top10_fraction_pct":{},
                  "top5_indices":{}, "shared_top5_dims_all_organs":[],
                  "shared_top5_dims_key3_organs":[], "organ_preferential_dims":{},
                  "liver_cumulative_top5":0, "liver_cumulative_top10":0}

top5_sets_all = {}
for fname in sorted(os.listdir(IG_DIR)):
    if not fname.endswith(".npy"): continue
    organ_name = fname.replace("latent_ig_","").replace(".npy","")
    ig  = np.load(os.path.join(IG_DIR, fname))
    tot = ig.sum()
    si  = np.argsort(ig)[::-1]
    t3  = 100*ig[si[:3]].sum()/tot
    t5  = 100*ig[si[:5]].sum()/tot
    t10 = 100*ig[si[:10]].sum()/tot
    top5_sets_all[organ_name] = set(si[:5].tolist())
    if organ_name in key_organs_ig:
        latent_results["top3_fraction_pct"][organ_name]  = round(float(t3),2)
        latent_results["top5_fraction_pct"][organ_name]  = round(float(t5),2)
        latent_results["top10_fraction_pct"][organ_name] = round(float(t10),2)
        latent_results["top5_indices"][organ_name]       = si[:5].tolist()
    print(f"  {organ_name:20s}: top3={t3:.1f}%  top5={t5:.1f}%  top10={t10:.1f}%  dims={si[:5].tolist()}")

liver_ig  = np.load(os.path.join(IG_DIR, "latent_ig_liver.npy"))
liver_si  = np.argsort(liver_ig)[::-1]
liver_tot = liver_ig.sum()
latent_results["liver_cumulative_top5"]  = round(float(100*liver_ig[liver_si[:5]].sum()/liver_tot),2)
latent_results["liver_cumulative_top10"] = round(float(100*liver_ig[liver_si[:10]].sum()/liver_tot),2)

shared_all  = set.intersection(*top5_sets_all.values()) if top5_sets_all else set()
latent_results["shared_top5_dims_all_organs"] = sorted(list(shared_all))
key3_sets = [top5_sets_all[o] for o in key_organs_ig if o in top5_sets_all]
if key3_sets:
    shared_k3 = set.intersection(*key3_sets)
    latent_results["shared_top5_dims_key3_organs"] = sorted(list(shared_k3))
    for o in key_organs_ig:
        if o not in top5_sets_all: continue
        pref = top5_sets_all[o] - set.union(*[top5_sets_all[oo] for oo in key_organs_ig if oo!=o and oo in top5_sets_all])
        latent_results["organ_preferential_dims"][o] = sorted(list(pref))

print(f"\n  Shared top-5 dims (all 10 organs): {sorted(list(shared_all))}")
print(f"  Shared top-5 dims (key 3 organs) : {latent_results['shared_top5_dims_key3_organs']}")
print(f"  Organ-preferential dims           : {latent_results['organ_preferential_dims']}")
print(f"  Liver cumulative top5/top10       : {latent_results['liver_cumulative_top5']}% / {latent_results['liver_cumulative_top10']}%")

with open(os.path.join(RESULTS, "latent_attribution_metrics.json"), "w") as f:
    json.dump(latent_results, f, indent=2)
print("  Saved latent_attribution_metrics.json")

# ================================================================== TASK 8
print("\n[Task 8] Placeholder replacement map ...")

stab  = json.load(open(os.path.join(RESULTS, "residual_stability_metrics.json")))
calib = json.load(open(os.path.join(RESULTS, "calibration_metrics.json")))
hero  = json.load(open(os.path.join(RESULTS, "hero_subject_GTEX-1117F.json")))
abl_df= pd.read_csv(os.path.join(RESULTS, "ablation_metrics.csv")).set_index("config")

def og(organ, col):
    return float(organ_full.loc[organ, col]) if organ in organ_full.index else None

lines = []
lines.append("="*72)
lines.append("ORGAN-AGE PLACEHOLDER REPLACEMENT MAP")
lines.append("="*72)
lines.append("")

lines.append("SECTION 3.1 — Unimodal results:")
for label, key, ph_mae, ph_mse, ph_r in [
    ("RNA",    "RNA only",    11.4, 198.3, 0.82),
    ("X-ray",  "X-ray only",  10.8, 176.2, 0.84),
    ("MRI",    "MRI only",    13.9, 268.4, 0.79),
]:
    mae = float(abl_df.loc[key,"MAE"]); mse = float(abl_df.loc[key,"MSE"])
    r   = float(abl_df.loc[key,"r"]) if not pd.isna(abl_df.loc[key,"r"]) else None
    lines.append(f"  {label} MAE: [placeholder {ph_mae}] -> {mae:.3f} yr")
    lines.append(f"  {label} MSE: [placeholder {ph_mse}] -> {mse:.2f}")
    lines.append(f"  {label} r:   [placeholder {ph_r}]   -> {r:.4f}" if r else f"  {label} r: N/A (no v3 pred parquet)")
lines.append("")

lines.append("SECTION 3.2 — Alignment results:")
v3m  = float(abl_df.loc["Naive\u0301 fusion (v3)","MAE"]) if "Naive\u0301 fusion (v3)" in abl_df.index else float(abl_df[abl_df.index.str.contains("v3")&abl_df.index.str.contains("Naive|Nai")]["MAE"].iloc[0]) if any(abl_df.index.str.contains("Naive|Nai")) else None
# get v3 row from ablation
v3_row = abl_df[abl_df.index.str.contains("v3") & ~abl_df.index.str.contains("v3.5")].iloc[0]
v35_row= abl_df.loc["Aligned fusion (v3.5)"]
lines.append(f"  Naive fusion (v3) MAE:       [placeholder 10.1] -> {float(v3_row['MAE']):.3f} yr")
lines.append(f"  Naive fusion (v3) MSE:       [placeholder 158.2]-> {float(v3_row['MSE']):.2f}")
lines.append(f"  Aligned fusion (v3.5) MAE:   [placeholder 9.3]  -> {float(v35_row['MAE']):.3f} yr")
lines.append(f"  Aligned fusion (v3.5) MSE:   [placeholder 138.0]-> {float(v35_row['MSE']):.2f}")
lines.append(f"  Aligned fusion (v3.5) r:     [placeholder 0.87] -> {float(v35_row['r']):.4f}")
lines.append(f"  Rel improvement (best uni -> aligned): [placeholder 14-19%] -> {rel_uni:.1f}%")
lines.append(f"  Rel improvement (v3 -> v3.5):          -> {rel_v3:.1f}%")
lines.append(f"  Residual SD v3.5 (calibrated):[placeholder 11.8] -> {stab['v35_residual_SD']:.3f} yr")
lines.append(f"  Residual SD v3 (from MSE):    [placeholder 13.4] -> {stab['v3_residual_SD_from_MSE']:.3f} yr")
lines.append(f"  Heteroscedasticity ratio v3.5:[placeholder 1.6x] -> {stab['v35_hetero_ratio']:.3f}x")
lines.append(f"  Heteroscedasticity ratio v3:  [placeholder 2.1x] -> {stab['v3_hetero_ratio']:.3f}x")
lines.append(f"  UMAP mixing score (aligned):  [placeholder >85%] -> {stab['umap_mixing_score_aligned_pct']:.1f}%")
lines.append(f"  UMAP mixing score (unaligned):[placeholder <50%] -> {stab['umap_mixing_score_unaligned_pct']:.1f}%")
lines.append("  NOTE: UMAP score computed on prediction features; embedding-space score may differ")
lines.append("")

lines.append("SECTION 3.3 — Calibration:")
lines.append(f"  Max |bias| across bins: [placeholder <0.5 yr] -> {calib['max_abs_bias_yr']:.3f} yr")
lines.append("  Bias by 10-yr bin (regression-to-mean effect):")
for k, v in calib["mean_bias_by_bin"].items():
    lines.append(f"    {k}: {v:+.3f} yr")
lines.append(f"  CI half-width 30-40 bin: [placeholder +-8.2 yr] -> +-{float(calib['ci_width_by_bin'].get('30-40',0))/2:.2f} yr  (full width={calib['ci_width_by_bin'].get('30-40','N/A')} yr)")
lines.append(f"  CI half-width 70-80 bin: [placeholder +-14.6 yr]-> +-{float(calib['ci_width_by_bin'].get('70-80',0))/2:.2f} yr  (full width={calib['ci_width_by_bin'].get('70-80','N/A')} yr)")
for organ, cov in calib["pred_range_coverage_pct"].items():
    lines.append(f"  Pred range coverage {organ}: -> {cov:.1f}%")
lines.append("")

lines.append("SECTION 3.4 — Per-organ:")
for organ in ["brain","brain_cortex","heart","lung"]:
    v = organ_full.loc[organ] if organ in organ_full.index else None
    if v is not None:
        lines.append(f"  {organ}: N={int(v['N'])}  MAE={v['MAE']:.3f}  residual_SD={v['residual_SD']:.3f}  mean_delta={v['mean_delta']:+.4f}")
lines.append(f"  Brain residual SD:  [placeholder 7.87]  -> {og('brain','residual_SD'):.3f}")
lines.append(f"  Lung  residual SD:  [placeholder 12.84] -> {og('lung','residual_SD'):.3f}")
lines.append(f"  Lung  mean delta:   [placeholder +1.1]  -> {og('lung','mean_delta'):+.4f}")
lines.append("")

lines.append("SECTION 3.6 — Hero subject (GTEX-1117F):")
lines.append(f"  Chronological age: {hero['chrono_age']}")
for organ, vals in hero["organs"].items():
    lines.append(f"  {organ:25s}: pred={vals['age_pred']}  delta={vals['delta']:+}  z={vals['zscore']}  CI=[{vals['ci_lower']},{vals['ci_upper']}]")
lines.append(f"  Mean organ delta:    [placeholder -9.7]         -> {hero['mean_delta']:+.3f}")
skel = hero["organs"].get("skeletal_muscle",{})
lines.append(f"  Skeletal muscle delta:[placeholder -14.2]       -> {skel.get('delta','N/A')}")
lines.append(f"  Skeletal muscle z:   [placeholder -1.01]        -> {skel.get('zscore','N/A')}")
lines.append(f"  Skeletal muscle CI:  [placeholder [-22.1,-6.3]] -> [{skel.get('ci_lower','N/A')},{skel.get('ci_upper','N/A')}]")
bc  = hero["organs"].get("brain_cortex",{})
lines.append(f"  Brain cortex delta:  [placeholder -5.8]         -> {bc.get('delta','N/A')}")
kid = hero["organs"].get("kidney",{})
lines.append(f"  Kidney delta:        [placeholder -10.5]        -> {kid.get('delta','N/A')}")
lines.append(f"  Kidney CI:           [placeholder [-15.8,-5.2]] -> [{kid.get('ci_lower','N/A')},{kid.get('ci_upper','N/A')}]")
lines.append(f"  Largest |delta| organ: {hero.get('largest_abs_delta_organ','N/A')}")
lines.append(f"  Narrowest CI organ:    {hero.get('narrowest_ci_organ','N/A')}")
lines.append("")

lines.append("SECTION 3.8 — Gene attribution:")
for organ in key_organs_genes:
    conc = gene_results["top5_concentration_pct"].get(organ,"N/A")
    lines.append(f"  {organ} top-5 concentration: -> {conc:.1f}%")
lines.append(f"  Liver   top-5: [placeholder ~62%] -> {gene_results['top5_concentration_pct'].get('liver','N/A'):.1f}%")
lines.append(f"  Kidney  top-5: [placeholder ~58%] -> {gene_results['top5_concentration_pct'].get('kidney','N/A'):.1f}%")
lines.append(f"  BrainCx top-5: [placeholder ~41%] -> {gene_results['top5_concentration_pct'].get('brain_cortex','N/A'):.1f}%")
lines.append(f"  Shared in all 4 organs: {gene_results['n_shared_in_all']} genes = {gene_results['shared_in_all_4_organs']}")
lines.append(f"  Organ-unique counts: {gene_results['n_unique_per_organ']}")
lines.append("")

lines.append("SECTION 3.8 — Latent attribution:")
for organ in key_organs_ig:
    lines.append(f"  {organ}: top3={latent_results['top3_fraction_pct'].get(organ,'N/A')}%  top5={latent_results['top5_fraction_pct'].get(organ,'N/A')}%  top10={latent_results['top10_fraction_pct'].get(organ,'N/A')}%  dims={latent_results['top5_indices'].get(organ,'N/A')}")
lines.append(f"  Liver  top-3 IG: [placeholder 68%] -> {latent_results['top3_fraction_pct'].get('liver','N/A')}%")
lines.append(f"  Kidney top-3 IG: [placeholder 59%] -> {latent_results['top3_fraction_pct'].get('kidney','N/A')}%")
lines.append(f"  BrainCx top-3:   [placeholder 47%] -> {latent_results['top3_fraction_pct'].get('brain_cortex','N/A')}%")
lines.append(f"  Liver top-5 cumulative:  [placeholder 82%] -> {latent_results['liver_cumulative_top5']}%")
lines.append(f"  Liver top-10 cumulative: [placeholder 94%] -> {latent_results['liver_cumulative_top10']}%")
lines.append(f"  Shared top-5 dims (all 10 organs): {latent_results['shared_top5_dims_all_organs']}")
lines.append(f"  Shared top-5 dims (key 3 organs):  {latent_results['shared_top5_dims_key3_organs']}")
lines.append(f"  Organ-preferential dims: {latent_results['organ_preferential_dims']}")
lines.append("")

lines.append("="*72)
lines.append("*** FLAGS — actual value differs from placeholder by >20% ***")
flags = []
def flag(label, placeholder, actual):
    try:
        if abs(actual - placeholder) / (abs(placeholder) + 1e-9) > 0.20:
            diff = 100*(actual-placeholder)/(abs(placeholder)+1e-9)
            flags.append(f"  *** {label}: placeholder={placeholder}, actual={actual:.3f}  ({diff:+.0f}%)")
    except: pass

flag("RNA MAE",       11.4,  float(abl_df.loc["RNA only","MAE"]))
flag("X-ray MAE",     10.8,  float(abl_df.loc["X-ray only","MAE"]))
flag("MRI MAE",       13.9,  float(abl_df.loc["MRI only","MAE"]))
flag("Aligned r",      0.87, float(v35_row["r"]))
flag("Rel improvement (uni->v3.5)", 0.165, rel_uni/100)
flag("Residual SD v3.5", 11.8, stab["v35_residual_SD"])
flag("Hetero ratio v3.5",  1.6, stab["v35_hetero_ratio"])
flag("Hetero ratio v3",    2.1, stab["v3_hetero_ratio"])
flag("UMAP aligned",      85.0, stab["umap_mixing_score_aligned_pct"])
flag("Max bias (expect <0.5)",  0.5, calib["max_abs_bias_yr"])
flag("Brain residual SD",  7.87, og("brain","residual_SD"))
flag("Lung  residual SD", 12.84, og("lung","residual_SD"))
flag("Lung  mean delta",   1.1,  og("lung","mean_delta"))
flag("Hero mean delta",   -9.7,  hero["mean_delta"])
flag("Liver gene conc",   62.0, gene_results["top5_concentration_pct"].get("liver",0))
flag("Kidney gene conc",  58.0, gene_results["top5_concentration_pct"].get("kidney",0))
flag("BC gene conc",      41.0, gene_results["top5_concentration_pct"].get("brain_cortex",0))
flag("Liver IG top3",     68.0, latent_results["top3_fraction_pct"].get("liver",0))
flag("Kidney IG top3",    59.0, latent_results["top3_fraction_pct"].get("kidney",0))
flag("BC IG top3",        47.0, latent_results["top3_fraction_pct"].get("brain_cortex",0))

if flags:
    lines.extend(flags)
    lines.append(f"\n  Total flags: {len(flags)}")
else:
    lines.append("  None — all within 20%")
lines.append("="*72)

out = "\n".join(lines)
print(out)
with open(os.path.join(RESULTS, "placeholder_replacements.txt"), "w", encoding="utf-8") as f:
    f.write(out)
print("\nDone. placeholder_replacements.txt written.")
