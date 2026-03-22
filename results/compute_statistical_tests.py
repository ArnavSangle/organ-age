"""
compute_statistical_tests.py
Adds statistical significance testing for the Organ-Age paper revision.
- Bootstrap 95% CIs on MAE (overall, per-modality, per-organ)
- One-sample Wilcoxon signed-rank tests on per-organ mean deltas
- Pearson r with p-values
"""

import os, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

PYROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ANA = os.path.join(PYROOT, "data", "analysis")
RESULTS  = os.path.join(PYROOT, "results")

print("Loading data …")
norm = pd.read_parquet(os.path.join(DATA_ANA, "organ_age_normative.parquet"))
cal  = pd.read_parquet(os.path.join(DATA_ANA, "organ_age_calibrated.parquet"))

rng = np.random.default_rng(42)
N_BOOT = 10000

def bootstrap_mae_ci(y_true, y_pred, n_boot=N_BOOT, ci=95):
    """Bootstrap 95% CI for MAE."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    abs_errors = np.abs(y_pred - y_true)
    mae_obs = float(np.mean(abs_errors))
    boot_maes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_maes[b] = np.mean(abs_errors[idx])
    lo = float(np.percentile(boot_maes, (100 - ci) / 2))
    hi = float(np.percentile(boot_maes, 100 - (100 - ci) / 2))
    return {"MAE": round(mae_obs, 3), "CI_lower": round(lo, 3), "CI_upper": round(hi, 3)}


# ============================================================
# 1. Bootstrap CIs on MAE — overall and per-modality (aligned model)
# ============================================================
print("\n[1] Bootstrap CIs on MAE (aligned fusion) …")
results = {}

# Overall
overall = bootstrap_mae_ci(norm["age_chrono"], norm["age_pred"])
norm_valid = norm.dropna(subset=["age_chrono", "age_pred"])
r_all, p_all = pearsonr(norm_valid["age_chrono"], norm_valid["age_pred"])
overall["r"] = round(float(r_all), 4)
overall["r_pvalue"] = float(p_all)
results["overall"] = overall
print(f"  Overall: MAE = {overall['MAE']} [{overall['CI_lower']}, {overall['CI_upper']}], r = {overall['r']}, p = {overall['r_pvalue']:.2e}")

# Per-modality
results["per_modality"] = {}
for mod in ["rna", "xray", "mri"]:
    sub = norm[norm.modality == mod].dropna(subset=["age_chrono", "age_pred"])
    ci = bootstrap_mae_ci(sub["age_chrono"], sub["age_pred"])
    r_mod, p_mod = pearsonr(sub["age_chrono"], sub["age_pred"])
    ci["r"] = round(float(r_mod), 4)
    ci["r_pvalue"] = float(p_mod)
    ci["N"] = len(sub)
    results["per_modality"][mod] = ci
    print(f"  {mod}: MAE = {ci['MAE']} [{ci['CI_lower']}, {ci['CI_upper']}], r = {ci['r']}, p = {ci['r_pvalue']:.2e}, N = {ci['N']}")

# ============================================================
# 2. Per-organ: bootstrap MAE CIs + Wilcoxon test on delta != 0
# ============================================================
print("\n[2] Per-organ statistical tests (calibrated) …")
results["per_organ"] = {}
key_organs = ["brain", "brain_cortex", "heart", "lung"]

for organ in key_organs:
    sub = cal[cal.organ == organ].dropna(subset=["age_chrono", "age_pred_cal", "age_delta_cal"])
    if len(sub) < 10:
        print(f"  {organ}: insufficient data (N={len(sub)})")
        continue

    delta = sub["age_delta_cal"].values
    ci = bootstrap_mae_ci(sub["age_chrono"], sub["age_pred_cal"])

    # Wilcoxon signed-rank test: H0: median delta = 0
    stat, p_wilcox = wilcoxon(delta, alternative="two-sided")
    mean_delta = float(np.mean(delta))
    median_delta = float(np.median(delta))

    ci["N"] = len(sub)
    ci["mean_delta"] = round(mean_delta, 3)
    ci["median_delta"] = round(median_delta, 3)
    ci["wilcoxon_stat"] = float(stat)
    ci["wilcoxon_p"] = float(p_wilcox)
    ci["delta_significant"] = p_wilcox < 0.05

    results["per_organ"][organ] = ci
    sig = "***" if p_wilcox < 0.001 else ("**" if p_wilcox < 0.01 else ("*" if p_wilcox < 0.05 else "ns"))
    print(f"  {organ}: MAE = {ci['MAE']} [{ci['CI_lower']}, {ci['CI_upper']}], "
          f"mean delta = {mean_delta:+.3f}, Wilcoxon p = {p_wilcox:.2e} {sig}")

# ============================================================
# 3. Per-organ pairwise MAE comparison (brain vs lung, etc.)
# ============================================================
print("\n[3] Pairwise organ MAE comparisons …")
results["pairwise_organ"] = {}
pairs = [("brain", "lung"), ("brain", "heart"), ("heart", "lung")]

for org_a, org_b in pairs:
    sub_a = cal[cal.organ == org_a].dropna(subset=["age_delta_cal"])
    sub_b = cal[cal.organ == org_b].dropna(subset=["age_delta_cal"])
    mae_a = float(np.mean(np.abs(sub_a["age_delta_cal"])))
    mae_b = float(np.mean(np.abs(sub_b["age_delta_cal"])))

    # Bootstrap difference in MAE
    abs_a = np.abs(sub_a["age_delta_cal"].values)
    abs_b = np.abs(sub_b["age_delta_cal"].values)
    n_a, n_b = len(abs_a), len(abs_b)
    boot_diff = np.empty(N_BOOT)
    for b in range(N_BOOT):
        boot_diff[b] = np.mean(rng.choice(abs_a, n_a, replace=True)) - np.mean(rng.choice(abs_b, n_b, replace=True))
    lo = float(np.percentile(boot_diff, 2.5))
    hi = float(np.percentile(boot_diff, 97.5))
    results["pairwise_organ"][f"{org_a}_vs_{org_b}"] = {
        "MAE_diff": round(mae_a - mae_b, 3),
        "CI_lower": round(lo, 3),
        "CI_upper": round(hi, 3),
        "significant": not (lo <= 0 <= hi),
    }
    sig_str = "significant" if not (lo <= 0 <= hi) else "not significant"
    print(f"  {org_a} vs {org_b}: dMAE = {mae_a - mae_b:+.3f} [{lo:+.3f}, {hi:+.3f}] — {sig_str}")

# ============================================================
# Save
# ============================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

out_path = os.path.join(RESULTS, "statistical_tests.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)
print(f"\nSaved to {out_path}")
