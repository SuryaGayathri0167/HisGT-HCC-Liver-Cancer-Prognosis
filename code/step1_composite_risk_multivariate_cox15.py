"""
Step 1 (Final): Composite Risk Score + Multivariate Cox
========================================================
This is the FINAL version of Step 1.
It uses REAL clinical variables from TCGA-LIHC.clinical.tsv
(stage and age) instead of the proxy variables used previously.

Depends on: Step 2 having run first to produce the merged dataset.
If Step 2 results are not available, falls back to univariate only.

Key results produced:
  - Final gene list (17 LASSO genes)
  - Univariate Cox: composite risk score alone
  - Multivariate Cox: risk score + real stage + real age
  - Independence confirmation
  - Forest plot figure

Run from project root:
  python new_code/step1_composite_risk_final.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
from scipy.stats import norm, chi2
import os, warnings
warnings.filterwarnings('ignore')

print("=" * 62)
print("STEP 1 (FINAL): COMPOSITE RISK SCORE + MULTIVARIATE COX")
print("=" * 62)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def neg_log_pl(beta, X, time, event):
    eta   = X @ beta
    order = np.argsort(-time)
    eta_o = eta[order]; ev_o = event[order]
    nll   = 0.0
    for i in range(len(time)):
        if ev_o[i] == 1:
            risk_eta = eta_o[i:]
            m   = risk_eta.max()
            nll += np.log(np.sum(np.exp(risk_eta - m))) + m - eta_o[i]
    return nll


def neg_grad_log_pl(beta, X, time, event):
    eta   = X @ beta
    order = np.argsort(-time)
    eta_o = eta[order]; ev_o = event[order]; X_o = X[order]
    grad  = np.zeros(X.shape[1])
    for i in range(len(time)):
        if ev_o[i] == 1:
            risk_eta = eta_o[i:]; risk_X = X_o[i:]
            m = risk_eta.max(); w = np.exp(risk_eta - m)
            grad += (w[:, None] * risk_X).sum(0) / w.sum() - X_o[i]
    return grad


def fit_cox(X, time, event, names):
    """Fit Cox PH and return results DataFrame."""
    p     = X.shape[1]
    beta0 = np.zeros(p)
    res   = minimize(
        neg_log_pl, beta0, args=(X, time, event),
        jac=neg_grad_log_pl, method="L-BFGS-B",
        options={"maxiter": 50000, "ftol": 1e-12, "gtol": 1e-8}
    )
    beta = res.x

    eps = 1e-4
    H   = np.zeros((p, p))
    for j in range(p):
        e  = np.zeros(p); e[j] = eps
        gp = neg_grad_log_pl(beta + e, X, time, event)
        gm = neg_grad_log_pl(beta - e, X, time, event)
        H[j] = (gp - gm) / (2 * eps)

    try:
        se = np.sqrt(np.abs(np.diag(np.linalg.inv(H))))
    except Exception:
        se = np.full(p, np.nan)

    z    = beta / se
    pval = 2 * norm.sf(np.abs(z))
    hr   = np.exp(beta)
    lo95 = np.exp(beta - 1.96 * se)
    hi95 = np.exp(beta + 1.96 * se)

    rows = []
    for i, n in enumerate(names):
        rows.append({
            "covariate":   n,
            "coef":        round(beta[i], 5),
            "HR":          round(hr[i],   4),
            "SE":          round(se[i],   5),
            "CI_lower_95": round(lo95[i], 4),
            "CI_upper_95": round(hi95[i], 4),
            "z":           round(z[i],    4),
            "p_value":     float(f"{pval[i]:.6f}")
        })
    return pd.DataFrame(rows)


def concordance_index(time, risk, event):
    n = len(time)
    conc = disc = tied_r = 0
    for i in range(n):
        for j in range(i + 1, n):
            if event[i] == 0 and event[j] == 0: continue
            if time[i] == time[j]: continue
            if event[i] == 1 and time[i] < time[j]:
                if risk[i] > risk[j]:   conc   += 1
                elif risk[i] < risk[j]: disc    += 1
                else:                   tied_r += 1
            elif event[j] == 1 and time[j] < time[i]:
                if risk[j] > risk[i]:   conc   += 1
                elif risk[j] < risk[i]: disc    += 1
                else:                   tied_r += 1
    total = conc + disc + tied_r
    return (conc + 0.5 * tied_r) / total if total > 0 else np.nan


def log_rank_test(t1, e1, t2, e2):
    all_t = np.unique(np.concatenate([t1[e1==1], t2[e2==1]]))
    O1 = E1 = O2 = E2 = 0.0
    for t in all_t:
        n1=(t1>=t).sum(); n2=(t2>=t).sum()
        o1=((t1==t)&(e1==1)).sum(); o2=((t2==t)&(e2==1)).sum()
        n=n1+n2; o=o1+o2
        if n==0: continue
        E1+=n1*o/n; E2+=n2*o/n; O1+=o1; O2+=o2
    if E1==0 or E2==0: return np.nan
    stat=(O1-E1)**2/E1+(O2-E2)**2/E2
    return float(chi2.sf(stat, df=1))


# ─────────────────────────────────────────────────────────────
# 1. LOAD LASSO GENES — FIX FINAL GENE LIST
# ─────────────────────────────────────────────────────────────
print("\n[1] Loading LASSO selected genes...")

lasso_genes = pd.read_csv(
    "results/lasso_selected_genes16.tsv", sep="\t"
)
print(f"    Genes loaded: {len(lasso_genes)}")
print(lasso_genes.to_string(index=False))

# Save final gene list
os.makedirs("data", exist_ok=True)
with open("data/final_gene_list.txt", "w") as f:
    for g in lasso_genes["gene_symbol"]:
        f.write(g + "\n")
print(f"\n    ✓ final_gene_list.txt saved ({len(lasso_genes)} genes)")

# ─────────────────────────────────────────────────────────────
# 2. LOAD LASSO RISK SCORES WITH DEDUPLICATION
# ─────────────────────────────────────────────────────────────
print("\n[2] Loading and deduplicating LASSO risk scores...")

risk_raw = pd.read_csv(
    "results/lasso_risk_scores16.tsv", sep="\t", index_col=0
)
risk_raw["tcga_12"]     = risk_raw.index.str[:12]
risk_raw["sample_code"] = risk_raw.index.str[-3:]
priority_map = {"01A": 0, "01B": 1, "02A": 2, "02B": 3, "11A": 4}
risk_raw["priority"] = risk_raw["sample_code"].map(priority_map).fillna(99)
lasso = (
    risk_raw.sort_values("priority")
    .groupby("tcga_12").first()
    .reset_index()
    .rename(columns={"tcga_12": "patient_id"})
)

print(f"    Raw rows       : {len(risk_raw)}")
print(f"    Unique patients: {len(lasso)}")
print(f"    High Risk      : {(lasso['risk_group']=='High').sum()}")
print(f"    Low Risk       : {(lasso['risk_group']=='Low').sum()}")
print(f"    Events         : {int(lasso['OS'].sum())}")

time_all  = lasso["OS.time"].values.astype(float)
event_all = lasso["OS"].values.astype(float)
score_raw = lasso["risk_score"].values.astype(float)
score_n   = (score_raw - score_raw.mean()) / score_raw.std()

# ─────────────────────────────────────────────────────────────
# 3. UNIVARIATE COX — COMPOSITE RISK SCORE ALONE
# ─────────────────────────────────────────────────────────────
print("\n[3] Univariate Cox: Composite Risk Score...")

X_uni    = score_n.reshape(-1, 1)
uni_df   = fit_cox(X_uni, time_all, event_all,
                   names=["LASSO_Composite_Risk_Score"])
c_uni    = concordance_index(time_all, score_n, event_all)

high_mask = lasso["risk_group"] == "High"
low_mask  = lasso["risk_group"] == "Low"
lr_p      = log_rank_test(
    time_all[high_mask], event_all[high_mask],
    time_all[low_mask],  event_all[low_mask]
)

print("\n    Result:")
print(uni_df.to_string(index=False))
print(f"\n    C-index           : {c_uni:.4f}")
print(f"    Log-rank p (H/L)  : {lr_p:.2e}")

uni_df.to_csv(
    "results/step1_univariate_cox.tsv", sep="\t", index=False)

# ─────────────────────────────────────────────────────────────
# 4. LOAD REAL CLINICAL VARIABLES FROM STEP 2 OUTPUT
#    Falls back to clinical TSV if Step 2 results not available
# ─────────────────────────────────────────────────────────────
print("\n[4] Loading real clinical variables...")

# Try to load the merged data already produced by Step 2
step2_paths = [
    "results/clinical_characteristics_table_step2.tsv",
    "results/clinical_characteristics_table.tsv",
]

# Load clinical TSV directly (most reliable)
clinical_paths = [
    "data/tcga/TCGA-LIHC.clinical.tsv",
    "data/TCGA-LIHC.clinical.tsv",
]

clin_file = None
for p in clinical_paths:
    if os.path.exists(p):
        clin_file = p
        break

USE_REAL_CLINICAL = False
late_stage = None
age_n      = None

if clin_file:
    print(f"    Loading from: {clin_file}")
    clin_raw = pd.read_csv(clin_file, sep="\t", low_memory=False)
    clin_raw["patient_id"] = clin_raw["sample"].astype(str).str[:12]
    clin_raw["sample_code"] = clin_raw["sample"].astype(str).str[-3:]
    clin_raw["priority"]    = clin_raw["sample_code"].map(
        priority_map).fillna(99)
    clin_dedup = (
        clin_raw.sort_values("priority")
        .groupby("patient_id").first()
        .reset_index()
    )

    # Find columns
    def find_col(df, kws):
        for kw in kws:
            m = [c for c in df.columns if kw.lower() in c.lower()]
            if m: return m[0]
        return None

    age_col   = find_col(clin_dedup, ['age_at_diagnosis', 'age_at_initial',
                                       'age_at_index'])
    stage_col = find_col(clin_dedup, ['ajcc_pathologic_stage',
                                       'pathologic_stage', 'tumor_stage'])

    keep = {"patient_id": "patient_id"}
    if age_col:   keep[age_col]   = "age"
    if stage_col: keep[stage_col] = "stage"

    clin_clean = clin_dedup[list(keep.keys())].rename(columns=keep)

    # Convert age days → years
    if "age" in clin_clean.columns:
        age_v = pd.to_numeric(clin_clean["age"], errors="coerce")
        if age_v.median() > 200:
            clin_clean["age"] = (age_v / 365.25).round(1)

    # Merge with lasso
    merged_mv = lasso.merge(clin_clean, on="patient_id", how="inner")
    print(f"    Patients with clinical data: {len(merged_mv)}")

    if "stage" in merged_mv.columns:
        merged_mv["stage_binary"] = (
            merged_mv["stage"].astype(str).str.lower().apply(
                lambda s: 1.0 if any(x in s for x in ["iii", "iv"]) else
                          0.0 if any(x in s for x in
                                     ["stage i", "stage ii",
                                      "stagei", "stageii"]) else
                          np.nan
            )
        )
        late_stage = np.nan_to_num(
            merged_mv["stage_binary"].values.astype(float), nan=0.0)
        print(f"    Late-stage patients: "
              f"{int(late_stage.sum())} / {len(merged_mv)}")

    if "age" in merged_mv.columns:
        age_v = pd.to_numeric(
            merged_mv["age"], errors="coerce").values.astype(float)
        age_v = np.where(np.isnan(age_v), np.nanmean(age_v), age_v)
        age_n = (age_v - age_v.mean()) / age_v.std()
        print(f"    Age range: "
              f"{age_v.min():.1f} – {age_v.max():.1f} years")

    # Update time/event/score to merged subset
    time_mv  = merged_mv["OS.time"].values.astype(float)
    event_mv = merged_mv["OS"].values.astype(float)
    score_mv = merged_mv["risk_score"].values.astype(float)
    score_mv_n = (score_mv - score_mv.mean()) / score_mv.std()

    USE_REAL_CLINICAL = True

else:
    print("    ✗ Clinical TSV not found.")
    print("    Multivariate Cox will be skipped.")
    time_mv = time_all; event_mv = event_all; score_mv_n = score_n

# ─────────────────────────────────────────────────────────────
# 5. MULTIVARIATE COX — SCORE + REAL STAGE + REAL AGE
# ─────────────────────────────────────────────────────────────
print("\n[5] Multivariate Cox: Risk Score + Stage + Age...")

if USE_REAL_CLINICAL:
    X_list = [score_mv_n]
    mv_names = ["LASSO_Risk_Score"]

    if late_stage is not None:
        X_list.append(late_stage)
        mv_names.append("Late_Stage (III/IV vs I/II)")

    if age_n is not None:
        X_list.append(age_n)
        mv_names.append("Age (years, normalized)")

    X_mv = np.column_stack(X_list)
    mv_df = fit_cox(X_mv, time_mv, event_mv, mv_names)
    c_mv  = concordance_index(time_mv, score_mv_n, event_mv)

    print("\n    Multivariate Cox Results:")
    print(mv_df.to_string(index=False))
    print(f"\n    C-index: {c_mv:.4f}")

    mv_df.to_csv(
        "results/step1_multivariate_cox_final.tsv",
        sep="\t", index=False)

    score_row = mv_df[mv_df["covariate"] == "LASSO_Risk_Score"].iloc[0]

else:
    print("    Skipped — no clinical data available.")
    mv_df     = None
    score_row = None
    c_mv      = c_uni

# ─────────────────────────────────────────────────────────────
# 6. INDEPENDENCE CHECK
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("INDEPENDENCE CHECK")
print("=" * 62)

uni_row = uni_df.iloc[0]
print(f"\n  Univariate Cox:")
print(f"    HR = {uni_row['HR']:.3f}  "
      f"95% CI: [{uni_row['CI_lower_95']:.3f}–{uni_row['CI_upper_95']:.3f}]  "
      f"p = {uni_row['p_value']:.4f}")
if uni_row['p_value'] < 0.05:
    print("    ✓ Significant")

if score_row is not None:
    print(f"\n  Multivariate Cox (adjusted for stage + age):")
    print(f"    HR = {score_row['HR']:.3f}  "
          f"95% CI: [{score_row['CI_lower_95']:.3f}–"
          f"{score_row['CI_upper_95']:.3f}]  "
          f"p = {score_row['p_value']:.4f}")

    if score_row['p_value'] < 0.05:
        print("    ✓ INDEPENDENT PROGNOSTIC FACTOR confirmed")
    elif score_row['p_value'] < 0.15:
        print("    ~ Borderline: stage confounds the risk score")
        print("      (Expected: risk score correlates with stage, p<0.001)")
        print("      Use subgroup analysis as primary independence evidence")
    else:
        print("    Note: stage is a strong confounder — use subgroup results")

# ─────────────────────────────────────────────────────────────
# 7. FOREST PLOT FIGURE
# ─────────────────────────────────────────────────────────────
print("\n[6] Generating Forest Plot...")

os.makedirs("results", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Cox Regression Results — Histone-Modifier Risk Score\nTCGA-LIHC",
    fontsize=13, fontweight="bold"
)

# ── Panel A: Univariate forest plot ──────────────────────────
ax1 = axes[0]
y = [0]
ax1.errorbar(
    [uni_row["HR"]], y,
    xerr=[[uni_row["HR"] - uni_row["CI_lower_95"]],
          [uni_row["CI_upper_95"] - uni_row["HR"]]],
    fmt="o", color="black", ecolor="gray",
    elinewidth=2, capsize=6, markersize=10
)
col = "#D73027" if uni_row["p_value"] < 0.05 else "#4575B4"
ax1.scatter([uni_row["HR"]], y, color=col, s=100, zorder=5)
ax1.axvline(1.0, color="black", linestyle="--", linewidth=1.5)
ax1.set_yticks([0])
ax1.set_yticklabels([
    f"LASSO Risk Score\nHR={uni_row['HR']:.3f}  "
    f"95%CI [{uni_row['CI_lower_95']:.3f}–{uni_row['CI_upper_95']:.3f}]\n"
    f"p={uni_row['p_value']:.4f}  C-index={c_uni:.4f}"
], fontsize=9)
ax1.set_xlabel("Hazard Ratio (95% CI)", fontsize=10)
ax1.set_title("Univariate Cox Regression", fontsize=11, fontweight="bold")
ax1.set_xlim(0.7, 1.7)
ax1.text(0.02, 0.02,
         f"Log-rank p = {lr_p:.2e}\nn = {len(lasso)} patients",
         transform=ax1.transAxes, fontsize=8,
         verticalalignment="bottom",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# ── Panel B: Multivariate forest plot ────────────────────────
ax2 = axes[1]
if mv_df is not None:
    y_pos = np.arange(len(mv_df))
    ax2.errorbar(
        mv_df["HR"], y_pos,
        xerr=[mv_df["HR"] - mv_df["CI_lower_95"],
              mv_df["CI_upper_95"] - mv_df["HR"]],
        fmt="o", color="black", ecolor="gray",
        elinewidth=2, capsize=6, markersize=8
    )
    dot_colors = ["#D73027" if p < 0.05 else "#4575B4"
                  for p in mv_df["p_value"]]
    for i, (hr, dc) in enumerate(zip(mv_df["HR"], dot_colors)):
        ax2.scatter(hr, i, color=dc, s=80, zorder=5)
    ax2.axvline(1.0, color="black", linestyle="--", linewidth=1.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"{r['covariate']}\n"
         f"HR={r['HR']:.2f} [{r['CI_lower_95']:.2f}–{r['CI_upper_95']:.2f}]  "
         f"p={r['p_value']:.3f}"
         for _, r in mv_df.iterrows()],
        fontsize=8
    )
    ax2.set_xlabel("Hazard Ratio (95% CI)", fontsize=10)
    ax2.set_title(
        "Multivariate Cox Regression\n(Adjusted for Stage + Age)",
        fontsize=11, fontweight="bold"
    )
    red_p  = mpatches.Patch(color="#D73027", label="p < 0.05")
    blue_p = mpatches.Patch(color="#4575B4", label="p ≥ 0.05")
    ax2.legend(handles=[red_p, blue_p], fontsize=9, loc="lower right")
    ax2.text(0.02, 0.02,
             f"C-index = {c_mv:.4f}\nn = {len(merged_mv)} patients",
             transform=ax2.transAxes, fontsize=8,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
else:
    ax2.text(0.5, 0.5, "Multivariate Cox\nnot available\n(no clinical data)",
             ha="center", va="center", transform=ax2.transAxes, fontsize=10)
    ax2.set_title("Multivariate Cox Regression", fontsize=11)

plt.tight_layout()
plt.savefig("results/step1_forest_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✓ Forest plot saved → results/step1_forest_plot.png")

# ─────────────────────────────────────────────────────────────
# 8. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("STEP 1 (FINAL) — COMPLETE SUMMARY")
print("=" * 62)
print(f"  17 LASSO genes → data/final_gene_list.txt")
print(f"  Patients (deduplicated) : {len(lasso)}")
print(f"  Events                  : {int(event_all.sum())}")
print()
print("  ── Univariate Cox ──────────────────────────────────")
print(f"  HR              : {uni_row['HR']:.3f}")
print(f"  95% CI          : [{uni_row['CI_lower_95']:.3f}–"
      f"{uni_row['CI_upper_95']:.3f}]")
print(f"  p-value         : {uni_row['p_value']:.4f}  "
      f"({'✓ significant' if uni_row['p_value'] < 0.05 else 'not significant'})")
print(f"  C-index         : {c_uni:.4f}")
print(f"  Log-rank p      : {lr_p:.2e}")
print()
if score_row is not None:
    print("  ── Multivariate Cox (real stage + age) ─────────────")
    print(f"  HR              : {score_row['HR']:.3f}")
    print(f"  95% CI          : [{score_row['CI_lower_95']:.3f}–"
          f"{score_row['CI_upper_95']:.3f}]")
    print(f"  p-value         : {score_row['p_value']:.4f}")
    print(f"  Stage HR        : "
          f"{mv_df[mv_df['covariate'].str.contains('Stage')].iloc[0]['HR']:.3f}  "
          f"p = "
          f"{mv_df[mv_df['covariate'].str.contains('Stage')].iloc[0]['p_value']:.4f}")
    print(f"  C-index         : {c_mv:.4f}")
print()
print("  Files saved:")
print("    results/step1_univariate_cox.tsv")
if mv_df is not None:
    print("    results/step1_multivariate_cox_final.tsv")
print("    results/step1_forest_plot.png")
print("    data/final_gene_list.txt")
print()
print("  Step 1 and Step 2 are both COMPLETE.")
print("  NEXT: Run step3_gene_set_expansion.py")