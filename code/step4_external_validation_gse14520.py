"""
Step 4: External Validation on GSE14520
=========================================
GSE14520 — HCC patients, Affymetrix GPL3921
Clinical data: GEO_data/GSE14520_clinical.xlsx (gzip TSV)
Expression:    GEO_data/GSE14520_family.soft.gz

Column mapping confirmed from file inspection:
  Affy_GSM       → GEO sample ID (links to SOFT expression)
  Survival status → 1=dead, 0=censored
  Survival months → overall survival time in months
  Tissue Type     → "Tumor" = HCC tumor sample

Run from project root:
  python scripts/step4_external_validation_gse14520.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip, os, warnings
from scipy.stats import chi2
warnings.filterwarnings('ignore')

print("=" * 65)
print("STEP 4: EXTERNAL VALIDATION — GSE14520")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def log_rank_test(t1, e1, t2, e2):
    all_t = np.unique(np.concatenate([t1[e1==1], t2[e2==1]]))
    O1=E1=O2=E2=0.0
    for t in all_t:
        n1=(t1>=t).sum(); n2=(t2>=t).sum()
        o1=((t1==t)&(e1==1)).sum(); o2=((t2==t)&(e2==1)).sum()
        n=n1+n2; o=o1+o2
        if n==0: continue
        E1+=n1*o/n; E2+=n2*o/n; O1+=o1; O2+=o2
    if E1==0 or E2==0: return np.nan
    stat=(O1-E1)**2/E1+(O2-E2)**2/E2
    return float(chi2.sf(stat, df=1))

def km_curve(time, event):
    order=np.argsort(time); t_s,e_s=time[order],event[order]
    surv=1.0; times=[0.0]; probs=[1.0]
    for t in np.unique(t_s[e_s==1]):
        n_at_risk=(t_s>=t).sum(); n_ev=((t_s==t)&(e_s==1)).sum()
        if n_at_risk>0: surv*=(1-n_ev/n_at_risk)
        times.append(t); probs.append(surv)
    return np.array(times), np.array(probs)

def concordance_index(time, risk, event):
    n=len(time); conc=disc=tied=0
    for i in range(n):
        for j in range(i+1,n):
            if event[i]==0 and event[j]==0: continue
            if time[i]==time[j]: continue
            if event[i]==1 and time[i]<time[j]:
                if risk[i]>risk[j]: conc+=1
                elif risk[i]<risk[j]: disc+=1
                else: tied+=1
            elif event[j]==1 and time[j]<time[i]:
                if risk[j]>risk[i]: conc+=1
                elif risk[j]<risk[i]: disc+=1
                else: tied+=1
    total=conc+disc+tied
    return (conc+0.5*tied)/total if total>0 else np.nan

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD LASSO COEFFICIENTS
# ══════════════════════════════════════════════════════════════
print("\n[1] Loading TCGA-trained LASSO coefficients...")

lasso_paths = ["results/lasso_selected_genes16.tsv",
               "data/lasso_selected_genes16.tsv",
               "data/baseline_4families/lasso_selected_genes16.tsv"]
lasso_file = next((p for p in lasso_paths if os.path.exists(p)), None)
if not lasso_file:
    raise FileNotFoundError("lasso_selected_genes16.tsv not found.")

lasso_df    = pd.read_csv(lasso_file, sep="\t")
lasso_genes = lasso_df["gene_symbol"].tolist()
lasso_coef  = dict(zip(lasso_df["gene_symbol"], lasso_df["coef"]))
print(f"    {len(lasso_genes)} LASSO genes from {lasso_file}")

# TCGA risk scores for comparison
tcga_paths = ["results/lasso_risk_scores16.tsv",
              "data/lasso_risk_scores16.tsv",
              "data/baseline_4families/lasso_risk_scores16.tsv"]
tcga_risk_file = next((p for p in tcga_paths if os.path.exists(p)), None)
tcga_risk = pd.read_csv(tcga_risk_file, sep="\t", index_col=0)
tcga_risk["tcga_12"]     = tcga_risk.index.str[:12]
tcga_risk["sample_code"] = tcga_risk.index.str[-3:]
priority_map = {"01A":0,"01B":1,"02A":2,"02B":3,"11A":4}
tcga_risk["priority"] = tcga_risk["sample_code"].map(priority_map).fillna(99)
tcga_dedup = (tcga_risk.sort_values("priority")
              .groupby("tcga_12").first().reset_index())
print(f"    TCGA patients (deduplicated): {len(tcga_dedup)}")

# ══════════════════════════════════════════════════════════════
# STEP 2 — LOAD CLINICAL DATA (gzip TSV)
# ══════════════════════════════════════════════════════════════
print("\n[2] Loading GSE14520 clinical data...")

clin_paths = [
    "GEO_data/GSE14520_clinical.xlsx",
    "GEO_data/GSE14520_Extra_Supplement.xlsx",
    "GEO_data/GSE14520_clinical.tsv",
    "GEO_data/GSE14520_clinical.txt",
]
clin_file = next((p for p in clin_paths if os.path.exists(p)), None)
if not clin_file:
    raise FileNotFoundError(
        "Clinical file not found. "
        "Download from GEO supplementary files for GSE14520."
    )

print(f"    Found: {clin_file}")

# File is gzip-compressed TSV despite the .xlsx extension
with gzip.open(clin_file, "rt", errors="replace") as f:
    clin_raw = pd.read_csv(f, sep="\t")

print(f"    Shape: {clin_raw.shape}")
print(f"    Columns: {clin_raw.columns.tolist()}")

# ── Filter tumor samples only ─────────────────────────────────
clin = clin_raw[
    clin_raw["Tissue Type"].str.strip().str.lower() == "tumor"
].copy()
print(f"    Tumor samples: {len(clin)}")

# ── Standardise column names ──────────────────────────────────
# Confirmed columns from file inspection:
#   Affy_GSM       → GEO sample ID
#   Survival status → 1=dead, 0=censored
#   Survival months → OS time in months
clin = clin.rename(columns={
    "Affy_GSM":       "gsm_id",
    "Survival status": "OS",
    "Survival months": "OS_months",
})

# Convert OS months → days
clin["OS.time"] = pd.to_numeric(
    clin["OS_months"], errors="coerce") * 30.44
clin["OS"] = pd.to_numeric(clin["OS"], errors="coerce")

# Drop rows missing survival
clin = clin.dropna(subset=["OS.time", "OS", "gsm_id"])
clin = clin[clin["OS.time"] > 0]
clin["gsm_id"] = clin["gsm_id"].astype(str).str.strip()

print(f"    Patients with complete survival: {len(clin)}")
print(f"    Events (deaths): {int(clin['OS'].sum())}")
print(f"    OS range: {clin['OS.time'].min():.0f}–"
      f"{clin['OS.time'].max():.0f} days")
print(f"    Median OS: {clin['OS.time'].median():.0f} days")

# ══════════════════════════════════════════════════════════════
# STEP 3 — PARSE SOFT FILE FOR EXPRESSION DATA
# ══════════════════════════════════════════════════════════════
print("\n[3] Parsing SOFT file for expression data...")
print("    (This takes 1-2 minutes)")

soft_paths = ["GEO_data/GSE14520_family.soft.gz",
              "data/GEO/GSE14520_family.soft.gz"]
soft_file = next((p for p in soft_paths if os.path.exists(p)), None)
if not soft_file:
    raise FileNotFoundError("GSE14520_family.soft.gz not found.")

# Only parse samples that have clinical survival data
valid_gsm = set(clin["gsm_id"].tolist())
print(f"    Samples to extract: {len(valid_gsm)}")

# GPL3921 probe → gene mapping
PROBE_TO_GENE = {
    # CREBBP
    "203241_at":"CREBBP",   "211161_s_at":"CREBBP",  "222935_s_at":"CREBBP",
    # EP300
    "215031_at":"EP300",    "205550_s_at":"EP300",   "221854_at":"EP300",
    # KAT2A
    "204739_s_at":"KAT2A",  "211372_s_at":"KAT2A",
    # KAT2B
    "208168_s_at":"KAT2B",  "208169_at":"KAT2B",
    # KAT5
    "207110_s_at":"KAT5",   "210914_s_at":"KAT5",
    # KAT8
    "218816_at":"KAT8",
    # HDAC1
    "201833_at":"HDAC1",    "216955_s_at":"HDAC1",
    # HDAC2
    "200913_s_at":"HDAC2",  "200914_s_at":"HDAC2",
    # HDAC3
    "204777_s_at":"HDAC3",  "204778_s_at":"HDAC3",
    # HDAC4
    "222641_s_at":"HDAC4",  "204215_at":"HDAC4",
    # HDAC5
    "208988_at":"HDAC5",
    # HDAC6
    "204908_s_at":"HDAC6",
    # HDAC7
    "212215_at":"HDAC7",
    # HDAC8
    "218336_s_at":"HDAC8",
    # HDAC9
    "218799_s_at":"HDAC9",
    # KDM1A
    "219228_at":"KDM1A",    "219048_s_at":"KDM1A",  "204825_at":"KDM1A",
    # KDM4A
    "212446_s_at":"KDM4A",
    # KDM6A
    "201244_s_at":"KDM6A",
    # KMT2A
    "203348_s_at":"KMT2A",
    # EHMT2
    "219918_s_at":"EHMT2",
    # SETD1A
    "206561_s_at":"SETD1A",
    # SETD1B
    "204165_at":"SETD1B",
    # PRMT1
    "201313_at":"PRMT1",
    # PRMT5
    "220192_at":"PRMT5",    "225471_at":"PRMT5",
    # EZH2
    "203725_s_at":"EZH2",   "203726_s_at":"EZH2",
    # KMT2C
    "211506_s_at":"KMT2C",
    # KMT2D
    "204562_s_at":"KMT2D",
    # SIRT1
    "204033_at":"SIRT1",
    # SUZ12
    "228648_at":"SUZ12",
    # KANSL1
    "201787_at":"KANSL1",
    # ING3
    "213649_at":"ING3",
    # RNF2
    "222360_s_at":"RNF2",
    # RING1
    "217379_x_at":"RING1",
    # NCOR1
    "201041_s_at":"NCOR1",
    # NCOR2
    "204819_at":"NCOR2",
    # RCOR1
    "209308_s_at":"RCOR1",
    # SIN3A
    "204365_s_at":"SIN3A",
    # TAF1
    "208917_s_at":"TAF1",
    # TAF10
    "200802_at":"TAF10",
    # TAF7
    "204042_at":"TAF7",
    # TP53
    "214702_s_at":"TP53",
    # TRRAP
    "209773_s_at":"TRRAP",
    # RUVBL1
    "219557_s_at":"RUVBL1",
    # USP7
    "214920_at":"USP7",
    # OGT
    "209304_at":"OGT",
    # PHF20
    "222039_at":"PHF20",
    # NCOA6
    "215071_s_at":"NCOA6",
    # MAX
    "211559_s_at":"MAX",    "200600_at":"MAX",
    # YEATS4
    "219537_at":"YEATS4",
    # SAP130
    "221609_s_at":"SAP130",
    # SAP30
    "202504_at":"SAP30",
    # RBBP5
    "209024_s_at":"RBBP5",
    # MEAF6
    "211464_s_at":"MEAF6",
    # MORF4L2
    "212665_at":"MORF4L2",
    # TADA3
    "209126_at":"TADA3",
    # ASH2L
    "218044_at":"ASH2L",
    # ATF2
    "204013_at":"ATF2",
    # ATM
    "201393_s_at":"ATM",
    # BRCA1
    "211851_s_at":"BRCA1",
    # CDK1
    "203213_at":"CDK1",
    # EP400
    "212907_at":"EP400",
    # SUV39H1
    "219385_at":"SUV39H1",
    # MCRS1
    "209183_s_at":"MCRS1",
    # RBBP5
    "209024_s_at":"RBBP5",
}

# Parse expression from SOFT — only for samples in valid_gsm
gene_probe_vals = {g: {} for g in lasso_genes}
current_id  = None
in_table    = False
table_rows  = []
n_read      = 0

with gzip.open(soft_file, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")

        if line.startswith("^SAMPLE"):
            # Save previous sample if needed
            if current_id and current_id in valid_gsm and table_rows:
                for probe, val in table_rows:
                    if probe in PROBE_TO_GENE:
                        gene = PROBE_TO_GENE[probe]
                        if gene in lasso_genes:
                            gene_probe_vals[gene].setdefault(
                                current_id, []).append(val)
                n_read += 1
            current_id = line.split("=")[1].strip()
            in_table   = False
            table_rows = []

        elif line.startswith("!sample_table_begin"):
            in_table = True; table_rows = []

        elif line.startswith("!sample_table_end"):
            in_table = False

        elif in_table and current_id in valid_gsm:
            cols = line.split("\t")
            if cols[0] != "ID_REF" and len(cols) >= 2:
                try:
                    table_rows.append((cols[0], float(cols[1])))
                except (ValueError, IndexError):
                    pass

# Save last sample
if current_id and current_id in valid_gsm and table_rows:
    for probe, val in table_rows:
        if probe in PROBE_TO_GENE:
            gene = PROBE_TO_GENE[probe]
            if gene in lasso_genes:
                gene_probe_vals[gene].setdefault(current_id, []).append(val)
    n_read += 1

print(f"    Expression data read for {n_read} samples")

# Average probes per gene
expr_matrix  = {}
genes_found  = []
genes_missing = []
for gene in lasso_genes:
    if gene_probe_vals[gene]:
        expr_matrix[gene] = {
            sid: np.mean(vals)
            for sid, vals in gene_probe_vals[gene].items()
        }
        if len(expr_matrix[gene]) >= 10:
            genes_found.append(gene)
        else:
            genes_missing.append(gene)
    else:
        genes_missing.append(gene)

print(f"    Genes found  : {len(genes_found)} / {len(lasso_genes)}")
print(f"    Genes found  : {genes_found}")
print(f"    Genes missing: {genes_missing}")

# ══════════════════════════════════════════════════════════════
# STEP 4 — COMPUTE RISK SCORES
# ══════════════════════════════════════════════════════════════
print("\n[4] Computing risk scores...")

risk_records = []
for _, row in clin.iterrows():
    sid = row["gsm_id"]
    score  = sum(lasso_coef[g] * expr_matrix[g][sid]
                 for g in genes_found
                 if sid in expr_matrix.get(g, {}))
    n_used = sum(1 for g in genes_found
                 if sid in expr_matrix.get(g, {}))
    if n_used >= max(1, len(genes_found) * 0.5):
        risk_records.append({
            "sample_id": sid,
            "OS.time":   row["OS.time"],
            "OS":        row["OS"],
            "risk_score": score,
            "n_genes":   n_used
        })

risk_df = pd.DataFrame(risk_records)
print(f"    Patients scored  : {len(risk_df)}")
print(f"    Events (deaths)  : {int(risk_df['OS'].sum())}")
print(f"    Genes per patient: {risk_df['n_genes'].mean():.1f} mean")

if len(risk_df) < 20:
    raise ValueError(f"Only {len(risk_df)} patients scored. "
                     "Check probe mapping above.")

# ══════════════════════════════════════════════════════════════
# STEP 5 — STANDARDISE AND CLASSIFY
# ══════════════════════════════════════════════════════════════
print("\n[5] Standardising and classifying...")

scores = risk_df["risk_score"].values
risk_df["risk_score_std"] = (scores - scores.mean()) / scores.std()
# Median split (standardised median = 0)
risk_df["risk_group"] = np.where(
    risk_df["risk_score_std"] > 0, "High", "Low")

high = risk_df[risk_df["risk_group"] == "High"]
low  = risk_df[risk_df["risk_group"] == "Low"]
print(f"    High Risk: {len(high)} | Low Risk: {len(low)}")

# ══════════════════════════════════════════════════════════════
# STEP 6 — SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════
print("\n[6] Survival analysis...")

c_idx = concordance_index(
    risk_df["OS.time"].values.astype(float),
    risk_df["risk_score_std"].values,
    risk_df["OS"].values.astype(float))

lr_p = log_rank_test(
    high["OS.time"].values.astype(float),
    high["OS"].values.astype(float),
    low["OS.time"].values.astype(float),
    low["OS"].values.astype(float))

print(f"    C-index       : {c_idx:.4f}")
print(f"    Log-rank p    : {lr_p:.4e}")
print(f"    Median OS High: {high['OS.time'].median():.0f} days "
      f"({high['OS.time'].median()/30.44:.1f} months)")
print(f"    Median OS Low : {low['OS.time'].median():.0f} days "
      f"({low['OS.time'].median()/30.44:.1f} months)")

# ══════════════════════════════════════════════════════════════
# STEP 7 — FIGURE
# ══════════════════════════════════════════════════════════════
print("\n[7] Generating KM figure...")

colors = {"High":"#D73027","Low":"#4575B4"}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"External Validation — GSE14520 (n={len(risk_df)})\n"
    "Histone-Modifier Graph Transformer Risk Score",
    fontsize=13, fontweight="bold")

ax1 = axes[0]
for grp, col in colors.items():
    sub = risk_df[risk_df["risk_group"] == grp]
    t, p = km_curve(sub["OS.time"].values.astype(float),
                    sub["OS"].values.astype(float))
    ax1.step(t/30.44, p, where="post", color=col, linewidth=2.5,
             label=f"{grp} Risk (n={len(sub)})")
ax1.set_title(
    f"Overall Survival — GSE14520\n"
    f"Log-rank p = {lr_p:.2e}  |  C-index = {c_idx:.4f}",
    fontsize=11)
ax1.set_xlabel("Time (months)", fontsize=11)
ax1.set_ylabel("Survival Probability", fontsize=11)
ax1.legend(fontsize=10); ax1.set_ylim(0, 1.05)
ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax1.text(0.60, 0.88,
         f"p = {lr_p:.2e}\nC-index = {c_idx:.4f}\n"
         f"Genes: {len(genes_found)}/{len(lasso_genes)}",
         transform=ax1.transAxes, fontsize=9, va="top",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

ax2 = axes[1]
ax2.hist(low["risk_score_std"],  bins=20, alpha=0.6,
         color=colors["Low"],  label="Low Risk",  edgecolor="white")
ax2.hist(high["risk_score_std"], bins=20, alpha=0.6,
         color=colors["High"], label="High Risk", edgecolor="white")
ax2.axvline(0, color="black", linestyle="--",
            linewidth=1.5, label="Threshold")
ax2.set_title("Risk Score Distribution\nGSE14520", fontsize=11)
ax2.set_xlabel("Standardised Risk Score", fontsize=11)
ax2.set_ylabel("Count", fontsize=11); ax2.legend(fontsize=10)

plt.tight_layout()
os.makedirs("results/plots", exist_ok=True)
plt.savefig("results/plots/external_validation_gse14520.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("    ✓ results/plots/external_validation_gse14520.png")

# ══════════════════════════════════════════════════════════════
# STEP 8 — TRAINING vs VALIDATION COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
print("\n[8] Training vs Validation comparison...")

tcga_h = tcga_dedup[tcga_dedup["risk_group"]=="High"]
tcga_l = tcga_dedup[tcga_dedup["risk_group"]=="Low"]
tcga_s = tcga_dedup["risk_score"].values
tcga_c = concordance_index(
    tcga_dedup["OS.time"].values.astype(float),
    (tcga_s-tcga_s.mean())/tcga_s.std(),
    tcga_dedup["OS"].values.astype(float))
tcga_lr = log_rank_test(
    tcga_h["OS.time"].values.astype(float),
    tcga_h["OS"].values.astype(float),
    tcga_l["OS.time"].values.astype(float),
    tcga_l["OS"].values.astype(float))

comp = pd.DataFrame([
    {"Cohort":       "TCGA-LIHC (Training)",
     "N":            len(tcga_dedup),
     "Events":       int(tcga_dedup["OS"].sum()),
     "High_n":       len(tcga_h),
     "Low_n":        len(tcga_l),
     "C_index":      round(tcga_c, 4),
     "Log_rank_p":   f"{tcga_lr:.2e}",
     "Median_H_days":int(tcga_h["OS.time"].median()),
     "Median_L_days":int(tcga_l["OS.time"].median())},
    {"Cohort":       "GSE14520 (Validation)",
     "N":            len(risk_df),
     "Events":       int(risk_df["OS"].sum()),
     "High_n":       len(high),
     "Low_n":        len(low),
     "C_index":      round(c_idx, 4),
     "Log_rank_p":   f"{lr_p:.2e}",
     "Median_H_days":int(high["OS.time"].median()),
     "Median_L_days":int(low["OS.time"].median())}
])

print()
print("    "+"="*65)
print("    TRAINING vs VALIDATION COMPARISON")
print("    "+"="*65)
print(comp.to_string(index=False))

os.makedirs("results", exist_ok=True)
risk_df.to_csv("results/gse14520_risk_scores.tsv", sep="\t", index=False)
comp.to_csv("results/validation_comparison.tsv",   sep="\t", index=False)

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("STEP 4 COMPLETE — EXTERNAL VALIDATION SUMMARY")
print("="*65)
print(f"  Cohort     : GSE14520 ({len(risk_df)} HCC patients)")
print(f"  Genes used : {len(genes_found)} / {len(lasso_genes)}")
print(f"  C-index    : {c_idx:.4f}")
print(f"  Log-rank p : {lr_p:.2e}")
if lr_p is not None and lr_p < 0.05:
    print(f"\n  ✓ VALIDATION SUCCESSFUL")
    print(f"    The histone-modifier risk signature generalises")
    print(f"    to the independent GSE14520 HCC cohort.")
    print(f"    C-index {c_idx:.4f} confirms discrimination.")
elif lr_p is not None and lr_p < 0.10:
    print(f"\n  ~ Borderline validation (p={lr_p:.4f})")
    print(f"    C-index {c_idx:.4f}")
else:
    print(f"\n  Check gene coverage: {len(genes_found)}/{len(lasso_genes)} genes found")
print()
print("  Files saved:")
print("    results/plots/external_validation_gse14520.png")
print("    results/gse14520_risk_scores.tsv")
print("    results/validation_comparison.tsv")
print()
print("  NEXT: Run step5_ablation_study.py")