"""
Step 3b: Rebuild Full Pipeline with Expanded Gene Set
======================================================
Step 3a already wrote directly to:
  data/histone_genes3.txt   (114 genes, 10 families)
  data/gene_groups10.tsv    (114 rows, 10 families)

This script simply verifies those files are ready and
re-runs every original pipeline script in sequence.
New results overwrite existing files automatically.
Baseline is safely archived in baseline_4families/.

Run from project root:
  python scripts/step3b_rebuild_expanded_pipeline.py
"""

import subprocess, sys, os
import pandas as pd
import numpy as np

print("=" * 65)
print("STEP 3b: REBUILD PIPELINE WITH EXPANDED GENE SET")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# VERIFY INPUT FILES — written directly by Step 3a
# ─────────────────────────────────────────────────────────────
print("\n[1] Verifying Step 3a outputs...")

for f in ["data/histone_genes3.txt", "data/gene_groups10.tsv"]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"{f} not found. Run step3a_extract_expanded_genes.py first."
        )

with open("data/histone_genes3.txt") as f:
    expanded_genes = [l.strip() for l in f if l.strip()]

gene_groups = pd.read_csv("data/gene_groups10.tsv", sep="\t")

print(f"    ✓ data/histone_genes3.txt  : {len(expanded_genes)} genes (was 62)")
print(f"    ✓ data/gene_groups10.tsv   : {len(gene_groups)} rows, "
      f"{gene_groups['group'].nunique()} families")
print(f"      Families: {sorted(gene_groups['group'].unique().tolist())}")

# Confirm these are the expanded versions, not the old baseline
if len(expanded_genes) < 100:
    raise ValueError(
        f"histone_genes3.txt has only {len(expanded_genes)} genes — "
        f"expected 114. Re-run step3a_extract_expanded_genes.py."
    )
if gene_groups['group'].nunique() < 8:
    raise ValueError(
        f"gene_groups10.tsv has only {gene_groups['group'].nunique()} families — "
        f"expected 10. Re-run step3a_extract_expanded_genes.py."
    )

print(f"\n    ✓ Confirmed: expanded gene set is ready")

# ─────────────────────────────────────────────────────────────
# PIPELINE SCRIPTS IN ORDER
# ─────────────────────────────────────────────────────────────
pipeline = [
    ("scripts/filter_histone_tcga5.py",
     "Filter TCGA expression for expanded gene set"),
    ("scripts/build_histone_graph6.py",
     "Build STRING PPI graph with expanded genes"),
    ("scripts/align_features_with_graph7.py",
     "Align expression features with new graph"),
    ("scripts/build_graph_matrices8.py",
     "Build adjacency matrix A and feature matrix X"),

    # create_gene_groups10.py — INTENTIONALLY SKIPPED
    # Step 3a already wrote the correct 10-family gene_groups10.tsv.
    # Running script 10 would overwrite it with old 4-family logic
    # and drop all new genes (DNMT, TET, CBX, BRD, SIRT) as "OTHER".

    # build_patient_graph9_optional.py — INTENTIONALLY SKIPPED
    # Superseded by build_patient_graph_lasso19.py which uses
    # LASSO-selected genes. Scripts 19-21 use the LASSO graph only.

    ("scripts/analyze_group_interactions11.py",
     "Analyze inter-family interactions (10 families)"),
    ("scripts/find_key_genes12.py",
     "Identify hub genes in expanded network"),
    ("scripts/top_hub_genes13.py",
     "Select top hub genes"),
    ("scripts/cox_analysis_correct14.py",
     "Univariate Cox screening of expanded gene set"),
    ("scripts/multivariate_cox_regression15.py",
     "Multivariate Cox on significant hub genes"),
    ("scripts/lasso_cox_survival_model16.py",
     "LASSO-Cox gene signature (expanded)"),
    ("scripts/kaplan_meier_analysis17.py",
     "Kaplan-Meier curve"),
    ("scripts/evaluate_cindex18.py",
     "C-index evaluation"),
    ("scripts/build_patient_graph_lasso19.py",
     "Build patient graph from LASSO scores"),
    ("scripts/train_graph_transformer20.py",
     "Train Graph Transformer on expanded graph"),
    ("scripts/km_graph_transformer_validation21.py",
     "Graph Transformer KM validation"),
]

# ─────────────────────────────────────────────────────────────
# RUN EACH SCRIPT
# ─────────────────────────────────────────────────────────────
print("\n[2] Running pipeline scripts in sequence...")
print("    Results overwrite existing files — baseline is in baseline_4families/")
print()

log    = []
failed = []

for script, description in pipeline:
    if not os.path.exists(script):
        print(f"    SKIP  : {script} — not found")
        log.append({"script": script, "status": "SKIPPED",
                    "description": description})
        continue

    print(f"    Running : {description}")
    print(f"              python {script}")

    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        out_lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        for line in out_lines[-3:]:
            print(f"              {line}")
        print(f"    ✓ Done\n")
        log.append({"script": script, "status": "SUCCESS",
                    "description": description})
    else:
        print(f"    ✗ FAILED")
        err = result.stderr.strip()
        if err:
            for line in err.split("\n")[-5:]:
                print(f"      {line}")
        print()
        log.append({"script": script, "status": "FAILED",
                    "description": description,
                    "error": err[-300:] if err else ""})
        failed.append(script)

# ─────────────────────────────────────────────────────────────
# SAVE RUN LOG
# ─────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
pd.DataFrame(log).to_csv(
    "results/step3b_pipeline_run_log.tsv", sep="\t", index=False)

# ─────────────────────────────────────────────────────────────
# BASELINE vs EXPANDED COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print("\n[3] Building baseline vs expanded comparison...")

comparison = []

# Gene count
b_gene_file = "data/baseline_4families/histone_genes3.txt"
if os.path.exists(b_gene_file):
    with open(b_gene_file) as f:
        b_genes = [l.strip() for l in f if l.strip()]
    comparison.append({
        "Metric":             "Genes in set",
        "Baseline_4families": len(b_genes),
        "Expanded_10families": len(expanded_genes),
        "Change":             f"+{len(expanded_genes) - len(b_genes)}"
    })

# Graph edges
b_edge = "data/baseline_4families/histone_graph_edges6.tsv"
e_edge = "data/ppi/histone_graph_edges6.tsv"
if os.path.exists(b_edge) and os.path.exists(e_edge):
    b_e = pd.read_csv(b_edge, sep="\t")
    e_e = pd.read_csv(e_edge, sep="\t")
    comparison.append({
        "Metric":             "PPI graph edges",
        "Baseline_4families": len(b_e),
        "Expanded_10families": len(e_e),
        "Change":             f"+{len(e_e) - len(b_e)}"
    })

# Significant hub genes
b_hub = "data/baseline_4families/significant_hub_genes14.tsv"
e_hub = "results/significant_hub_genes14.tsv"
if os.path.exists(b_hub) and os.path.exists(e_hub):
    b_h = pd.read_csv(b_hub, sep="\t")
    e_h = pd.read_csv(e_hub, sep="\t")
    comparison.append({
        "Metric":             "Significant hub genes",
        "Baseline_4families": len(b_h),
        "Expanded_10families": len(e_h),
        "Change":             f"{len(e_h) - len(b_h):+d}"
    })

# LASSO selected genes
b_lasso = "data/baseline_4families/lasso_selected_genes16.tsv"
e_lasso = "results/lasso_selected_genes16.tsv"
if os.path.exists(b_lasso) and os.path.exists(e_lasso):
    b_l = pd.read_csv(b_lasso, sep="\t")
    e_l = pd.read_csv(e_lasso, sep="\t")
    comparison.append({
        "Metric":             "LASSO selected genes",
        "Baseline_4families": len(b_l),
        "Expanded_10families": len(e_l),
        "Change":             f"{len(e_l) - len(b_l):+d}"
    })

# C-index
def compute_cindex(filepath):
    df = pd.read_csv(filepath, sep="\t", index_col=0)
    t  = df["OS.time"].values.astype(float)
    e  = df["OS"].values.astype(float)
    r  = df["risk_score"].values.astype(float)
    conc = disc = tied = 0
    n = len(t)
    for i in range(n):
        for j in range(i+1, n):
            if e[i]==0 and e[j]==0: continue
            if t[i]==t[j]: continue
            if e[i]==1 and t[i]<t[j]:
                if r[i]>r[j]:   conc+=1
                elif r[i]<r[j]: disc+=1
                else:            tied+=1
            elif e[j]==1 and t[j]<t[i]:
                if r[j]>r[i]:   conc+=1
                elif r[j]<r[i]: disc+=1
                else:            tied+=1
    total = conc+disc+tied
    return round((conc+0.5*tied)/total, 4) if total > 0 else float("nan")

b_risk = "data/baseline_4families/lasso_risk_scores16.tsv"
e_risk = "results/lasso_risk_scores16.tsv"
if os.path.exists(b_risk) and os.path.exists(e_risk):
    b_ci = compute_cindex(b_risk)
    e_ci = compute_cindex(e_risk)
    comparison.append({
        "Metric":             "C-index (LASSO-Cox)",
        "Baseline_4families": b_ci,
        "Expanded_10families": e_ci,
        "Change":             f"{e_ci - b_ci:+.4f}"
    })

if comparison:
    comp_df = pd.DataFrame(comparison)
    print()
    print("    " + "=" * 58)
    print("    BASELINE (4 families) vs EXPANDED (10 families)")
    print("    " + "=" * 58)
    print(comp_df.to_string(index=False))
    comp_df.to_csv(
        "results/baseline_vs_expanded_comparison.tsv",
        sep="\t", index=False
    )
    print(f"\n    ✓ Saved → results/baseline_vs_expanded_comparison.tsv")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
n_ok   = sum(1 for r in log if r["status"] == "SUCCESS")
n_fail = sum(1 for r in log if r["status"] == "FAILED")
n_skip = sum(1 for r in log if r["status"] == "SKIPPED")

print("\n" + "=" * 65)
print("STEP 3b COMPLETE — SUMMARY")
print("=" * 65)
print(f"  Gene set        : {len(expanded_genes)} genes (10 families)")
print(f"  Scripts SUCCESS : {n_ok}")
print(f"  Scripts FAILED  : {n_fail}")
print(f"  Scripts SKIPPED : {n_skip}")
if failed:
    print()
    print("  Failed scripts (fix and re-run individually):")
    for s in failed:
        print(f"    ✗ {s}")
print()
print("  Baseline preserved in:")
print("    results/baseline_4families/")
print("    data/baseline_4families/")
print()
print("  NEXT STEPS:")
print("  1. python new_code/step1_composite_risk_final.py")
print("  2. python new_code/step2_clinical_association_fixed.py")
print("  3. External validation on GSE14520")