"""
Step 3a: Extract Expanded Histone Gene Set from EpiFactors
===========================================================
WHAT THIS REPLACES:
  extract_histone_genes_from_epifactors3.py  (4 families: HAT, HDAC, HMT, KDM)

WHAT IS NEW:
  Adds 6 additional biologically relevant families:
    PRMT  — Protein Arginine Methyltransferases
    SIRT  — Sirtuins (Class III NAD-dependent deacetylases)
    DNMT  — DNA Methyltransferases (epigenetic writers)
    TET   — DNA Demethylases (hydroxymethylation erasers)
    CBX   — Chromobox histone methyl readers (PRC1 complex)
    BRD   — Bromodomain acetyl readers (BET family, drug targets)

BIOLOGICAL JUSTIFICATION:
  - PRMT: arginine methylation of H3/H4 regulates gene activation in HCC
  - SIRT: NAD-dependent deacetylation links metabolism to epigenetics in HCC
  - DNMT: DNA methylation cross-talks with histone modification in HCC subtypes
  - TET: oxidative DNA demethylation cooperates with KDM demethylases
  - CBX: read H3K27me3/H3K9me3 marks placed by your existing HMT genes
  - BRD: read H3K27ac/H3K9ac marks placed by your existing HAT genes

TOTAL GENES: ~100 (up from 62 in baseline)
10 families: HAT, HDAC, HMT, KDM, PRMT, SIRT, DNMT, TET, CBX, BRD

Output: data/histone_genes3.txt

Run from project root:
  python scripts/step3a_extract_expanded_genes.py
"""

import pandas as pd
import os

print("=" * 62)
print("STEP 3a: EXPANDED HISTONE GENE EXTRACTION")
print("=" * 62)

# ─────────────────────────────────────────────────────────────
# LOAD EPIFACTORS
# ─────────────────────────────────────────────────────────────
epi_paths = [
    "data/epifactors_proteins.csv",
    "data/epifactors/epifactors_proteins.csv",
]
epi_file = None
for p in epi_paths:
    if os.path.exists(p):
        epi_file = p
        break

if epi_file is None:
    raise FileNotFoundError(
        "epifactors_proteins.csv not found. "
        "Place it in data/ and rerun."
    )

df = pd.read_csv(epi_file)
df["Function_l"]     = df["Function"].str.lower().fillna("")
df["Modification_l"] = df["Modification"].str.lower().fillna("")
df["Target_l"]       = df["Target"].str.lower().fillna("")

print(f"\n    EpiFactors loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"    Unique genes in database: "
      f"{df['HGNC_symbol'].dropna().nunique()}")

# ─────────────────────────────────────────────────────────────
# FAMILY DEFINITIONS
# ─────────────────────────────────────────────────────────────
families = {}

# ── EXISTING 4 FAMILIES (kept identical to baseline) ────────

# HAT: Histone acetyltransferases
families["HAT"] = df[
    df["HGNC_symbol"].str.contains("^KAT", na=False) |
    df["HGNC_symbol"].isin(["EP300", "CREBBP"])
]["HGNC_symbol"].dropna().unique().tolist()

# HDAC: Histone deacetylases (Class I, II, IV)
# Note: EpiFactors stores these as 'Histone acetylation' (they remove it)
families["HDAC"] = df[
    df["HGNC_symbol"].str.contains("^HDAC", na=False) &
    df["Modification_l"].str.contains("acetyl", na=False)
]["HGNC_symbol"].dropna().unique().tolist()

# HMT: Histone methyltransferases — expanded to include EHMT, NSD, SMYD, ASH, DOT1L, PRDM
families["HMT"] = df[
    df["HGNC_symbol"].str.contains(
        "^KMT|^SET|^EZH|^SMYD|^EHMT|^NSD|^ASH|^DOT1|^PRDM|^WHSC",
        na=False
    ) &
    df["Modification_l"].str.contains("histone methyl", na=False)
]["HGNC_symbol"].dropna().unique().tolist()

# KDM: Histone lysine demethylases — expanded to include JMJD family
families["KDM"] = df[
    df["HGNC_symbol"].str.contains("^KDM|^JMJD", na=False) &
    df["Modification_l"].str.contains("methyl|demethy", na=False)
]["HGNC_symbol"].dropna().unique().tolist()

# ── NEW FAMILY 1: PRMT — Arginine methyltransferases ────────
families["PRMT"] = df[
    df["HGNC_symbol"].str.contains("^PRMT", na=False) &
    df["Modification_l"].str.contains("methyl", na=False)
]["HGNC_symbol"].dropna().unique().tolist()

# ── NEW FAMILY 2: SIRT — Sirtuins (Class III HDACs) ─────────
families["SIRT"] = df[
    df["HGNC_symbol"].str.contains("^SIRT", na=False) &
    df["Modification_l"].str.contains("acetyl|methyl", na=False)
]["HGNC_symbol"].dropna().unique().tolist()

# ── NEW FAMILY 3: DNMT — DNA methyltransferases ─────────────
families["DNMT"] = df[
    df["HGNC_symbol"].str.contains("^DNMT", na=False) &
    df["Modification_l"].str.contains("dna methyl", na=False)
]["HGNC_symbol"].dropna().unique().tolist()

# ── NEW FAMILY 4: TET — DNA demethylases ────────────────────
# EpiFactors lists as 'DNA hydroxymethylation'
families["TET"] = df[
    df["HGNC_symbol"].str.contains("^TET", na=False)
]["HGNC_symbol"].dropna().unique().tolist()

# ── NEW FAMILY 5: CBX — Chromobox methyl readers ────────────
families["CBX"] = df[
    df["HGNC_symbol"].str.contains("^CBX", na=False) &
    df["Function_l"].str.contains("read", na=False)
]["HGNC_symbol"].dropna().unique().tolist()

# ── NEW FAMILY 6: BRD — Bromodomain acetyl readers ──────────
families["BRD"] = df[
    df["HGNC_symbol"].str.contains("^BRD", na=False) &
    df["Function_l"].str.contains("read", na=False)
]["HGNC_symbol"].dropna().unique().tolist()

# ─────────────────────────────────────────────────────────────
# COMPILE AND DEDUPLICATE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("FAMILY BREAKDOWN")
print("=" * 62)

all_genes   = []
family_rows = []

for fam, genes in families.items():
    genes_sorted = sorted(set(genes))
    tag = "NEW" if fam in ["PRMT","SIRT","DNMT","TET","CBX","BRD"] else "existing"
    print(f"  {fam:6s} ({tag:8s}): {len(genes_sorted):3d} genes")
    if genes_sorted:
        preview = ", ".join(genes_sorted[:5])
        if len(genes_sorted) > 5:
            preview += f", ... (+{len(genes_sorted)-5} more)"
        print(f"           {preview}")
    all_genes.extend(genes_sorted)
    for g in genes_sorted:
        family_rows.append({"gene": g, "group": fam})

# Deduplicate — keep first assignment if a gene appears in multiple families
seen = set()
unique_rows = []
for row in family_rows:
    if row["gene"] not in seen:
        seen.add(row["gene"])
        unique_rows.append(row)

gene_df = pd.DataFrame(unique_rows)
unique_genes = gene_df["gene"].tolist()

print()
print(f"  Baseline (4 families)  : 62 genes")
print(f"  Expanded (10 families) : {len(unique_genes)} genes")
print(f"  Net addition           : +{len(unique_genes) - 62} genes")

# ─────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

# Main gene list — replaces histone_genes3.txt for expanded pipeline
with open("data/histone_genes3.txt", "w") as f:
    for g in unique_genes:
        f.write(g + "\n")
print(f"\n    ✓ data/histone_genes3.txt  ({len(unique_genes)} genes)")

# Family assignment table — replaces gene_groups10.tsv
gene_df.to_csv("data/gene_groups10.tsv", sep="\t", index=False)
print(f"    ✓ data/gene_groups10.tsv   ({len(gene_df)} rows)")

# Summary table
summary_rows = []
for fam, genes in families.items():
    summary_rows.append({
        "family":   fam,
        "n_genes":  len(set(genes)),
        "genes":    ", ".join(sorted(set(genes))),
        "is_new":   "Yes" if fam in ["PRMT","SIRT","DNMT","TET","CBX","BRD"] else "No"
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("data/expanded_family_summary.tsv", sep="\t", index=False)
print(f"    ✓ data/expanded_family_summary.tsv")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("STEP 3a COMPLETE — SUMMARY")
print("=" * 62)
print(f"  Total unique genes : {len(unique_genes)}")
print(f"  Total families     : {len(families)}")
print()
print("  Family counts:")
for fam in families:
    n = len(set(families[fam]))
    print(f"    {fam:6s}: {n:3d} genes")
print()
print("  Files saved:")
print("    data/histone_genes3.txt    ← input for step3b")
print("    data/gene_groups10.tsv      ← input for step3b")
print("    data/expanded_family_summary.tsv   ← for reporting")
print()
print("  NEXT: Run step3b_rebuild_expanded_pipeline.py")
print("  This will re-run the full pipeline with the expanded gene set")