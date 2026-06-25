# ============================================================
# STEP 24 — PATHWAY ENRICHMENT ANALYSIS (PHASE 8)
#
# Runs Enrichr-based gene set enrichment on the 56
# LASSO-selected histone-modifier genes against three
# curated pathway databases:
#   - KEGG_2021_Human       (metabolic + cancer pathways)
#   - MSigDB_Hallmark_2020  (hallmark cancer gene sets)
#   - Reactome_2022         (epigenetic regulation sets)
#
# Requires internet access (Enrichr API) and:
#   pip install gseapy --break-system-packages
#
# Input:
#   results/lasso_selected_genes16.tsv  (56 LASSO genes)
#
# Output:
#   results/pathway_enrichment24.tsv    (all enriched terms)
#   results/pathway_enrichment24_significant.tsv (adj.p<0.05)
# ============================================================

import pandas as pd
import os

print("=" * 62)
print("STEP 24: PATHWAY ENRICHMENT ANALYSIS")
print("=" * 62)

try:
    import gseapy as gp
except ImportError:
    raise SystemExit(
        "\ngseapy is not installed.\n"
        "Run: pip install gseapy --break-system-packages\n"
        "Then re-run this script."
    )

os.makedirs("results", exist_ok=True)

# ============================================================
# STEP 1: LOAD 56 LASSO GENE LIST
# ============================================================

print("\n[1] Loading LASSO-selected gene list...")

lasso_genes = pd.read_csv(
    "results/lasso_selected_genes16.tsv",
    sep="\t",
    index_col=0
)

gene_list = lasso_genes.index.tolist()

print(f"    Gene set size: {len(gene_list)} genes")
print(f"    Genes: {gene_list[:8]}... (showing first 8)")

# ============================================================
# STEP 2: RUN ENRICHMENT AGAINST THREE LIBRARIES
# ============================================================

print("\n[2] Running Enrichr enrichment (requires internet)...")

gene_sets = [
    "KEGG_2021_Human",
    "MSigDB_Hallmark_2020",
    "Reactome_2022"
]

all_results = []

for lib in gene_sets:

    print(f"\n    Querying {lib}...")

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=lib,
            organism="human",
            outdir=None,
            no_plot=True
        )

        res = enr.results.copy()
        res["Library"] = lib
        all_results.append(res)

        n_sig = (res["Adjusted P-value"] < 0.05).sum()
        print(f"    {len(res)} terms returned | "
              f"{n_sig} significant (adj.p < 0.05)")

    except Exception as e:
        print(f"    Failed for {lib}: {e}")
        print("    (Enrichr API requires internet access)")

if not all_results:
    raise SystemExit(
        "\nNo enrichment results returned.\n"
        "Check your internet connection and try again.\n"
        "The Enrichr API (https://maayanlab.cloud/Enrichr) "
        "must be reachable."
    )

# ============================================================
# STEP 3: COMBINE AND SORT RESULTS
# ============================================================

print("\n[3] Combining results...")

combined = pd.concat(all_results, ignore_index=True)
combined = combined.sort_values("Adjusted P-value")

combined = combined[[
    "Library", "Term", "Overlap",
    "P-value", "Adjusted P-value", "Genes"
]]

print(f"    Total terms: {len(combined)}")
print(f"    Significant (adj.p<0.05): "
      f"{(combined['Adjusted P-value'] < 0.05).sum()}")

# ============================================================
# STEP 4: SAVE RESULTS
# ============================================================

print("\n[4] Saving results...")

combined.to_csv(
    "results/pathway_enrichment24.tsv",
    sep="\t",
    index=False
)

sig = combined[combined["Adjusted P-value"] < 0.05]
sig.to_csv(
    "results/pathway_enrichment24_significant.tsv",
    sep="\t",
    index=False
)

print("    Saved: results/pathway_enrichment24.tsv")
print("    Saved: results/pathway_enrichment24_significant.tsv")

# ============================================================
# STEP 5: TOP RESULTS
# ============================================================

print("\n[5] Top 20 enriched pathways (by adjusted p-value):")
print(combined.head(20)[
    ["Library", "Term", "Overlap", "Adjusted P-value"]
].to_string(index=False))

# ============================================================
# STEP 6: KEY BIOLOGICAL THEME CHECK
# ============================================================

print("\n" + "=" * 62)
print("KEY BIOLOGICAL THEME CHECK")
print("=" * 62)

keywords = {
    "Cell Cycle":        "Validates CDK1 and cell-cycle gene involvement",
    "DNA Damage":        "Connects to BRCA1, ATM in signature",
    "Chromatin":         "Directly validates histone-modifier biology",
    "Histone":           "Core biological theme of this work",
    "Liver":             "HCC-specific pathway enrichment",
    "Hepatocellular":    "Direct disease relevance",
    "Epigenetic":        "Core biological theme",
    "Transcription":     "Downstream effect of histone modification",
    "Methylation":       "HMT/DNMT/TET family relevance",
    "Acetylation":       "HAT/HDAC family relevance",
    "Apoptosis":         "Tumor suppressor pathway",
    "p53":               "TP53 is top hub gene in your network",
}

found_themes = []

for kw, biological_relevance in keywords.items():

    matches = combined[
        combined["Term"].str.contains(kw, case=False, na=False)
    ]
    sig_matches = matches[matches["Adjusted P-value"] < 0.05]

    if len(sig_matches) > 0:
        print(f"\n  ✓ '{kw}' — {len(sig_matches)} significant term(s)")
        print(f"    Relevance: {biological_relevance}")
        for _, row in sig_matches.head(2).iterrows():
            print(
                f"    -> {row['Term'][:60]}"
                f"  (adj.p={row['Adjusted P-value']:.2e}, "
                f"Library={row['Library']})"
            )
        found_themes.append(kw)
    else:
        print(f"\n  - '{kw}' — no significant terms")

# ============================================================
# STEP 7: THESIS SUMMARY
# ============================================================

print("\n" + "=" * 62)
print("STEP 24 COMPLETE — SUMMARY FOR THESIS")
print("=" * 62)

total_sig = (combined["Adjusted P-value"] < 0.05).sum()

print(f"""
Gene set size          : {len(gene_list)} LASSO-selected genes
Libraries queried      : {len(gene_sets)}
Total terms returned   : {len(combined)}
Significant (adj.p<0.05): {total_sig}

Key biological themes confirmed: {found_themes}

Top 5 enriched terms:
""")

for _, row in combined.head(5).iterrows():
    print(f"  {row['Term'][:55]:55s}  "
          f"adj.p={row['Adjusted P-value']:.2e}  "
          f"({row['Library']})")

print(f"""
Files saved:
  results/pathway_enrichment24.tsv
  results/pathway_enrichment24_significant.tsv

Next step: Step 25 is COMPLETE (this was the final
computational step). Proceed to thesis writing and
paper preparation (Phase 9).
""")