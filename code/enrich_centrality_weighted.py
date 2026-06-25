# ============================================================
# PHASE 3 (FINAL v2 — STRONGER WEIGHT) — CENTRALITY-WEIGHTED
# EXPRESSION (5x SCALE)
#
# PREVIOUS RESULT (weight = 1 + hub_score, ~1.0x-2.0x range):
# C-index = 0.5932 -- a real but small perturbation that,
# after StandardScaler re-normalization, produced only a
# minor deviation from baseline yet still underperformed.
#
# THIS VERSION uses a 5x stronger scale:
#     weight = 1 + hub_score * 5
# giving a weight range of approximately 1.0x (hub_score=0)
# to ~6.0x (TP53, hub_score≈0.994). This is a substantially
# larger perturbation of the relative gene contributions
# before StandardScaler is applied, while still preserving
# per-patient variance in every column (no constant columns).
#
# Output:
#   data/processed/patient_X_lasso19_centrality_weighted_strong.npy
#       shape: (patients x 56)
#   results/centrality_weights_applied_strong.tsv
# ============================================================

import numpy as np
import pandas as pd

print("===== PHASE 3 (FINAL v2 - STRONG WEIGHT): CENTRALITY-WEIGHTED EXPRESSION (5x) =====")

# ------------------------------------------------------------
# STEP 1: LOAD LASSO GENE ORDER (56 genes)
# ------------------------------------------------------------

lasso_genes = pd.read_csv(
    "results/lasso_selected_genes16.tsv",
    sep="\t",
    index_col=0
)

gene_order = lasso_genes.index.tolist()

print(f"\nLASSO gene set: {len(gene_order)} genes")

# ------------------------------------------------------------
# STEP 2: LOAD HUB SCORES (18 genes have measured centrality)
# ------------------------------------------------------------

hub = pd.read_csv(
    "data/hub_genes13.tsv",
    sep="\t",
    index_col=0
)

print(f"\nHub gene table: {len(hub)} genes with measured centrality")

# ------------------------------------------------------------
# STEP 3: BUILD PER-GENE WEIGHT VECTOR (5x SCALE)
# weight[g] = 1 + 5 * hub_score[g]
#   hub_score=0    -> weight=1.0  (unchanged)
#   hub_score=0.99 -> weight≈6.0  (TP53: 6x amplification)
# Genes not in hub table get hub_score = 0 -> weight = 1
# ------------------------------------------------------------

WEIGHT_SCALE = 5.0

weights = pd.Series(0.0, index=gene_order)  # hub_score, default 0

for gene in gene_order:
    if gene in hub.index:
        weights[gene] = hub.loc[gene, "hub_score"]

weight_vector = 1.0 + WEIGHT_SCALE * weights.values  # shape (56,)

print(f"\nWeight scale factor: {WEIGHT_SCALE}x")
print("Per-gene weights (1 + 5*hub_score):")
weight_df = pd.DataFrame({
    "gene": gene_order,
    "hub_score": weights.values,
    "weight": weight_vector
}).sort_values("weight", ascending=False)

print(weight_df.head(12).to_string(index=False))
print("...")
print(f"\n{(weights > 0).sum()} of {len(gene_order)} genes have "
      f"measured centrality (weight > 1); the remaining "
      f"{(weights == 0).sum()} genes keep weight = 1 (unchanged).")

weight_df.to_csv(
    "results/centrality_weights_applied_strong.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# STEP 4: LOAD EXPRESSION MATRIX AND APPLY WEIGHTING
# ------------------------------------------------------------

X = np.load("data/processed/patient_X_lasso19.npy")

print(f"\nOriginal expression matrix X shape: {X.shape}")

# Element-wise multiply each gene-column by its weight
X_weighted = X * weight_vector[np.newaxis, :]

print(f"Centrality-weighted matrix shape: {X_weighted.shape}")
print("(same shape as baseline -- no dimension change)")

# ------------------------------------------------------------
# STEP 5: SAVE
# ------------------------------------------------------------

np.save(
    "data/processed/patient_X_lasso19_centrality_weighted_strong.npy",
    X_weighted
)

print("\nSaved: data/processed/patient_X_lasso19_centrality_weighted_strong.npy")

# ------------------------------------------------------------
# BIOLOGICAL INTERPRETATION
# ------------------------------------------------------------

print("\n===== BIOLOGICAL INTERPRETATION =====")
print(
    "\nThis is a 5x-stronger version of the centrality-weighted "
    "encoding. Genes such as TP53 (hub_score≈0.994) now receive "
    "a weight of approximately 6.0x rather than the previous "
    "~2.0x, while genes with hub_score≈0 remain at weight=1.0 "
    "(unchanged). This produces a much larger pre-scaling "
    "perturbation of the relative gene expression magnitudes, "
    "intended to survive StandardScaler normalization and "
    "produce a measurable change in the patient similarity "
    "structure used by the Graph Transformer."
)
print(
    "\nAs with the 1x version, every column retains full "
    "patient-to-patient variance (no constant columns), so "
    "this encoding is mathematically valid input for "
    "StandardScaler -- the question is now whether a 6x "
    "amplification of hub genes is sufficient to alter the "
    "learned risk stratification."
)

print("\nPhase 3 (final) completed successfully!")