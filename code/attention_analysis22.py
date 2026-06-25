# ============================================================
# STEP 22 — ATTENTION WEIGHT EXTRACTION AND
# BIOLOGICAL INTERPRETATION (PHASE 5)
#
# Uses attention weights from the IMPROVED Graph Transformer
# (C-index = 0.7425, hidden dims 16/8, lr=0.01, 500 epochs)
# saved during train_graph_transformer_improved.py.
#
# Steps:
#  13. Load attention matrices (alpha1, alpha2) from improved model
#  14. Separate mean attention for High vs Low Risk patients
#  15. Correlate attention weights with per-gene expression
#      similarity to find the genes driving attention
#  16. Visualize as a 20x20 heatmap (top 10 High + top 10 Low)
#
# Inputs:
#   results/attention_layer1_improved.npy    (418 x 418)
#   results/attention_layer2_improved.npy    (418 x 418)
#   results/graph_transformer_risk_predictions_improved.tsv
#   data/processed/patient_X_lasso19.npy    (418 x 56)
#   results/lasso_selected_genes16.tsv       (56 genes)
#
# Outputs:
#   results/attention_block_summary22.tsv
#   results/attention_gene_correlation22.tsv
#   results/plots/attention_heatmap22.png
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

print("=" * 62)
print("STEP 22: ATTENTION WEIGHT ANALYSIS (IMPROVED MODEL)")
print("=" * 62)

os.makedirs("results/plots", exist_ok=True)

# ============================================================
# STEP 13: LOAD ATTENTION MATRICES AND RISK PREDICTIONS
# ============================================================

print("\n[13] Loading attention matrices from improved model...")

alpha1 = np.load("results/attention_layer1_improved.npy")
alpha2 = np.load("results/attention_layer2_improved.npy")

print(f"    Attention layer 1 shape: {alpha1.shape}")
print(f"    Attention layer 2 shape: {alpha2.shape}")

results = pd.read_csv(
    "results/graph_transformer_risk_predictions_improved.tsv",
    sep="\t"
)

print(f"    Risk predictions loaded: {results.shape}")

# Use layer-2 attention (closer to final risk output) as the
# primary attention matrix for biological interpretation.
alpha = alpha2

mean_attention_all = alpha.mean(axis=0)
print(f"    Mean attention (all patients): "
      f"{mean_attention_all.mean():.6f}")

# ============================================================
# STEP 14: SEPARATE HIGH VS LOW RISK ATTENTION PATTERNS
# ============================================================

print("\n[14] Separating High vs Low Risk attention patterns...")

median_risk = results["predicted_risk"].median()

print(f"    Median risk score: {median_risk:.4f}")

high_idx = results.index[
    results["predicted_risk"] > median_risk
].to_numpy()

low_idx = results.index[
    results["predicted_risk"] <= median_risk
].to_numpy()

print(f"    High Risk patients: {len(high_idx)}")
print(f"    Low Risk patients:  {len(low_idx)}")

# Compute four block-mean attention values
attn_high_high = alpha[np.ix_(high_idx, high_idx)]
attn_low_low   = alpha[np.ix_(low_idx,  low_idx)]
attn_high_low  = alpha[np.ix_(high_idx, low_idx)]
attn_low_high  = alpha[np.ix_(low_idx,  high_idx)]

hh = round(attn_high_high.mean(), 6)
ll = round(attn_low_low.mean(),   6)
hl = round(attn_high_low.mean(),  6)
lh = round(attn_low_high.mean(),  6)

print(f"\n    Mean attention High->High : {hh}")
print(f"    Mean attention Low->Low   : {ll}")
print(f"    Mean attention High->Low  : {hl}")
print(f"    Mean attention Low->High  : {lh}")

# Biological interpretation of block pattern
if hh > hl:
    block_interpretation = (
        "High Risk patients show stronger mutual attention "
        "(High->High > High->Low), indicating the model "
        "clusters aggressive-disease patients together when "
        "forming risk embeddings."
    )
elif ll > lh:
    block_interpretation = (
        "Low Risk patients show stronger intra-group attention "
        "(Low->Low > Low->High), indicating the model treats "
        "low-risk patients as a more coherent group."
    )
else:
    block_interpretation = (
        "Attention is distributed broadly across risk groups, "
        "suggesting the model relies on cross-group patient "
        "similarity for risk embedding rather than tight "
        "intra-group clustering."
    )

print(f"\n    Interpretation: {block_interpretation}")

# Save block summary
attention_summary = pd.DataFrame({
    "Block": [
        "High->High",
        "Low->Low",
        "High->Low",
        "Low->High"
    ],
    "Mean_Attention": [hh, ll, hl, lh]
})

attention_summary.to_csv(
    "results/attention_block_summary22.tsv",
    sep="\t",
    index=False
)

print("\n    Saved: results/attention_block_summary22.tsv")

# ============================================================
# STEP 15: MAP ATTENTION TO GENE EXPRESSION SIMILARITY
# ============================================================

print("\n[15] Mapping attention to gene expression similarity...")

X = np.load("data/processed/patient_X_lasso19.npy")

lasso_genes = pd.read_csv(
    "results/lasso_selected_genes16.tsv",
    sep="\t",
    index_col=0
)

gene_names = lasso_genes.index.tolist()

print(f"    Expression matrix shape (raw): {X.shape}")

# Align to the same 418 patients used during training
# (patient_X_lasso19.npy has 424 rows; the model used
# only the first len(df) = 418 after survival data alignment)
n_aligned = alpha.shape[0]   # 418
X = X[:n_aligned]

print(f"    Expression matrix shape (aligned): {X.shape}")
print(f"    LASSO genes: {len(gene_names)}")

n_patients = X.shape[0]

# Flatten attention excluding self-attention diagonal
mask = ~np.eye(n_patients, dtype=bool)
attn_flat = alpha[mask].flatten()

gene_attention_corr = {}

print("    Computing pairwise gene-expression similarity "
      "vs attention correlation...")

for gi, gene in enumerate(gene_names):

    expr_col = X[:, gi]

    # Pairwise similarity: negative absolute difference
    # (higher value = more similar expression between patients)
    diff = np.abs(expr_col[:, None] - expr_col[None, :])
    sim = -diff

    sim_flat = sim[mask].flatten()

    r, p = pearsonr(attn_flat, sim_flat)

    gene_attention_corr[gene] = {
        "pearson_r": round(r, 6),
        "p_value":   p
    }

corr_df = pd.DataFrame(gene_attention_corr).T
corr_df = corr_df.sort_values("pearson_r", ascending=False)

print("\n    Gene-attention correlation (top 15):")
print(corr_df.head(15).to_string())

print("\n    Gene-attention correlation (bottom 5):")
print(corr_df.tail(5).to_string())

corr_df.to_csv(
    "results/attention_gene_correlation22.tsv",
    sep="\t"
)

print("\n    Saved: results/attention_gene_correlation22.tsv")

top5 = corr_df.head(5).index.tolist()
bot5 = corr_df.tail(5).index.tolist()

print(f"\n    Top 5 genes POSITIVELY driving attention similarity:")
print(f"    {top5}")
print(f"\n    Top 5 genes with NEGATIVE attention-expression coupling:")
print(f"    {bot5}")

# ============================================================
# STEP 16: VISUALIZE 20x20 ATTENTION HEATMAP
# ============================================================

print("\n[16] Generating 20x20 attention heatmap...")

# Select top 10 High Risk + top 10 Low Risk patients by
# absolute deviation from median risk (most confidently
# classified patients in each group)
results["abs_dev"] = (
    results["predicted_risk"] - median_risk
).abs()

top_high = (
    results.loc[high_idx]
    .sort_values("abs_dev", ascending=False)
    .head(10)
    .index
    .to_numpy()
)

top_low = (
    results.loc[low_idx]
    .sort_values("abs_dev", ascending=False)
    .head(10)
    .index
    .to_numpy()
)

top20_idx = np.concatenate([top_high, top_low])

heatmap_data = alpha[np.ix_(top20_idx, top20_idx)]

fig, ax = plt.subplots(figsize=(9, 8))

im = ax.imshow(
    heatmap_data,
    cmap="viridis",
    aspect="auto",
    vmin=heatmap_data.min(),
    vmax=heatmap_data.max()
)

ax.set_title(
    "Patient-Patient Graph Attention Weights\n"
    "Top 10 High Risk (left) + Top 10 Low Risk (right) patients\n"
    f"Improved Graph Transformer (C-index = 0.7425)",
    fontsize=10,
    pad=12
)

ax.set_xlabel("Patient index (attended-to)", fontsize=9)
ax.set_ylabel("Patient index (attending)", fontsize=9)

# White dividing line between High and Low Risk blocks
ax.axhline(9.5, color="white", linewidth=2.0)
ax.axvline(9.5, color="white", linewidth=2.0)

# Block labels above the heatmap
ax.text(
    4.5, -1.2, "High Risk",
    ha="center", fontsize=9,
    color="crimson", fontweight="bold"
)
ax.text(
    14.5, -1.2, "Low Risk",
    ha="center", fontsize=9,
    color="steelblue", fontweight="bold"
)

plt.colorbar(im, ax=ax, label="Attention weight (α)")
plt.tight_layout()

heatmap_path = "results/plots/attention_heatmap22.png"

plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"    Heatmap saved: {heatmap_path}")

# ============================================================
# SUMMARY FOR THESIS
# ============================================================

print("\n" + "=" * 62)
print("STEP 22 COMPLETE — SUMMARY FOR THESIS")
print("=" * 62)

print(f"""
Attention Block Analysis (Layer 2, Improved GT):
  High->High : {hh}
  Low->Low   : {ll}
  High->Low  : {hl}
  Low->High  : {lh}

  Interpretation: {block_interpretation}

Top 5 genes whose expression similarity
most strongly predicts patient attention weight:
  {top5}

These are the genes driving the model's patient
clustering, providing a model-identified biological
interpretation of the Graph Transformer's learned
patient interaction structure.

Files saved:
  results/attention_block_summary22.tsv
  results/attention_gene_correlation22.tsv
  results/plots/attention_heatmap22.png
""")