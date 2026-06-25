import pandas as pd
import numpy as np

print("===== SELECTING HUB HISTONE REGULATORS =====")

# =====================================================
# STEP 1: LOAD RANKED HUB GENES
# =====================================================

key_genes = pd.read_csv(
    "data/key_genes12.tsv",
    sep="\t"
)

print("\nLoaded ranked hub genes:")
print(key_genes.shape)

# =====================================================
# STEP 2: SORT BY HUB SCORE
# =====================================================

key_genes = key_genes.sort_values(
    by="hub_score",
    ascending=False
)

# =====================================================
# STEP 3: COMPUTE AUTOMATIC HUB THRESHOLD
# =====================================================

print("\n===== COMPUTING HUB SCORE THRESHOLD =====")

# Mean hub score
mean_score = key_genes["hub_score"].mean()

# Standard deviation
std_score = key_genes["hub_score"].std()

# -----------------------------------------------------
# Automatic threshold
# -----------------------------------------------------
# Recommended balanced threshold:
# genes significantly above average
# -----------------------------------------------------

threshold = mean_score + (0.5 * std_score)

print(f"\nMean hub score: {round(mean_score, 4)}")
print(f"Standard deviation: {round(std_score, 4)}")
print(f"Hub selection threshold: {round(threshold, 4)}")

# =====================================================
# STEP 4: SELECT HUB GENES AUTOMATICALLY
# =====================================================

hub_genes = key_genes[
    key_genes["hub_score"] > threshold
]

print(
    f"\nAutomatically selected "
    f"{len(hub_genes)} hub regulators."
)

# =====================================================
# STEP 5: DISPLAY SELECTED HUB GENES
# =====================================================

print("\n===== SELECTED HUB GENES =====\n")

print(
    hub_genes[
        [
            "gene",
            "group",
            "hub_score"
        ]
    ]
)

# =====================================================
# STEP 6: SAVE FULL HUB GENE TABLE
# =====================================================

hub_genes.to_csv(
    "data/hub_genes13.tsv",
    sep="\t",
    index=False
)

# =====================================================
# STEP 7: SAVE ONLY GENE NAMES
# IMPORTANT:
# This file will be directly used in Cox regression
# =====================================================

hub_genes["gene"].to_csv(
    "data/hub_gene_list13.txt",
    index=False,
    header=False
)

print("\nSaved hub gene files successfully!")

# =====================================================
# STEP 8: GROUP DISTRIBUTION
# =====================================================

print("\n===== HUB GENE FAMILY DISTRIBUTION =====\n")

print(
    hub_genes["group"].value_counts()
)

# =====================================================
# STEP 9: BIOLOGICAL INTERPRETATION
# =====================================================

top_gene = hub_genes.iloc[0]

print("\n===== BIOLOGICAL INTERPRETATION =====")

print(
    f"\nHighest-ranked hub regulator: "
    f"{top_gene['gene']}"
)

print(
    f"Regulator family: "
    f"{top_gene['group']}"
)

print(
    f"Hub score: "
    f"{round(top_gene['hub_score'], 4)}"
)

print(
    "\nInterpretation:"
)

print(
    "Hub genes were selected automatically "
    "using a data-driven hub-score threshold "
    "based on the network importance distribution. "
    "Only genes with significantly above-average "
    "network influence were retained for downstream "
    "survival analysis."
)

# =====================================================
# STEP 10: SUMMARY STATISTICS
# =====================================================

print("\n===== HUB SELECTION SUMMARY =====")

print(
    f"\nTotal ranked genes: {len(key_genes)}"
)

print(
    f"Selected hub genes: {len(hub_genes)}"
)

print(
    f"Selection ratio: "
    f"{round((len(hub_genes) / len(key_genes)) * 100, 2)}%"
)