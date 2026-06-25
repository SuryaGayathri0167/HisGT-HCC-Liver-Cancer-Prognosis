import pandas as pd

# Load datasets
expr = pd.read_csv("data/tcga/TCGA-LIHC.star_tpm.tsv", sep="\t", index_col=0)
survival = pd.read_csv("data/tcga/TCGA-LIHC.survival.tsv", sep="\t")

# Extract sample IDs
expr_samples = set(expr.columns)
survival_samples = set(survival["sample"])

# Print counts
print("Expression samples:", len(expr_samples))
print("Survival samples:", len(survival_samples))

# Check intersections
common_samples = expr_samples & survival_samples

print("\nCommon samples across all datasets:", len(common_samples))

# Check missing samples
missing_in_survival = expr_samples - survival_samples

print("Samples missing in survival dataset:", len(missing_in_survival))