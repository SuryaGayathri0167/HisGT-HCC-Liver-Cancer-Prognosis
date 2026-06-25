import pandas as pd

print("========== VERIFYING STRING INTERACTION DATASET ==========\n")

# Load STRING interaction network
links = pd.read_csv(
    "data/ppi/9606.protein.links.v12.0.txt",
    sep=" "
)

print("Interaction dataset shape:", links.shape)

print("\nColumns in interaction dataset:")
print(links.columns)

print("\nFirst 5 interaction rows:")
print(links.head())


print("\n========== VERIFYING STRING PROTEIN INFO DATASET ==========\n")

# Load STRING protein info file
info = pd.read_csv(
    "data/ppi/9606.protein.info.v12.0.txt",
    sep="\t"
)

print("Protein info dataset shape:", info.shape)

print("\nColumns in protein info dataset:")
print(info.columns)

print("\nFirst 5 protein entries:")
print(info.head())


print("\n========== QUICK MAPPING CHECK ==========\n")

# Example: map protein IDs to gene names
sample_links = links.head()

merged = sample_links.merge(
    info,
    left_on="protein1",
    right_on="#string_protein_id",
    how="left"
)

print("Example protein → gene mapping:")
print(merged[["protein1", "preferred_name"]].head())