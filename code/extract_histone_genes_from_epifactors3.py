import pandas as pd

df = pd.read_csv("data/epifactors_proteins.csv")

# Convert to lowercase
df["Target"] = df["Target"].str.lower()
df["Function"] = df["Function"].str.lower()
df["Modification"] = df["Modification"].str.lower()

# Keep only PURE writer or erase (no extra roles)
histone_df = df[
    df["Target"].str.contains("histone", na=False) &
    df["Modification"].str.contains("acetyl|methyl", na=False) &
    df["HGNC_symbol"].str.contains("KAT|HDAC|KDM|KMT|SET|EZH|PRMT")
]

# Extract genes
genes = histone_df["HGNC_symbol"].dropna().unique()

# Save file
with open("data/histone_genes3.txt", "w") as f:
    for gene in genes:
        f.write(gene + "\n")

print("Total histone modifying genes:", len(genes))