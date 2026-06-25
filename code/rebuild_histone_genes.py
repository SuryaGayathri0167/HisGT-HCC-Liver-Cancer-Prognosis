import pandas as pd

print("===== REBUILDING HISTONE GENE LIST =====")

df = pd.read_csv("data/epifactors_proteins.csv")

# Keep only genes with histone-related modifications
df = df[df["Modification"].str.contains("Histone", case=False, na=False)]

# Extract gene names
genes = df["HGNC_symbol"].dropna().unique()

print("Total histone-related genes:", len(genes))

# Save
with open("data/histone_genes.txt", "w") as f:
    for g in genes:
        f.write(g + "\n")

print("New histone_genes.txt created!")

with open("data/histone_genes3.txt") as f:
    genes = [line.strip() for line in f]

print("Total genes:", len(genes))