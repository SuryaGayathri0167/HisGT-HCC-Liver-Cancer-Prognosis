import pandas as pd

# Load TCGA data
expr = pd.read_csv(
    "data/tcga/TCGA-LIHC.star_tpm.tsv",
    sep="\t",
    index_col=0
)

# 🚨 DO NOT REMOVE VERSION NUMBERS
# expr.index = expr.index.str.split(".").str[0]  ❌ REMOVE THIS

# Load mapping
mapping = pd.read_csv(
    "data/tcga/gencode.v36.annotation.gtf.gene.probemap",
    sep="\t"
)

print("Mapping columns:", mapping.columns)

# 🚨 DO NOT MODIFY mapping IDs
# mapping["id"] = mapping["id"].str.split(".").str[0] ❌ REMOVE THIS

# Create mapping dictionary
gene_map = dict(zip(mapping["id"], mapping["gene"]))

# Map genes
expr["gene_symbol"] = expr.index.map(gene_map)

# Check how many mapped
print("Mapped genes:", expr["gene_symbol"].notna().sum())

# Remove unmapped genes
expr = expr.dropna(subset=["gene_symbol"])

# Set gene symbols as index
expr = expr.set_index("gene_symbol")

# Remove duplicates
expr = expr.groupby(expr.index).mean()

print("Final expression shape:", expr.shape)

# Save
expr.to_csv("data/tcga/TCGA-LIHC_gene_symbols.tsv", sep="\t")

print("Conversion completed successfully!")