import pandas as pd

# Load processed TCGA data (gene symbols)
expr = pd.read_csv(
    "data/tcga/TCGA-LIHC_gene_symbols4.tsv",
    sep="\t",
    index_col=0
)

# Load histone gene list
with open("data/histone_genes.txt") as f:
    histone_genes = [line.strip() for line in f]

# Filter TCGA for histone genes
histone_expr = expr.loc[expr.index.intersection(histone_genes)]

print("Histone gene expression shape:", histone_expr.shape)

# Save filtered data
histone_expr.to_csv(
    "data/tcga/TCGA-LIHC_histone_expression5.tsv",
    sep="\t"
)

print("Histone expression dataset created!")