import pandas as pd
import numpy as np

print("===== BUILDING GENE ADJACENCY =====")

# ✅ USE YOUR EXISTING FILE
edges = pd.read_csv(
    "data/ppi/histone_graph_edges.tsv",
    sep="\t"
)

# Extract gene list from edges
genes = sorted(set(edges["gene1"]).union(set(edges["gene2"])))

print("Total genes:", len(genes))

# Mapping
gene_to_idx = {g: i for i, g in enumerate(genes)}

# Initialize adjacency
A = np.zeros((len(genes), len(genes)))

# Fill adjacency
for _, row in edges.iterrows():

    g1 = row["gene1"]
    g2 = row["gene2"]

    i = gene_to_idx[g1]
    j = gene_to_idx[g2]

    A[i, j] = 1
    A[j, i] = 1

# Save
np.save("data/processed/gene_A.npy", A)

print("Adjacency matrix shape:", A.shape)
print("Saved successfully!")