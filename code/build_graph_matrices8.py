import pandas as pd
import numpy as np

print("===== BUILDING GRAPH MATRICES =====")

# Load aligned node features
expr = pd.read_csv(
    "data/tcga/TCGA-LIHC_graph_features7.tsv",
    sep="\t",
    index_col=0
)

# Load graph edges
edges = pd.read_csv(
    "data/ppi/histone_graph_edges6.tsv",
    sep="\t"
)

# Get node list
nodes = list(expr.index)
node_to_idx = {node: i for i, node in enumerate(nodes)}

print("Number of nodes:", len(nodes))

# ==============================
# Build adjacency matrix A
# ==============================

A = np.zeros((len(nodes), len(nodes)))

for _, row in edges.iterrows():
    g1 = row["gene1"]
    g2 = row["gene2"]
    score = row["combined_score"]

    if g1 in node_to_idx and g2 in node_to_idx:
        i = node_to_idx[g1] #row index
        j = node_to_idx[g2] #column index

        A[i, j] = score
        A[j, i] = score  # undirected graph

print("Adjacency matrix shape:", A.shape)

# ==============================
# Build feature matrix X
# ==============================

X = expr.values

print("Feature matrix shape:", X.shape)

# ==============================
# Save matrices
# ==============================

np.save("data/processed/A.npy", A)
np.save("data/processed/X.npy", X)

print("Matrices saved successfully!")