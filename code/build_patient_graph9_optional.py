import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph

print("===== BUILDING PATIENT GRAPH (KNN) =====")

# ==============================
# Load aligned histone expression
# ==============================

expr = pd.read_csv(
    "data/tcga/TCGA-LIHC_graph_features.tsv",
    sep="\t",
    index_col=0
)

# Transpose → patients as nodes
X = expr.T.values  # shape: (patients, genes)

print("Feature matrix (patients × genes):", X.shape)

# ==============================
# Build KNN graph
# ==============================

print("\nBuilding KNN graph...")

k = 10  # You can tune this (5–15 recommended)

A = kneighbors_graph(
    X,
    n_neighbors=k,
    mode='connectivity',
    include_self=False
).toarray()

# ==============================
# Make graph symmetric (IMPORTANT)
# ==============================

A = np.maximum(A, A.T)

print("Adjacency matrix shape:", A.shape)
print("Number of edges:", int(A.sum() / 2))  # divide by 2 for undirected graph

# ==============================
# Save matrices
# ==============================

np.save("data/processed/patient_X9.npy", X)
np.save("data/processed/patient_A9.npy", A)

print("\nPatient graph saved successfully!")