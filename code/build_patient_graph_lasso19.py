import pandas as pd
import numpy as np

from sklearn.neighbors import kneighbors_graph

print("===== BUILDING LASSO-REFINED PATIENT GRAPH =====")

# =====================================================
# STEP 1: LOAD LASSO-SELECTED GENES
# =====================================================

lasso_genes = pd.read_csv(
    "results/lasso_selected_genes16.tsv",
    sep="\t",
    index_col=0
)

selected_genes = lasso_genes.index.tolist()

print("\nLASSO-selected prognostic genes:",
      len(selected_genes))

print(selected_genes)

# =====================================================
# STEP 2: LOAD HISTONE EXPRESSION DATA
# =====================================================

expr = pd.read_csv(
    "data/tcga/TCGA-LIHC_graph_features7.tsv",
    sep="\t",
    index_col=0
)

print("\nOriginal expression matrix shape:",
      expr.shape)

# =====================================================
# STEP 3: FILTER TO LASSO GENES
# =====================================================

expr = expr.loc[
    expr.index.intersection(selected_genes)
]

print("\nFiltered expression matrix shape:",
      expr.shape)

# =====================================================
# STEP 4: TRANSPOSE MATRIX
# PATIENTS BECOME NODES
# =====================================================

X = expr.T.values

print("\nFeature matrix (patients × LASSO genes):",
      X.shape)

# =====================================================
# STEP 5: BUILD KNN PATIENT GRAPH
# =====================================================

print("\nBuilding KNN graph...")

k = 10

A = kneighbors_graph(
    X,
    n_neighbors=k,
    mode="connectivity",
    include_self=False
).toarray()

# =====================================================
# STEP 6: MAKE GRAPH SYMMETRIC
# =====================================================

A = np.maximum(A, A.T)

# =====================================================
# STEP 7: DISPLAY GRAPH INFORMATION
# =====================================================

print("\nAdjacency matrix shape:", A.shape)

print(
    "Number of edges:",
    int(A.sum() / 2)
)

# =====================================================
# STEP 8: SAVE MATRICES
# =====================================================

np.save(
    "data/processed/patient_X_lasso19.npy",
    X
)

np.save(
    "data/processed/patient_A_lasso19.npy",
    A
)

print("\nLASSO-refined patient graph saved!")

# =====================================================
# STEP 9: BIOLOGICAL INTERPRETATION
# =====================================================

print("\n===== BIOLOGICAL INTERPRETATION =====")

print(
    "\nThis patient graph was constructed "
    "using only LASSO-selected prognostic "
    "histone regulators."
)

print(
    "\nPatients connected in this graph "
    "share similar prognostic histone "
    "expression patterns."
)

print(
    "\nThis refined graph will be used "
    "for interaction-aware survival "
    "prediction using the Graph Transformer."
)