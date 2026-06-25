import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

print("===== IDENTIFYING HUB HISTONE REGULATORS =====")

# =====================================================
# STEP 1: LOAD STRING INTERACTION NETWORK
# =====================================================

edges = pd.read_csv(
    "data/ppi/histone_graph_edges6.tsv",
    sep="\t"
)

print("\nLoaded interaction edges:")
print(edges.shape)

# =====================================================
# STEP 2: LOAD GENE GROUP INFORMATION
# =====================================================

groups = pd.read_csv(
    "data/gene_groups10.tsv",
    sep="\t"
)

print("\nLoaded gene groups:")
print(groups.shape)

# Create mapping:
# gene → histone regulator family
group_map = dict(zip(groups["gene"], groups["group"]))

# =====================================================
# STEP 3: BUILD WEIGHTED STRING GRAPH
# =====================================================

print("\nBuilding weighted interaction graph...")

G = nx.Graph()

for _, row in edges.iterrows():

    gene1 = row["gene1"]
    gene2 = row["gene2"]
    score = row["combined_score"]

    # Add weighted interaction edge
    G.add_edge(
        gene1,
        gene2,
        weight=score
    )

print("\nGraph statistics:")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# =====================================================
# STEP 4: COMPUTE CENTRALITY METRICS
# =====================================================

print("\nComputing centrality metrics...")

# -----------------------------------------------------
# 4A. Degree Centrality
# Measures:
# how connected a gene is
# -----------------------------------------------------

degree_centrality = nx.degree_centrality(G)

# -----------------------------------------------------
# 4B. Weighted Degree (Node Strength)
# Measures:
# total interaction strength of a gene
# -----------------------------------------------------

weighted_degree = dict(G.degree(weight="weight"))

# -----------------------------------------------------
# 4C. Betweenness Centrality
# Measures:
# how strongly a gene bridges pathways
# -----------------------------------------------------

betweenness_centrality = nx.betweenness_centrality(
    G,
    weight="weight"
)

# -----------------------------------------------------
# 4D. Eigenvector Centrality
# Measures:
# connection to other important genes
# -----------------------------------------------------

eigenvector_centrality = nx.eigenvector_centrality(
    G,
    weight="weight",
    max_iter=500
)

# =====================================================
# STEP 5: BUILD CENTRALITY DATAFRAME
# =====================================================

print("\nConstructing hub-gene table...")

df = pd.DataFrame({
    "gene": list(G.nodes())
})

# Add regulator family
df["group"] = df["gene"].map(group_map)

# Add centrality metrics
df["degree_centrality"] = df["gene"].map(degree_centrality)

df["weighted_degree"] = df["gene"].map(weighted_degree)

df["betweenness_centrality"] = df["gene"].map(
    betweenness_centrality
)

df["eigenvector_centrality"] = df["gene"].map(
    eigenvector_centrality
)

# =====================================================
# STEP 6: NORMALIZE METRICS
# IMPORTANT:
# Metrics exist on different scales
# =====================================================

print("\nNormalizing centrality metrics...")

metrics = [
    "degree_centrality",
    "weighted_degree",
    "betweenness_centrality",
    "eigenvector_centrality"
]

scaler = MinMaxScaler()

df[metrics] = scaler.fit_transform(df[metrics])

# =====================================================
# STEP 7: COMPUTE COMPOSITE HUB SCORE
# =====================================================

print("\nComputing composite hub score...")

df["hub_score"] = (
    df["degree_centrality"] +
    df["weighted_degree"] +
    df["betweenness_centrality"] +
    df["eigenvector_centrality"]
) / 4

# =====================================================
# STEP 8: SORT HUB GENES
# =====================================================

df = df.sort_values(
    by="hub_score",
    ascending=False
)

# =====================================================
# STEP 9: DISPLAY TOP HUB GENES
# =====================================================

print("\n===== TOP HUB HISTONE REGULATORS =====\n")

print(
    df[
        [
            "gene",
            "group",
            "degree_centrality",
            "weighted_degree",
            "betweenness_centrality",
            "eigenvector_centrality",
            "hub_score"
        ]
    ].head(20)
)

# =====================================================
# STEP 10: SAVE RESULTS
# =====================================================

df.to_csv(
    "data/key_genes12.tsv",
    sep="\t",
    index=False
)

print("\nHub-gene analysis saved successfully!")

# =====================================================
# STEP 11: GROUP-LEVEL HUB SUMMARY
# =====================================================

print("\n===== HUB DISTRIBUTION BY GENE FAMILY =====")

group_summary = (
    df.groupby("group")["hub_score"]
    .mean()
    .sort_values(ascending=False)
)

print(group_summary)

# =====================================================
# STEP 12: BIOLOGICAL INTERPRETATION
# =====================================================

top_gene = df.iloc[0]

print("\n===== BIOLOGICAL INTERPRETATION =====")

print(
    f"\nTop hub regulator: {top_gene['gene']}"
)

print(
    f"Regulator family: {top_gene['group']}"
)

print(
    f"Composite hub score: "
    f"{round(top_gene['hub_score'], 4)}"
)

print(
    "\nInterpretation:"
)

print(
    f"{top_gene['gene']} occupies a highly central "
    "position within the histone interaction network, "
    "suggesting that it may function as a major "
    "epigenetic regulator in hepatocellular carcinoma."
)