import pandas as pd

# Load histone expression data
expr = pd.read_csv(
    "data/tcga/TCGA-LIHC_histone_expression5.tsv",
    sep="\t",
    index_col=0
)

# Load graph edges
edges = pd.read_csv(
    "data/ppi/histone_graph_edges6.tsv",
    sep="\t"
)

# Extract graph nodes
graph_nodes = set(edges["gene1"]).union(set(edges["gene2"]))

print("Graph nodes:", len(graph_nodes))
print("Expression genes:", expr.shape[0])

# Filter expression to graph nodes
expr = expr.loc[expr.index.intersection(graph_nodes)] #important

print("Aligned expression shape:", expr.shape)

# Save aligned features
expr.to_csv(
    "data/tcga/TCGA-LIHC_graph_features7.tsv",
    sep="\t"
)

print("Feature alignment completed!")
print("Graph nodes:", len(graph_nodes))
print("Feature rows:", expr.shape[0])
print(set(expr.index) == graph_nodes)
extra = set(expr.index) - graph_nodes
missing = graph_nodes - set(expr.index)

print("Extra genes:", extra)
print("Missing genes:", missing)