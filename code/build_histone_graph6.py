import pandas as pd

print("===== LOADING DATA =====")

# Load STRING interaction dataset
links = pd.read_csv(
    "data/ppi/9606.protein.links.v12.0.txt",
    sep=" "
)

# Load STRING protein info (for mapping)
info = pd.read_csv(
    "data/ppi/9606.protein.info.v12.0.txt",
    sep="\t"
)

# Load histone gene list
with open("data/histone_genes.txt") as f:
    histone_genes = set(line.strip() for line in f)

print("Total histone genes:", len(histone_genes))


# ==============================
# STEP 1: MAP PROTEIN → GENE
# ==============================

print("\n===== MAPPING PROTEINS TO GENES =====")

# Create mapping: protein ID → gene name
protein_to_gene = dict(zip(
    info["#string_protein_id"],
    info["preferred_name"]
))

# Map protein IDs in STRING links
links["gene1"] = links["protein1"].map(protein_to_gene)
links["gene2"] = links["protein2"].map(protein_to_gene)

# Remove rows with missing mapping
links = links.dropna(subset=["gene1", "gene2"])

print("After mapping:", links.shape)


# ==============================
# STEP 2: FILTER HISTONE GENES
# ==============================

print("\n===== FILTERING HISTONE INTERACTIONS =====")

histone_links = links[
    (links["gene1"].isin(histone_genes)) &
    (links["gene2"].isin(histone_genes))
]

print("Histone interactions:", histone_links.shape)


# ==============================
# STEP 3: FILTER HIGH CONFIDENCE
# ==============================

print("\n===== APPLYING CONFIDENCE FILTER =====")

histone_links = histone_links[
    histone_links["combined_score"] >= 700
]

print("High-confidence interactions:", histone_links.shape)


# ==============================
# STEP 4: SAVE EDGE LIST
# ==============================

print("\n===== SAVING GRAPH =====")

edges = histone_links[["gene1", "gene2", "combined_score"]]

edges.to_csv(
    "data/ppi/histone_graph_edges6.tsv",
    sep="\t",
    index=False
)

print("Graph saved successfully!")

print("\nFinal graph stats:")
print("Nodes (approx):", len(histone_genes))
print("Edges:", len(edges))
nodes = set(edges["gene1"]).union(set(edges["gene2"]))
print("Actual nodes in graph:", len(nodes))