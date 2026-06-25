"""
Fixed: analyze_group_interactions11.py
=======================================
Fixes two issues from the original script:
  FIX 1 — interaction_type column was referenced in groupby
           before being assigned to the DataFrame
  FIX 2 — Works correctly with 10-family expanded gene set

Output: data/group_interactions11.tsv (overwrites original)

Run from project root:
  python scripts/fix_analyze_group_interactions11.py
"""

import pandas as pd
import os

print("===== ANALYZING GROUP INTERACTIONS (FIXED) =====")

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD EDGES
# ─────────────────────────────────────────────────────────────
edges = pd.read_csv("data/ppi/histone_graph_edges6.tsv", sep="\t")
print(f"\nLoaded interaction edges: {edges.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 2: LOAD GENE GROUPS
# ─────────────────────────────────────────────────────────────
groups = pd.read_csv("data/gene_groups10.tsv", sep="\t")
print(f"Loaded gene groups: {groups.shape}")
print(f"Families: {sorted(groups['group'].unique().tolist())}")

# ─────────────────────────────────────────────────────────────
# STEP 3: MAP GROUPS ONTO EDGES
# ─────────────────────────────────────────────────────────────
group_map = dict(zip(groups["gene"], groups["group"]))
edges["group1"] = edges["gene1"].map(group_map)
edges["group2"] = edges["gene2"].map(group_map)

# Remove edges where either gene has no group assignment
edges = edges.dropna(subset=["group1", "group2"])
print(f"\nEdges after removing unassigned genes: {len(edges)}")

# ─────────────────────────────────────────────────────────────
# STEP 4: NORMALIZE DIRECTION (A↔B = B↔A)
# ─────────────────────────────────────────────────────────────
sorted_pairs = edges.apply(
    lambda row: sorted([row["group1"], row["group2"]]), axis=1
).tolist()
edges[["groupA", "groupB"]] = pd.DataFrame(
    sorted_pairs, index=edges.index
)

# ─────────────────────────────────────────────────────────────
# STEP 5: ASSIGN INTERACTION TYPE
# FIX: This must happen BEFORE the groupby that uses it
# ─────────────────────────────────────────────────────────────
edges["interaction_type"] = edges.apply(
    lambda row: "INTRA_GROUP" if row["groupA"] == row["groupB"]
                else "INTER_GROUP",
    axis=1
)

# ─────────────────────────────────────────────────────────────
# STEP 6: COMPUTE INTERACTION STATISTICS
# ─────────────────────────────────────────────────────────────
interaction_stats = (
    edges.groupby(["groupA", "groupB", "interaction_type"])
    .agg(
        interaction_count=("combined_score", "count"),
        mean_confidence=("combined_score", "mean"),
        total_confidence=("combined_score", "sum"),
        max_confidence=("combined_score", "max")
    )
    .reset_index()
    .sort_values("total_confidence", ascending=False)
)

# ─────────────────────────────────────────────────────────────
# STEP 7: DISPLAY AND SAVE
# ─────────────────────────────────────────────────────────────
print("\n===== GROUP INTERACTION SUMMARY =====\n")
print(interaction_stats.to_string(index=False))

interaction_stats.to_csv(
    "data/group_interactions11.tsv", sep="\t", index=False
)
print("\nSaved → data/group_interactions11.tsv")

# ─────────────────────────────────────────────────────────────
# STEP 8: BIOLOGICAL INTERPRETATION
# ─────────────────────────────────────────────────────────────
print("\n===== BIOLOGICAL INTERPRETATION =====")

top = interaction_stats.iloc[0]
print(f"\nStrongest interaction: {top['groupA']} ↔ {top['groupB']}")
print(f"Interaction type    : {top['interaction_type']}")
print(f"Total strength      : {top['total_confidence']:.0f}")
print(f"Mean confidence     : {top['mean_confidence']:.1f}")
print(f"Edge count          : {int(top['interaction_count'])}")

# Inter-group summary
inter = interaction_stats[
    interaction_stats["interaction_type"] == "INTER_GROUP"
].head(5)
print("\nTop 5 INTER-GROUP interactions:")
print(inter[["groupA","groupB","interaction_count",
             "mean_confidence","total_confidence"]].to_string(index=False))

# Intra-group summary
intra = interaction_stats[
    interaction_stats["interaction_type"] == "INTRA_GROUP"
].head(5)
print("\nTop 5 INTRA-GROUP interactions:")
print(intra[["groupA","groupB","interaction_count",
             "mean_confidence","total_confidence"]].to_string(index=False))

print("\n===== ANALYSIS COMPLETE =====")