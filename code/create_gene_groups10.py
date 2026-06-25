import pandas as pd

print("===== ASSIGNING GENE GROUPS (CURATED) =====")

# Load histone genes
with open("data/histone_genes3.txt") as f:
    genes = [line.strip() for line in f]

# ==============================
# Curated gene family rules
# ==============================

def assign_group(gene):

    # HDAC family
    if gene.startswith("HDAC") or gene.startswith("SIRT"):
        return "HDAC"

    # KDM (demethylases)
    if gene.startswith("KDM"):
        return "KDM"

    # HMT (methyltransferases)
    if gene.startswith("KMT") or gene.startswith("SET") or gene.startswith("EZH") or gene.startswith("PRMT"):
        return "HMT"

    # HAT (acetyltransferases)
    if gene.startswith("KAT") or gene in ["EP300", "CREBBP"]:
        return "HAT"

    return "OTHER"

# Apply grouping
df = pd.DataFrame({"gene": genes})
df["group"] = df["gene"].apply(assign_group)

# 🔥 KEEP ONLY CORE HISTONE MODIFIERS
df = df[df["group"] != "OTHER"]

print("\nGroup distribution:")
print(df["group"].value_counts())

df.to_csv("data/gene_groups10.tsv", sep="\t", index=False)

print("\nGene groups saved!")