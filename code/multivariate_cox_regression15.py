import pandas as pd
from lifelines import CoxPHFitter

print("===== MULTIVARIATE COX REGRESSION =====")

# =====================================================
# STEP 1: LOAD TCGA EXPRESSION DATA
# =====================================================

expr = pd.read_csv(
    "data/tcga/TCGA-LIHC_graph_features7.tsv",
    sep="\t",
    index_col=0
).T

# -----------------------------------------------------
# After transpose:
# rows    -> patients
# columns -> genes
# -----------------------------------------------------

print("\nExpression matrix shape:")
print(expr.shape)

# =====================================================
# STEP 2: LOAD SURVIVAL DATA
# =====================================================

survival = pd.read_csv(
    "data/tcga/TCGA-LIHC.survival.tsv",
    sep="\t"
)[["sample", "OS.time", "OS"]]

# Remove missing survival values
survival = survival.dropna()

print("\nSurvival data shape:")
print(survival.shape)

# =====================================================
# STEP 3: LOAD SIGNIFICANT PROGNOSTIC HUB GENES
# =====================================================

sig_genes_df = pd.read_csv(
    "data/significant_hub_genes14.tsv",
    sep="\t"
)

# Extract only gene names
sig_genes = sig_genes_df["gene"].tolist()

print("\nSignificant prognostic hub genes:")
print(len(sig_genes))

print("\nGenes used in multivariate Cox model:")
print(sig_genes)

# =====================================================
# STEP 4: KEEP ONLY SIGNIFICANT GENES
# =====================================================

# Keep only genes existing in expression matrix
sig_genes = [
    gene for gene in sig_genes
    if gene in expr.columns
]

expr = expr[sig_genes]

print("\nFiltered expression matrix:")
print(expr.shape)

# =====================================================
# STEP 5: ALIGN PATIENTS
# =====================================================

common_patients = expr.index.intersection(
    survival["sample"]
)

expr = expr.loc[common_patients]

survival = (
    survival
    .set_index("sample")
    .loc[common_patients]
)

print("\nPatients used:")
print(len(common_patients))

# =====================================================
# STEP 6: BUILD MULTIVARIATE DATAFRAME
# =====================================================

# Create dataframe containing:
# multiple prognostic genes + survival data

df = expr.copy()

df["time"] = survival["OS.time"]

df["event"] = survival["OS"]

print("\nMultivariate Cox dataframe shape:")
print(df.shape)

# =====================================================
# STEP 7: REMOVE LOW-VARIANCE GENES
# =====================================================

print("\nChecking low-variance genes...")

remove_cols = []

for col in expr.columns:

    variance = df[col].var()

    if variance < 0.01:

        print(f"Removing {col} (low variance)")

        remove_cols.append(col)

# Remove low-variance genes
df = df.drop(columns=remove_cols)

print("\nRemaining genes:")
print(df.shape[1] - 2)  # minus time + event

# =====================================================
# STEP 8: FIT MULTIVARIATE COX MODEL
# =====================================================

print("\nRunning multivariate Cox regression...")

cph = CoxPHFitter()

try:

    cph.fit(
        df,
        duration_col="time",
        event_col="event"
    )

    print("\nModel fitted successfully!")

except Exception as e:

    print("\nError fitting model:")
    print(e)

    raise

# =====================================================
# STEP 9: EXTRACT RESULTS
# =====================================================

summary = cph.summary

# Sort by significance
summary = summary.sort_values(
    by="p"
)

print("\n===== MULTIVARIATE COX RESULTS =====\n")

print(
    summary[
        [
            "coef",
            "exp(coef)",
            "p"
        ]
    ]
)

# =====================================================
# STEP 10: IDENTIFY FINAL SIGNIFICANT GENES
# =====================================================

final_sig = summary[
    summary["p"] < 0.05
]

print("\n===== FINAL INDEPENDENT PROGNOSTIC GENES =====\n")

print(
    final_sig[
        [
            "coef",
            "exp(coef)",
            "p"
        ]
    ]
)

# =====================================================
# STEP 11: SAVE RESULTS
# =====================================================

summary.to_csv(
    "data/multivariate_cox_results15.tsv",
    sep="\t"
)

final_sig.to_csv(
    "data/final_prognostic_genes15.tsv",
    sep="\t"
)

print("\nSaved multivariate Cox results!")

# =====================================================
# STEP 12: SAVE FINAL GENE LIST
# =====================================================

final_sig.index.to_series().to_csv(
    "data/final_gene_list15.txt",
    index=False,
    header=False
)

print("\nSaved final prognostic gene list!")

# =====================================================
# STEP 13: BIOLOGICAL INTERPRETATION
# =====================================================

print("\n===== BIOLOGICAL INTERPRETATION =====")

print(
    "\nThe multivariate Cox model identified "
    "histone hub regulators that independently "
    "contribute to survival outcome in "
    "hepatocellular carcinoma patients."
)

print(
    "\nThese genes represent prognostic "
    "epigenetic regulators after adjusting "
    "for the combined effects of other "
    "hub genes in the network."
)

# =====================================================
# STEP 14: SUMMARY
# =====================================================

print("\n===== MODEL SUMMARY =====")

print(
    f"\nInitial prognostic hub genes: "
    f"{len(sig_genes)}"
)

print(
    f"Final independent prognostic genes: "
    f"{len(final_sig)}"
)

print(
    f"Reduction after multivariate modeling: "
    f"{len(sig_genes) - len(final_sig)} genes"
)

print(
    "\nMultivariate Cox regression completed successfully!"
)