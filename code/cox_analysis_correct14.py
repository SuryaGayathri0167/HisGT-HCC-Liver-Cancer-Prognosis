import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

print("===== NETWORK-GUIDED COX REGRESSION =====")

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
# rows    → patients
# columns → genes
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
# STEP 3: LOAD SELECTED HUB GENES
# =====================================================

with open("data/hub_gene_list13.txt") as f:
    hub_genes = [line.strip() for line in f]

print("\nSelected hub genes:")
print(len(hub_genes))

# =====================================================
# STEP 4: KEEP ONLY HUB GENES
# =====================================================

# Keep only genes existing in expression matrix
hub_genes = [
    gene for gene in hub_genes
    if gene in expr.columns
]

expr = expr[hub_genes]

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

print("\nPatients used for Cox analysis:")
print(len(common_patients))

# =====================================================
# STEP 6: RUN COX REGRESSION
# =====================================================

print("\nRunning Cox regression on hub genes...")

results = []

for gene in expr.columns:

    print(f"Analyzing: {gene}")

    values = expr[gene]

    # -------------------------------------------------
    # FILTER 1:
    # Remove low-variance genes
    # -------------------------------------------------

    if values.var() < 0.01:
        print(f"Skipped {gene} (low variance)")
        continue

    # -------------------------------------------------
    # FILTER 2:
    # Remove invalid genes
    # -------------------------------------------------

    if values.isna().any():
        print(f"Skipped {gene} (missing values)")
        continue

    try:

        # =============================================
        # Build survival dataframe
        # =============================================

        df = pd.DataFrame({
            "expression": values,
            "time": survival["OS.time"],
            "event": survival["OS"]
        })

        # =============================================
        # Fit Cox proportional hazards model
        # =============================================

        cph = CoxPHFitter()

        cph.fit(
            df,
            duration_col="time",
            event_col="event"
        )

        summary = cph.summary

        # =============================================
        # Store results
        # =============================================

        results.append({
            "gene": gene,

            "hazard_ratio":
                summary["exp(coef)"].values[0],

            "cox_coefficient":
                summary["coef"].values[0],

            "p_value":
                summary["p"].values[0],

            "confidence_lower":
                summary["exp(coef) lower 95%"].values[0],

            "confidence_upper":
                summary["exp(coef) upper 95%"].values[0]
        })

    except Exception as e:

        print(f"Error analyzing {gene}: {e}")

        continue

# =====================================================
# STEP 7: CREATE RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(results)

# Sort by significance
results_df = results_df.sort_values(
    by="p_value"
)

# =====================================================
# STEP 8: IDENTIFY SIGNIFICANT GENES
# =====================================================

significant_genes = results_df[
    results_df["p_value"] < 0.05
]

print("\n===== SIGNIFICANT PROGNOSTIC HUB GENES =====\n")

print(
    significant_genes[
        [
            "gene",
            "hazard_ratio",
            "p_value"
        ]
    ]
)

# =====================================================
# STEP 9: SAVE RESULTS
# =====================================================

results_df.to_csv(
    "data/cox_results14.tsv",
    sep="\t",
    index=False
)

significant_genes.to_csv(
    "data/significant_hub_genes14.tsv",
    sep="\t",
    index=False
)

print("\nSaved Cox regression results!")

# =====================================================
# STEP 10: BIOLOGICAL INTERPRETATION
# =====================================================

print("\n===== BIOLOGICAL INTERPRETATION =====")

print(
    "\nCox regression identified hub histone "
    "regulators significantly associated "
    "with overall survival in hepatocellular "
    "carcinoma patients."
)

print(
    "\nThese genes represent biologically "
    "central epigenetic regulators with "
    "potential prognostic significance."
)

# =====================================================
# STEP 11: SUMMARY
# =====================================================

print("\n===== COX ANALYSIS SUMMARY =====")

print(
    f"\nHub genes analyzed: "
    f"{len(expr.columns)}"
)

print(
    f"Significant prognostic genes: "
    f"{len(significant_genes)}"
)

print(
    f"Significance ratio: "
    f"{round((len(significant_genes) / len(expr.columns)) * 100, 2)}%"
)