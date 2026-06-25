import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis

print("===== LASSO COX SURVIVAL MODEL =====")

# =====================================================
# STEP 1: LOAD HUB GENES
# =====================================================

print("\n===== LOADING HUB GENES =====")

hub_df = pd.read_csv(
    "data/hub_genes13.tsv",
    sep="\t"
)

hub_genes = hub_df["gene"].tolist()

print("Hub genes loaded:", len(hub_genes))

# =====================================================
# STEP 2: LOAD EXPRESSION DATA
# =====================================================

print("\n===== LOADING EXPRESSION DATA =====")

expr = pd.read_csv(
    "data/tcga/TCGA-LIHC_graph_features7.tsv",
    sep="\t",
    index_col=0
)

print("Original expression shape:", expr.shape)

# =====================================================
# STEP 3: KEEP ONLY HUB GENES
# =====================================================

expr = expr.loc[
    expr.index.intersection(hub_genes)
]

print("Filtered expression shape:", expr.shape)

# =====================================================
# STEP 4: TRANSPOSE MATRIX
# patients × genes
# =====================================================

expr = expr.T

print("Transposed shape:", expr.shape)

# =====================================================
# STEP 5: LOAD SURVIVAL DATA
# =====================================================

print("\n===== LOADING SURVIVAL DATA =====")

survival = pd.read_csv(
    "data/tcga/TCGA-LIHC.survival.tsv",
    sep="\t"
)[
    ["sample", "OS.time", "OS"]
].dropna()

print("Survival samples:", survival.shape[0])

# =====================================================
# STEP 6: ALIGN PATIENTS
# =====================================================

print("\n===== ALIGNING PATIENTS =====")

common = expr.index.intersection(
    survival["sample"]
)

expr = expr.loc[common]

survival = survival.set_index(
    "sample"
).loc[common]

print("Patients after alignment:", len(common))

# =====================================================
# STEP 7: CREATE SURVIVAL OBJECT
# =====================================================

print("\n===== CREATING SURVIVAL OBJECT =====")

y = Surv.from_arrays(
    event=survival["OS"].astype(bool),
    time=survival["OS.time"]
)

print("Survival object created!")

# =====================================================
# STEP 8: STANDARDIZE FEATURES
# =====================================================

print("\n===== STANDARDIZING FEATURES =====")

scaler = StandardScaler()

X = scaler.fit_transform(expr)

print("Feature matrix shape:", X.shape)

# =====================================================
# STEP 9: TRAIN LASSO COX MODEL
# =====================================================

print("\n===== TRAINING LASSO COX MODEL =====")

model = CoxnetSurvivalAnalysis(

    l1_ratio=1.0,

    alpha_min_ratio=0.01,

    max_iter=100000
)

model.fit(X, y)

print("LASSO Cox training completed!")

# =====================================================
# STEP 10: EXTRACT COEFFICIENTS
# =====================================================

print("\n===== EXTRACTING COEFFICIENTS =====")

coef_df = pd.DataFrame(
    model.coef_,
    index=expr.columns,
    columns=model.alphas_
)

print("Coefficient matrix shape:", coef_df.shape)

# =====================================================
# STEP 11: SELECT FINAL COEFFICIENTS
# =====================================================

final_coef = coef_df.iloc[:, -1]

# =====================================================
# STEP 12: KEEP NON-ZERO GENES
# =====================================================

selected = final_coef[
    final_coef != 0
]

print("\n===== SELECTED PROGNOSTIC GENES =====\n")

print(selected)

# =====================================================
# STEP 13: SAVE SELECTED GENES
# =====================================================

selected.to_csv(
    "results/lasso_selected_genes16.tsv",
    sep="\t",
    header=["coef"]
)

print("\nSelected genes saved!")

# =====================================================
# STEP 14: CONSTRUCT RISK SCORES
# =====================================================

print("\n===== CONSTRUCTING RISK SCORES =====")

risk_score = np.dot(

    expr[selected.index],

    selected.values
)

survival["risk_score"] = risk_score

print("Risk scores computed!")

# =====================================================
# =====================================================
# STEP 15: CREATE OPTIMAL RISK GROUPS
# =====================================================

from lifelines.statistics import logrank_test

print("\n===== FINDING OPTIMAL RISK CUTOFF =====")

scores = survival["risk_score"]

best_p = 1.0
best_cutoff = None

# -----------------------------------------------------
# Search possible cutoffs
# Avoid extreme splits using 20%–80% range
# -----------------------------------------------------

candidate_cutoffs = np.percentile(
    scores,
    np.arange(20, 81)
)

for cutoff in candidate_cutoffs:

    high = survival[scores > cutoff]
    low = survival[scores <= cutoff]

    # Skip invalid splits
    if len(high) < 10 or len(low) < 10:
        continue

    # Log-rank test
    result = logrank_test(

        high["OS.time"],
        low["OS.time"],

        event_observed_A=high["OS"],
        event_observed_B=low["OS"]
    )

    p = result.p_value

    # Keep best cutoff
    if p < best_p:

        best_p = p
        best_cutoff = cutoff

# -----------------------------------------------------
# Apply optimal cutoff
# -----------------------------------------------------

print("\nOptimal cutoff found:", round(best_cutoff, 4))

print("Best log-rank p-value:", best_p)

survival["risk_group"] = np.where(

    survival["risk_score"] > best_cutoff,

    "High",

    "Low"
)

print("\nOptimal risk groups created!")

# -----------------------------------------------------
# Display summary
# -----------------------------------------------------

print(
    survival[
        [
            "OS.time",
            "OS",
            "risk_score",
            "risk_group"
        ]
    ].head()
)

print("\nGroup counts:")

print(
    survival["risk_group"].value_counts()
)

# =====================================================
# STEP 16: SAVE RISK MODEL
# =====================================================

survival.to_csv(
    "results/lasso_risk_scores16.tsv",
    sep="\t"
)

print("\nRisk model saved successfully!")

# =====================================================
# STEP 17: FINAL SUMMARY
# =====================================================

print("\n===== FINAL SUMMARY =====")

print(
    "Final prognostic genes:",
    len(selected)
)

print("\nSelected genes:")

print(
    selected.index.tolist()
)

print(
    "\nHigh-risk patients:",
    (survival["risk_group"] == "High").sum()
)

print(
    "Low-risk patients:",
    (survival["risk_group"] == "Low").sum()
)

print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====")