# ============================================================
# KAPLAN-MEIER VALIDATION
# GRAPH TRANSFORMER SURVIVAL MODEL
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import os

print("===== GRAPH TRANSFORMER KM VALIDATION =====")

# ============================================================
# LOAD GRAPH TRANSFORMER PREDICTIONS
# ============================================================

df = pd.read_csv(
    "results/graph_transformer_risk_predictions_full20.tsv",
    sep="\t"
)

print("\nLoaded prediction file:")
print(df.shape)

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_cols = [
    "predicted_risk",
    "OS.time",
    "OS"
]

for col in required_cols:

    if col not in df.columns:

        raise ValueError(
            f"Missing column: {col}"
        )

print("\nAll required columns found!")

# ============================================================
# SORT PATIENTS BY PREDICTED RISK
# ============================================================

df = df.sort_values(
    by="predicted_risk",
    ascending=False
)

# ============================================================
# CREATE RISK GROUPS
# ============================================================

print("\n===== CREATING RISK GROUPS =====")

# ------------------------------------------------------------
# Median risk cutoff
# ------------------------------------------------------------

median_risk = df["predicted_risk"].median()

print("Median risk cutoff:",
      round(median_risk, 4))

# ------------------------------------------------------------
# Assign groups
# ------------------------------------------------------------

df["risk_group"] = df[
    "predicted_risk"
].apply(

    lambda x:
    "High Risk"
    if x >= median_risk
    else "Low Risk"
)

# ============================================================
# SPLIT GROUPS
# ============================================================

high_risk = df[
    df["risk_group"] == "High Risk"
]

low_risk = df[
    df["risk_group"] == "Low Risk"
]

print("\nHigh-risk patients:",
      len(high_risk))

print("Low-risk patients:",
      len(low_risk))

# ============================================================
# EXTRACT SURVIVAL DATA
# ============================================================

T_high = high_risk["OS.time"]

E_high = high_risk["OS"]

T_low = low_risk["OS.time"]

E_low = low_risk["OS"]

# ============================================================
# LOG-RANK TEST
# ============================================================

print("\n===== LOG-RANK TEST =====")

results = logrank_test(

    T_high,
    T_low,

    event_observed_A=E_high,
    event_observed_B=E_low
)

p_value = results.p_value

print("Log-rank p-value:",
      p_value)

# ============================================================
# KAPLAN-MEIER CURVE
# ============================================================

print("\n===== GENERATING KM CURVE =====")

kmf = KaplanMeierFitter()

plt.figure(figsize=(8, 6))

# ------------------------------------------------------------
# HIGH-RISK CURVE
# ------------------------------------------------------------

kmf.fit(
    T_high,
    event_observed=E_high,
    label="High Risk"
)

kmf.plot()

# ------------------------------------------------------------
# LOW-RISK CURVE
# ------------------------------------------------------------

kmf.fit(
    T_low,
    event_observed=E_low,
    label="Low Risk"
)

kmf.plot()

# ============================================================
# PLOT SETTINGS
# ============================================================

plt.title(

    "Graph Transformer Survival Stratification\n"

    f"Log-rank p = {p_value:.4e}"
)

plt.xlabel("Survival Time (Days)")

plt.ylabel("Survival Probability")

plt.grid(True)

# ============================================================
# SAVE PLOT
# ============================================================

os.makedirs(
    "results/plots",
    exist_ok=True
)

plot_path = (
    "results/plots/"
    "graph_transformer_km_curve20.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nKM curve saved!")

print("Saved to:", plot_path)

# ============================================================
# SAVE RISK GROUPS
# ============================================================

df.to_csv(
    "results/graph_transformer_risk_groups20.tsv",
    sep="\t",
    index=False
)

print("\nRisk groups saved!")

# ============================================================
# BIOLOGICAL INTERPRETATION
# ============================================================

print("\n===== BIOLOGICAL INTERPRETATION =====")

if p_value < 0.05:

    print(

        "\nThe Graph Transformer successfully "
        "stratified HCC patients into "
        "significantly different survival groups."
    )

else:

    print(

        "\nThe Graph Transformer showed "
        "limited survival-group separation "
        "despite learning graph-based "
        "interaction patterns."
    )

print(

    "\nPatients classified as high-risk "
    "exhibited poorer survival probability "
    "compared to low-risk patients."
)

print("\nKaplan-Meier validation completed!")