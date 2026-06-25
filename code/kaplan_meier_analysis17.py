import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import os

print("===== FINAL KAPLAN-MEIER SURVIVAL ANALYSIS =====")

# =====================================================
# STEP 1: LOAD LASSO RISK MODEL OUTPUT
# =====================================================

survival = pd.read_csv(
    "results/lasso_risk_scores16.tsv",
    sep="\t",
    index_col=0
)

print("\nLoaded survival risk model:")
print(survival.shape)

# =====================================================
# STEP 2: CHECK REQUIRED COLUMNS
# =====================================================

required_cols = [
    "OS.time",
    "OS",
    "risk_score",
    "risk_group"
]

for col in required_cols:

    if col not in survival.columns:
        raise ValueError(f"Missing column: {col}")

print("\nAll required columns found!")

# =====================================================
# STEP 3: SPLIT HIGH / LOW RISK PATIENTS
# =====================================================

high_risk = survival[
    survival["risk_group"] == "High"
]

low_risk = survival[
    survival["risk_group"] == "Low"
]

print("\nHigh-risk patients:", len(high_risk))
print("Low-risk patients:", len(low_risk))

# =====================================================
# STEP 4: EXTRACT SURVIVAL INFORMATION
# =====================================================

T_high = high_risk["OS.time"]
E_high = high_risk["OS"]

T_low = low_risk["OS.time"]
E_low = low_risk["OS"]

# =====================================================
# STEP 5: PERFORM LOG-RANK TEST
# =====================================================

results = logrank_test(
    T_high,
    T_low,
    event_observed_A=E_high,
    event_observed_B=E_low
)

p_value = results.p_value

print("\nLog-rank p-value:", p_value)

# =====================================================
# STEP 6: CREATE KM PLOT
# =====================================================

kmf = KaplanMeierFitter()

plt.figure(figsize=(8, 6))

# -----------------------------
# HIGH-RISK CURVE
# -----------------------------

kmf.fit(
    T_high,
    event_observed=E_high,
    label="High Risk"
)

kmf.plot()

# -----------------------------
# LOW-RISK CURVE
# -----------------------------

kmf.fit(
    T_low,
    event_observed=E_low,
    label="Low Risk"
)

kmf.plot()

# =====================================================
# STEP 7: PLOT SETTINGS
# =====================================================

plt.title(
    f"LASSO Risk Model Survival Analysis\n"
    f"Log-rank p = {p_value:.4e}"
)

plt.xlabel("Survival Time (Days)")
plt.ylabel("Survival Probability")

plt.grid(True)

# =====================================================
# STEP 8: SAVE FIGURE
# =====================================================

os.makedirs("results/plots", exist_ok=True)

plot_path = "results/plots/final_km_curve17.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nKM curve saved!")
print("Saved to:", plot_path)

# =====================================================
# STEP 9: BIOLOGICAL INTERPRETATION
# =====================================================

print("\n===== BIOLOGICAL INTERPRETATION =====")

if p_value < 0.05:

    print(
        "\nThe LASSO-derived histone risk model "
        "significantly stratifies HCC patients "
        "into high-risk and low-risk survival groups."
    )

else:

    print(
        "\nThe survival difference between "
        "risk groups was not statistically significant."
    )

print("\nKaplan-Meier analysis completed successfully!")

    