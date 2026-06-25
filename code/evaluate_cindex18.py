import pandas as pd
from lifelines.utils import concordance_index

print("===== C-INDEX EVALUATION =====")

# Load risk model
df = pd.read_csv(
    "results/lasso_risk_scores16.tsv",
    sep="\t",
    index_col=0
)

# Compute C-index
c_index = concordance_index(
    df["OS.time"],
    -df["risk_score"],   # negative because higher risk = worse survival
    df["OS"]
)

print("C-index:", round(c_index, 4))