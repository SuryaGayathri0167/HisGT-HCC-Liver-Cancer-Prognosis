# ============================================================
# GRAPH SAGE INDUCTIVE SURVIVAL MODEL
# Replaces the transductive GAT (train_graph_transformer_improved.py)
# with an inductive GraphSAGE architecture.
#
# WHY GraphSAGE:
#   The original GAT was transductive — it learned fixed
#   embeddings for 418 training patients. On held-out patients
#   it produced C-index ~0.41 (below chance). GraphSAGE learns
#   an AGGREGATION FUNCTION instead of patient-specific
#   embeddings. This function maps any patient's feature
#   vector + neighbor features -> risk score, and transfers
#   to new patients without retraining.
#
# HOW GraphSAGE DIFFERS FROM GAT:
#   GAT:       h_i = softmax(e_ij) * W * h_j   (attention over neighbors)
#   GraphSAGE: h_i = MLP([h_i || MEAN(h_j for j in N(i))])
#              (concatenate own features with mean-pooled neighbor features,
#               then pass through MLP — fully inductive)
#
# EVALUATION:
#   5-fold cross-validation — each fold trains on 80% of
#   patients and evaluates on the held-out 20%. The mean
#   C-index across 5 folds is the reported generalisation
#   performance.
#
# Output:
#   results/graphsage_cv_results.tsv   (per-fold C-index)
#   results/graphsage_risk_predictions.tsv  (full-cohort predictions)
#   results/plots/graphsage_km_curve.png
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# HYPERPARAMETERS
# ============================================================

HIDDEN_1     = 16
HIDDEN_2     = 8
DROPOUT      = 0.3
LEARNING_RATE= 0.01
WEIGHT_DECAY = 1e-4
N_EPOCHS     = 500
N_FOLDS      = 5

print(f"\nGraphSAGE hyperparameters:")
print(f"  Architecture  : 56 -> {HIDDEN_1} -> {HIDDEN_2} -> 1 (SAGE)")
print(f"  Learning rate : {LEARNING_RATE}")
print(f"  Weight decay  : {WEIGHT_DECAY}")
print(f"  Epochs        : {N_EPOCHS}")
print(f"  Dropout       : {DROPOUT}")
print(f"  CV folds      : {N_FOLDS}")

# ============================================================
# LOAD DATA
# ============================================================

print("\n===== LOADING DATA =====")

X_raw = np.load("data/processed/patient_X_lasso19.npy")
A_raw = np.load("data/processed/patient_A_lasso19.npy")

df = pd.read_csv(
    "results/lasso_risk_scores16.tsv",
    sep="\t",
    index_col=0
)

n = len(df)

X_raw = X_raw[:n]
A_raw = A_raw[:n, :n]

time  = df["OS.time"].values
event = df["OS"].values

print(f"Expression matrix : {X_raw.shape}")
print(f"Adjacency matrix  : {A_raw.shape}")
print(f"Patients: {n} | Events: {int(event.sum())}")

# Normalise adjacency (symmetric, with self-loops) once
A_norm = A_raw + np.eye(n)
D      = np.diag(np.sum(A_norm, axis=1))
D_inv  = np.linalg.inv(np.sqrt(D))
A_norm = D_inv @ A_norm @ D_inv
A_t    = torch.tensor(A_norm, dtype=torch.float32).to(device)

# ============================================================
# GRAPHSAGE LAYER
#
# For each node i:
#   1. Compute mean of neighbor features:
#      agg_i = MEAN(h_j for j in N(i))  via A_norm @ H
#   2. Concatenate own features with aggregated neighbors:
#      concat_i = [h_i || agg_i]
#   3. Apply linear + normalise:
#      h_i' = ReLU(W * concat_i)
#
# This is fully inductive: W is a shared weight matrix that
# applies to any patient's (feature, neighbor_mean) pair,
# including patients not seen during training.
# ============================================================

class SAGELayer(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        # Input is concatenation of [self || neighbor_mean]
        # so input dim is 2 * in_features
        self.W = nn.Linear(2 * in_features, out_features)

    def forward(self, H, A):
        # H: (n_patients x in_features)
        # A: (n_patients x n_patients) normalised adjacency

        # Mean-pool neighbor features via normalised adjacency
        agg = torch.matmul(A, H)       # (n x in_features)

        # Concatenate self features with aggregated neighbors
        concat = torch.cat([H, agg], dim=1)  # (n x 2*in_features)

        return self.W(concat)          # (n x out_features)


class GraphSAGE(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.sage1   = SAGELayer(input_dim, HIDDEN_1)
        self.sage2   = SAGELayer(HIDDEN_1,  HIDDEN_2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.out     = nn.Linear(HIDDEN_2, 1)

    def forward(self, X, A):
        X = self.dropout(self.relu(self.sage1(X, A)))
        X = self.dropout(self.relu(self.sage2(X, A)))
        return self.out(X).squeeze()

# ============================================================
# COX LOSS
# ============================================================

def cox_loss(risk, time, event):
    order      = torch.argsort(time, descending=True)
    risk       = risk[order]
    event      = event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss       = -torch.sum((risk - log_cumsum) * event)
    return loss / torch.sum(event)

# ============================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n===== 5-FOLD CROSS-VALIDATION =====")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results = []
all_risk_scores = np.zeros(n)

for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):

    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
    print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")

    # ── Reset seed per fold for reproducibility ───────────────
    torch.manual_seed(42 + fold)
    np.random.seed(42 + fold)

    # ── Normalise features (fit on train, apply to all) ───────
    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_raw[train_idx])
    X_all_n = scaler.transform(X_raw)   # full cohort, normalised

    # ── Tensors ───────────────────────────────────────────────
    X_all_t    = torch.tensor(X_all_n, dtype=torch.float32).to(device)
    time_tr_t  = torch.tensor(time[train_idx],  dtype=torch.float32).to(device)
    event_tr_t = torch.tensor(event[train_idx], dtype=torch.float32).to(device)

    # ── Initialise model ──────────────────────────────────────
    model = GraphSAGE(input_dim=X_all_t.shape[1]).to(device)
    opt   = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ── Training (Cox loss on train patients only) ────────────
    for epoch in range(N_EPOCHS):
        model.train()
        risk_all  = model(X_all_t, A_t)
        loss      = cox_loss(
            risk_all[train_idx], time_tr_t, event_tr_t
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 100 == 0 or epoch == N_EPOCHS - 1:
            print(f"  Epoch {epoch:03d} | Train Loss: {loss.item():.4f}")

    # ── Evaluation on HELD-OUT test patients ──────────────────
    model.eval()
    with torch.no_grad():
        risk_all_np = model(X_all_t, A_t).cpu().numpy()

    # Store test-fold predictions
    all_risk_scores[test_idx] = risk_all_np[test_idx]

    risk_test = risk_all_np[test_idx]
    c_idx     = concordance_index(
        time[test_idx], -risk_test, event[test_idx]
    )

    # Log-rank on test fold (median split)
    cut  = np.median(risk_test)
    high = risk_test > cut
    if high.sum() >= 5 and (~high).sum() >= 5:
        res   = logrank_test(
            time[test_idx][high],  time[test_idx][~high],
            event[test_idx][high], event[test_idx][~high]
        )
        lr_p  = res.p_value
    else:
        lr_p  = np.nan

    print(f"  Fold {fold+1} C-index: {c_idx:.4f} | "
          f"Log-rank p: {lr_p:.3e}")

    fold_results.append({
        "fold":       fold + 1,
        "n_train":    len(train_idx),
        "n_test":     len(test_idx),
        "C_index":    round(c_idx, 4),
        "logrank_p":  lr_p
    })

# ============================================================
# CROSS-VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 62)
print("CROSS-VALIDATION SUMMARY")
print("=" * 62)

cv_df = pd.DataFrame(fold_results)

mean_c  = cv_df["C_index"].mean()
std_c   = cv_df["C_index"].std()

print("\nPer-fold results:")
print(cv_df[["fold","n_train","n_test","C_index","logrank_p"]].to_string(index=False))
print(f"\nMean C-index : {mean_c:.4f} ± {std_c:.4f}")
print(f"Min C-index  : {cv_df['C_index'].min():.4f}")
print(f"Max C-index  : {cv_df['C_index'].max():.4f}")
print(f"\nComparison:")
print(f"  GraphSAGE (5-fold CV mean)            : {mean_c:.4f} ± {std_c:.4f}")
print(f"  GAT transductive (full cohort)        : 0.7425")
print(f"  GAT held-out test (single 30% split)  : 0.4122")
print(f"  LASSO-Cox (held-out test)             : 0.7289")

cv_df.to_csv(
    "results/graphsage_cv_results.tsv",
    sep="\t",
    index=False
)

print("\nSaved: results/graphsage_cv_results.tsv")

# ============================================================
# FINAL MODEL — TRAIN ON FULL COHORT FOR PREDICTIONS
# ============================================================

print("\n===== TRAINING FINAL MODEL ON FULL COHORT =====")

torch.manual_seed(42)
np.random.seed(42)

scaler_final = StandardScaler()
X_final_n    = scaler_final.fit_transform(X_raw)
X_final_t    = torch.tensor(X_final_n, dtype=torch.float32).to(device)
time_t       = torch.tensor(time,  dtype=torch.float32).to(device)
event_t      = torch.tensor(event, dtype=torch.float32).to(device)

final_model = GraphSAGE(input_dim=X_final_t.shape[1]).to(device)
opt_final   = optim.Adam(
    final_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

for epoch in range(N_EPOCHS):
    final_model.train()
    risk = final_model(X_final_t, A_t)
    loss = cox_loss(risk, time_t, event_t)
    opt_final.zero_grad()
    loss.backward()
    opt_final.step()
    if epoch % 100 == 0 or epoch == N_EPOCHS - 1:
        print(f"  Epoch {epoch:03d} | Loss: {loss.item():.4f}")

final_model.eval()
with torch.no_grad():
    final_risk = final_model(X_final_t, A_t).cpu().numpy()

full_c = concordance_index(time, -final_risk, event)
print(f"\nFull-cohort C-index (final model): {full_c:.4f}")

# ── Save predictions ─────────────────────────────────────────
pred_df = pd.DataFrame({
    "patient":        df.index,
    "predicted_risk": final_risk,
    "OS.time":        time,
    "OS":             event
})
pred_df.to_csv(
    "results/graphsage_risk_predictions.tsv",
    sep="\t",
    index=False
)
print("Saved: results/graphsage_risk_predictions.tsv")

# ── Save model ───────────────────────────────────────────────
torch.save(
    final_model.state_dict(),
    "results/graphsage_model_final.pth"
)
print("Saved: results/graphsage_model_final.pth")

# ============================================================
# KM CURVE (FINAL MODEL, FULL COHORT)
# ============================================================

print("\n===== GENERATING KM CURVE =====")

os.makedirs("results/plots", exist_ok=True)

cut  = np.median(final_risk)
high = final_risk > cut

lr_res = logrank_test(
    time[high],  time[~high],
    event[high], event[~high]
)

kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))

kmf.fit(time[high], event[high], label="High Risk")
kmf.plot(ci_show=True)

kmf.fit(time[~high], event[~high], label="Low Risk")
kmf.plot(ci_show=True)

plt.title(
    f"GraphSAGE Survival Stratification\n"
    f"Log-rank p = {lr_res.p_value:.3e} | "
    f"C-index (full cohort) = {full_c:.4f}",
    fontsize=10
)
plt.xlabel("Survival Time (Days)")
plt.ylabel("Survival Probability")
plt.grid(True, alpha=0.3)

km_path = "results/plots/graphsage_km_curve.png"
plt.savefig(km_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {km_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 62)
print("GRAPHSAGE INDUCTIVE MODEL — COMPLETE SUMMARY")
print("=" * 62)
print(f"""
Architecture    : GraphSAGE (56->{HIDDEN_1}->{HIDDEN_2}->1)
Aggregation     : Mean-pooling of neighbor features (inductive)
Evaluation      : 5-fold cross-validation + full-cohort

5-Fold CV Results:
  Mean C-index  : {mean_c:.4f} ± {std_c:.4f}
  (Each fold trains on 80%, evaluates on unseen 20%)

Full-cohort C-index: {full_c:.4f}
Full-cohort log-rank p: {lr_res.p_value:.3e}

vs. Transductive GAT:
  GAT full cohort  : 0.7425  (transductive, seen patients)
  GAT held-out 30% : 0.4122  (transductive, unseen patients)
  GraphSAGE CV mean: {mean_c:.4f}  (inductive, unseen folds)

Files saved:
  results/graphsage_cv_results.tsv
  results/graphsage_risk_predictions.tsv
  results/graphsage_model_final.pth
  results/plots/graphsage_km_curve.png
""")