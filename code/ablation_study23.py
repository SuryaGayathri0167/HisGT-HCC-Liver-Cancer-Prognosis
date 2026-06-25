# ============================================================
# STEP 23 — ABLATION STUDY (CORRECTED VERSION)
#
# KEY FIXES over the first version:
#
# 1. TRAIN/TEST SPLIT (70/30):
#    Training loss is computed on the 70% train set.
#    C-index and log-rank are evaluated on the held-out
#    30% test set only. This prevents Model B (flat MLP)
#    from achieving artificially inflated C-index via
#    memorization of training data — which produced the
#    unrealistic 0.9464 in the first run.
#
# 2. PER-MODEL SEED RESET:
#    torch.manual_seed(42) and np.random.seed(42) are
#    reset before each model's training loop, removing
#    cross-model RNG contamination that caused Model A's
#    C-index (0.7090) to diverge from the standalone
#    train_graph_transformer_improved.py result (0.7425).
#
# All other settings unchanged:
#   - 56-gene plain expression features
#   - Same patient similarity graph
#   - Same Cox partial likelihood loss
#   - lr=0.01, wd=1e-4, 500 epochs, dropout=0.3
#   - Hidden dims 16/8 for all neural models
#
# Output: results/ablation_comparison23.tsv
# ============================================================

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# SHARED HYPERPARAMETERS
# ============================================================

LEARNING_RATE = 0.01
WEIGHT_DECAY  = 1e-4
N_EPOCHS      = 500
HIDDEN_1      = 16
HIDDEN_2      = 8
DROPOUT       = 0.3
TEST_SIZE     = 0.30
RANDOM_STATE  = 42

print(f"\nShared hyperparameters:")
print(f"  Hidden dims   : 56 -> {HIDDEN_1} -> {HIDDEN_2} -> 1")
print(f"  Learning rate : {LEARNING_RATE}")
print(f"  Weight decay  : {WEIGHT_DECAY}")
print(f"  Epochs        : {N_EPOCHS}")
print(f"  Dropout       : {DROPOUT}")
print(f"  Train/Test    : {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split")

# ============================================================
# LOAD DATA
# ============================================================

print("\n===== LOADING DATA =====")

X_plain = np.load("data/processed/patient_X_lasso19.npy")
A_all   = np.load("data/processed/patient_A_lasso19.npy")

df = pd.read_csv(
    "results/lasso_risk_scores16.tsv",
    sep="\t",
    index_col=0
)

n = len(df)

X_plain = X_plain[:n]
A       = A_all[:n, :n]

time  = df["OS.time"].values
event = df["OS"].values

print(f"Expression matrix: {X_plain.shape}")
print(f"Adjacency matrix : {A.shape}")
print(f"Patients: {n} | Events: {int(event.sum())}")

# ── Train/Test split indices ──────────────────────────────────
all_idx = np.arange(n)

train_idx, test_idx = train_test_split(
    all_idx,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Train: {len(train_idx)} patients | "
      f"Test: {len(test_idx)} patients")

# ── Normalise features (fit on TRAIN only) ────────────────────
scaler    = StandardScaler()
X_train_n = scaler.fit_transform(X_plain[train_idx])
X_test_n  = scaler.transform(X_plain[test_idx])
X_all_n   = scaler.transform(X_plain)  # for graph models (full graph)

# ── Survival data subsets ─────────────────────────────────────
time_train  = time[train_idx];  event_train  = event[train_idx]
time_test   = time[test_idx];   event_test   = event[test_idx]

# ── Normalise graph ───────────────────────────────────────────
A_norm = A + np.eye(n)
D      = np.diag(np.sum(A_norm, axis=1))
D_inv  = np.linalg.inv(np.sqrt(D))
A_norm = D_inv @ A_norm @ D_inv

# ── Full-graph tensors (graph models train on full graph) ─────
X_full_t   = torch.tensor(X_all_n,  dtype=torch.float32).to(device)
A_t        = torch.tensor(A_norm,   dtype=torch.float32).to(device)
time_t     = torch.tensor(time,     dtype=torch.float32).to(device)
event_t    = torch.tensor(event,    dtype=torch.float32).to(device)

# ── MLP tensors (train subset only) ──────────────────────────
X_mlp_tr_t = torch.tensor(X_train_n, dtype=torch.float32).to(device)
X_mlp_te_t = torch.tensor(X_test_n,  dtype=torch.float32).to(device)
time_tr_t  = torch.tensor(time_train,  dtype=torch.float32).to(device)
event_tr_t = torch.tensor(event_train, dtype=torch.float32).to(device)

# ============================================================
# SHARED UTILITIES
# ============================================================

def cox_loss(risk, time, event):
    order      = torch.argsort(time, descending=True)
    risk       = risk[order]
    event      = event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss       = -torch.sum((risk - log_cumsum) * event)
    return loss / torch.sum(event)


def evaluate_on_test(risk_np_full, label, threshold="optimal"):
    """Evaluate C-index and log-rank on held-out TEST set only."""
    risk_test = risk_np_full[test_idx]

    c_idx = concordance_index(time_test, -risk_test, event_test)

    if threshold == "median":
        cut = np.median(risk_test)
    else:
        best_stat, cut = -np.inf, np.median(risk_test)
        for pct in np.arange(10, 91, 1):
            c    = np.percentile(risk_test, pct)
            high = risk_test > c
            if high.sum() < 5 or (~high).sum() < 5:
                continue
            res = logrank_test(
                time_test[high],  time_test[~high],
                event_test[high], event_test[~high]
            )
            if res.test_statistic > best_stat:
                best_stat, cut = res.test_statistic, c

    high = risk_test > cut
    res  = logrank_test(
        time_test[high],  time_test[~high],
        event_test[high], event_test[~high]
    )

    print(
        f"  {label:40s} | C-index={c_idx:.4f} | "
        f"log-rank p={res.p_value:.3e} | "
        f"High n={high.sum()}, Low n={(~high).sum()} "
        f"[TEST SET n={len(test_idx)}]"
    )

    return {
        "Model":     label,
        "C_index":   round(c_idx, 4),
        "logrank_p": res.p_value,
        "n_high":    int(high.sum()),
        "n_low":     int((~high).sum()),
        "eval_set":  f"test (n={len(test_idx)})"
    }


results_table = []

# ============================================================
# MODEL A — GRAPH ATTENTION TRANSFORMER
# Trains on full graph (transductive), evaluates on test_idx
# ============================================================

print("\n===== MODEL A: GRAPH ATTENTION TRANSFORMER =====")

torch.manual_seed(42)
np.random.seed(42)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.W    = nn.Linear(in_f, out_f)
        self.attn = nn.Linear(2 * out_f, 1)

    def forward(self, X, A):
        H  = self.W(X)
        N  = H.size(0)
        H1 = H.repeat(1, N).view(N * N, -1)
        H2 = H.repeat(N, 1)
        e  = self.attn(torch.cat([H1, H2], dim=1)).view(N, N)
        A_loop = A + torch.eye(N, device=A.device)
        e  = e.masked_fill(A_loop == 0, float('-inf'))
        return torch.matmul(torch.softmax(e, dim=1), H)


class ModelA(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gat1    = GraphAttentionLayer(input_dim, HIDDEN_1)
        self.gat2    = GraphAttentionLayer(HIDDEN_1,  HIDDEN_2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.out     = nn.Linear(HIDDEN_2, 1)

    def forward(self, X, A):
        X = self.dropout(self.relu(self.gat1(X, A)))
        X = self.dropout(self.relu(self.gat2(X, A)))
        return self.out(X).squeeze()


model_a = ModelA(X_full_t.shape[1]).to(device)
opt_a   = optim.Adam(
    model_a.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

for epoch in range(N_EPOCHS):
    model_a.train()
    risk_full = model_a(X_full_t, A_t)
    # Train loss computed on TRAIN patients only
    loss = cox_loss(risk_full[train_idx], time_tr_t, event_tr_t)
    opt_a.zero_grad()
    loss.backward()
    opt_a.step()
    if epoch % 100 == 0 or epoch == N_EPOCHS - 1:
        print(f"  Epoch {epoch:03d} | Train Loss: {loss.item():.4f}")

model_a.eval()
with torch.no_grad():
    risk_a = model_a(X_full_t, A_t).cpu().numpy()

results_table.append(
    evaluate_on_test(risk_a, "A: Graph Attention Transformer")
)

# ============================================================
# MODEL B — FLAT MLP (no graph)
# Trains on train set only, evaluates on test set
# ============================================================

print("\n===== MODEL B: FLAT MLP (no graph) =====")

torch.manual_seed(42)
np.random.seed(42)

class ModelB(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_2, 1)
        )

    def forward(self, X):
        return self.net(X).squeeze()


model_b = ModelB(X_mlp_tr_t.shape[1]).to(device)
opt_b   = optim.Adam(
    model_b.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

for epoch in range(N_EPOCHS):
    model_b.train()
    risk_tr = model_b(X_mlp_tr_t)
    loss    = cox_loss(risk_tr, time_tr_t, event_tr_t)
    opt_b.zero_grad()
    loss.backward()
    opt_b.step()
    if epoch % 100 == 0 or epoch == N_EPOCHS - 1:
        print(f"  Epoch {epoch:03d} | Train Loss: {loss.item():.4f}")

model_b.eval()
with torch.no_grad():
    # Evaluate on ALL patients for consistent test_idx indexing
    X_all_b = torch.tensor(X_all_n, dtype=torch.float32).to(device)
    risk_b  = model_b(X_all_b).cpu().numpy()

results_table.append(
    evaluate_on_test(risk_b, "B: Flat MLP (no graph)")
)

# ============================================================
# MODEL C — STANDARD GCN (no attention)
# ============================================================

print("\n===== MODEL C: STANDARD GCN (no attention) =====")

torch.manual_seed(42)
np.random.seed(42)

class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.W = nn.Linear(in_f, out_f)

    def forward(self, X, A):
        return torch.matmul(A, self.W(X))


class ModelC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gcn1    = GCNLayer(input_dim, HIDDEN_1)
        self.gcn2    = GCNLayer(HIDDEN_1,  HIDDEN_2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.out     = nn.Linear(HIDDEN_2, 1)

    def forward(self, X, A):
        X = self.dropout(self.relu(self.gcn1(X, A)))
        X = self.dropout(self.relu(self.gcn2(X, A)))
        return self.out(X).squeeze()


model_c = ModelC(X_full_t.shape[1]).to(device)
opt_c   = optim.Adam(
    model_c.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

for epoch in range(N_EPOCHS):
    model_c.train()
    risk_full = model_c(X_full_t, A_t)
    loss      = cox_loss(risk_full[train_idx], time_tr_t, event_tr_t)
    opt_c.zero_grad()
    loss.backward()
    opt_c.step()
    if epoch % 100 == 0 or epoch == N_EPOCHS - 1:
        print(f"  Epoch {epoch:03d} | Train Loss: {loss.item():.4f}")

model_c.eval()
with torch.no_grad():
    risk_c = model_c(X_full_t, A_t).cpu().numpy()

results_table.append(
    evaluate_on_test(risk_c, "C: Standard GCN (no attention)")
)

# ============================================================
# MODEL D — LASSO-COX ALONE
# ============================================================

print("\n===== MODEL D: LASSO-COX ALONE =====")

if "risk_score" in df.columns:
    risk_d = df["risk_score"].values
    print("  Loaded from lasso_risk_scores16.tsv")
else:
    coefs  = pd.read_csv(
        "results/lasso_selected_genes16.tsv",
        sep="\t", index_col=0
    )["coef"].values
    risk_d = (X_plain * coefs).sum(axis=1)
    print("  Recomputed from coefficients x expression")

results_table.append(
    evaluate_on_test(risk_d, "D: LASSO-Cox alone")
)

# ============================================================
# MODEL E — GRAPH TRANSFORMER, MEDIAN-SPLIT THRESHOLD
# ============================================================

print("\n===== MODEL E: GRAPH TRANSFORMER (median split) =====")

results_table.append(
    evaluate_on_test(
        risk_a,
        "E: Graph Transformer (median split)",
        threshold="median"
    )
)

# ============================================================
# COMPARISON TABLE
# ============================================================

print("\n" + "=" * 62)
print("ABLATION STUDY RESULTS (Test set evaluation)")
print("=" * 62)

comparison = pd.DataFrame(results_table)
comparison["logrank_p_fmt"] = comparison["logrank_p"].apply(
    lambda x: f"{x:.3e}"
)

print("\n" + comparison[[
    "Model", "C_index", "logrank_p_fmt", "n_high", "n_low"
]].to_string(index=False))

comparison.to_csv(
    "results/ablation_comparison23.tsv",
    sep="\t",
    index=False
)

print("\nSaved: results/ablation_comparison23.tsv")

# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 62)
print("INTERPRETATION")
print("=" * 62)

c_a = results_table[0]["C_index"]
c_b = results_table[1]["C_index"]
c_c = results_table[2]["C_index"]
c_d = results_table[3]["C_index"]

print(f"\nModel A (Graph Attention Transformer) : {c_a:.4f}")
print(f"Model B (Flat MLP, no graph)          : {c_b:.4f}  ->  "
      f"{'A > B: graph structure adds value' if c_a > c_b else 'A <= B: graph structure does not add value here'}")
print(f"Model C (Standard GCN, no attention)  : {c_c:.4f}  ->  "
      f"{'A > C: attention adds value over plain GCN' if c_a > c_c else 'A <= C: attention does not outperform plain GCN'}")
print(f"Model D (LASSO-Cox alone)             : {c_d:.4f}  ->  "
      f"{'A > D: graph learning adds value beyond LASSO' if c_a > c_d else 'LASSO-Cox matches/outperforms Graph Transformer'}")

print("\nStep 23 completed successfully!")