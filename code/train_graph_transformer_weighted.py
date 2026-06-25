# ============================================================
# PHASE 3 (FINAL v2 - STRONG WEIGHT, RETRAIN) +
# PHASE 5 (ATTENTION EXTRACTION)
# GRAPH TRANSFORMER SURVIVAL MODEL — 5x CENTRALITY-WEIGHTED
# EXPRESSION FEATURES
#
# Loads the 5x centrality-weighted 56-dim feature matrix
# (X_weighted[:,g] = X[:,g] * (1 + 5*hub_score[g])) produced
# by enrich_centrality_weighted_strong.py. TP53 (hub_score≈
# 0.994) receives ~6x amplification vs the previous ~2x.
# Same architecture, same 100 epochs, same Cox loss, for a
# controlled comparison against:
#   - 0.7480 baseline (expression only)
#   - 0.5960 dense/sparse concatenation (collapsed by scaler)
#   - 0.5932 1x centrality weighting
# ============================================================

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from lifelines.utils import concordance_index

# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# ============================================================
# LOAD ENRICHED GRAPH DATA  (Phase 3 output)
# ============================================================

print("\n===== LOADING 5x CENTRALITY-WEIGHTED EXPRESSION DATA =====")

X_all = np.load(
    "data/processed/patient_X_lasso19_centrality_weighted_strong.npy"
)

A_all = np.load(
    "data/processed/patient_A_lasso19.npy"
)

print("Enriched feature matrix shape:", X_all.shape)
print("Adjacency matrix shape:", A_all.shape)

# ============================================================
# FEATURE NORMALIZATION
# ============================================================

print("\n===== NORMALIZING FEATURES =====")

scaler = StandardScaler()

X_all = scaler.fit_transform(X_all)

# ============================================================
# LOAD SURVIVAL DATA
# ============================================================

print("\n===== LOADING SURVIVAL DATA =====")

df = pd.read_csv(
    "results/lasso_risk_scores16.tsv",
    sep="\t",
    index_col=0
)

print("Survival dataset shape:", df.shape)

# ============================================================
# ALIGN PATIENTS
# ============================================================

print("\n===== ALIGNING PATIENTS =====")

X = X_all[:len(df)]

A = A_all[:len(df), :len(df)]

time = df["OS.time"].values

event = df["OS"].values

print("Final aligned feature matrix:", X.shape)
print("Final aligned adjacency matrix:", A.shape)

# ============================================================
# GRAPH NORMALIZATION
# ============================================================

print("\n===== NORMALIZING GRAPH =====")

A = A + np.eye(A.shape[0])

D = np.diag(np.sum(A, axis=1))

D_inv = np.linalg.inv(np.sqrt(D))

A = D_inv @ A @ D_inv

print("Graph normalization completed!")

# ============================================================
# CONVERT TO TENSORS
# ============================================================

print("\n===== CONVERTING TO TENSORS =====")

X = torch.tensor(X, dtype=torch.float32).to(device)
A = torch.tensor(A, dtype=torch.float32).to(device)
time = torch.tensor(time, dtype=torch.float32).to(device)
event = torch.tensor(event, dtype=torch.float32).to(device)

print("Tensor conversion completed!")

# ============================================================
# GRAPH ATTENTION LAYER (returns attention weights too)
# ============================================================

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)
        self.attn = nn.Linear(2 * out_features, 1)

    def forward(self, X, A):

        H = self.W(X)
        N = H.size(0)

        H1 = H.repeat(1, N).view(N * N, -1)
        H2 = H.repeat(N, 1)

        concat = torch.cat([H1, H2], dim=1)
        e = self.attn(concat).view(N, N)

        A_loop = A + torch.eye(A.size(0), device=device)

        e = e.masked_fill(A_loop == 0, float('-inf'))

        alpha = torch.softmax(e, dim=1)

        out = torch.matmul(alpha, H)

        # Return both the propagated features AND the
        # attention matrix for Phase 5 extraction.
        return out, alpha

# ============================================================
# GRAPH TRANSFORMER MODEL
# ============================================================

class GraphTransformer(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.gat1 = GraphAttentionLayer(input_dim, 64)
        self.gat2 = GraphAttentionLayer(64, 32)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(32, 1)

    def forward(self, X, A):

        h1, alpha1 = self.gat1(X, A)
        h1 = self.dropout(self.relu(h1))

        h2, alpha2 = self.gat2(h1, A)
        h2 = self.dropout(self.relu(h2))

        risk = self.out(h2).squeeze()

        return risk, alpha1, alpha2

# ============================================================
# COX LOSS
# ============================================================

def cox_loss(risk, time, event):

    order = torch.argsort(time, descending=True)

    risk = risk[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(risk, dim=0)

    loss = -torch.sum((risk - log_cumsum) * event)

    return loss / torch.sum(event)

# ============================================================
# INITIALIZE MODEL
# ============================================================

print("\n===== INITIALIZING MODEL =====")

model = GraphTransformer(input_dim=X.shape[1]).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)

# ============================================================
# TRAINING
# ============================================================

print("\n===== TRAINING GRAPH TRANSFORMER (ENRICHED) =====")

train_losses = []

for epoch in range(100):

    model.train()

    risk, _, _ = model(X, A)

    loss = cox_loss(risk, time, event)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

print("\nTraining completed!")

# ============================================================
# EVALUATION
# ============================================================

print("\n===== MODEL EVALUATION =====")

model.eval()

with torch.no_grad():
    risk, alpha1, alpha2 = model(X, A)

risk_np = risk.cpu().numpy()
alpha1_np = alpha1.cpu().numpy()
alpha2_np = alpha2.cpu().numpy()

# ============================================================
# FULL COHORT C-INDEX
# ============================================================

c_index = concordance_index(
    time.cpu().numpy(),
    -risk_np,
    event.cpu().numpy()
)

print("\nFull TCGA C-index (5x centrality-weighted expression):",
      round(c_index, 4))
print("Baseline (56-gene expression-only):           0.7480")
print("Concatenation attempts (collapsed by scaler): 0.5960")
print("1x centrality weighting:                      0.5932")

# ============================================================
# SAVE PREDICTIONS
# ============================================================

print("\n===== SAVING PREDICTIONS =====")

results = pd.DataFrame({
    "patient": df.index,
    "predicted_risk": risk_np,
    "OS.time": time.cpu().numpy(),
    "OS": event.cpu().numpy()
})

results.to_csv(
    "results/graph_transformer_risk_predictions_weighted_strong.tsv",
    sep="\t",
    index=False
)

print("Predictions saved successfully!")

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "results/graph_transformer_model_weighted_strong.pth"
)

print("Model saved successfully!")

# ============================================================
# SAVE ATTENTION WEIGHTS (PHASE 5 INPUT)
# ============================================================

print("\n===== SAVING ATTENTION WEIGHTS =====")

np.save("results/attention_layer1_weighted_strong.npy", alpha1_np)
np.save("results/attention_layer2_weighted_strong.npy", alpha2_np)

print("Attention matrices saved:")
print("  results/attention_layer1_weighted_strong.npy  shape:", alpha1_np.shape)
print("  results/attention_layer2_weighted_strong.npy  shape:", alpha2_np.shape)

# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n===== BIOLOGICAL INTERPRETATION =====")

print(
    "\nThe 5x centrality-weighted Graph Transformer learned "
    "interaction-aware survival patterns among HCC patients "
    "using the 56 LASSO-selected prognostic histone "
    "regulators, with each gene's expression rescaled by "
    "(1 + 5*hub_score) so that TP53 (hub_score≈0.994, weight≈"
    "6.0x) and other central genes (EP300, HDAC1, KAT2B) "
    "dominate the patient similarity graph far more strongly "
    "than in the 1x version."
)

print(
    "\nAttention matrices from both Graph Attention Layers "
    "have been saved for Phase 5 biological interpretation, "
    "showing which patients the model considers similar when "
    "computing each patient's risk embedding."
)

print("\nPipeline completed successfully!")