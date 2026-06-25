# ============================================================
# GRAPH TRANSFORMER SURVIVAL MODEL — IMPROVED HYPERPARAMETERS
#
# Based on the original train_graph_transformer20.py
# (C-index = 0.5932), this version addresses the
# under-training symptoms visible in the original loss curve
# (5.2272 -> 5.1855 over 100 epochs, essentially flat):
#
#   1. LEARNING RATE: 0.001 -> 0.01 (10x higher)
#   2. EPOCHS: 100 -> 500 (5x more, with full-epoch loss log)
#   3. WEIGHT DECAY: 0 -> 1e-4 (L2 regularisation, since the
#      original model has ~6000+ params for only 418 samples)
#   4. ARCHITECTURE: 56 -> 64 -> 32 -> 1 reduced to
#      56 -> 16 -> 8 -> 1 (smaller hidden dims, less
#      overparameterisation for n=418)
#
# Architecture (class names, forward signatures), Cox loss,
# data loading, normalisation, and patient alignment are
# UNCHANGED from the original script. Same random seed (42)
# for reproducibility.
#
# Output:
#   results/graph_transformer_risk_predictions_improved.tsv
#   results/graph_transformer_model_improved.pth
#   results/training_loss_improved.tsv (full per-epoch loss)
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
# LOAD GRAPH DATA
# (UNCHANGED from original)
# ============================================================

print("\n===== LOADING LASSO-REFINED GRAPH DATA =====")

X_all = np.load(
    "data/processed/patient_X_lasso19.npy"
)

A_all = np.load(
    "data/processed/patient_A_lasso19.npy"
)

print("Feature matrix shape:", X_all.shape)
print("Adjacency matrix shape:", A_all.shape)

# ============================================================
# FEATURE NORMALIZATION
# (UNCHANGED from original)
# ============================================================

print("\n===== NORMALIZING FEATURES =====")

scaler = StandardScaler()

X_all = scaler.fit_transform(X_all)

# ============================================================
# LOAD SURVIVAL DATA
# (UNCHANGED from original)
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
# (UNCHANGED from original)
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
# (UNCHANGED from original)
# ============================================================

print("\n===== NORMALIZING GRAPH =====")

A = A + np.eye(A.shape[0])

D = np.diag(np.sum(A, axis=1))

D_inv = np.linalg.inv(np.sqrt(D))

A = D_inv @ A @ D_inv

print("Graph normalization completed!")

# ============================================================
# CONVERT TO TENSORS
# (UNCHANGED from original)
# ============================================================

print("\n===== CONVERTING TO TENSORS =====")

X = torch.tensor(X, dtype=torch.float32).to(device)
A = torch.tensor(A, dtype=torch.float32).to(device)
time = torch.tensor(time, dtype=torch.float32).to(device)
event = torch.tensor(event, dtype=torch.float32).to(device)

print("Tensor conversion completed!")

# ============================================================
# GRAPH ATTENTION LAYER
# (UNCHANGED structure from original, returns alpha too
#  for consistency with the Phase-5-ready version)
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

        return out, alpha

# ============================================================
# GRAPH TRANSFORMER MODEL — REDUCED CAPACITY
#
# CHANGE FROM ORIGINAL:
#   gat1: input -> 64   becomes   input -> 16
#   gat2: 64 -> 32      becomes   16 -> 8
#   out:  32 -> 1       becomes   8 -> 1
#
# Rationale: 56 input features and 418 samples give a much
# smaller effective sample-to-parameter ratio with 64/32
# hidden units than with 16/8. Reducing capacity is a
# standard response to underfitting-via-noise / flat-loss
# symptoms on small tabular datasets.
# ============================================================

HIDDEN_1 = 16
HIDDEN_2 = 8

class GraphTransformer(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.gat1 = GraphAttentionLayer(input_dim, HIDDEN_1)
        self.gat2 = GraphAttentionLayer(HIDDEN_1, HIDDEN_2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(HIDDEN_2, 1)

    def forward(self, X, A):

        h1, alpha1 = self.gat1(X, A)
        X = self.dropout(self.relu(h1))

        h2, alpha2 = self.gat2(X, A)
        X = self.dropout(self.relu(h2))

        risk = self.out(X).squeeze()

        return risk, alpha1, alpha2

# ============================================================
# COX LOSS
# (UNCHANGED from original)
# ============================================================

def cox_loss(risk, time, event):

    order = torch.argsort(time, descending=True)

    risk = risk[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(risk, dim=0)

    loss = -torch.sum((risk - log_cumsum) * event)

    return loss / torch.sum(event)

# ============================================================
# INITIALIZE MODEL — NEW HYPERPARAMETERS
#
# CHANGES FROM ORIGINAL:
#   lr: 0.001 -> 0.01            (10x higher)
#   weight_decay: 0 -> 1e-4      (L2 regularisation, new)
# ============================================================

print("\n===== INITIALIZING MODEL (IMPROVED HYPERPARAMETERS) =====")

LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
N_EPOCHS = 500

model = GraphTransformer(input_dim=X.shape[1]).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print(model)
print(f"\nHyperparameters:")
print(f"  Hidden dims      : {X.shape[1]} -> {HIDDEN_1} -> {HIDDEN_2} -> 1")
print(f"  Learning rate    : {LEARNING_RATE}  (was 0.001)")
print(f"  Weight decay     : {WEIGHT_DECAY}  (was 0)")
print(f"  Epochs           : {N_EPOCHS}  (was 100)")
print(f"  Dropout          : 0.3  (unchanged)")

# ============================================================
# TRAINING — 500 EPOCHS, FULL LOSS LOGGING
# ============================================================

print("\n===== TRAINING GRAPH TRANSFORMER (IMPROVED) =====")

train_losses = []

for epoch in range(N_EPOCHS):

    model.train()

    risk, _, _ = model(X, A)

    loss = cox_loss(risk, time, event)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if epoch % 50 == 0 or epoch == N_EPOCHS - 1:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

print("\nTraining completed!")

print(f"\nLoss trajectory summary:")
print(f"  Epoch 0   loss: {train_losses[0]:.4f}")
print(f"  Epoch 100 loss: {train_losses[100]:.4f}")
print(f"  Epoch 250 loss: {train_losses[250]:.4f}")
print(f"  Epoch {N_EPOCHS-1} loss: {train_losses[-1]:.4f}")
print(f"  Total decrease : {train_losses[0] - train_losses[-1]:.4f}")
print(f"  (Original 100-epoch decrease was only 0.0357: "
      f"5.2272 -> 5.1855)")

# Save full per-epoch loss for plotting/inspection
loss_df = pd.DataFrame({
    "epoch": range(N_EPOCHS),
    "loss": train_losses
})
loss_df.to_csv(
    "results/training_loss_improved.tsv",
    sep="\t",
    index=False
)

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

print("\nFull TCGA C-index (improved hyperparameters):",
      round(c_index, 4))
print("\nComparison:")
print(f"  Original GAT (56->64->32->1, lr=0.001, 100 epochs): 0.5932")
print(f"  LASSO-Cox composite risk score (linear, n=365):     0.7478")
print(f"  This run (56->{HIDDEN_1}->{HIDDEN_2}->1, lr={LEARNING_RATE}, "
      f"{N_EPOCHS} epochs, wd={WEIGHT_DECAY}): {round(c_index, 4)}")

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
    "results/graph_transformer_risk_predictions_improved.tsv",
    sep="\t",
    index=False
)

print("Predictions saved successfully!")

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "results/graph_transformer_model_improved.pth"
)

print("Model saved successfully!")

# ============================================================
# SAVE ATTENTION WEIGHTS
# ============================================================

print("\n===== SAVING ATTENTION WEIGHTS =====")

np.save("results/attention_layer1_improved.npy", alpha1_np)
np.save("results/attention_layer2_improved.npy", alpha2_np)

print("Attention matrices saved:")
print("  results/attention_layer1_improved.npy  shape:", alpha1_np.shape)
print("  results/attention_layer2_improved.npy  shape:", alpha2_np.shape)

# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n===== INTERPRETATION =====")

if c_index > 0.5932:
    print(
        f"\nThe improved hyperparameters increased the C-index "
        f"from 0.5932 to {round(c_index, 4)}, an improvement of "
        f"{round(c_index - 0.5932, 4)}. "
    )
    if c_index >= 0.748:
        print(
            "This now MATCHES OR EXCEEDS the LASSO-Cox composite "
            "risk score (0.7478) -- the Graph Transformer's "
            "underperformance was primarily a training/capacity "
            "issue rather than a fundamental limitation of the "
            "graph-based approach for this task."
        )
    else:
        print(
            f"This is still below the LASSO-Cox composite risk "
            f"score (0.7478), but the gap has narrowed from "
            f"{round(0.7478-0.5932,4)} to "
            f"{round(0.7478-c_index,4)}, indicating the "
            f"hyperparameter changes recovered some of the lost "
            f"discriminative performance."
        )
else:
    print(
        f"\nThe improved hyperparameters did not increase the "
        f"C-index (0.5932 -> {round(c_index, 4)}). This suggests "
        f"the bottleneck is not pure under-training/over-capacity, "
        f"and further architectural changes (e.g. residual "
        f"connection to the linear LASSO score, or revisiting the "
        f"K-nearest-neighbour patient graph construction) would be "
        f"needed."
    )

print("\nPipeline completed successfully!")