# ============================================================
# GRAPH TRANSFORMER SURVIVAL MODEL
# FULL-GRAPH VERSION + ATTENTION WEIGHT SAVING (Phase 5)
#
# This is train_graph_transformer20.py with ONE addition:
# the GraphAttentionLayer and GraphTransformer now also
# return the attention matrices (alpha1, alpha2), which are
# saved to disk after evaluation. Architecture, training
# loop, loss function, optimizer, seed, and epoch count are
# UNCHANGED -- this should reproduce the original C-index of
# 0.7480 exactly while additionally producing:
#   results/attention_layer1_baseline20.npy
#   results/attention_layer2_baseline20.npy
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

# Add self-loops
A = A + np.eye(A.shape[0])

# Symmetric normalization
D = np.diag(np.sum(A, axis=1))

D_inv = np.linalg.inv(np.sqrt(D))

A = D_inv @ A @ D_inv

print("Graph normalization completed!")

# ============================================================
# CONVERT TO TENSORS
# ============================================================

print("\n===== CONVERTING TO TENSORS =====")

X = torch.tensor(
    X,
    dtype=torch.float32
).to(device)

A = torch.tensor(
    A,
    dtype=torch.float32
).to(device)

time = torch.tensor(
    time,
    dtype=torch.float32
).to(device)

event = torch.tensor(
    event,
    dtype=torch.float32
).to(device)

print("Tensor conversion completed!")

# ============================================================
# GRAPH ATTENTION LAYER
# (MODIFIED: now also returns the attention matrix alpha)
# ============================================================

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.W = nn.Linear(
            in_features,
            out_features
        )

        self.attn = nn.Linear(
            2 * out_features,
            1
        )

    def forward(self, X, A):

        H = self.W(X)

        N = H.size(0)

        H1 = H.repeat(1, N).view(N * N, -1)

        H2 = H.repeat(N, 1)

        concat = torch.cat([H1, H2], dim=1)

        e = self.attn(concat).view(N, N)

        A_loop = A + torch.eye(
            A.size(0),
            device=device
        )

        e = e.masked_fill(
            A_loop == 0,
            float('-inf')
        )

        alpha = torch.softmax(e, dim=1)

        out = torch.matmul(alpha, H)

        # MODIFIED: return attention matrix alongside output
        return out, alpha

# ============================================================
# GRAPH TRANSFORMER MODEL
# (MODIFIED: forward now also returns alpha1, alpha2)
# ============================================================

class GraphTransformer(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.gat1 = GraphAttentionLayer(
            input_dim,
            64
        )

        self.gat2 = GraphAttentionLayer(
            64,
            32
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.out = nn.Linear(32, 1)

    def forward(self, X, A):

        h1, alpha1 = self.gat1(X, A)

        X = self.dropout(
            self.relu(h1)
        )

        h2, alpha2 = self.gat2(X, A)

        X = self.dropout(
            self.relu(h2)
        )

        risk = self.out(X).squeeze()

        # MODIFIED: return attention matrices too
        return risk, alpha1, alpha2

# ============================================================
# COX LOSS
# (UNCHANGED)
# ============================================================

def cox_loss(risk, time, event):

    order = torch.argsort(
        time,
        descending=True
    )

    risk = risk[order]

    event = event[order]

    log_cumsum = torch.logcumsumexp(
        risk,
        dim=0
    )

    loss = -torch.sum(
        (risk - log_cumsum) * event
    )

    return loss / torch.sum(event)

# ============================================================
# INITIALIZE MODEL
# (UNCHANGED)
# ============================================================

print("\n===== INITIALIZING MODEL =====")

model = GraphTransformer(
    input_dim=X.shape[1]
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

print(model)

# ============================================================
# TRAINING
# (UNCHANGED except unpacking the extra return values,
#  which are discarded during training)
# ============================================================

print("\n===== TRAINING GRAPH TRANSFORMER =====")

train_losses = []

for epoch in range(100):

    model.train()

    # Predict survival risk (alpha1, alpha2 discarded during training)
    risk, _, _ = model(X, A)

    # FULL-GRAPH COX LOSS
    loss = cox_loss(
        risk,
        time,
        event
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    train_losses.append(loss.item())

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f}"
        )

print("\nTraining completed!")

# ============================================================
# EVALUATION
# (MODIFIED: also capture alpha1, alpha2)
# ============================================================

print("\n===== MODEL EVALUATION =====")

model.eval()

with torch.no_grad():

    risk, alpha1, alpha2 = model(X, A)

    risk = risk.cpu().numpy()
    alpha1_np = alpha1.cpu().numpy()
    alpha2_np = alpha2.cpu().numpy()

# ============================================================
# FULL COHORT C-INDEX
# (UNCHANGED)
# ============================================================

c_index = concordance_index(
    time.cpu().numpy(),
    -risk,
    event.cpu().numpy()
)

print("\nFull TCGA C-index:", round(c_index, 4))
print("(Expected: 0.7480, matching the original baseline run)")

# ============================================================
# SAVE PREDICTIONS
# (UNCHANGED)
# ============================================================

print("\n===== SAVING PREDICTIONS =====")

results = pd.DataFrame({

    "patient": df.index,

    "predicted_risk": risk,

    "OS.time": time.cpu().numpy(),

    "OS": event.cpu().numpy()
})

results.to_csv(
    "results/graph_transformer_risk_predictions_full20.tsv",
    sep="\t",
    index=False
)

print("Predictions saved successfully!")

# ============================================================
# SAVE MODEL
# (UNCHANGED)
# ============================================================

torch.save(
    model.state_dict(),
    "results/graph_transformer_model_full20.pth"
)

print("Model saved successfully!")

# ============================================================
# SAVE ATTENTION WEIGHTS (NEW — PHASE 5 INPUT)
# ============================================================

print("\n===== SAVING ATTENTION WEIGHTS (PHASE 5) =====")

np.save("results/attention_layer1_baseline20.npy", alpha1_np)
np.save("results/attention_layer2_baseline20.npy", alpha2_np)

print("Attention matrices saved:")
print("  results/attention_layer1_baseline20.npy  shape:", alpha1_np.shape)
print("  results/attention_layer2_baseline20.npy  shape:", alpha2_np.shape)

# ============================================================
# FINAL INTERPRETATION
# (UNCHANGED)
# ============================================================

print("\n===== BIOLOGICAL INTERPRETATION =====")

print(
    "\nThe Graph Transformer learned "
    "interaction-aware survival patterns "
    "among HCC patients using "
    "LASSO-selected prognostic "
    "histone regulators."
)

print(
    "\nPatient relationships were modeled "
    "through graph attention mechanisms "
    "over the refined patient similarity graph."
)

print(
    "\nThe final model predicts "
    "continuous survival risk while "
    "incorporating graph-based "
    "patient interaction structure."
)

print("\nAttention matrices from both Graph Attention Layers "
      "have been saved for Phase 5 biological interpretation.")

print("\nPipeline completed successfully!")