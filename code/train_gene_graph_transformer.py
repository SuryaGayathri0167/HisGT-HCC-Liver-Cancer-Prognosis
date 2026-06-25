# ============================================================
# GENE-LEVEL GRAPH TRANSFORMER (CORRECT MODEL)
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index

print("===== LOADING DATA =====")

# ==============================
# LOAD GENE GRAPH
# ==============================

A = np.load("data/processed/gene_A.npy")   # (genes × genes)

# ==============================
# LOAD EXPRESSION
# ==============================

expr = pd.read_csv(
    "data/tcga/TCGA-LIHC_graph_features7.tsv",
    sep="\t",
    index_col=0
)

# expr = (genes × patients)
X = expr.values

# ==============================
# LOAD SURVIVAL
# ==============================

surv = pd.read_csv(
    "data/tcga/TCGA-LIHC.survival.tsv",
    sep="\t"
)[["sample", "OS.time", "OS"]].dropna()

# Align patients
common = expr.columns.intersection(surv["sample"])

expr = expr[common]
surv = surv.set_index("sample").loc[common]

X = expr.values   # (genes × patients)

time = surv["OS.time"].values
event = surv["OS"].values

print("X shape (genes × patients):", X.shape)
print("A shape:", A.shape)

# ==============================
# NORMALIZE FEATURES
# ==============================

scaler = StandardScaler()
X = scaler.fit_transform(X.T).T  # normalize per gene

# ==============================
# SPLIT PATIENTS
# ==============================

idx = np.arange(len(common))

train_idx, test_idx = train_test_split(
    idx, test_size=0.3, random_state=42
)

# ==============================
# TENSORS
# ==============================

X = torch.tensor(X, dtype=torch.float32)     # (genes × patients)
A = torch.tensor(A, dtype=torch.float32)
time = torch.tensor(time, dtype=torch.float32)
event = torch.tensor(event, dtype=torch.float32)

# ============================================================
# GRAPH ATTENTION LAYER
# ============================================================

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.W = nn.Linear(in_features, out_features)
        self.attn = nn.Linear(2 * out_features, 1)

    def forward(self, X, A):

        H = self.W(X)   # (genes × hidden)
        N = H.size(0)

        H1 = H.repeat(1, N).view(N*N, -1)
        H2 = H.repeat(N, 1)

        concat = torch.cat([H1, H2], dim=1)

        e = self.attn(concat).view(N, N)

        e = e.masked_fill(A == 0, float('-inf'))

        alpha = torch.softmax(e, dim=1)

        return torch.matmul(alpha, H)

# ============================================================
# MODEL
# ============================================================

class GeneGraphTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.gat1 = GraphAttentionLayer(input_dim, 64)
        self.gat2 = GraphAttentionLayer(64, 32)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Final prediction layer
        self.fc = nn.Linear(32, 1)

    def forward(self, X, A):

        # X = (genes × patients)

        # ===== STEP 1: Learn gene embeddings =====
        H = self.relu(self.gat1(X, A))
        H = self.dropout(H)

        H = self.relu(self.gat2(H, A))
        H = self.dropout(H)

        # H = (genes × hidden_dim) → (94 × 32)

        # ===== STEP 2: Convert to patient-level features =====
        patient_features = torch.matmul(X.T, H)
        # (patients × genes) × (genes × hidden)
        # → (418 × 32)

        # ===== STEP 3: Predict risk =====
        return self.fc(patient_features).squeeze()
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
# TRAINING
# ============================================================

print("===== TRAINING GENE GRAPH TRANSFORMER =====")

model = GeneGraphTransformer(input_dim=X.shape[1])

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for epoch in range(epochs):

    model.train()

    risk = model(X, A)

    loss = cox_loss(
        risk[train_idx],
        time[train_idx],
        event[train_idx]
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ============================================================
# EVALUATION
# ============================================================

print("\n===== EVALUATION =====")

model.eval()

with torch.no_grad():
    risk = model(X, A).numpy()

c_index = concordance_index(
    time[test_idx],
    -risk[test_idx],
    event[test_idx]
)

print("Test C-index:", round(c_index, 4))