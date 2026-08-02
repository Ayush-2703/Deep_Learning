"""
Phase 6 - Topic 1: Graph Neural Networks (GCN, implemented from scratch)
CPU-only, synthetic stochastic-block-model graph, PyTorch (no torch_geometric).

Run: python3 implementation.py
Produces: outputs/graph_structure.png, outputs/training_curves.png,
          outputs/gcn_vs_mlp_comparison.png, outputs/embedding_evolution.png
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(3)
np.random.seed(3)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# 1. Synthetic graph: stochastic block model with K communities
# ---------------------------------------------------------------------------
K = 4                 # number of communities/classes
NODES_PER_COMM = 30
N = K * NODES_PER_COMM
P_IN = 0.35            # edge prob within same community
P_OUT = 0.02           # edge prob across communities

labels = np.repeat(np.arange(K), NODES_PER_COMM)
A = np.zeros((N, N), dtype=np.float32)
for i in range(N):
    for j in range(i + 1, N):
        p = P_IN if labels[i] == labels[j] else P_OUT
        if np.random.rand() < p:
            A[i, j] = A[j, i] = 1.0

G = nx.from_numpy_array(A)
degrees = A.sum(axis=1)
print(f"Synthetic SBM graph: N={N} nodes, K={K} communities, "
      f"{int(A.sum() / 2)} edges, avg degree={degrees.mean():.2f}, "
      f"isolated nodes={int((degrees == 0).sum())}")

# Node features: noisy one-hot-ish signal + random noise (deliberately weak,
# so the graph structure -- not the raw features -- does most of the work)
feature_dim = 8
X_feat = np.random.normal(0, 1, size=(N, feature_dim)).astype(np.float32)
# inject a weak class signal into features (not fully separable alone)
for i in range(N):
    X_feat[i, labels[i] % feature_dim] += 0.6  # weak, not dominant, bump
X = torch.tensor(X_feat)
y = torch.tensor(labels, dtype=torch.long)

# ---------------------------------------------------------------------------
# 2. GCN symmetric normalization: D_hat^-1/2 A_hat D_hat^-1/2
# ---------------------------------------------------------------------------
A_hat = A + np.eye(N, dtype=np.float32)
D_hat = A_hat.sum(axis=1)
D_hat_inv_sqrt = np.diag(1.0 / np.sqrt(D_hat))
A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
A_norm_t = torch.tensor(A_norm.astype(np.float32))

# ---------------------------------------------------------------------------
# 3. GCN layer and model (implemented from scratch, no library)
# ---------------------------------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, A_norm, H):
        # H: [N, in_dim] -> aggregate neighbors via A_norm, then linear transform
        agg = A_norm @ H                # [N, in_dim], symmetric-normalized neighbor sum
        return self.linear(agg)         # [N, out_dim]


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, n_classes)

    def forward(self, A_norm, X):
        h1 = F.relu(self.gc1(A_norm, X))
        h1 = F.dropout(h1, p=0.3, training=self.training)
        out = self.gc2(A_norm, h1)
        return out, h1  # return hidden embedding too, for visualization


class MLPBaseline(nn.Module):
    """Identical depth/width, but ignores graph structure entirely -- classifies
    each node purely from its own feature vector."""
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, X):
        h1 = F.relu(self.fc1(X))
        h1 = F.dropout(h1, p=0.3, training=self.training)
        return self.fc2(h1), h1

# ---------------------------------------------------------------------------
# 4. Semi-supervised split: only a small fraction of node labels are visible
# ---------------------------------------------------------------------------
train_frac = 0.15  # only 15% of nodes' labels used for training -- realistic GCN setting
perm = np.random.permutation(N)
n_train = int(train_frac * N)
train_idx = torch.tensor(perm[:n_train])
val_idx = torch.tensor(perm[n_train:n_train + 40])
test_idx = torch.tensor(perm[n_train + 40:])
print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test nodes "
      f"(labels hidden for val/test during training)")

# ---------------------------------------------------------------------------
# 5. Train GCN
# ---------------------------------------------------------------------------
def train_model(model, is_gcn, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "test_acc": []}
    embed_snapshots = {}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        if is_gcn:
            out, h = model(A_norm_t, X)
        else:
            out, h = model(X)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if is_gcn:
                out_eval, h_eval = model(A_norm_t, X)
            else:
                out_eval, h_eval = model(X)
            preds = out_eval.argmax(dim=1)
            train_acc = (preds[train_idx] == y[train_idx]).float().mean().item()
            val_acc = (preds[val_idx] == y[val_idx]).float().mean().item()
            test_acc = (preds[test_idx] == y[test_idx]).float().mean().item()

        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        if is_gcn and epoch in (1, 50, 100, 200):
            embed_snapshots[epoch] = h_eval.detach().numpy().copy()

        if epoch % 40 == 0 or epoch == 1:
            tag = "GCN" if is_gcn else "MLP"
            print(f"[{tag}] Epoch {epoch:3d}/{epochs} | loss={loss.item():.3f} "
                  f"| train_acc={train_acc:.3f} val_acc={val_acc:.3f} test_acc={test_acc:.3f}")

    return history, embed_snapshots


print("\n--- Training GCN (uses graph structure) ---")
gcn_model = GCN(feature_dim, hidden_dim=16, n_classes=K)
gcn_history, embed_snapshots = train_model(gcn_model, is_gcn=True)

print("\n--- Training MLP baseline (identical features, graph structure hidden) ---")
mlp_model = MLPBaseline(feature_dim, hidden_dim=16, n_classes=K)
mlp_history, _ = train_model(mlp_model, is_gcn=False)

gcn_final_test = gcn_history["test_acc"][-1]
mlp_final_test = mlp_history["test_acc"][-1]
print(f"\nFinal TEST accuracy -- GCN: {gcn_final_test:.3f} | MLP (no graph): {mlp_final_test:.3f}")
gap = gcn_final_test - mlp_final_test
if gap > 0.05:
    print(f"NOTE: GCN outperforms the graph-blind MLP baseline by {gap:.3f} -- "
          f"confirms the graph structure carries real, exploitable signal.")
elif gap < -0.02:
    print(f"NOTE: GCN underperformed the MLP baseline by {-gap:.3f} -- reporting honestly; "
          f"possible causes: oversmoothing, too few training epochs, or weak community separation "
          f"(P_IN={P_IN}, P_OUT={P_OUT}).")
else:
    print("NOTE: GCN and MLP performed similarly -- graph structure may not have added much "
          "signal beyond node features in this particular run.")

# ---------------------------------------------------------------------------
# 6. Visualizations
# ---------------------------------------------------------------------------
# (a) Graph structure, colored by true community
pos = nx.spring_layout(G, seed=3, k=0.3)
plt.figure(figsize=(7, 7))
nx.draw_networkx_edges(G, pos, alpha=0.15)
nx.draw_networkx_nodes(G, pos, node_color=labels, cmap="tab10", node_size=60)
plt.title(f"Synthetic SBM graph: N={N}, K={K} communities (P_in={P_IN}, P_out={P_OUT})")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "graph_structure.png"), dpi=110)
plt.close()

# (b) Training curves: GCN vs MLP
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(gcn_history["train_loss"], label="GCN train loss")
axes[0].plot(mlp_history["train_loss"], label="MLP train loss")
axes[0].set_title("Training Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(gcn_history["test_acc"], label="GCN test acc")
axes[1].plot(mlp_history["test_acc"], label="MLP test acc")
axes[1].set_title("Test Accuracy over training"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=110)
plt.close()

# (c) Final GCN vs MLP bar comparison (train/val/test)
fig, ax = plt.subplots(figsize=(7, 5))
splits = ["train_acc", "val_acc", "test_acc"]
gcn_vals = [gcn_history[s][-1] for s in splits]
mlp_vals = [mlp_history[s][-1] for s in splits]
x = np.arange(len(splits))
width = 0.35
ax.bar(x - width / 2, gcn_vals, width, label="GCN")
ax.bar(x + width / 2, mlp_vals, width, label="MLP (graph-blind)")
ax.set_xticks(x); ax.set_xticklabels(["Train", "Val", "Test"])
ax.set_ylabel("Accuracy"); ax.set_title("GCN vs. graph-blind MLP baseline")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gcn_vs_mlp_comparison.png"), dpi=110)
plt.close()

# (d) Embedding evolution: project GCN hidden layer to 2D (PCA via SVD) at 4 checkpoints
fig, axes = plt.subplots(1, len(embed_snapshots), figsize=(5 * len(embed_snapshots), 5))
for i, (ep, emb) in enumerate(sorted(embed_snapshots.items())):
    emb_c = emb - emb.mean(axis=0)
    U, S, Vt = np.linalg.svd(emb_c, full_matrices=False)
    proj = emb_c @ Vt[:2].T
    sc = axes[i].scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=25, alpha=0.8)
    axes[i].set_title(f"epoch {ep}")
plt.suptitle("GCN hidden-layer embeddings over training (PCA-projected, colored by true community)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "embedding_evolution.png"), dpi=110)
plt.close()

print("\nSaved: graph_structure.png, training_curves.png, gcn_vs_mlp_comparison.png, embedding_evolution.png")
print("Topic 1 (GNN/GCN) run complete.")
