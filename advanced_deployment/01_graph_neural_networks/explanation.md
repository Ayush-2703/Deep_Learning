# Explanation: GCN Implementation Walkthrough

## 1. Building the synthetic stochastic block model

```python
for i in range(N):
    for j in range(i + 1, N):
        p = P_IN if labels[i] == labels[j] else P_OUT
        if np.random.rand() < p:
            A[i, j] = A[j, i] = 1.0
```

`P_IN=0.35` (edge probability within a community) is deliberately much
higher than `P_OUT=0.02` (across communities) — this is what gives the
graph genuine, exploitable community structure. The printed sanity check
(`avg degree=12.63, isolated nodes=0`) confirms the generated graph is
well-connected, not degenerate (an all-isolated or all-connected graph
would make this a meaningless test).

## 2. Deliberately weak node features — this is the whole point of the experiment

```python
X_feat[i, labels[i] % feature_dim] += 0.6  # weak, not dominant, bump
```

The features get only a small class-correlated nudge on top of pure random
noise. This is intentional: if the node features alone were strongly
separable, an MLP baseline would do fine without ever looking at the graph,
and the comparison would prove nothing about GNNs specifically. Keeping the
per-node signal weak forces any model that succeeds to actually exploit the
*graph structure* (who's connected to whom), not just the raw features —
which is exactly what the GCN vs. MLP comparison is designed to isolate.

## 3. `A_norm` — the symmetric normalization, computed once up front

```python
A_hat = A + np.eye(N, dtype=np.float32)
D_hat = A_hat.sum(axis=1)
D_hat_inv_sqrt = np.diag(1.0 / np.sqrt(D_hat))
A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
```

This exactly implements theory.md section 3's formula. It's computed once,
outside the training loop, since the graph structure is fixed — only the
node features and weights change during training. `A_norm` is then reused
as a constant in every forward pass (`A_norm @ H`).

## 4. `GCNLayer.forward` — message passing in one matrix multiply

```python
agg = A_norm @ H       # [N, in_dim]: each row = symmetric-normalized sum of neighbor features
return self.linear(agg)
```

`A_norm @ H` is doing all the "message passing" work in a single dense
matrix multiply: row `v` of the result is
`sum_u( A_norm[v,u] * H[u] )` — exactly the weighted neighbor-aggregation
from theory.md section 2/3, vectorized across all `N` nodes simultaneously
rather than looping node-by-node. This is why GCNs scale well: the whole
layer is one matmul, not an explicit per-node loop.

## 5. Semi-supervised split — only 15% of labels are ever used for training

```python
train_frac = 0.15
```

This mirrors the real GCN use case (Kipf & Welling's original citation-
network benchmarks used a similarly small labeled fraction): most nodes'
labels are hidden, and the model must generalize to the other 85% using
graph structure + the few labeled examples. `val_idx`/`test_idx` are
strictly disjoint index sets from `train_idx`, so the reported test
accuracy is on nodes the model never saw a label for.

## 6. The actual result — GCN vs. MLP, and why the gap is so large

```
Final TEST accuracy -- GCN: 0.984 | MLP (no graph): 0.419
```

This is a genuinely large, honest gap (not cherry-picked — it's the
natural consequence of the experimental design in section 2 above): the
MLP baseline has access to *identical* node features and *identical*
training labels, but no way to see the graph. It ends up only slightly
better than random guessing (1/K = 0.25 for K=4 classes), because the
injected feature signal was deliberately weak. The GCN, seeing the exact
same weak features *plus* the graph structure, reaches near-perfect
accuracy — direct, concrete evidence that message passing over the graph
recovers signal that isn't present in any single node's features alone,
which is the entire premise of GNNs from theory.md section 1.

## 7. `embedding_evolution.png` — watching representations separate over training

```python
if is_gcn and epoch in (1, 50, 100, 200):
    embed_snapshots[epoch] = h_eval.detach().numpy().copy()
```

Captures the GCN's hidden-layer representation (`h1`, post-first-GCN-layer,
pre-classification) at four points during training, each projected to 2D
via SVD/PCA. At epoch 1 the four communities should appear as a jumbled,
overlapping cloud (the network hasn't learned anything yet); by epoch 200
they should form four visually distinct, well-separated clusters — a
direct visual confirmation that the message-passing mechanism is doing
exactly what theory.md describes: pulling same-community nodes' hidden
representations together through repeated neighbor aggregation.

## 8. Why `MLPBaseline` is architecturally identical except for the graph

```python
class MLPBaseline(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
```

Same input dimension, same hidden width (16), same output classes, same
dropout rate, same optimizer settings, same number of training epochs as
the GCN. The *only* difference is `self.fc1(X)` (each node processed
independently) vs. `self.gc1(A_norm, X)` (each node's update mixes in
neighbor information). This tight control is what makes the final
comparison a fair, isolated test of what the graph structure specifically
contributes — not a confound of model capacity or training budget.
