# Graph Neural Networks (GNNs)

## 1. Why graphs need a different architecture

CNNs assume a fixed grid structure (pixels have a consistent notion of
"neighbor above/below/left/right"). RNNs/Transformers assume a sequence
(fixed notion of "before/after"). Graphs have neither: each node can have a
different number of neighbors, in no particular order, and the graph
structure itself is part of the input. A GNN must be:

- **Permutation-invariant**: relabeling nodes 1..N in any order shouldn't
  change what the network computes for a given node.
- **Able to handle variable node degree**: node 3 might have 2 neighbors,
  node 7 might have 20.

## 2. Message passing — the general GNN framework

Almost all GNN variants (GCN, GraphSAGE, GAT, etc.) are instances of
**message passing**: at each layer, every node aggregates information from
its neighbors, combines it with its own current representation, and
produces an updated representation.

```
h_v^(l+1) = UPDATE( h_v^(l),  AGGREGATE( { h_u^(l) : u in N(v) } ) )
```

- `h_v^(l)`: node `v`'s feature vector at layer `l`
- `N(v)`: the set of `v`'s neighbors
- `AGGREGATE`: a permutation-invariant function (sum, mean, max) — this is
  what guarantees the permutation-invariance property above
- `UPDATE`: typically a linear layer + nonlinearity

Stacking `L` message-passing layers lets information propagate up to `L`
hops away from each node — after `L` layers, `h_v^(L)` has "seen"
everything within `L` hops of `v`.

## 3. Graph Convolutional Network (GCN) — the specific variant used here

Kipf & Welling (2017) formulate this as a single, efficient matrix
operation across the whole graph at once:

```
H^(l+1) = sigma( D_hat^(-1/2) * A_hat * D_hat^(-1/2) * H^(l) * W^(l) )
```

- `A_hat = A + I`: adjacency matrix with self-loops added (so a node's own
  features are included in its own update, not just its neighbors')
- `D_hat`: the degree matrix of `A_hat` (diagonal matrix of node degrees)
- `D_hat^(-1/2) * A_hat * D_hat^(-1/2)`: the **symmetric normalization** —
  without it, high-degree nodes would dominate purely because they sum over
  more neighbors, not because their neighbors are more informative
- `H^(l)`: matrix of all node features at layer `l`, shape `[N, d_l]`
- `W^(l)`: learned weight matrix, shape `[d_l, d_{l+1}]`
- `sigma`: nonlinearity (ReLU)

This is exactly the message-passing framework above, with `AGGREGATE` =
symmetric-normalized sum over neighbors (including self), and `UPDATE` =
linear transform + ReLU.

## 4. Why the normalization matters (concretely)

Without `D_hat^(-1/2) * ... * D_hat^(-1/2)`, a node with 50 neighbors would
have a feature magnitude roughly 50x a node with 1 neighbor after a plain
sum-aggregation, purely from summing more terms — this would swamp the
actual learned signal and make training unstable. The symmetric
normalization rescales each edge's contribution by
`1 / sqrt(deg(v) * deg(u))`, which keeps aggregated magnitudes comparable
across nodes regardless of degree.

## 5. Task used in this implementation: node classification

Given a graph where nodes belong to unknown communities/classes, and only a
few nodes' labels are known (semi-supervised setting), predict the class of
every other node using the graph structure. This is the canonical GCN
benchmark task (originally demonstrated on citation networks like Cora).

Here, the graph is a synthetic **stochastic block model** (SBM): nodes are
assigned to `K` communities, and edges are added with high probability
between nodes in the same community and low probability between different
communities. This gives the GNN genuine structure to exploit — nodes should
be classifiable largely by "who they're connected to," which is exactly
what message passing is designed to capture, as opposed to a bag of
unconnected feature vectors where only node features (not graph structure)
would matter.

## 6. What a GNN gets that an MLP-on-node-features doesn't

An MLP applied independently to each node's own feature vector (ignoring
the graph) has no way to use the fact that a node's neighbors are mostly in
community 2, say. The GCN layer explicitly mixes in neighbor information at
every layer, so even nodes with uninformative or noisy own-features can be
classified correctly purely from their neighborhood's identity — this
implementation verifies that claim directly by comparing a GCN against an
identical-depth MLP baseline trained on the same node features with the
graph structure hidden from it.
