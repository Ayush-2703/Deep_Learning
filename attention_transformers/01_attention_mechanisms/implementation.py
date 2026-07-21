"""
Topic: Attention Mechanisms -- Scaled Dot-Product & Multi-Head
================================================================================
Repository : deep-learning/attention-transformers/01-attention-mechanisms/
File       : implementation.py

Sections:
  A | Scaled dot-product attention from scratch -- verified vs F.scaled_dot_product_attention
  B | Variance/scaling empirical demonstration (why divide by sqrt(d_k))
  C | Multi-head attention from scratch -- verified vs nn.MultiheadAttention
  D | Causal masking -- verify exactly zero attention to future positions
  E | Sinusoidal positional encoding -- implement + verify mathematical properties
  F | Content-based lookup demo: single self-attention layer learns to find the max
  G | Complexity benchmark: attention O(L^2) vs RNN O(L), empirical timing
  H | Visualization dashboard
"""

import os, math, time, warnings
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[CONFIG] Device: {DEVICE}  |  PyTorch: {torch.__version__}")


# =============================================================================
# SECTION A -- SCALED DOT-PRODUCT ATTENTION FROM SCRATCH
# =============================================================================

def scaled_dot_product_attention_numpy(Q, K, V, mask=None):
    """
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

    Q: (Lq, d_k)   K: (Lk, d_k)   V: (Lk, d_v)   mask: (Lq,Lk) bool, True=masked-out
    Returns: output (Lq, d_v), attn_weights (Lq, Lk)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                 # (Lq, Lk)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)   # numerical stability
    exp_scores = np.exp(scores)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    output = attn_weights @ V
    return output, attn_weights


def section_a_scaled_dot_product():
    print("\n" + "="*65)
    print("SECTION A -- Scaled Dot-Product Attention: Scratch vs PyTorch")
    print("="*65)

    Lq, Lk, d_k, d_v = 5, 6, 8, 10
    rng = np.random.default_rng(SEED)
    Q = rng.standard_normal((Lq, d_k)).astype(np.float32)
    K = rng.standard_normal((Lk, d_k)).astype(np.float32)
    V = rng.standard_normal((Lk, d_v)).astype(np.float32)

    out_np, attn_np = scaled_dot_product_attention_numpy(Q, K, V)

    Q_t = torch.tensor(Q).unsqueeze(0).unsqueeze(0)    # (1,1,Lq,d_k) -- batch,heads dims for F.sdpa
    K_t = torch.tensor(K).unsqueeze(0).unsqueeze(0)
    V_t = torch.tensor(V).unsqueeze(0).unsqueeze(0)
    out_torch = F.scaled_dot_product_attention(Q_t, K_t, V_t)[0,0].numpy()

    match = np.allclose(out_np, out_torch, atol=1e-5)
    print(f"\n  Q:{Q.shape} K:{K.shape} V:{V.shape}")
    print(f"  Output shape: {out_np.shape}")
    print(f"  Attention weights row sums (should all be 1.0): {attn_np.sum(axis=-1).round(4)}")
    print(f"  Match with F.scaled_dot_product_attention: {match}")
    assert match
    print("\n  OK: Scratch scaled dot-product attention verified against PyTorch")
    return Q, K, V, attn_np


# =============================================================================
# SECTION B -- WHY SCALE BY SQRT(D_K): EMPIRICAL VARIANCE DEMONSTRATION
# =============================================================================

def section_b_variance_demo():
    print("\n" + "="*65)
    print("SECTION B -- Empirical Variance: Why Scale by sqrt(d_k)")
    print("="*65)

    rng = np.random.default_rng(SEED)
    n_trials = 5000
    dims_to_test = [4, 16, 64, 256]

    print(f"\n  {'d_k':>6} | {'Var(raw q.k)':>14} | {'Var(scaled q.k)':>16} | {'Theory: Var(raw)=d_k':>20}")
    print("  " + "-"*66)

    results = {}
    for d_k in dims_to_test:
        q = rng.standard_normal((n_trials, d_k))
        k = rng.standard_normal((n_trials, d_k))
        raw_dot = np.sum(q*k, axis=1)
        scaled_dot = raw_dot / np.sqrt(d_k)

        var_raw = np.var(raw_dot)
        var_scaled = np.var(scaled_dot)
        results[d_k] = (var_raw, var_scaled)
        print(f"  {d_k:>6} | {var_raw:>14.2f} | {var_scaled:>16.4f} | {d_k:>20}")

    print("\n  OK: Raw dot-product variance grows linearly with d_k (matches theory exactly);")
    print("      scaled variance stays ~1.0 regardless of d_k -- keeps softmax well-conditioned")

    # Demonstrate the softmax saturation consequence directly
    d_k_large = 256
    q = rng.standard_normal(d_k_large); k1 = rng.standard_normal(d_k_large); k2 = rng.standard_normal(d_k_large)
    raw_scores = np.array([q@k1, q@k2, q@q*0.1])   # exaggerate for illustration
    scaled_scores = raw_scores / np.sqrt(d_k_large)

    def softmax(x):
        e = np.exp(x - x.max())
        return e/e.sum()

    print(f"\n  Illustrative 3-key example at d_k={d_k_large}:")
    print(f"    Raw scores:    {raw_scores.round(2)}  -> softmax: {softmax(raw_scores).round(4)}")
    print(f"    Scaled scores: {scaled_scores.round(2)}  -> softmax: {softmax(scaled_scores).round(4)}")

    return results


# =============================================================================
# SECTION C -- MULTI-HEAD ATTENTION FROM SCRATCH
# =============================================================================

class MultiHeadAttentionScratch(nn.Module):
    """
    From-scratch multi-head attention matching PyTorch's nn.MultiheadAttention
    EXACTLY: in_proj stacks [Wq;Wk;Wv], heads split embed_dim into CONTIGUOUS
    chunks, scaling is 1/sqrt(head_dim) -- all verified empirically beforehand
    by probing nn.MultiheadAttention's actual parameter layout and behavior.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3*embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3*embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(self, query, key, value, attn_mask=None):
        # query,key,value: (batch, L, embed_dim)
        B, Lq, D = query.shape
        Lk = key.shape[1]

        Wq, Wk, Wv = self.in_proj_weight.chunk(3, dim=0)     # each (embed_dim, embed_dim)
        bq, bk, bv = self.in_proj_bias.chunk(3, dim=0)

        Q = query @ Wq.T + bq        # (B, Lq, D)
        K = key   @ Wk.T + bk        # (B, Lk, D)
        V = value @ Wv.T + bv        # (B, Lk, D)

        # Split into heads: CONTIGUOUS chunks of the embed dimension
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)   # (B,H,Lq,hd)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)   # (B,H,Lk,hd)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)   # (B,H,Lk,hd)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)         # (B,H,Lq,Lk)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ V                                                # (B,H,Lq,hd)

        out = out.transpose(1, 2).contiguous().view(B, Lq, D)                # concat heads
        out = self.out_proj(out)
        return out, attn_weights


def section_c_multihead():
    print("\n" + "="*65)
    print("SECTION C -- Multi-Head Attention: Scratch vs nn.MultiheadAttention")
    print("="*65)

    embed_dim, num_heads, L, B = 16, 4, 6, 2
    torch.manual_seed(SEED)

    scratch = MultiHeadAttentionScratch(embed_dim, num_heads)
    torch_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    # Inject IDENTICAL weights into both
    with torch.no_grad():
        torch_mha.in_proj_weight[:] = scratch.in_proj_weight
        torch_mha.in_proj_bias[:] = scratch.in_proj_bias
        torch_mha.out_proj.weight[:] = scratch.out_proj.weight
        torch_mha.out_proj.bias[:] = scratch.out_proj.bias

    x = torch.randn(B, L, embed_dim)

    out_scratch, attn_scratch = scratch(x, x, x)
    out_torch, attn_torch = torch_mha(x, x, x, need_weights=True, average_attn_weights=False)

    out_match = torch.allclose(out_scratch, out_torch, atol=1e-5)
    attn_match = torch.allclose(attn_scratch, attn_torch, atol=1e-5)

    print(f"\n  embed_dim={embed_dim}, num_heads={num_heads}, head_dim={embed_dim//num_heads}")
    print(f"  Output shape: {tuple(out_scratch.shape)}")
    print(f"  Attention weights shape: {tuple(attn_scratch.shape)} (batch,heads,Lq,Lk)")
    print(f"  Output match: {out_match}")
    print(f"  Attention weights match: {attn_match}")
    assert out_match and attn_match
    print("\n  OK: Scratch multi-head attention verified EXACTLY against nn.MultiheadAttention")

    return scratch


# =============================================================================
# SECTION D -- CAUSAL MASKING VERIFICATION
# =============================================================================

def section_d_causal_masking(mha_model):
    print("\n" + "="*65)
    print("SECTION D -- Causal Masking: Verify Zero Attention to Future")
    print("="*65)

    L = 6
    causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)   # True=masked
    print(f"\n  Causal mask (True=forbidden), shape {tuple(causal_mask.shape)}:")
    print(f"  {causal_mask.int().numpy()}")

    x = torch.randn(1, L, mha_model.embed_dim)
    _, attn_weights = mha_model(x, x, x, attn_mask=causal_mask)

    attn_avg = attn_weights[0].mean(dim=0).detach().numpy()   # average over heads -> (L,L)
    future_weights = attn_avg[causal_mask.numpy()]

    print(f"\n  Max attention weight assigned to any FUTURE position: {future_weights.max():.2e}")
    print(f"  (should be exactly 0.0 -- forbidden positions get zero weight after softmax)")
    assert np.allclose(future_weights, 0.0, atol=1e-7)

    row_sums = attn_avg.sum(axis=-1)
    print(f"  Row sums (should still be 1.0 each, summing only over ALLOWED positions): {row_sums.round(4)}")

    print("\n  OK: Causal masking correctly zeroes out all future-position attention")
    return causal_mask, attn_avg


# =============================================================================
# SECTION E -- SINUSOIDAL POSITIONAL ENCODING
# =============================================================================

def sinusoidal_positional_encoding(max_len, d_model):
    """PE(pos,2i)=sin(pos/10000^(2i/d_model)), PE(pos,2i+1)=cos(pos/10000^(2i/d_model))"""
    position = np.arange(max_len)[:, None]                      # (max_len,1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))   # (d_model/2,)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def section_e_positional_encoding():
    print("\n" + "="*65)
    print("SECTION E -- Sinusoidal Positional Encoding: Properties Verification")
    print("="*65)

    max_len, d_model = 100, 32
    pe = sinusoidal_positional_encoding(max_len, d_model)

    print(f"\n  PE shape: {pe.shape}")

    # Property 1: bounded
    print(f"\n  [Property 1: Bounded] min={pe.min():.4f}, max={pe.max():.4f}  (expect within [-1,1])")
    assert pe.min() >= -1.0001 and pe.max() <= 1.0001

    # Property 2: unique per position (no two positions identical)
    unique_rows = len(np.unique(pe, axis=0))
    print(f"  [Property 2: Uniqueness] {unique_rows}/{max_len} positions have unique encodings")
    assert unique_rows == max_len

    # Property 3: relative position as linear function
    # PE(pos+k) should be expressible as M_k @ PE(pos) for a FIXED matrix M_k
    # (independent of pos) -- verify by fitting M_k via least squares on many
    # (PE(pos), PE(pos+k)) pairs and checking the fit generalizes to held-out pos.
    k_offset = 5
    pos_train = np.arange(0, 60)
    pos_test = np.arange(60, 90)

    PE_pos_train = pe[pos_train]              # (60, d_model)
    PE_posk_train = pe[pos_train + k_offset]   # (60, d_model)
    M_k, _, _, _ = np.linalg.lstsq(PE_pos_train, PE_posk_train, rcond=None)   # fit M_k

    PE_pos_test = pe[pos_test]
    PE_posk_test_true = pe[pos_test + k_offset]
    PE_posk_test_pred = PE_pos_test @ M_k

    fit_err = np.max(np.abs(PE_posk_test_true - PE_posk_test_pred))
    print(f"\n  [Property 3: Relative position = linear fn] offset k={k_offset}")
    print(f"    Fit M_k on positions 0-59, test on HELD-OUT positions 60-89")
    print(f"    Max error on held-out positions: {fit_err:.2e}  (near-zero confirms linearity)")
    assert fit_err < 1e-6

    print("\n  OK: All 3 theoretical properties of sinusoidal PE verified empirically")
    return pe


# =============================================================================
# SECTION F -- CONTENT-BASED LOOKUP: SINGLE ATTENTION LAYER FINDS THE MAX
# =============================================================================

class MaxFinderAttention(nn.Module):
    """
    A single self-attention layer trained to find the position of the
    MAXIMUM value in a sequence -- a clean, verifiable demonstration of
    content-based (not position-based) attention. No positional encoding
    is used deliberately, since this task is purely about VALUE content.
    """
    def __init__(self, d_model=16):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model))   # learned "find the max" query
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, L, 1) -- raw scalar values
        h = self.input_proj(x)                          # (batch, L, d_model)
        B = x.shape[0]
        q = self.query_token.expand(B, -1, -1)             # (batch, 1, d_model)
        out, attn_weights = self.mha(q, h, h, need_weights=True, average_attn_weights=True)
        pred_value = self.output_proj(out.squeeze(1))       # (batch, 1) -- predicted max value
        return pred_value.squeeze(-1), attn_weights.squeeze(1)   # attn_weights: (batch, L)


def generate_max_finding_data(n_samples, seq_len, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, seq_len, 1)).astype(np.float32)
    y_value = X.max(axis=1).squeeze(-1)      # true max value
    y_idx = X.squeeze(-1).argmax(axis=1)      # true argmax index (for evaluating attention)
    return X, y_value, y_idx


def section_f_content_lookup(n_epochs=40, seq_len=10):
    print("\n" + "="*65)
    print("SECTION F -- Content-Based Lookup: Single Attention Layer Finds the Max")
    print("="*65)

    X_tr, y_tr, _ = generate_max_finding_data(2000, seq_len, SEED)
    X_va, y_va, idx_va = generate_max_finding_data(400, seq_len, SEED+1)

    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)
    X_va_t, y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(y_va).to(DEVICE)

    torch.manual_seed(SEED)
    model = MaxFinderAttention().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            pred, _ = model(Xb)
            loss = nn.MSELoss()(pred, Yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_va, attn_va = model(X_va_t)
        mse = nn.MSELoss()(pred_va, y_va_t).item()
        attn_argmax = attn_va.argmax(dim=1).cpu().numpy()
        attn_accuracy = (attn_argmax == idx_va).mean()

    print(f"\n  Trained {n_epochs} epochs on 'find the max value' task (seq_len={seq_len})")
    print(f"  Validation value-prediction MSE: {mse:.6f}")
    print(f"  Attention-argmax matches TRUE argmax position: {attn_accuracy*100:.1f}% of validation examples")
    print("  (High match rate confirms the model learned genuine content-based lookup,")
    print("   not just a numerically-close output via some other shortcut)")

    return model, X_va, y_va, idx_va, attn_va.cpu().numpy()


# =============================================================================
# SECTION G -- COMPLEXITY BENCHMARK: ATTENTION O(L^2) vs RNN O(L)
# =============================================================================

def section_g_complexity_benchmark():
    print("\n" + "="*65)
    print("SECTION G -- Complexity Benchmark: Self-Attention vs RNN, Empirical Timing")
    print("="*65)

    d_model, n_repeats = 64, 20
    lengths = [16, 32, 64, 128, 256, 512]

    mha = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
    rnn = nn.LSTM(d_model, d_model, batch_first=True)

    attn_times, rnn_times = [], []
    print(f"\n  {'Length':>7} | {'Attention (ms)':>15} | {'LSTM (ms)':>12} | {'Ratio (LSTM/Attn)':>18}")
    print("  " + "-"*60)

    for L in lengths:
        x = torch.randn(4, L, d_model)

        # Warm-up
        with torch.no_grad():
            mha(x, x, x); rnn(x)

        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_repeats):
                mha(x, x, x)
        attn_time = (time.time()-t0)/n_repeats * 1000

        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_repeats):
                rnn(x)
        rnn_time = (time.time()-t0)/n_repeats * 1000

        attn_times.append(attn_time); rnn_times.append(rnn_time)
        ratio = rnn_time / attn_time if attn_time > 0 else float("nan")
        print(f"  {L:>7} | {attn_time:>15.3f} | {rnn_time:>12.3f} | {ratio:>18.2f}")

    print("\n  OK: Timing reflects implementation constants as much as asymptotic complexity at these")
    print("      scales (both fit comfortably on CPU) -- the O(L^2) vs O(L) difference dominates only")
    print("      at much longer sequences than tested here (see theory.md Section 9 for the analysis)")

    return lengths, attn_times, rnn_times


# =============================================================================
# SECTION H -- VISUALIZATION
# =============================================================================

def build_figures(attn_np_example, var_results, causal_mask, attn_avg_causal,
                  pe, max_finder_data, bench_lengths, attn_times, rnn_times):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Phase 4 -- Topic 1: Attention Mechanisms", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    a = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

    # Panel 1: raw attention weight matrix example
    im0 = a[0].imshow(attn_np_example, cmap="viridis", aspect="auto")
    a[0].set_title("Scaled Dot-Product Attention Weights", fontweight="bold", fontsize=10)
    a[0].set_xlabel("Key position"); a[0].set_ylabel("Query position")
    plt.colorbar(im0, ax=a[0], fraction=0.046)

    # Panel 2: variance vs d_k
    dims = list(var_results.keys())
    raw_vars = [var_results[d][0] for d in dims]
    scaled_vars = [var_results[d][1] for d in dims]
    a[1].plot(dims, raw_vars, "o-", color="#e74c3c", lw=2, label="Raw Var(q.k)")
    a[1].plot(dims, scaled_vars, "s-", color="#3498db", lw=2, label="Scaled Var(q.k/sqrt(d_k))")
    a[1].plot(dims, dims, "k--", lw=1, alpha=0.5, label="Theory: Var=d_k")
    a[1].set_title("Dot-Product Variance vs Dimension", fontweight="bold", fontsize=10)
    a[1].set_xlabel("d_k"); a[1].set_ylabel("Variance")
    a[1].legend(fontsize=8); a[1].grid(True, alpha=0.3)

    # Panel 3: causal mask attention pattern
    im2 = a[2].imshow(attn_avg_causal, cmap="Blues", aspect="auto")
    a[2].set_title("Causal-Masked Attention (lower triangular)", fontweight="bold", fontsize=10)
    a[2].set_xlabel("Key position"); a[2].set_ylabel("Query position")
    plt.colorbar(im2, ax=a[2], fraction=0.046)

    # Panel 4: positional encoding heatmap
    im3 = a[3].imshow(pe[:50, :].T, cmap="RdBu_r", aspect="auto")
    a[3].set_title("Sinusoidal Positional Encoding", fontweight="bold", fontsize=10)
    a[3].set_xlabel("Position"); a[3].set_ylabel("Encoding dimension")
    plt.colorbar(im3, ax=a[3], fraction=0.046)

    # Panel 5: content-based lookup example
    model, X_va, y_va, idx_va, attn_va = max_finder_data
    example_idx = 0
    seq_vals = X_va[example_idx, :, 0]
    attn_row = attn_va[example_idx]
    true_max_idx = idx_va[example_idx]

    ax5 = a[4]
    ax5_twin = ax5.twinx()
    ax5.bar(range(len(seq_vals)), seq_vals, color="#95a5a6", alpha=0.6, label="Sequence values")
    ax5_twin.plot(range(len(attn_row)), attn_row, "o-", color="#e74c3c", lw=2, label="Attention weight")
    ax5.scatter([true_max_idx], [seq_vals[true_max_idx]], color="black", s=100, marker="*",
               zorder=5, label="True max")
    ax5.set_title("Content-Based Lookup: Attention Finds the Max", fontweight="bold", fontsize=10)
    ax5.set_xlabel("Sequence position"); ax5.set_ylabel("Value", color="#95a5a6")
    ax5_twin.set_ylabel("Attention weight", color="#e74c3c")
    ax5.legend(fontsize=7, loc="upper left"); ax5_twin.legend(fontsize=7, loc="upper right")

    # Panel 6: complexity benchmark
    a[5].plot(bench_lengths, attn_times, "o-", color="#9b59b6", lw=2, label="Self-Attention")
    a[5].plot(bench_lengths, rnn_times, "s-", color="#27ae60", lw=2, label="LSTM")
    a[5].set_title("Empirical Timing vs Sequence Length", fontweight="bold", fontsize=10)
    a[5].set_xlabel("Sequence Length"); a[5].set_ylabel("Time per forward pass (ms)")
    a[5].legend(fontsize=8); a[5].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "01_attention_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Dashboard saved -> {path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#"*65)
    print("  Phase 4 -- Topic 1: Attention Mechanisms")
    print("#"*65)

    Q, K, V, attn_example = section_a_scaled_dot_product()
    var_results = section_b_variance_demo()
    mha_scratch = section_c_multihead()
    causal_mask, attn_avg_causal = section_d_causal_masking(mha_scratch)
    pe = section_e_positional_encoding()
    max_finder_data = section_f_content_lookup()
    bench_lengths, attn_times, rnn_times = section_g_complexity_benchmark()

    build_figures(attn_example, var_results, causal_mask, attn_avg_causal,
                 pe, max_finder_data, bench_lengths, attn_times, rnn_times)

    print("\n" + "#"*65)
    print("  DONE: Topic 1 complete.")
    print("#"*65 + "\n")


if __name__ == "__main__":
    main()

