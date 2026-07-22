"""
Topic: The Transformer Architecture
======================================================
Repository : deep-learning/attention-transformers/02-transformer-architecture/
File       : implementation.py

Task: same character-level digit-sequence REVERSAL task as Phase 3 Topic 3,
enabling a DIRECT comparison between the RNN-based Seq2Seq+Attention model
and this fully attention-based Transformer.

Sections:
  A | LayerNorm from scratch -- verified against nn.LayerNorm
  B | Position-wise FFN + Sinusoidal Positional Encoding module
  C | Encoder Layer & Decoder Layer (Pre-LN and Post-LN variants)
  D | Full Transformer assembly (embeddings, encoder/decoder stacks, output head)
  E | Parallel teacher-forcing training (single forward pass, THE key advantage over RNN Seq2Seq)
  F | Core experiment: accuracy vs source length -- Transformer vs Phase 3's Seq2Seq+Attention
  G | Pre-LN vs Post-LN training stability comparison
  H | Cross-attention visualization
  I | Visualization dashboard
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

PAD_IDX, SOS_IDX, EOS_IDX = 10, 11, 12
VOCAB_SIZE = 13


# =============================================================================
# SECTION A -- LAYERNORM FROM SCRATCH
# =============================================================================

class LayerNormScratch(nn.Module):
    """LayerNorm(x) = gamma*(x-mu)/sqrt(var+eps) + beta, stats over LAST dim."""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


def section_a_layernorm():
    print("\n" + "="*65)
    print("SECTION A -- LayerNorm From Scratch vs nn.LayerNorm")
    print("="*65)

    d_model = 16
    torch.manual_seed(SEED)
    scratch = LayerNormScratch(d_model)
    torch_ln = nn.LayerNorm(d_model)

    with torch.no_grad():
        torch_ln.weight[:] = scratch.gamma
        torch_ln.bias[:] = scratch.beta

    x = torch.randn(4, 6, d_model) * 5 + 2   # arbitrary scale/shift to stress-test normalization

    out_scratch = scratch(x)
    out_torch = torch_ln(x)

    match = torch.allclose(out_scratch, out_torch, atol=1e-5)
    print(f"\n  Input shape: {tuple(x.shape)}")
    print(f"  Post-norm mean (should be ~0): {out_scratch.mean(dim=-1)[0,0].item():.6f}")
    print(f"  Post-norm std (should be ~1):  {out_scratch.std(dim=-1, unbiased=False)[0,0].item():.6f}")
    print(f"  Match with nn.LayerNorm: {match}")
    assert match
    print("\n  OK: Scratch LayerNorm verified against nn.LayerNorm")


# =============================================================================
# SECTION B -- POSITION-WISE FFN + POSITIONAL ENCODING
# =============================================================================

class PositionwiseFFN(nn.Module):
    """FFN(x) = max(0, xW1+b1)W2+b2 -- applied IDENTICALLY and INDEPENDENTLY at every position."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, L, d_model)
        return x + self.pe[:x.size(1)].unsqueeze(0)


def section_b_ffn_pe_demo():
    print("\n" + "="*65)
    print("SECTION B -- Position-wise FFN + Positional Encoding")
    print("="*65)

    d_model, d_ff, L, B = 16, 64, 5, 2
    ffn = PositionwiseFFN(d_model, d_ff)
    pe = PositionalEncoding(d_model)

    x = torch.randn(B, L, d_model)
    ffn_out = ffn(x)
    pe_out = pe(x)

    # Verify FFN is applied IDENTICALLY per-position: same weights used for
    # every position, so applying it to one position in isolation should
    # match applying it to the full sequence and slicing that position out.
    single_pos_out = ffn(x[:, 2:3, :])
    match_positionwise = torch.allclose(ffn_out[:, 2:3, :], single_pos_out, atol=1e-6)

    print(f"\n  FFN output shape: {tuple(ffn_out.shape)}")
    print(f"  PE output shape:  {tuple(pe_out.shape)}")
    print(f"  Position-wise independence verified (isolated pos matches full-seq slice): {match_positionwise}")
    assert match_positionwise
    print("\n  OK: FFN applies identically per-position; PE adds fixed position signal")


# =============================================================================
# SECTION C -- ENCODER & DECODER LAYERS (Pre-LN / Post-LN)
# =============================================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, pre_ln=True):
        super().__init__()
        self.pre_ln = pre_ln
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.pre_ln:
            h = self.norm1(x)
            attn_out, _ = self.self_attn(h, h, h)
            x = x + attn_out
            h = self.norm2(x)
            x = x + self.ffn(h)
        else:
            attn_out, _ = self.self_attn(x, x, x)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.ffn(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, pre_ln=True):
        super().__init__()
        self.pre_ln = pre_ln
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, causal_mask):
        if self.pre_ln:
            h = self.norm1(x)
            attn_out, _ = self.self_attn(h, h, h, attn_mask=causal_mask)
            x = x + attn_out
            h = self.norm2(x)
            cross_out, cross_attn_w = self.cross_attn(h, enc_out, enc_out, need_weights=True, average_attn_weights=True)
            x = x + cross_out
            h = self.norm3(x)
            x = x + self.ffn(h)
        else:
            attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
            x = self.norm1(x + attn_out)
            cross_out, cross_attn_w = self.cross_attn(x, enc_out, enc_out, need_weights=True, average_attn_weights=True)
            x = self.norm2(x + cross_out)
            x = self.norm3(x + self.ffn(x))
        return x, cross_attn_w


def section_c_layer_shapes():
    print("\n" + "="*65)
    print("SECTION C -- Encoder/Decoder Layer Shape Verification")
    print("="*65)

    d_model, num_heads, d_ff, Ls, Lt, B = 32, 4, 64, 7, 5, 2
    enc_layer = EncoderLayer(d_model, num_heads, d_ff, pre_ln=True)
    dec_layer = DecoderLayer(d_model, num_heads, d_ff, pre_ln=True)

    src = torch.randn(B, Ls, d_model)
    tgt = torch.randn(B, Lt, d_model)
    causal_mask = torch.triu(torch.ones(Lt, Lt, dtype=torch.bool), diagonal=1)

    enc_out = enc_layer(src)
    dec_out, cross_attn = dec_layer(tgt, enc_out, causal_mask)

    print(f"\n  Encoder: input {tuple(src.shape)} -> output {tuple(enc_out.shape)}")
    print(f"  Decoder: input {tuple(tgt.shape)} + enc_out {tuple(enc_out.shape)} -> output {tuple(dec_out.shape)}")
    print(f"  Cross-attention weights shape: {tuple(cross_attn.shape)} (batch, Lt_query, Ls_key)")
    assert enc_out.shape == src.shape
    assert dec_out.shape == tgt.shape
    assert cross_attn.shape == (B, Lt, Ls)
    print("\n  OK: Shapes correct -- encoder preserves shape, decoder cross-attends to (different-length) encoder output")


# =============================================================================
# SECTION D -- FULL TRANSFORMER ASSEMBLY
# =============================================================================

class Transformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, num_heads=4, d_ff=256,
                num_layers=2, max_len=40, pre_ln=True):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, pre_ln) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, pre_ln) for _ in range(num_layers)])
        self.final_norm_enc = nn.LayerNorm(d_model) if pre_ln else nn.Identity()
        self.final_norm_dec = nn.LayerNorm(d_model) if pre_ln else nn.Identity()

        self.output_proj = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.final_norm_enc(x)

    def decode(self, tgt, enc_out, causal_mask):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        cross_attn = None
        for layer in self.decoder_layers:
            x, cross_attn = layer(x, enc_out, causal_mask)
        x = self.final_norm_dec(x)
        return self.output_proj(x), cross_attn

    def forward(self, src, tgt_input):
        # tgt_input: decoder input (teacher-forced) -- ALL positions processed IN PARALLEL
        Lt = tgt_input.size(1)
        causal_mask = torch.triu(torch.ones(Lt, Lt, dtype=torch.bool, device=src.device), diagonal=1)
        enc_out = self.encode(src)
        logits, cross_attn = self.decode(tgt_input, enc_out, causal_mask)
        return logits, cross_attn

    @torch.no_grad()
    def greedy_decode(self, src, max_len=30):
        self.eval()
        enc_out = self.encode(src)
        tokens = torch.full((src.size(0), 1), SOS_IDX, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            Lt = tokens.size(1)
            causal_mask = torch.triu(torch.ones(Lt, Lt, dtype=torch.bool, device=src.device), diagonal=1)
            logits, _ = self.decode(tokens, enc_out, causal_mask)
            next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == EOS_IDX).all():
                break
        return tokens[:, 1:]    # drop SOS


def section_d_full_model_demo():
    print("\n" + "="*65)
    print("SECTION D -- Full Transformer: Forward Pass Demo")
    print("="*65)

    torch.manual_seed(SEED)
    model = Transformer(num_layers=2)
    src = torch.randint(0, 10, (2, 8))
    tgt_input = torch.randint(0, 10, (2, 6))

    logits, cross_attn = model(src, tgt_input)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n  Total parameters: {n_params:,}")
    print(f"  Source shape: {tuple(src.shape)}  Decoder-input shape: {tuple(tgt_input.shape)}")
    print(f"  Output logits shape: {tuple(logits.shape)} (batch, Lt, vocab_size)")
    assert logits.shape == (2, 6, VOCAB_SIZE)
    print("\n  OK: Full encoder-decoder Transformer forward pass produces correct output shape")


# =============================================================================
# SECTION E -- DATA + PARALLEL TEACHER-FORCING TRAINING
# =============================================================================

def generate_reversal_data(n_samples, src_len, seed):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, 10, size=(n_samples, src_len)).astype(np.int64)
    reversed_src = src[:, ::-1].copy()
    tgt = np.concatenate([
        np.full((n_samples, 1), SOS_IDX, dtype=np.int64),
        reversed_src,
        np.full((n_samples, 1), EOS_IDX, dtype=np.int64),
    ], axis=1)
    return src, tgt


def train_epoch(model, loader, opt):
    model.train()
    total_loss, total_correct, total_tok = 0.0, 0, 0
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]     # SOS + reversed source (everything but final EOS)
        tgt_output = tgt[:, 1:]      # reversed source + EOS (everything but initial SOS)

        # KEY ADVANTAGE: entire target sequence processed in ONE forward pass
        # (causal mask enforces correct information flow) -- no token-by-token
        # Python loop needed during TRAINING, unlike the RNN decoder (Phase 3 Topic 3)
        logits, _ = model(src, tgt_input)
        loss = crit(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(-1) == tgt_output).sum().item()
        total_tok += tgt_output.numel()

    return total_loss/len(loader), total_correct/total_tok


def _truncate_at_eos(seq_1d):
    """Truncate a 1D token list at its FIRST EOS (inclusive), dropping any
    trailing tokens generated after EOS -- necessary because BATCHED greedy
    decoding only stops the whole batch's loop once EVERY sequence has
    produced EOS, so sequences that finish EARLY keep being fed through
    the model (and can emit further, semantically-meaningless tokens)
    until the LAST sequence in the batch also finishes."""
    seq_list = seq_1d.tolist()
    if EOS_IDX in seq_list:
        return seq_list[:seq_list.index(EOS_IDX)+1]
    return seq_list


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_exact, total_seqs = 0, 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        pred = model.greedy_decode(src, max_len=tgt.size(1)+5)
        true_out = tgt[:, 1:]

        for i in range(src.size(0)):
            pred_trunc = _truncate_at_eos(pred[i])
            true_trunc = _truncate_at_eos(true_out[i])
            if pred_trunc == true_trunc:
                total_exact += 1
        total_seqs += src.size(0)
    return total_exact/total_seqs


def train_transformer(src_len, n_epochs=25, n_train=600, n_val=150, pre_ln=True, num_layers=2):
    X_tr, Y_tr = generate_reversal_data(n_train, src_len, SEED)
    X_va, Y_va = generate_reversal_data(n_val, src_len, SEED+1)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(Y_va)), batch_size=32, shuffle=False)

    torch.manual_seed(SEED)
    model = Transformer(num_layers=num_layers, pre_ln=pre_ln, max_len=src_len+15).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "train_acc": []}
    for epoch in range(n_epochs):
        tl, ta = train_epoch(model, train_loader, opt)
        history["train_loss"].append(tl); history["train_acc"].append(ta)

    exact_acc = evaluate(model, val_loader)
    return model, history, exact_acc


# =============================================================================
# SECTION F -- CORE EXPERIMENT: TRANSFORMER vs PHASE 3 SEQ2SEQ+ATTENTION
# =============================================================================

def section_f_length_experiment():
    print("\n" + "="*65)
    print("SECTION F -- Transformer Accuracy vs Source Length")
    print("  (compare directly against Phase 3 Topic 3's Seq2Seq+Attention results)")
    print("="*65)

    lengths = [5, 10, 15, 20, 25]
    exact_accs = []
    times = []

    print(f"\n  {'Length':>7} | {'Exact-Match Acc':>16} | {'Train Time (s)':>14}")
    print("  " + "-"*44)

    saved_models = {}
    for L in lengths:
        t0 = time.time()
        model, hist, exact_acc = train_transformer(L, n_epochs=80)
        elapsed = time.time() - t0
        exact_accs.append(exact_acc); times.append(elapsed)
        saved_models[L] = model
        print(f"  {L:>7} | {exact_acc*100:>15.1f}% | {elapsed:>14.1f}")

    print("\n  Phase 3 Topic 3 Seq2Seq+Attention reference (RNN-based, for comparison):")
    print("    Reversal task exact-match was ~99.0% (trained 30 epochs, similar data budget)")

    return lengths, exact_accs, times, saved_models


# =============================================================================
# SECTION G -- PRE-LN vs POST-LN TRAINING STABILITY
# =============================================================================

def section_g_preln_postln():
    print("\n" + "="*65)
    print("SECTION G -- Pre-LN vs Post-LN Training Stability")
    print("="*65)

    src_len = 15
    print(f"\n  Training BOTH variants at source_len={src_len}, num_layers=4 (deeper -- stresses LN placement)")

    _, hist_pre, acc_pre = train_transformer(src_len, n_epochs=25, pre_ln=True, num_layers=4)
    _, hist_post, acc_post = train_transformer(src_len, n_epochs=25, pre_ln=False, num_layers=4)

    print(f"\n  Pre-LN:  final train_loss={hist_pre['train_loss'][-1]:.4f}  exact_acc={acc_pre*100:.1f}%")
    print(f"  Post-LN: final train_loss={hist_post['train_loss'][-1]:.4f}  exact_acc={acc_post*100:.1f}%")

    loss_std_pre = np.std(hist_pre["train_loss"][5:])    # stability AFTER initial descent
    loss_std_post = np.std(hist_post["train_loss"][5:])
    print(f"\n  Loss std-dev (epochs 5+, lower=more stable): Pre-LN={loss_std_pre:.4f}  Post-LN={loss_std_post:.4f}")

    return hist_pre, hist_post, acc_pre, acc_post


# =============================================================================
# SECTION H -- CROSS-ATTENTION VISUALIZATION
# =============================================================================

@torch.no_grad()
def get_cross_attention(model, src):
    model.eval()
    enc_out = model.encode(src)
    tokens = torch.full((1, 1), SOS_IDX, dtype=torch.long, device=src.device)
    all_cross_attn = []
    for _ in range(src.size(1) + 2):
        Lt = tokens.size(1)
        causal_mask = torch.triu(torch.ones(Lt, Lt, dtype=torch.bool, device=src.device), diagonal=1)
        logits, cross_attn = model.decode(tokens, enc_out, causal_mask)
        next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
        tokens = torch.cat([tokens, next_tok], dim=1)
        if next_tok.item() == EOS_IDX:
            break
    final_cross_attn = cross_attn[0].cpu().numpy()   # (Lt, Ls) -- from the LAST decode call, all positions
    return tokens[0, 1:].cpu().numpy(), final_cross_attn


def section_h_cross_attention_viz(saved_models, example_len=15):
    print("\n" + "="*65)
    print("SECTION H -- Cross-Attention Visualization")
    print("="*65)

    model = saved_models[example_len]
    src, _ = generate_reversal_data(1, example_len, seed=999)
    src_t = torch.tensor(src).to(DEVICE)

    pred_tokens, cross_attn = get_cross_attention(model, src_t)
    pred_str = "".join(str(t) for t in pred_tokens if t not in (EOS_IDX, PAD_IDX))
    true_str = "".join(str(d) for d in src[0][::-1])

    print(f"\n  Source: {''.join(str(d) for d in src[0])}")
    print(f"  Predicted: {pred_str}")
    print(f"  True reversed: {true_str}")
    print(f"  Exact match: {pred_str == true_str}")
    print(f"  Cross-attention matrix shape: {cross_attn.shape}")

    return src[0], pred_tokens, cross_attn


# =============================================================================
# SECTION I -- VISUALIZATION
# =============================================================================

def build_figures(lengths, exact_accs, times, hist_pre, hist_post, acc_pre, acc_post,
                  src_ex, pred_tokens, cross_attn):
    fig = plt.figure(figsize=(17, 10))
    fig.suptitle("Phase 4 -- Topic 2: The Transformer Architecture", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    a1, a2, a3, a4 = (fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
                      fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]))

    a1.plot(lengths, [v*100 for v in exact_accs], "o-", color="#9b59b6", lw=2, ms=7)
    a1.axhline(99.0, color="gray", ls="--", lw=1, label="Phase 3 Seq2Seq+Attn (RNN) reference: ~99%")
    a1.set_title("Transformer Exact-Match Accuracy vs Source Length", fontweight="bold", fontsize=10)
    a1.set_xlabel("Source Sequence Length"); a1.set_ylabel("Exact-Match Accuracy (%)")
    a1.legend(fontsize=8); a1.grid(True, alpha=0.3); a1.set_ylim(-5, 105)

    ep = range(1, len(hist_pre["train_loss"])+1)
    a2.plot(ep, hist_pre["train_loss"], color="#3498db", lw=2, label=f"Pre-LN (acc={acc_pre*100:.0f}%)")
    a2.plot(ep, hist_post["train_loss"], color="#e74c3c", lw=2, label=f"Post-LN (acc={acc_post*100:.0f}%)")
    a2.set_title("Pre-LN vs Post-LN Training Loss (4 layers)", fontweight="bold", fontsize=10)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Train Loss")
    a2.legend(fontsize=8); a2.grid(True, alpha=0.3)

    im = a3.imshow(cross_attn, cmap="viridis", aspect="auto")
    a3.set_title("Decoder Cross-Attention: Reversal Task", fontweight="bold", fontsize=10)
    a3.set_xlabel("Source position"); a3.set_ylabel("Decoding step")
    a3.set_xticks(range(len(src_ex))); a3.set_xticklabels([str(d) for d in src_ex], fontsize=7)
    plt.colorbar(im, ax=a3, fraction=0.046)

    a4.bar([str(l) for l in lengths], times, color="#27ae60")
    a4.set_title("Transformer Training Time vs Source Length", fontweight="bold", fontsize=10)
    a4.set_xlabel("Source Sequence Length"); a4.set_ylabel("Training Time (s, 25 epochs)")
    a4.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "02_transformer_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Dashboard saved -> {path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#"*65)
    print("  Phase 4 -- Topic 2: The Transformer Architecture")
    print("#"*65)

    section_a_layernorm()
    section_b_ffn_pe_demo()
    section_c_layer_shapes()
    section_d_full_model_demo()

    lengths, exact_accs, times, saved_models = section_f_length_experiment()
    hist_pre, hist_post, acc_pre, acc_post = section_g_preln_postln()
    src_ex, pred_tokens, cross_attn = section_h_cross_attention_viz(saved_models, example_len=15)

    build_figures(lengths, exact_accs, times, hist_pre, hist_post, acc_pre, acc_post,
                 src_ex, pred_tokens, cross_attn)

    print("\n" + "#"*65)
    print("  DONE: Topic 2 complete.")
    print("#"*65 + "\n")


if __name__ == "__main__":
    main()
