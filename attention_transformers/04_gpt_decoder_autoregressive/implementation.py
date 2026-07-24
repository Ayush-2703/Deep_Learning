"""
Phase 4 -- Topic 4: GPT -- Decoder-Only Autoregressive Generation
======================================================================
Repository : deep-learning-mastery/phase-4-attention-transformers/04-gpt-decoder-autoregressive/
File       : implementation.py

Synthetic task: a noisy "counting" Markov chain -- token i is followed by
(i+1)%10 with 80% probability, else a uniformly random token. This gives
next-token prediction genuine SEQUENTIAL structure to learn (unlike Topic
3's i.i.d.-within-topic corpus), appropriate for testing a causal LM.

Sections:
  A | GPT decoder block (causal self-attention + FFN, NO cross-attention)
  B | Full GPT model assembly
  C | Synthetic Markov-chain "counting" corpus
  D | Next-token prediction training (parallel via causal mask)
  E | Perplexity evaluation
  F | Autoregressive generation: greedy / temperature / top-k sampling
  G | Sampling strategy comparison (diversity vs quality)
  H | Training speed: GPT (parallel) vs LSTM (sequential) on the same task
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

CONTENT_VOCAB = 10     # tokens 0-9
BOS_IDX = 10
VOCAB_SIZE = 11
SEQ_LEN = 15           # generated Markov-chain length (excludes BOS)


# =============================================================================
# SECTION A -- GPT DECODER BLOCK (causal self-attention + FFN, NO cross-attn)
# =============================================================================

class GPTBlock(nn.Module):
    """Pre-LN decoder block: masked self-attention + FFN. NO cross-attention
    sub-layer at all -- there is no separate encoder/source sequence."""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, causal_mask):
        h = self.norm1(x)
        attn_out, attn_w = self.self_attn(h, h, h, attn_mask=causal_mask,
                                          need_weights=True, average_attn_weights=True)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x, attn_w


def section_a_block_demo():
    print("\n" + "="*65)
    print("SECTION A -- GPT Decoder Block: Causal-Only, No Cross-Attention")
    print("="*65)

    d_model, num_heads, d_ff, L, B = 32, 4, 64, 8, 2
    block = GPTBlock(d_model, num_heads, d_ff)
    x = torch.randn(B, L, d_model)
    causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)

    out, attn_w = block(x, causal_mask)
    future_weights = attn_w[0].detach().numpy()[causal_mask.numpy()]

    print(f"\n  Input/Output shape: {tuple(x.shape)} -> {tuple(out.shape)}")
    print(f"  Max attention weight to future positions: {future_weights.max():.2e} (should be 0.0)")
    assert np.allclose(future_weights, 0.0, atol=1e-7)
    print("\n  OK: GPT block correctly restricts attention to causal (past-only) positions")
    print("      Note: only TWO sub-layers (self-attn, FFN) -- no cross-attention exists")


# =============================================================================
# SECTION B -- FULL GPT MODEL ASSEMBLY
# =============================================================================

class GPT(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, num_heads=4, d_ff=256,
                num_layers=3, max_len=20):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=token_ids.device), diagonal=1)

        attn_w = None
        for block in self.blocks:
            x, attn_w = block(x, causal_mask)
        x = self.final_norm(x)
        return self.output_proj(x), attn_w

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, strategy="greedy", temperature=1.0, top_k=None):
        """
        prompt: (1, L0) starting token sequence.
        strategy: "greedy" | "temperature" | "top_k"
        """
        self.eval()
        tokens = prompt.clone()
        for _ in range(max_new_tokens):
            logits, _ = self(tokens)
            next_logits = logits[0, -1, :]    # (vocab_size,)

            if strategy == "greedy":
                next_tok = next_logits.argmax().unsqueeze(0)
            elif strategy == "temperature":
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            elif strategy == "top_k":
                scaled = next_logits / temperature
                topk_vals, topk_idx = scaled.topk(top_k)
                topk_probs = F.softmax(topk_vals, dim=-1)
                sampled_idx = torch.multinomial(topk_probs, num_samples=1)
                next_tok = topk_idx[sampled_idx]
            else:
                raise ValueError(strategy)

            tokens = torch.cat([tokens, next_tok.unsqueeze(0)], dim=1)
        return tokens


def section_b_model_demo():
    print("\n" + "="*65)
    print("SECTION B -- Full GPT Model: Shape Verification")
    print("="*65)

    torch.manual_seed(SEED)
    model = GPT(num_layers=3)
    token_ids = torch.randint(0, CONTENT_VOCAB, (4, 12))
    logits, attn_w = model(token_ids)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n  Total parameters: {n_params:,}")
    print(f"  Input shape: {tuple(token_ids.shape)}")
    print(f"  Output logits shape: {tuple(logits.shape)} (batch, L, vocab_size)")
    assert logits.shape == (4, 12, VOCAB_SIZE)
    print("\n  OK: Full GPT decoder-only model produces correct output shape")


# =============================================================================
# SECTION C -- SYNTHETIC MARKOV-CHAIN "COUNTING" CORPUS
# =============================================================================

def generate_markov_sequence(rng, length, continue_prob=0.8):
    """token[t+1] = (token[t]+1)%10 with prob continue_prob, else uniform random."""
    seq = [int(rng.integers(0, CONTENT_VOCAB))]
    for _ in range(length - 1):
        if rng.random() < continue_prob:
            seq.append((seq[-1] + 1) % CONTENT_VOCAB)
        else:
            seq.append(int(rng.integers(0, CONTENT_VOCAB)))
    return seq


def generate_corpus(n_sequences, seed, length=SEQ_LEN):
    rng = np.random.default_rng(seed)
    return np.array([generate_markov_sequence(rng, length) for _ in range(n_sequences)], dtype=np.int64)


def section_c_corpus_demo():
    print("\n" + "="*65)
    print("SECTION C -- Synthetic Markov-Chain 'Counting' Corpus")
    print("="*65)

    sequences = generate_corpus(5, SEED)
    print(f"\n  Rule: token[t+1] = (token[t]+1)%10 with 80% probability, else random")
    print(f"  Example sequences:")
    for seq in sequences:
        print(f"    {seq.tolist()}")
    print("\n  (Notice the mostly-increasing count pattern with occasional random 'jumps')")


# =============================================================================
# SECTION D -- NEXT-TOKEN PREDICTION TRAINING (parallel via causal mask)
# =============================================================================

def build_lm_batch(sequences):
    """Prepend BOS; input=[BOS,x1,...,x_{L-1}], target=[x1,...,xL] (predict EVERY position)."""
    N = len(sequences)
    bos_col = np.full((N, 1), BOS_IDX, dtype=np.int64)
    lm_input = np.concatenate([bos_col, sequences[:, :-1]], axis=1)
    lm_target = sequences
    return lm_input, lm_target


def train_gpt(n_epochs=40, n_sequences=3000, batch_size=32, num_layers=3):
    print("\n" + "="*65)
    print("SECTION D -- Next-Token Prediction Training")
    print("="*65)

    sequences = generate_corpus(n_sequences, SEED)
    n_val = int(n_sequences * 0.15)
    seq_tr, seq_va = sequences[n_val:], sequences[:n_val]

    X_tr, Y_tr = build_lm_batch(seq_tr)
    X_va, Y_va = build_lm_batch(seq_va)

    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)), batch_size=batch_size, shuffle=True)
    X_va_t, Y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(Y_va).to(DEVICE)

    torch.manual_seed(SEED)
    model = GPT(num_layers=num_layers, max_len=SEQ_LEN+2).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    print(f"\n  Corpus: {len(seq_tr)} train / {len(seq_va)} val sequences (length={SEQ_LEN})")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = {"train_loss": [], "val_ppl": []}
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        tl = 0.0
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(Xb)
            loss = crit(logits.reshape(-1, VOCAB_SIZE), Yb.reshape(-1))
            loss.backward()
            opt.step()
            tl += loss.item()
        history["train_loss"].append(tl/len(loader))

        if (epoch+1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X_va_t)
                val_loss = crit(val_logits.reshape(-1, VOCAB_SIZE), Y_va_t.reshape(-1)).item()
                val_ppl = math.exp(val_loss)
            history["val_ppl"].append(val_ppl)
            print(f"    Epoch {epoch+1:3d}/{n_epochs} | train_loss={tl/len(loader):.4f} | val_perplexity={val_ppl:.3f}")

    elapsed = time.time() - t0
    print(f"\n  OK: GPT training complete in {elapsed:.1f}s")
    return model, history, elapsed, seq_va


# =============================================================================
# SECTION E -- PERPLEXITY: THEORETICAL BOUNDS CHECK
# =============================================================================

def section_e_perplexity_bounds(model, seq_va):
    print("\n" + "="*65)
    print("SECTION E -- Perplexity: Theoretical Bounds Verification")
    print("="*65)

    X_va, Y_va = build_lm_batch(seq_va)
    X_va_t, Y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(Y_va).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits, _ = model(X_va_t)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, VOCAB_SIZE), Y_va_t.reshape(-1)).item()
    trained_ppl = math.exp(loss)

    # Theoretical bounds:
    uniform_ppl = CONTENT_VOCAB    # a uniform model over 10 tokens
    # Theoretical BEST possible (accounting for the task's OWN irreducible randomness):
    # 80% of the time the next token is perfectly predictable (prob=1); 20% of the time
    # it's uniform over all 10 tokens (prob=1/10 each). Expected -log(p) per token:
    theoretical_best_nll = -(0.8*math.log(0.8 + 0.2/10) + 0.2*9*(1/10)*math.log(0.2/10))
    # Simplify: true generative probability of the ACTUAL next token under the known
    # Markov rule: p(correct_continuation) = 0.8 + 0.2*(1/10) [continuation could ALSO
    # coincidentally be drawn as the random 20% pick], other 9 tokens each get 0.2/10.
    p_expected_token = 0.8 + 0.2*(1/10)
    theoretical_best_nll = -math.log(p_expected_token)
    theoretical_best_ppl = math.exp(theoretical_best_nll)

    print(f"\n  Trained model perplexity:        {trained_ppl:.3f}")
    print(f"  Theoretical BEST possible (oracle knowing the exact Markov rule): {theoretical_best_ppl:.3f}")
    print(f"  Uniform-random baseline (chance): {uniform_ppl:.3f}")
    print(f"\n  Trained model is {(uniform_ppl-trained_ppl)/(uniform_ppl-theoretical_best_ppl)*100:.1f}% "
          f"of the way from chance to the theoretical optimum")

    return trained_ppl, theoretical_best_ppl, uniform_ppl


# =============================================================================
# SECTION F & G -- AUTOREGRESSIVE GENERATION: SAMPLING STRATEGY COMPARISON
# =============================================================================

def section_fg_sampling_comparison(model, n_samples=20, max_new_tokens=14):
    print("\n" + "="*65)
    print("SECTION F/G -- Sampling Strategies: Greedy / Temperature / Top-k")
    print("="*65)

    prompt = torch.tensor([[BOS_IDX]]).to(DEVICE)

    def check_rule_adherence(tokens):
        """Fraction of transitions that follow the +1 mod 10 rule."""
        toks = tokens[0, 1:].cpu().numpy()   # drop BOS
        correct = sum(1 for i in range(1, len(toks)) if toks[i] == (toks[i-1]+1) % CONTENT_VOCAB)
        return correct / (len(toks)-1) if len(toks) > 1 else 0.0

    strategies = {
        "Greedy": dict(strategy="greedy"),
        "Temperature T=0.5": dict(strategy="temperature", temperature=0.5),
        "Temperature T=1.5": dict(strategy="temperature", temperature=1.5),
        "Top-k (k=3, T=1.0)": dict(strategy="top_k", top_k=3, temperature=1.0),
    }

    results = {}
    print(f"\n  {'Strategy':>22} | {'Example generation':>35} | {'Rule-adherence rate':>20}")
    print("  " + "-"*85)

    for name, kwargs in strategies.items():
        torch.manual_seed(SEED)
        adherence_rates = []
        example_seq = None
        for i in range(n_samples):
            gen = model.generate(prompt, max_new_tokens, **kwargs)
            adherence_rates.append(check_rule_adherence(gen))
            if i == 0:
                example_seq = gen[0, 1:].cpu().numpy().tolist()
        avg_adherence = np.mean(adherence_rates)
        results[name] = {"adherence": avg_adherence, "example": example_seq, "all_rates": adherence_rates}
        print(f"  {name:>22} | {str(example_seq):>35} | {avg_adherence*100:>19.1f}%")

    print(f"\n  (Rule-adherence = fraction of consecutive steps following the +1 mod 10 pattern;")
    print(f"   the TRUE underlying rate is 80% in the training data itself)")

    return results


# =============================================================================
# SECTION H -- TRAINING SPEED: GPT (PARALLEL) vs LSTM (SEQUENTIAL)
# =============================================================================

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, hidden_size=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers=num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        h, _ = self.lstm(x)
        return self.output_proj(h)


def section_h_training_speed_comparison(n_epochs=25, n_sequences=3000):
    print("\n" + "="*65)
    print("SECTION H -- Training Speed: GPT (parallel) vs LSTM (sequential)")
    print("  (both trained via next-token prediction on the IDENTICAL corpus)")
    print("="*65)

    sequences = generate_corpus(n_sequences, SEED)
    n_val = int(n_sequences*0.15)
    seq_tr, seq_va = sequences[n_val:], sequences[:n_val]
    X_tr, Y_tr = build_lm_batch(seq_tr)
    X_va, Y_va = build_lm_batch(seq_va)

    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)), batch_size=32, shuffle=True)
    X_va_t, Y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(Y_va).to(DEVICE)
    crit = nn.CrossEntropyLoss()

    results = {}
    for name, build_fn in [("GPT", lambda: GPT(num_layers=3, max_len=SEQ_LEN+2)),
                           ("LSTM", lambda: LSTMLanguageModel())]:
        torch.manual_seed(SEED)
        model = build_fn().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        n_params = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        for epoch in range(n_epochs):
            model.train()
            for Xb, Yb in loader:
                Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                opt.zero_grad()
                out = model(Xb)
                logits = out[0] if isinstance(out, tuple) else out
                loss = crit(logits.reshape(-1, VOCAB_SIZE), Yb.reshape(-1))
                loss.backward()
                opt.step()
        elapsed = time.time() - t0

        model.eval()
        with torch.no_grad():
            out = model(X_va_t)
            logits = out[0] if isinstance(out, tuple) else out
            val_ppl = math.exp(crit(logits.reshape(-1, VOCAB_SIZE), Y_va_t.reshape(-1)).item())

        results[name] = {"time": elapsed, "params": n_params, "val_ppl": val_ppl}
        print(f"\n  {name}: params={n_params:,} | {n_epochs} epochs in {elapsed:.1f}s | val_perplexity={val_ppl:.3f}")

    return results


# =============================================================================
# SECTION I -- VISUALIZATION
# =============================================================================

def build_figures(train_hist, ppl_bounds, sampling_results, speed_results):
    fig = plt.figure(figsize=(17, 10))
    fig.suptitle("Phase 4 -- Topic 4: GPT Decoder & Autoregressive Generation", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    a1, a2, a3, a4 = (fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
                      fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]))

    ep = range(1, len(train_hist["train_loss"])+1)
    a1.plot(ep, train_hist["train_loss"], color="#9b59b6", lw=2)
    a1.set_title("GPT Next-Token Prediction Training Loss", fontweight="bold", fontsize=10)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Cross-Entropy Loss")
    a1.grid(True, alpha=0.3)

    trained_ppl, theoretical_best, uniform_ppl = ppl_bounds
    bars_names = ["Trained\nModel", "Theoretical\nOptimum", "Uniform\n(chance)"]
    bars_vals = [trained_ppl, theoretical_best, uniform_ppl]
    bars = a2.bar(bars_names, bars_vals, color=["#3498db", "#27ae60", "#e74c3c"])
    for bar, v in zip(bars, bars_vals):
        a2.text(bar.get_x()+bar.get_width()/2, v+0.1, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    a2.set_title("Perplexity: Trained vs Theoretical Bounds", fontweight="bold", fontsize=10)
    a2.set_ylabel("Perplexity (lower=better)")
    a2.grid(True, axis="y", alpha=0.3)

    names = list(sampling_results.keys())
    adherence = [sampling_results[n]["adherence"]*100 for n in names]
    colors_s = ["#2c3e50", "#3498db", "#e74c3c", "#27ae60"]
    bars2 = a3.bar(names, adherence, color=colors_s)
    a3.axhline(80, color="gray", ls="--", lw=1, label="True rate (80%)")
    for bar, v in zip(bars2, adherence):
        a3.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.0f}%", ha="center", fontsize=8)
    a3.set_title("Sampling Strategy: Rule-Adherence Rate", fontweight="bold", fontsize=10)
    a3.set_ylabel("Adherence to +1 mod 10 rule (%)")
    a3.tick_params(axis="x", rotation=15, labelsize=8)
    a3.legend(fontsize=8); a3.grid(True, axis="y", alpha=0.3)

    speed_names = list(speed_results.keys())
    speed_times = [speed_results[n]["time"] for n in speed_names]
    speed_ppls = [speed_results[n]["val_ppl"] for n in speed_names]
    ax4b = a4.twinx()
    bars3 = a4.bar(np.arange(len(speed_names))-0.2, speed_times, 0.35, color="#9b59b6", label="Train time (s)")
    bars4 = ax4b.bar(np.arange(len(speed_names))+0.2, speed_ppls, 0.35, color="#e67e22", label="Val perplexity")
    a4.set_xticks(range(len(speed_names))); a4.set_xticklabels(speed_names)
    a4.set_title("GPT vs LSTM: Training Time & Final Perplexity", fontweight="bold", fontsize=10)
    a4.set_ylabel("Training Time (s)", color="#9b59b6"); ax4b.set_ylabel("Perplexity", color="#e67e22")
    a4.legend(loc="upper left", fontsize=8); ax4b.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS, "04_gpt_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Dashboard saved -> {path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#"*65)
    print("  Phase 4 -- Topic 4: GPT -- Decoder-Only Autoregressive Generation")
    print("#"*65)

    section_a_block_demo()
    section_b_model_demo()
    section_c_corpus_demo()

    model, train_hist, elapsed, seq_va = train_gpt(n_epochs=40)
    ppl_bounds = section_e_perplexity_bounds(model, seq_va)
    sampling_results = section_fg_sampling_comparison(model)
    speed_results = section_h_training_speed_comparison(n_epochs=25)

    build_figures(train_hist, ppl_bounds, sampling_results, speed_results)

    print("\n" + "#"*65)
    print("  DONE: Topic 4 complete.")
    print("#"*65 + "\n")


if __name__ == "__main__":
    main()
