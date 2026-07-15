"""
Topic: Seq2Seq, Attention & Teacher Forcing
==========================================================
Repository : deep-learning/sequential/03-seq2seq-nlp/
File       : implementation.py

Since this environment has no internet access to real NLP corpora, we use two
synthetic sequence-to-sequence tasks that faithfully exercise every architectural
component described in theory.md:

  Task A — Character Reversal: "abcde" → "edcba"
            Tests basic encoder-decoder memory across a gap.
  Task B — Number Sorting:     "3 1 4 1 5 9" → "1 1 3 4 5 9"
            Tests attention (decoder must selectively look at different encoder
            positions to build a sorted output) — harder than reversal.

Sections:
  A │ Tokenizer + synthetic dataset generation (Tasks A and B)
  B │ Encoder (bidirectional GRU)
  C │ Bahdanau Attention module (additive attention, theory.md §3)
  D │ Decoder (GRU + attention context, with teacher-forcing support)
  E │ Seq2Seq wrapper + train loop with scheduled teacher-forcing decay
  F │ Greedy decode, Beam Search decode, BLEU evaluation
  G │ Train on Task A (reversal) and Task B (sorting), compare
  H │ Visualization (attention heatmap + learning curves + BLEU)
"""

import os, math, random, warnings, itertools
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[CONFIG] Device: {DEVICE}  |  PyTorch: {torch.__version__}")

PAD, SOS, EOS = 0, 1, 2
PAD_TOK, SOS_TOK, EOS_TOK = "<pad>", "<sos>", "<eos>"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — TOKENIZER + SYNTHETIC TASKS
# ═════════════════════════════════════════════════════════════════════════════

class CharVocab:
    """Simple character-level vocabulary."""
    def __init__(self, chars):
        self.tok2id = {PAD_TOK: PAD, SOS_TOK: SOS, EOS_TOK: EOS}
        for ch in chars:
            if ch not in self.tok2id:
                self.tok2id[ch] = len(self.tok2id)
        self.id2tok = {v: k for k, v in self.tok2id.items()}

    def encode(self, seq: list) -> list:
        return [self.tok2id[t] for t in seq]

    def decode(self, ids: list) -> list:
        return [self.id2tok.get(i, "?") for i in ids if i not in (PAD, SOS, EOS)]

    def __len__(self):
        return len(self.tok2id)


def make_reversal_data(n_samples, min_len=5, max_len=10, seed=SEED):
    """Task A: reverse a sequence of lowercase letters."""
    rng = random.Random(seed)
    chars = list("abcdefghijklmnopqrstuvwxyz")
    vocab = CharVocab(chars)
    pairs = []
    for _ in range(n_samples):
        L = rng.randint(min_len, max_len)
        src = [rng.choice(chars) for _ in range(L)]
        tgt = src[::-1]
        pairs.append((src, tgt))
    return vocab, pairs


def make_sorting_data(n_samples, min_len=4, max_len=8, seed=SEED):
    """Task B: sort a sequence of digit characters."""
    rng = random.Random(seed)
    digits = list("0123456789")
    vocab = CharVocab(digits)
    pairs = []
    for _ in range(n_samples):
        L = rng.randint(min_len, max_len)
        src = [rng.choice(digits) for _ in range(L)]
        tgt = sorted(src)
        pairs.append((src, tgt))
    return vocab, pairs


class Seq2SeqDataset(Dataset):
    def __init__(self, vocab, pairs):
        self.vocab = vocab
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = [SOS] + self.vocab.encode(src) + [EOS]
        tgt_ids = [SOS] + self.vocab.encode(tgt) + [EOS]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_seq2seq(batch):
    """Pad src and tgt to max lengths in the batch."""
    src_seqs = [b[0] for b in batch]
    tgt_seqs = [b[1] for b in batch]
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD)
    src_lens = torch.tensor([len(s) for s in src_seqs])
    return src_padded, tgt_padded, src_lens


def build_dataloaders(vocab, pairs, val_frac=0.15, batch_size=64):
    n_val = int(len(pairs)*val_frac)
    rng = random.Random(SEED)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    val_pairs, train_pairs = shuffled[:n_val], shuffled[n_val:]
    train_ds = Seq2SeqDataset(vocab, train_pairs)
    val_ds   = Seq2SeqDataset(vocab, val_pairs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_seq2seq)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_seq2seq)
    return train_loader, val_loader, val_pairs


def section_a_data():
    print("\n" + "="*65)
    print("SECTION A — Synthetic Seq2Seq Tasks")
    print("="*65)

    vocab_rev, pairs_rev = make_reversal_data(3000)
    vocab_srt, pairs_srt = make_sorting_data(3000)

    print(f"\n  Task A (Reversal):  {len(pairs_rev)} samples | "
          f"vocab_size={len(vocab_rev)} | example: {pairs_rev[0]}")
    print(f"  Task B (Sorting):   {len(pairs_srt)} samples | "
          f"vocab_size={len(vocab_srt)} | example: {pairs_srt[0]}")
    return vocab_rev, pairs_rev, vocab_srt, pairs_srt


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — ENCODER
# ═════════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """
    Bidirectional GRU encoder.
    Returns all hidden states (for attention) and a projected context vector.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True,
                          bidirectional=True)
        # Project 2*hidden (bidir) -> hidden for decoder initialization
        self.fc_h = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, src, src_lens):
        """
        src: (batch, T_src)  src_lens: (batch,)
        Returns: enc_out (batch, T_src, 2*hidden), h (batch, hidden)
        """
        x = self.dropout(self.embed(src))                # (B,T,embed)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        enc_out_packed, h = self.gru(packed)              # h: (2,B,hidden)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out_packed, batch_first=True)
        # Merge bidir: concat fwd and bwd final hidden states
        h_fwd = h[0]; h_bwd = h[1]                        # each: (B,hidden)
        h_combined = torch.tanh(self.fc_h(torch.cat([h_fwd, h_bwd], dim=-1)))
        return enc_out, h_combined


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — BAHDANAU (ADDITIVE) ATTENTION
# ═════════════════════════════════════════════════════════════════════════════

class BahdanauAttention(nn.Module):
    """
    Additive attention (Bahdanau et al. 2015):
      score(s_{t-1}, h_j) = vᵀ tanh(W_dec·s_{t-1} + W_enc·h_j)
      α = softmax(scores)
      context = Σⱼ αⱼ hⱼ
    """
    def __init__(self, dec_hidden, enc_hidden_2):
        """
        dec_hidden:   decoder hidden size
        enc_hidden_2: encoder hidden size * 2 (bidirectional output)
        """
        super().__init__()
        attn_dim = dec_hidden
        self.W_dec = nn.Linear(dec_hidden, attn_dim, bias=False)
        self.W_enc = nn.Linear(enc_hidden_2, attn_dim, bias=False)
        self.v     = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, s, enc_out, mask=None):
        """
        s:       (batch, dec_hidden)          decoder hidden state at step t
        enc_out: (batch, T_src, enc_hidden_2) all encoder hidden states
        mask:    (batch, T_src) bool, True where PAD (to mask out in softmax)

        Returns: context (batch, enc_hidden_2), weights (batch, T_src)
        """
        T_src = enc_out.size(1)
        s_exp = self.W_dec(s).unsqueeze(1).expand(-1, T_src, -1)   # (B,T_src,attn_dim)
        e_exp = self.W_enc(enc_out)                                   # (B,T_src,attn_dim)
        scores = self.v(torch.tanh(s_exp + e_exp)).squeeze(-1)       # (B,T_src)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        alpha = F.softmax(scores, dim=1)                              # (B,T_src)
        context = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)  # (B,enc_hidden_2)
        return context, alpha


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — DECODER WITH ATTENTION
# ═════════════════════════════════════════════════════════════════════════════

class AttentionDecoder(nn.Module):
    """
    GRU decoder with Bahdanau attention.
    At each step: embed(y_{t-1}) → GRU(embed + context) → project → output
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, enc_hidden_2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.attention = BahdanauAttention(hidden_size, enc_hidden_2)
        self.gru = nn.GRU(embed_dim + enc_hidden_2, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size + enc_hidden_2 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, y_prev, s, enc_out, mask=None):
        """
        One decoding step.
        y_prev: (batch,)    — previous token id
        s:      (batch, H)  — current hidden state
        Returns: logits (batch, vocab), s_next (batch, H), alpha (batch, T_src)
        """
        emb = self.dropout(self.embed(y_prev.unsqueeze(1)))      # (B,1,embed)
        context, alpha = self.attention(s, enc_out, mask)         # (B,enc_H2), (B,T)

        gru_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)  # (B,1,embed+enc_H2)
        gru_out, h_new = self.gru(gru_in, s.unsqueeze(0))        # (B,1,H), (1,B,H)
        s_next = h_new.squeeze(0)

        # Combine gru output, context, embedding for output projection
        pred_in = torch.cat([gru_out.squeeze(1), context, emb.squeeze(1)], dim=-1)
        logits = self.fc_out(pred_in)                              # (B, vocab_size)
        return logits, s_next, alpha


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — SEQ2SEQ WRAPPER + TRAINING
# ═════════════════════════════════════════════════════════════════════════════

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_size=64):
        super().__init__()
        enc_hidden_2 = hidden_size * 2
        self.encoder = Encoder(vocab_size, embed_dim, hidden_size)
        self.decoder = AttentionDecoder(vocab_size, embed_dim, hidden_size, enc_hidden_2)
        self.vocab_size = vocab_size

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        """
        src:  (B, T_src)
        tgt:  (B, T_tgt)   — includes SOS and EOS
        Returns: logits (B, T_tgt-1, vocab_size)
        """
        B, T_tgt = tgt.size()
        enc_out, s = self.encoder(src, src_lens)

        # Padding mask: True where src == PAD
        mask = (src == PAD)  # (B, T_src)

        outputs = []
        y_input = tgt[:, 0]                    # first input token = SOS

        for t in range(1, T_tgt):
            logits, s, _ = self.decoder.forward_step(y_input, s, enc_out, mask)
            outputs.append(logits.unsqueeze(1))

            # Teacher forcing: use ground truth vs own prediction
            use_tf = random.random() < teacher_forcing_ratio
            if use_tf:
                y_input = tgt[:, t]
            else:
                y_input = logits.argmax(dim=-1)

        return torch.cat(outputs, dim=1)        # (B, T_tgt-1, vocab)


def train_seq2seq(vocab, pairs, task_name, n_epochs=30, embed_dim=32, hidden_size=64):
    print(f"\n  Training Seq2Seq on Task {task_name}...")
    train_loader, val_loader, val_pairs = build_dataloaders(vocab, pairs)

    torch.manual_seed(SEED)
    model = Seq2Seq(len(vocab), embed_dim, hidden_size).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        # Scheduled teacher forcing: decay from 1.0 → 0.0 over training
        tf_ratio = 1.0 - (epoch / n_epochs)

        model.train()
        tl, tn = 0.0, 0
        for src, tgt, src_lens in train_loader:
            src, tgt, src_lens = src.to(DEVICE), tgt.to(DEVICE), src_lens
            opt.zero_grad()
            logits = model(src, src_lens, tgt, teacher_forcing_ratio=tf_ratio)
            # logits: (B, T_tgt-1, V), target: tgt[:,1:] (skip SOS)
            loss = crit(logits.reshape(-1, len(vocab)), tgt[:,1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * src.size(0); tn += src.size(0)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for src, tgt, src_lens in val_loader:
                src, tgt, src_lens = src.to(DEVICE), tgt.to(DEVICE), src_lens
                logits = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
                loss = crit(logits.reshape(-1, len(vocab)), tgt[:,1:].reshape(-1))
                vl += loss.item() * src.size(0); vn += src.size(0)

        history["train_loss"].append(tl/tn)
        history["val_loss"].append(vl/vn)

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{n_epochs} | tf={tf_ratio:.2f} | "
                  f"train={tl/tn:.4f} | val={vl/vn:.4f}")

    return model, history, val_pairs


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — DECODING + BLEU EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_decode(model, vocab, src_tokens, max_len=20):
    """Decode a single source sequence greedily."""
    model.eval()
    src_ids = [SOS] + vocab.encode(src_tokens) + [EOS]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_lens = torch.tensor([len(src_ids)])

    enc_out, s = model.encoder(src, src_lens)
    mask = (src == PAD)

    y = torch.tensor([SOS], dtype=torch.long).to(DEVICE)
    result = []; all_alphas = []

    for _ in range(max_len):
        logits, s, alpha = model.decoder.forward_step(y, s, enc_out, mask)
        all_alphas.append(alpha[0].cpu().numpy())
        pred = logits.argmax(-1)
        if pred.item() == EOS:
            break
        result.append(pred.item())
        y = pred

    decoded = vocab.decode(result)
    return decoded, all_alphas


@torch.no_grad()
def beam_decode(model, vocab, src_tokens, beam_size=3, max_len=20):
    """Beam search decoding for a single source sequence."""
    model.eval()
    src_ids = [SOS] + vocab.encode(src_tokens) + [EOS]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_lens = torch.tensor([len(src_ids)])
    enc_out, s_init = model.encoder(src, src_lens)
    mask = (src == PAD)

    # Each beam: (log_prob, token_ids, hidden_state)
    beams = [(0.0, [SOS], s_init)]
    completed = []

    for _ in range(max_len):
        next_beams = []
        for log_prob, tokens, s in beams:
            y = torch.tensor([tokens[-1]], dtype=torch.long).to(DEVICE)
            logits, s_next, _ = model.decoder.forward_step(y, s, enc_out, mask)
            log_probs = F.log_softmax(logits, dim=-1)[0]
            top_k = log_probs.topk(beam_size)
            for lp, tok in zip(top_k.values, top_k.indices):
                new_lp = log_prob + lp.item()
                new_tokens = tokens + [tok.item()]
                if tok.item() == EOS:
                    completed.append((new_lp, new_tokens))
                else:
                    next_beams.append((new_lp, new_tokens, s_next))

        next_beams.sort(key=lambda x: x[0], reverse=True)
        beams = next_beams[:beam_size]
        if len(beams) == 0:
            break

    if completed:
        best = max(completed, key=lambda x: x[0])
    else:
        best = max(beams, key=lambda x: x[0])

    return vocab.decode(best[1])


def bleu_score(hypothesis: list, reference: list, max_n: int = 4) -> float:
    """Compute corpus BLEU-4 (simplified, sentence level)."""
    if len(hypothesis) == 0:
        return 0.0

    bp = math.exp(min(0, 1 - len(reference)/len(hypothesis)))
    log_prec = 0.0
    for n in range(1, max_n+1):
        hyp_ngrams = [tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)]
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]
        if not hyp_ngrams:
            log_prec += float('-inf')
            continue
        ref_set = {}
        for ng in ref_ngrams:
            ref_set[ng] = ref_set.get(ng, 0) + 1
        matches = sum(min(hyp_ngrams.count(ng), cnt) for ng, cnt in ref_set.items())
        prec = matches / len(hyp_ngrams) if hyp_ngrams else 0.0
        log_prec += math.log(max(prec, 1e-10)) / max_n

    return bp * math.exp(log_prec)


def evaluate_model(model, vocab, val_pairs, use_beam=False):
    """Compute exact-match accuracy and average BLEU-4 on validation pairs."""
    exact, total_bleu, n = 0, 0.0, 0
    for src, tgt in val_pairs:
        if use_beam:
            pred = beam_decode(model, vocab, src)
        else:
            pred, _ = greedy_decode(model, vocab, src)
        exact += int(pred == tgt)
        total_bleu += bleu_score(pred, tgt)
        n += 1
    return exact/n, total_bleu/n


# ═════════════════════════════════════════════════════════════════════════════
# SECTION G — TRAINING EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════════════

def section_g_train_both(vocab_rev, pairs_rev, vocab_srt, pairs_srt):
    print("\n" + "="*65)
    print("SECTION G — Training on Both Tasks")
    print("="*65)

    model_rev, hist_rev, val_pairs_rev = train_seq2seq(vocab_rev, pairs_rev, "A (Reversal)", n_epochs=30)
    model_srt, hist_srt, val_pairs_srt = train_seq2seq(vocab_srt, pairs_srt, "B (Sorting)", n_epochs=40)

    print("\n  Evaluating models (greedy and beam):")
    acc_rev_g, bleu_rev_g = evaluate_model(model_rev, vocab_rev, val_pairs_rev[:100])
    acc_rev_b, bleu_rev_b = evaluate_model(model_rev, vocab_rev, val_pairs_rev[:100], use_beam=True)
    acc_srt_g, bleu_srt_g = evaluate_model(model_srt, vocab_srt, val_pairs_srt[:100])
    acc_srt_b, bleu_srt_b = evaluate_model(model_srt, vocab_srt, val_pairs_srt[:100], use_beam=True)

    print(f"\n  Task A (Reversal):")
    print(f"    Greedy: exact_match={acc_rev_g*100:.1f}%  BLEU={bleu_rev_g:.4f}")
    print(f"    Beam:   exact_match={acc_rev_b*100:.1f}%  BLEU={bleu_rev_b:.4f}")
    print(f"\n  Task B (Sorting):")
    print(f"    Greedy: exact_match={acc_srt_g*100:.1f}%  BLEU={bleu_srt_g:.4f}")
    print(f"    Beam:   exact_match={acc_srt_b*100:.1f}%  BLEU={bleu_srt_b:.4f}")

    # Show some examples
    print(f"\n  Task A examples (Reversal):")
    for src, tgt in val_pairs_rev[:5]:
        pred, _ = greedy_decode(model_rev, vocab_rev, src)
        ok = "✓" if pred == tgt else "✗"
        print(f"    {ok} src={''.join(src):12s} | tgt={''.join(tgt):12s} | pred={''.join(pred)}")

    print(f"\n  Task B examples (Sorting):")
    for src, tgt in val_pairs_srt[:5]:
        pred, _ = greedy_decode(model_srt, vocab_srt, src)
        ok = "✓" if pred == tgt else "✗"
        print(f"    {ok} src={''.join(src):12s} | tgt={''.join(tgt):12s} | pred={''.join(pred)}")

    metrics = {
        "rev": {"greedy": (acc_rev_g, bleu_rev_g), "beam": (acc_rev_b, bleu_rev_b)},
        "srt": {"greedy": (acc_srt_g, bleu_srt_g), "beam": (acc_srt_b, bleu_srt_b)},
    }
    return model_rev, hist_rev, model_srt, hist_srt, val_pairs_rev, val_pairs_srt, metrics, vocab_rev, vocab_srt


# ═════════════════════════════════════════════════════════════════════════════
# SECTION H — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def build_figures(hist_rev, hist_srt, model_rev, vocab_rev, val_pairs_rev,
                  model_srt, vocab_srt, val_pairs_srt, metrics):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Phase 3 — Topic 3: Seq2Seq with Bahdanau Attention", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

    # Panel 1: Task A training curves
    ep_r = range(1, len(hist_rev["train_loss"])+1)
    axes[0].plot(ep_r, hist_rev["train_loss"], color="#e74c3c", lw=2, label="Train")
    axes[0].plot(ep_r, hist_rev["val_loss"], color="#3498db", lw=2, label="Val")
    axes[0].set_title("Task A (Reversal) — Loss Curves", fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("CE Loss")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # Panel 2: Task B training curves
    ep_s = range(1, len(hist_srt["train_loss"])+1)
    axes[1].plot(ep_s, hist_srt["train_loss"], color="#e74c3c", lw=2, label="Train")
    axes[1].plot(ep_s, hist_srt["val_loss"], color="#3498db", lw=2, label="Val")
    axes[1].set_title("Task B (Sorting) — Loss Curves", fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("CE Loss")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    # Panel 3: BLEU and accuracy comparison bars
    tasks = ["Reversal\n(Greedy)", "Reversal\n(Beam)", "Sorting\n(Greedy)", "Sorting\n(Beam)"]
    bleus = [metrics["rev"]["greedy"][1], metrics["rev"]["beam"][1],
             metrics["srt"]["greedy"][1], metrics["srt"]["beam"][1]]
    accs  = [metrics["rev"]["greedy"][0], metrics["rev"]["beam"][0],
             metrics["srt"]["greedy"][0], metrics["srt"]["beam"][0]]
    x = np.arange(4)
    ax3b = axes[2].twinx()
    axes[2].bar(x-0.2, bleus, 0.35, color="#27ae60", label="BLEU-4")
    ax3b.bar(x+0.2, accs, 0.35, color="#9b59b6", alpha=0.7, label="Exact Match")
    axes[2].set_xticks(x); axes[2].set_xticklabels(tasks, fontsize=8)
    axes[2].set_title("BLEU-4 & Exact Match", fontweight="bold", fontsize=10)
    axes[2].set_ylabel("BLEU-4", color="#27ae60")
    ax3b.set_ylabel("Exact Match %", color="#9b59b6")
    axes[2].set_ylim(0, 1.1); ax3b.set_ylim(0, 1.1)
    axes[2].grid(True, axis="y", alpha=0.3)

    # Panel 4: Attention heatmap for a reversal example
    src_ex = val_pairs_rev[0][0]
    _, alphas_rev = greedy_decode(model_rev, vocab_rev, src_ex)
    if alphas_rev:
        alpha_mat = np.stack(alphas_rev)  # (T_tgt, T_src)
        # T_src includes SOS/EOS — trim to match display
        src_display = [SOS_TOK] + src_ex + [EOS_TOK]
        im = axes[3].imshow(alpha_mat, cmap="Blues", aspect="auto")
        axes[3].set_title(f"Attention: Reversal of '{''.join(src_ex)}'",
                         fontweight="bold", fontsize=10)
        axes[3].set_xlabel("Source position")
        axes[3].set_ylabel("Decoding step")
        plt.colorbar(im, ax=axes[3], fraction=0.046)
        axes[3].set_xticks(range(min(len(src_display), alpha_mat.shape[1])))
        axes[3].set_xticklabels(src_display[:alpha_mat.shape[1]], fontsize=8)

    # Panel 5: Attention heatmap for a sorting example
    src_srt = val_pairs_srt[2][0]  # pick an interesting example
    _, alphas_srt = greedy_decode(model_srt, vocab_srt, src_srt)
    if alphas_srt:
        alpha_mat_s = np.stack(alphas_srt)
        src_display_s = [SOS_TOK] + src_srt + [EOS_TOK]
        im2 = axes[4].imshow(alpha_mat_s, cmap="Greens", aspect="auto")
        axes[4].set_title(f"Attention: Sorting '{''.join(src_srt)}'",
                         fontweight="bold", fontsize=10)
        axes[4].set_xlabel("Source position")
        axes[4].set_ylabel("Decoding step")
        plt.colorbar(im2, ax=axes[4], fraction=0.046)
        axes[4].set_xticks(range(min(len(src_display_s), alpha_mat_s.shape[1])))
        axes[4].set_xticklabels(src_display_s[:alpha_mat_s.shape[1]], fontsize=8)

    # Panel 6: Scheduled teacher forcing curve
    epochs_plot = np.arange(30)
    tf_values = [1.0 - e/30 for e in epochs_plot]
    axes[5].plot(epochs_plot, tf_values, color="#e67e22", lw=2.5)
    axes[5].fill_between(epochs_plot, 0, tf_values, alpha=0.15, color="#e67e22")
    axes[5].set_title("Scheduled Teacher Forcing Decay", fontweight="bold", fontsize=10)
    axes[5].set_xlabel("Epoch"); axes[5].set_ylabel("Teacher Forcing Ratio")
    axes[5].set_ylim(-0.05, 1.1); axes[5].grid(True, alpha=0.3)
    axes[5].text(15, 0.55, "1.0 (full TF)\n→ 0.0 (free run)", ha="center", fontsize=9, color="#e67e22")

    plt.tight_layout()
    path = os.path.join(RESULTS, "03_seq2seq_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Dashboard saved → {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  Phase 3 — Topic 3: Seq2Seq, Attention & Teacher Forcing")
    print("▓"*65)

    vocab_rev, pairs_rev, vocab_srt, pairs_srt = section_a_data()

    (model_rev, hist_rev, model_srt, hist_srt,
     val_pairs_rev, val_pairs_srt, metrics,
     vocab_rev, vocab_srt) = section_g_train_both(
        vocab_rev, pairs_rev, vocab_srt, pairs_srt)

    build_figures(hist_rev, hist_srt, model_rev, vocab_rev, val_pairs_rev,
                  model_srt, vocab_srt, val_pairs_srt, metrics)

    print("\n" + "▓"*65)
    print("  ✓ Topic 3 complete.")
    print("▓"*65 + "\n")


if __name__ == "__main__":
    main()
