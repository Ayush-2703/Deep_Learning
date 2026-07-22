"""
Topic: BERT -- Bidirectional Encoder Pretraining
==================================================================
Repository : deep-learning/attention-transformers/03-bert-encoder-pretraining/
File       : implementation.py

Synthetic corpus: sentences generated from one of 4 "topics", each topic
having a characteristic (noisy) token distribution -- gives MLM pretraining
genuine learnable structure (predicting a masked token requires inferring
the sentence's topic from surrounding context).

Sections:
  A | Synthetic topic-structured corpus generator
  B | 80/10/10 masking strategy -- implemented + verified
  C | BERT-style encoder (token+position+segment embeddings, bidirectional stack)
  D | MLM pretraining loop (loss on masked positions only)
  E | Pretrain + evaluate MLM accuracy
  F | Downstream fine-tuning: From-scratch vs Feature-Extraction vs Full-Fine-tune
      (directly mirrors Phase 2 Topic 5's transfer learning methodology)
  G | Visualization dashboard
"""

import os, math, time, warnings, copy
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

NUM_TOPICS = 4
TOKENS_PER_TOPIC = 5
CONTENT_VOCAB = NUM_TOPICS * TOKENS_PER_TOPIC   # 20 regular content tokens
PAD_IDX, MASK_IDX, CLS_IDX, SEP_IDX = 20, 21, 22, 23
VOCAB_SIZE = 24
SENT_LEN = 8    # content tokens per sentence (excludes CLS/SEP)


# =============================================================================
# SECTION A -- SYNTHETIC TOPIC-STRUCTURED CORPUS
# =============================================================================

def generate_topic_sentence(rng, topic, sent_len=SENT_LEN, own_topic_prob=0.85):
    """
    Each topic has a preferred 5-token subset. Tokens sampled 85% from the
    topic's OWN subset, 15% uniformly from the full 20-token vocabulary --
    enough structure to be learnable, enough noise to be non-trivial.
    """
    own_tokens = np.arange(topic*TOKENS_PER_TOPIC, (topic+1)*TOKENS_PER_TOPIC)
    tokens = []
    for _ in range(sent_len):
        if rng.random() < own_topic_prob:
            tokens.append(int(rng.choice(own_tokens)))
        else:
            tokens.append(int(rng.integers(0, CONTENT_VOCAB)))
    return tokens


def generate_corpus(n_sentences, seed):
    rng = np.random.default_rng(seed)
    topics = rng.integers(0, NUM_TOPICS, size=n_sentences)
    sentences = np.array([generate_topic_sentence(rng, t) for t in topics], dtype=np.int64)
    return sentences, topics


def section_a_corpus_demo():
    print("\n" + "="*65)
    print("SECTION A -- Synthetic Topic-Structured Corpus")
    print("="*65)

    sentences, topics = generate_corpus(6, SEED)
    print(f"\n  Vocab: {CONTENT_VOCAB} content tokens across {NUM_TOPICS} topics "
          f"({TOKENS_PER_TOPIC} tokens/topic) + PAD/MASK/CLS/SEP")
    print(f"  Example sentences:")
    for i in range(6):
        print(f"    Topic {topics[i]}: {sentences[i].tolist()}")

    print("\n  (Notice: topic 0 sentences favor tokens 0-4, topic 1 favors 5-9, etc.,")
    print("   with occasional 'noise' tokens from other topics)")


# =============================================================================
# SECTION B -- 80/10/10 MASKING STRATEGY
# =============================================================================

def apply_mlm_masking(sentences, mask_prob=0.15, seed=SEED):
    """
    Returns:
      masked_input: (N, L) -- input with 80/10/10 corruption applied
      mlm_labels:   (N, L) -- original token at MASKED positions, PAD_IDX elsewhere
                    (PAD_IDX here doubles as an "ignore_index" sentinel for positions
                    that were NOT selected for masking)
    """
    rng = np.random.default_rng(seed)
    masked_input = sentences.copy()
    mlm_labels = np.full_like(sentences, PAD_IDX)     # PAD_IDX = "not a masked position"

    N, L = sentences.shape
    select_mask = rng.random((N, L)) < mask_prob        # ~15% of positions selected

    for i in range(N):
        for j in range(L):
            if not select_mask[i, j]:
                continue
            mlm_labels[i, j] = sentences[i, j]            # record the TRUE original token
            roll = rng.random()
            if roll < 0.8:
                masked_input[i, j] = MASK_IDX               # 80%: [MASK]
            elif roll < 0.9:
                masked_input[i, j] = rng.integers(0, CONTENT_VOCAB)   # 10%: random token
            # else 10%: leave unchanged (masked_input already has original token)

    return masked_input, mlm_labels, select_mask


def section_b_masking_verification():
    print("\n" + "="*65)
    print("SECTION B -- 80/10/10 Masking Strategy Verification")
    print("="*65)

    sentences, _ = generate_corpus(2000, SEED)
    masked_input, mlm_labels, select_mask = apply_mlm_masking(sentences, mask_prob=0.15)

    n_selected = select_mask.sum()
    frac_selected = n_selected / select_mask.size
    n_mask_token = (masked_input[select_mask] == MASK_IDX).sum()
    n_unchanged = (masked_input[select_mask] == sentences[select_mask]).sum()
    n_random = n_selected - n_mask_token - n_unchanged

    print(f"\n  Total positions: {select_mask.size}  |  Selected for masking: {n_selected} ({frac_selected*100:.1f}%)")
    print(f"  Of selected positions:")
    print(f"    -> [MASK] token:      {n_mask_token/n_selected*100:.1f}%  (target: 80%)")
    print(f"    -> Random token:      {n_random/n_selected*100:.1f}%  (target: 10%)")
    print(f"    -> Unchanged:         {n_unchanged/n_selected*100:.1f}%  (target: 10%)")

    print("\n  OK: Masking statistics match the 80/10/10 target within sampling noise")


# =============================================================================
# SECTION C -- BERT-STYLE ENCODER
# =============================================================================

class TransformerEncoderLayer(nn.Module):
    """Pre-LN encoder layer (bidirectional self-attention + FFN), matching Topic 2's design."""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, attn_w = self.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x, attn_w


class BERTEncoder(nn.Module):
    """
    Token + learned Position + Segment embeddings (summed), fed through a
    stack of bidirectional Transformer encoder layers -- no causal masking
    anywhere (theory.md sec 1).
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, num_heads=4, d_ff=128,
                num_layers=3, max_len=16):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(max_len, d_model)            # LEARNED position embedding
        self.seg_emb = nn.Embedding(2, d_model)                    # 2 segments (single-segment task uses seg=0 throughout)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids, segment_ids=None):
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)

        x = self.token_emb(token_ids) + self.pos_emb(positions) + self.seg_emb(segment_ids)
        attn_w = None
        for layer in self.layers:
            x, attn_w = layer(x)
        return self.final_norm(x), attn_w


class BERTForMLM(nn.Module):
    def __init__(self, encoder: BERTEncoder):
        super().__init__()
        self.encoder = encoder
        self.mlm_head = nn.Linear(encoder.d_model, VOCAB_SIZE)

    def forward(self, token_ids, segment_ids=None):
        h, attn_w = self.encoder(token_ids, segment_ids)
        return self.mlm_head(h), attn_w


def section_c_shapes_demo():
    print("\n" + "="*65)
    print("SECTION C -- BERT Encoder: Shape Verification")
    print("="*65)

    torch.manual_seed(SEED)
    encoder = BERTEncoder()
    model = BERTForMLM(encoder)

    token_ids = torch.randint(0, CONTENT_VOCAB, (4, 10))
    logits, attn_w = model(token_ids)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n  Total parameters: {n_params:,}")
    print(f"  Input shape: {tuple(token_ids.shape)}")
    print(f"  MLM logits shape: {tuple(logits.shape)} (batch, L, vocab_size)")
    print(f"  Attention weights shape: {tuple(attn_w.shape)} (batch, Lq, Lk) -- BIDIRECTIONAL, no mask")
    assert logits.shape == (4, 10, VOCAB_SIZE)
    print("\n  OK: BERT encoder + MLM head produce correct output shapes")


# =============================================================================
# SECTION D & E -- MLM PRETRAINING
# =============================================================================

def build_mlm_batch(sentences_batch, seed):
    """Wrap each sentence with [CLS]...[SEP], apply masking to the CONTENT tokens only."""
    N = len(sentences_batch)
    masked_content, mlm_labels_content, _ = apply_mlm_masking(sentences_batch, mask_prob=0.15, seed=seed)

    cls_col = np.full((N, 1), CLS_IDX, dtype=np.int64)
    sep_col = np.full((N, 1), SEP_IDX, dtype=np.int64)
    pad_label_col = np.full((N, 1), PAD_IDX, dtype=np.int64)

    full_input = np.concatenate([cls_col, masked_content, sep_col], axis=1)
    full_labels = np.concatenate([pad_label_col, mlm_labels_content, pad_label_col], axis=1)
    return full_input, full_labels


def pretrain_mlm(n_epochs=40, n_sentences=3000, batch_size=32):
    print("\n" + "="*65)
    print("SECTION D/E -- MLM Pretraining")
    print("="*65)

    sentences, topics = generate_corpus(n_sentences, SEED)
    n_val = int(n_sentences * 0.15)
    sent_tr, sent_va = sentences[n_val:], sentences[:n_val]

    torch.manual_seed(SEED)
    encoder = BERTEncoder().to(DEVICE)
    model = BERTForMLM(encoder).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    print(f"\n  Pretraining corpus: {len(sent_tr)} sentences (train) / {len(sent_va)} (val)")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = {"train_loss": [], "val_mlm_acc": []}

    for epoch in range(n_epochs):
        model.train()
        # Re-mask fresh every epoch (mirrors real BERT: masking is stochastic per-epoch/step)
        rng_seed = SEED + epoch
        full_input, full_labels = build_mlm_batch(sent_tr, seed=rng_seed)
        loader = DataLoader(TensorDataset(torch.tensor(full_input), torch.tensor(full_labels)),
                            batch_size=batch_size, shuffle=True)

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
                va_input, va_labels = build_mlm_batch(sent_va, seed=SEED+9999)
                va_input_t = torch.tensor(va_input).to(DEVICE)
                va_labels_t = torch.tensor(va_labels).to(DEVICE)
                logits, _ = model(va_input_t)
                mask_positions = va_labels_t != PAD_IDX
                correct = (logits.argmax(-1)[mask_positions] == va_labels_t[mask_positions]).float().mean().item()
            history["val_mlm_acc"].append(correct)
            print(f"    Epoch {epoch+1:3d}/{n_epochs} | train_loss={tl/len(loader):.4f} | val_MLM_acc={correct*100:.1f}%")

    print(f"\n  OK: MLM pretraining complete")
    return model, encoder, history, sentences, topics


# =============================================================================
# SECTION F -- DOWNSTREAM FINE-TUNING: 3-WAY COMPARISON (mirrors Phase 2 Topic 5)
# =============================================================================

class TopicClassifier(nn.Module):
    """Uses the [CLS] token's final representation for 4-way topic classification."""
    def __init__(self, encoder: BERTEncoder, num_classes=NUM_TOPICS):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.d_model, num_classes)

    def forward(self, token_ids):
        h, _ = self.encoder(token_ids)
        cls_repr = h[:, 0, :]      # [CLS] is always position 0
        return self.classifier(cls_repr)


def build_classification_batch(sentences_batch):
    N = len(sentences_batch)
    cls_col = np.full((N, 1), CLS_IDX, dtype=np.int64)
    sep_col = np.full((N, 1), SEP_IDX, dtype=np.int64)
    return np.concatenate([cls_col, sentences_batch, sep_col], axis=1)


def train_classifier_variant(encoder_source, X_tr, y_tr, X_va, y_va, n_epochs=40,
                             freeze_encoder=False, lr=1e-3):
    if encoder_source is not None:
        encoder = copy.deepcopy(encoder_source)
    else:
        torch.manual_seed(SEED)
        encoder = BERTEncoder()

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    model = TopicClassifier(encoder).to(DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()

    X_tr_t, y_tr_t = torch.tensor(X_tr).to(DEVICE), torch.tensor(y_tr).to(DEVICE)
    X_va_t, y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(y_va).to(DEVICE)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=16, shuffle=True)

    history = {"val_acc": []}
    for epoch in range(n_epochs):
        model.train()
        if freeze_encoder:
            model.encoder.eval()    # keep frozen encoder's dropout/norm behavior consistent (defensive; no dropout used here, but good practice)
        for Xb, Yb in loader:
            opt.zero_grad()
            logits = model(Xb)
            loss = crit(logits, Yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_va_t).argmax(-1) == y_va_t).float().mean().item()
        history["val_acc"].append(val_acc)

    return history, trainable_params


def section_f_finetuning_comparison(pretrained_encoder, n_epochs=40):
    print("\n" + "="*65)
    print("SECTION F -- Downstream Fine-Tuning: 3-Way Comparison")
    print("  (mirrors Phase 2 Topic 5's transfer learning methodology, in NLP)")
    print("="*65)

    # SMALL downstream labeled dataset (deliberately scarce, echoing Phase 2 Topic 5)
    sentences, topics = generate_corpus(120, seed=SEED+777)
    n_val = 40
    sent_va, top_va = sentences[:n_val], topics[:n_val]
    sent_tr, top_tr = sentences[n_val:], topics[n_val:]

    X_tr = build_classification_batch(sent_tr)
    X_va = build_classification_batch(sent_va)

    print(f"\n  Downstream labeled data: {len(sent_tr)} train / {len(sent_va)} val (DELIBERATELY small)")

    results = {}

    print("\n  [1/3] From-scratch (random init, train directly on small labeled data)")
    hist_scratch, params_scratch = train_classifier_variant(None, X_tr, top_tr, X_va, top_va, n_epochs)
    results["From-scratch"] = {"history": hist_scratch, "params": params_scratch}
    print(f"        trainable_params={params_scratch:,} | final_val_acc={hist_scratch['val_acc'][-1]*100:.1f}%")

    print("\n  [2/3] Feature Extraction (frozen pretrained encoder, new head only)")
    hist_fe, params_fe = train_classifier_variant(pretrained_encoder, X_tr, top_tr, X_va, top_va,
                                                   n_epochs, freeze_encoder=True)
    results["Feature Extraction"] = {"history": hist_fe, "params": params_fe}
    print(f"        trainable_params={params_fe:,} | final_val_acc={hist_fe['val_acc'][-1]*100:.1f}%")

    print("\n  [3/3] Full Fine-tuning (unfrozen pretrained encoder, low LR)")
    hist_ft, params_ft = train_classifier_variant(pretrained_encoder, X_tr, top_tr, X_va, top_va,
                                                   n_epochs, freeze_encoder=False, lr=1e-4)
    results["Full Fine-tuning"] = {"history": hist_ft, "params": params_ft}
    print(f"        trainable_params={params_ft:,} | final_val_acc={hist_ft['val_acc'][-1]*100:.1f}%")

    return results


# =============================================================================
# SECTION G -- VISUALIZATION
# =============================================================================

def build_figures(pretrain_hist, finetune_results, mlm_model, sentences, topics):
    colors = {"From-scratch": "#e74c3c", "Feature Extraction": "#3498db", "Full Fine-tuning": "#27ae60"}

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle("Phase 4 -- Topic 3: BERT Encoder Pretraining", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    a1, a2, a3, a4 = (fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
                      fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]))

    ep = range(1, len(pretrain_hist["train_loss"])+1)
    a1.plot(ep, pretrain_hist["train_loss"], color="#9b59b6", lw=2)
    a1.set_title("MLM Pretraining Loss", fontweight="bold", fontsize=10)
    a1.set_xlabel("Epoch"); a1.set_ylabel("MLM Cross-Entropy Loss")
    a1.grid(True, alpha=0.3)

    eval_epochs = list(range(10, len(pretrain_hist["train_loss"])+1, 10))
    if len(eval_epochs) < len(pretrain_hist["val_mlm_acc"]):
        eval_epochs = [1] + eval_epochs
    a2.plot(eval_epochs[:len(pretrain_hist["val_mlm_acc"])], [v*100 for v in pretrain_hist["val_mlm_acc"]],
           "o-", color="#e67e22", lw=2)
    a2.axhline(100/VOCAB_SIZE, color="gray", ls="--", lw=1, label=f"Chance ({100/VOCAB_SIZE:.1f}%)")
    a2.set_title("MLM Validation Accuracy (masked-token prediction)", fontweight="bold", fontsize=10)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy (%)")
    a2.legend(fontsize=8); a2.grid(True, alpha=0.3)

    for name, data in finetune_results.items():
        ep_f = range(1, len(data["history"]["val_acc"])+1)
        a3.plot(ep_f, [v*100 for v in data["history"]["val_acc"]], color=colors[name], lw=2, label=name)
    a3.axhline(100/NUM_TOPICS, color="gray", ls="--", lw=1, label="Chance")
    a3.set_title("Downstream Topic Classification: Val Accuracy", fontweight="bold", fontsize=10)
    a3.set_xlabel("Epoch"); a3.set_ylabel("Accuracy (%)")
    a3.legend(fontsize=7); a3.grid(True, alpha=0.3)

    names = list(finetune_results.keys())
    finals = [finetune_results[n]["history"]["val_acc"][-1]*100 for n in names]
    params = [finetune_results[n]["params"] for n in names]
    bars = a4.bar(names, finals, color=[colors[n] for n in names])
    for bar, v, p in zip(bars, finals, params):
        a4.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.0f}%\n({p:,}p)", ha="center", fontsize=8)
    a4.set_title("Final Accuracy vs Trainable Parameters", fontweight="bold", fontsize=10)
    a4.set_ylabel("Val Accuracy (%)"); a4.set_ylim(0, 110)
    a4.tick_params(axis="x", rotation=10, labelsize=8)
    a4.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "03_bert_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Dashboard saved -> {path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#"*65)
    print("  Phase 4 -- Topic 3: BERT -- Bidirectional Encoder Pretraining")
    print("#"*65)

    section_a_corpus_demo()
    section_b_masking_verification()
    section_c_shapes_demo()

    mlm_model, pretrained_encoder, pretrain_hist, sentences, topics = pretrain_mlm(n_epochs=40)
    finetune_results = section_f_finetuning_comparison(pretrained_encoder, n_epochs=40)

    build_figures(pretrain_hist, finetune_results, mlm_model, sentences, topics)

    print("\n" + "#"*65)
    print("  DONE: Topic 3 complete.")
    print("#"*65 + "\n")


if __name__ == "__main__":
    main()
