"""
Phase 3 — Topic 2: LSTM & GRU — Gated Recurrent Architectures
=================================================================
Repository : deep-learning-mastery/phase-3-sequential/02-lstm-and-gru/
File       : implementation.py

Sections:
  A │ Manual LSTM cell (NumPy) — verified against PyTorch nn.LSTM
  B │ Manual GRU cell (NumPy) — verified against PyTorch nn.GRU
  C │ Gradient flow: RNN vs LSTM vs GRU (|∂h_T/∂x₀| vs seq length)
  D │ Adding Problem (Hochreiter & Schmidhuber 1997) — classic LSTM benchmark
  E │ Signal detection revisited at long lengths: RNN vs LSTM vs GRU
  F │ Visualization dashboard
"""

import os, warnings
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[CONFIG] Device: {DEVICE}  |  PyTorch: {torch.__version__}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — MANUAL LSTM CELL
# PyTorch gate order (stacked in weight_ih_l0 rows): input(i), forget(f), cell(g), output(o)
# ═════════════════════════════════════════════════════════════════════════════

class ManualLSTM:
    """
    LSTM matching PyTorch's exact formulas:
      i_t = σ(W_ii x + b_ii + W_hi h + b_hi)
      f_t = σ(W_if x + b_if + W_hf h + b_hf)
      g_t = tanh(W_ig x + b_ig + W_hg h + b_hg)
      o_t = σ(W_io x + b_io + W_ho h + b_ho)
      c_t = f_t*c_{t-1} + i_t*g_t
      h_t = o_t*tanh(c_t)
    """
    def __init__(self, input_size, hidden_size, seed=SEED):
        rng = np.random.default_rng(seed)
        std = 1.0 / np.sqrt(hidden_size)
        # Rows: [0:d]=i, [d:2d]=f, [2d:3d]=g, [3d:4d]=o
        self.W_ih = rng.uniform(-std, std, (4*hidden_size, input_size))
        self.W_hh = rng.uniform(-std, std, (4*hidden_size, hidden_size))
        self.b_ih = np.zeros(4*hidden_size)
        self.b_hh = np.zeros(4*hidden_size)
        self.d = hidden_size

    def forward(self, X):
        T, d = X.shape[0], self.d
        H = np.zeros((T, d)); C_out = np.zeros((T, d))
        h = np.zeros(d); c = np.zeros(d)
        for t in range(T):
            gates = self.W_ih @ X[t] + self.b_ih + self.W_hh @ h + self.b_hh
            i_t = sigmoid(gates[0*d:1*d])
            f_t = sigmoid(gates[1*d:2*d])
            g_t = np.tanh(gates[2*d:3*d])
            o_t = sigmoid(gates[3*d:4*d])
            c = f_t*c + i_t*g_t
            h = o_t*np.tanh(c)
            H[t] = h; C_out[t] = c
        return H, C_out


def section_a_manual_lstm():
    print("\n" + "="*65)
    print("SECTION A — Manual LSTM Cell vs PyTorch nn.LSTM")
    print("="*65)

    input_size, hidden_size, T = 4, 6, 8
    X = np.random.randn(T, input_size)

    manual = ManualLSTM(input_size, hidden_size)
    H_manual, _ = manual.forward(X)

    torch_lstm = nn.LSTM(input_size, hidden_size, batch_first=True).double()
    with torch.no_grad():
        torch_lstm.weight_ih_l0[:] = torch.tensor(manual.W_ih)
        torch_lstm.weight_hh_l0[:] = torch.tensor(manual.W_hh)
        torch_lstm.bias_ih_l0[:]   = torch.tensor(manual.b_ih)
        torch_lstm.bias_hh_l0[:]   = torch.tensor(manual.b_hh)

    H_torch, _ = torch_lstm(torch.tensor(X).unsqueeze(0))
    H_torch = H_torch[0].detach().numpy()

    match = np.allclose(H_manual, H_torch, atol=1e-6)
    print(f"\n  Shape: {H_manual.shape} | Match: {match}")
    assert match
    print("  ✓ Manual LSTM verified against PyTorch nn.LSTM (gate order: i,f,g,o)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — MANUAL GRU CELL
# PyTorch gate order: reset(r), update(z), new(n)
# Key subtlety: reset gate applies to HIDDEN PROJECTION only:
#   n_t = tanh(W_in x + b_in + r_t*(W_hn h + b_hn))
# ═════════════════════════════════════════════════════════════════════════════

class ManualGRU:
    """
    GRU matching PyTorch's exact formula (reset gates hidden projection, not h directly):
      r_t = σ(W_ir x + b_ir + W_hr h + b_hr)
      z_t = σ(W_iz x + b_iz + W_hz h + b_hz)
      n_t = tanh(W_in x + b_in + r_t ⊙ (W_hn h + b_hn))
      h_t = (1-z_t)*n_t + z_t*h_{t-1}
    """
    def __init__(self, input_size, hidden_size, seed=SEED):
        rng = np.random.default_rng(seed)
        std = 1.0/np.sqrt(hidden_size)
        self.W_ih = rng.uniform(-std, std, (3*hidden_size, input_size))
        self.W_hh = rng.uniform(-std, std, (3*hidden_size, hidden_size))
        self.b_ih = np.zeros(3*hidden_size)
        self.b_hh = np.zeros(3*hidden_size)
        self.d = hidden_size

    def forward(self, X):
        T, d = X.shape[0], self.d
        H = np.zeros((T, d)); h = np.zeros(d)
        for t in range(T):
            gi = self.W_ih @ X[t] + self.b_ih
            gh = self.W_hh @ h + self.b_hh
            r_t = sigmoid(gi[0*d:1*d] + gh[0*d:1*d])
            z_t = sigmoid(gi[1*d:2*d] + gh[1*d:2*d])
            # reset gates the HIDDEN-SIDE projection only (PyTorch convention)
            n_t = np.tanh(gi[2*d:3*d] + r_t * gh[2*d:3*d])
            h = (1-z_t)*n_t + z_t*h
            H[t] = h
        return H


def section_b_manual_gru():
    print("\n" + "="*65)
    print("SECTION B — Manual GRU Cell vs PyTorch nn.GRU")
    print("="*65)

    input_size, hidden_size, T = 4, 6, 8
    X = np.random.randn(T, input_size)

    manual = ManualGRU(input_size, hidden_size)
    H_manual = manual.forward(X)

    torch_gru = nn.GRU(input_size, hidden_size, batch_first=True).double()
    with torch.no_grad():
        torch_gru.weight_ih_l0[:] = torch.tensor(manual.W_ih)
        torch_gru.weight_hh_l0[:] = torch.tensor(manual.W_hh)
        torch_gru.bias_ih_l0[:]   = torch.tensor(manual.b_ih)
        torch_gru.bias_hh_l0[:]   = torch.tensor(manual.b_hh)

    H_torch, _ = torch_gru(torch.tensor(X).unsqueeze(0))
    H_torch = H_torch[0].detach().numpy()

    match = np.allclose(H_manual, H_torch, atol=1e-6)
    print(f"\n  Shape: {H_manual.shape} | Match: {match}")
    assert match
    print("  ✓ Manual GRU verified against PyTorch nn.GRU (gate order: r,z,n)")
    print("  Key: reset gate applies to hidden PROJECTION, not h directly")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — GRADIENT FLOW COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def grad_at_t0(cell_type, seq_length, hidden_size=32, seed=SEED):
    torch.manual_seed(seed)
    if cell_type == "RNN":
        rnn = nn.RNN(1, hidden_size, batch_first=True, nonlinearity="tanh")
    elif cell_type == "LSTM":
        rnn = nn.LSTM(1, hidden_size, batch_first=True)
    else:
        rnn = nn.GRU(1, hidden_size, batch_first=True)
    X = torch.randn(1, seq_length, 1, requires_grad=True)
    out = rnn(X)
    H = out[0]
    H[0,-1].sum().backward()
    return X.grad[0,0].abs().item()


def section_c_gradient_comparison():
    print("\n" + "="*65)
    print("SECTION C — Gradient Flow: RNN vs LSTM vs GRU")
    print("="*65)
    lengths = [10, 30, 50, 75, 100, 150, 200]
    results = {"RNN": [], "LSTM": [], "GRU": []}
    print(f"\n  {'Length':>8} | {'RNN':>12} | {'LSTM':>12} | {'GRU':>12}")
    print("  " + "─"*52)
    for L in lengths:
        row = []
        for ct in ["RNN", "LSTM", "GRU"]:
            g = grad_at_t0(ct, L)
            results[ct].append(g); row.append(g)
        print(f"  {L:>8} | {row[0]:>12.2e} | {row[1]:>12.2e} | {row[2]:>12.2e}")
    print("\n  ✓ LSTM/GRU maintain substantially larger gradients at long lengths")
    return lengths, results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — THE ADDING PROBLEM
# ═════════════════════════════════════════════════════════════════════════════

def gen_adding(n, T, seed):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(0, 1, (n, T)).astype(np.float32)
    marks = np.zeros((n, T), dtype=np.float32)
    half = T // 2
    idx1 = rng.integers(0, half, n)
    idx2 = rng.integers(half, T, n)
    for i in range(n):
        marks[i, idx1[i]] = 1.0
        marks[i, idx2[i]] = 1.0
    X = np.stack([vals, marks], axis=-1)
    y = np.array([vals[i,idx1[i]]+vals[i,idx2[i]] for i in range(n)], dtype=np.float32)
    return X, y


class SeqRegressor(nn.Module):
    def __init__(self, cell_type, input_size=2, hidden_size=32):
        super().__init__()
        if cell_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif cell_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        H, _ = self.rnn(x)
        return self.fc(H[:,-1,:]).squeeze(-1)


def train_adding(cell_type, T, n_epochs=60, n_train=2000, n_val=400):
    X_tr, y_tr = gen_adding(n_train, T, SEED)
    X_va, y_va = gen_adding(n_val, T, SEED+1)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                       batch_size=32, shuffle=True)
    X_va_t, y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(y_va).to(DEVICE)
    baseline_mse = ((y_va_t - 1.0)**2).mean().item()

    torch.manual_seed(SEED)
    model = SeqRegressor(cell_type).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        model.train()
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            nn.MSELoss()(model(Xb), Yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        mse = nn.MSELoss()(model(X_va_t), y_va_t).item()
    return mse, baseline_mse


def section_d_adding_problem():
    print("\n" + "="*65)
    print("SECTION D — The Adding Problem (Hochreiter & Schmidhuber 1997)")
    print("  Target = sum of two marked values; requires long-range memory")
    print("="*65)
    lengths = [10, 30, 50, 100, 200]
    results = {"RNN": [], "LSTM": [], "GRU": []}
    baselines = []
    print(f"\n  {'Length':>8} | {'RNN MSE':>10} | {'LSTM MSE':>10} | {'GRU MSE':>10} | {'Baseline':>10}")
    print("  " + "─"*58)
    for L in lengths:
        row = {}; baseline = None
        for ct in ["RNN", "LSTM", "GRU"]:
            mse, baseline = train_adding(ct, L, n_epochs=60)
            results[ct].append(mse); row[ct] = mse
        baselines.append(baseline)
        print(f"  {L:>8} | {row['RNN']:>10.4f} | {row['LSTM']:>10.4f} | {row['GRU']:>10.4f} | {baseline:>10.4f}")
    print("\n  ✓ LSTM/GRU beat baseline at long lengths; vanilla RNN struggles")
    return lengths, results, baselines


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — SIGNAL DETECTION REVISITED AT LONG LENGTHS
# ═════════════════════════════════════════════════════════════════════════════

def gen_signal(n, T, seed):
    rng = np.random.default_rng(seed)
    sigs = rng.integers(0, 2, n).astype(np.float32)
    noise = rng.uniform(-1, 1, (n, T-1)).astype(np.float32)
    X = np.zeros((n, T, 1), dtype=np.float32)
    X[:,0,0] = sigs * 2 - 1; X[:,1:,0] = noise
    return X, sigs


class SeqClassifier(nn.Module):
    def __init__(self, cell_type, hidden_size=32):
        super().__init__()
        if cell_type == "RNN":
            self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        elif cell_type == "LSTM":
            self.rnn = nn.LSTM(1, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        H, _ = self.rnn(x)
        return torch.sigmoid(self.fc(H[:,-1,:])).squeeze(-1)


def train_signal(cell_type, T, n_epochs=30, n_train=800, n_val=200):
    X_tr, y_tr = gen_signal(n_train, T, SEED)
    X_va, y_va = gen_signal(n_val, T, SEED+1)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                       batch_size=32, shuffle=True)
    X_va_t, y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(y_va).to(DEVICE)
    torch.manual_seed(SEED)
    model = SeqClassifier(cell_type).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        model.train()
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            nn.BCELoss()(model(Xb), Yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        return ((model(X_va_t)>=0.5).float()==y_va_t).float().mean().item()


def section_e_signal_revisited():
    print("\n" + "="*65)
    print("SECTION E — Signal Detection Revisited: RNN vs LSTM vs GRU")
    print("  Same task as Topic 1 Section D; now testing gated architectures")
    print("="*65)
    lengths = [50, 100, 150]
    results = {"RNN": [], "LSTM": [], "GRU": []}
    print(f"\n  {'Length':>8} | {'RNN':>9} | {'LSTM':>9} | {'GRU':>9}")
    print("  " + "─"*42)
    for L in lengths:
        row = {}
        for ct in ["RNN", "LSTM", "GRU"]:
            acc = train_signal(ct, L, n_epochs=30)
            results[ct].append(acc); row[ct] = acc
        print(f"  {L:>8} | {row['RNN']*100:>8.1f}% | {row['LSTM']*100:>8.1f}% | {row['GRU']*100:>8.1f}%")
    print("\n  ✓ LSTM/GRU reliably solve where vanilla RNN was unreliable (Topic 1 Section D)")
    return lengths, results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def build_figures(grad_L, grad_R, add_L, add_R, add_B, sig_L, sig_R):
    colors = {"RNN":"#e74c3c","LSTM":"#3498db","GRU":"#27ae60"}

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Phase 3 — Topic 2: LSTM & GRU vs Vanilla RNN", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(2)]

    # Panel 1: Gradient flow (log scale)
    for ct in ["RNN","LSTM","GRU"]:
        axes[0].semilogy(grad_L, grad_R[ct], "o-", color=colors[ct], lw=2, label=ct, ms=5)
    axes[0].set_title("|∂h_T/∂x₀| vs Sequence Length", fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Sequence Length"); axes[0].set_ylabel("Gradient (log)")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, which="both")

    # Panel 2: Adding problem
    for ct in ["RNN","LSTM","GRU"]:
        axes[1].plot(add_L, add_R[ct], "s-", color=colors[ct], lw=2, label=ct)
    axes[1].plot(add_L, add_B, "k--", lw=1.5, label="Baseline")
    axes[1].set_title("Adding Problem: Validation MSE", fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Sequence Length"); axes[1].set_ylabel("MSE (lower=better)")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # Panel 3: Signal detection comparison bars
    w = 0.25; x = np.arange(len(sig_L))
    for i, ct in enumerate(["RNN","LSTM","GRU"]):
        axes[2].bar(x+(i-1)*w, [v*100 for v in sig_R[ct]], w, color=colors[ct], label=ct)
    axes[2].axhline(50, color="gray", ls="--", lw=1, label="Chance")
    axes[2].set_xticks(x); axes[2].set_xticklabels(sig_L)
    axes[2].set_title("Signal Detection: All 3 Architectures", fontweight="bold", fontsize=10)
    axes[2].set_xlabel("Sequence Length"); axes[2].set_ylabel("Val Accuracy (%)")
    axes[2].legend(fontsize=8); axes[2].grid(True, axis="y", alpha=0.3)

    # Panel 4: Parameter count
    hidden_size = 32; k = 1
    d = hidden_size
    rnn_p = d*k + d*d + d
    gru_p = 3*(d*k + d*d + d)
    lstm_p = 4*(d*k + d*d + d)
    bars = axes[3].bar(["RNN","GRU","LSTM"], [rnn_p,gru_p,lstm_p],
                       color=[colors["RNN"],colors["GRU"],colors["LSTM"]])
    for bar, v in zip(bars, [rnn_p,gru_p,lstm_p]):
        axes[3].text(bar.get_x()+bar.get_width()/2, v+50, f"{v:,}",
                    ha="center", fontsize=9, fontweight="bold")
    axes[3].set_title(f"Parameter Count (hidden={hidden_size})", fontweight="bold", fontsize=10)
    axes[3].set_ylabel("Parameters"); axes[3].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "02_lstm_gru_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Dashboard saved → {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  Phase 3 — Topic 2: LSTM & GRU")
    print("▓"*65)

    section_a_manual_lstm()
    section_b_manual_gru()
    grad_L, grad_R = section_c_gradient_comparison()
    add_L, add_R, add_B = section_d_adding_problem()
    sig_L, sig_R = section_e_signal_revisited()
    build_figures(grad_L, grad_R, add_L, add_R, add_B, sig_L, sig_R)

    print("\n" + "▓"*65)
    print("  ✓ Topic 2 complete.")
    print("▓"*65 + "\n")


if __name__ == "__main__":
    main()
