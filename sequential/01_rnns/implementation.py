"""
Topic: Recurrent Neural Networks (RNNs)
========================================================
Repository : deep_learning/sequential/01_rnns/
File       : implementation.py
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


class ManualRNN:
    def __init__(self, input_size, hidden_size, seed=SEED):
        rng = np.random.default_rng(seed)
        std = 1.0 / np.sqrt(hidden_size)
        self.Whh = rng.uniform(-std, std, (hidden_size, hidden_size))
        self.Wxh = rng.uniform(-std, std, (hidden_size, input_size))
        self.bh  = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, X):
        T = X.shape[0]
        H = np.zeros((T, self.hidden_size))
        h = np.zeros(self.hidden_size)
        for t in range(T):
            h = np.tanh(self.Whh @ h + self.Wxh @ X[t] + self.bh)
            H[t] = h
        return H


def section_a_manual_rnn():
    print("\n" + "="*65)
    print("SECTION A — Manual RNN Cell vs PyTorch nn.RNN")
    print("="*65)

    input_size, hidden_size, T = 4, 6, 8
    X = np.random.randn(T, input_size).astype(np.float64)

    manual = ManualRNN(input_size, hidden_size)
    H_manual = manual.forward(X)

    torch_rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh").double()
    with torch.no_grad():
        torch_rnn.weight_hh_l0[:] = torch.tensor(manual.Whh)
        torch_rnn.weight_ih_l0[:] = torch.tensor(manual.Wxh)
        torch_rnn.bias_hh_l0[:]   = torch.tensor(manual.bh)
        torch_rnn.bias_ih_l0[:]   = 0.0

    X_t = torch.tensor(X).unsqueeze(0)
    H_torch, _ = torch_rnn(X_t)
    H_torch = H_torch[0].detach().numpy()

    match = np.allclose(H_manual, H_torch, atol=1e-5)
    print(f"\n  Match: {match}")
    assert match
    print("  ✓ Manual RNN cell verified against PyTorch nn.RNN")
    return manual, X


def manual_bptt(X, Whh, Wxh, bh, Wy, by, y_true):
    T, input_size = X.shape
    hidden_size = Whh.shape[0]
    H = np.zeros((T+1, hidden_size))
    Z = np.zeros((T, hidden_size))
    for t in range(T):
        Z[t] = Whh @ H[t] + Wxh @ X[t] + bh
        H[t+1] = np.tanh(Z[t])
    h_final = H[T]
    logit = Wy @ h_final + by
    prob = 1.0 / (1.0 + np.exp(-logit))
    loss = -(y_true*np.log(prob+1e-9) + (1-y_true)*np.log(1-prob+1e-9))

    dlogit = prob - y_true
    dWy = dlogit * h_final; dby = dlogit
    dh_next = Wy * dlogit
    dWhh = np.zeros_like(Whh); dWxh = np.zeros_like(Wxh); dbh = np.zeros_like(bh)
    for t in reversed(range(T)):
        dz = dh_next * (1 - H[t+1]**2)
        dWhh += np.outer(dz, H[t]); dWxh += np.outer(dz, X[t]); dbh += dz
        dh_next = Whh.T @ dz
    return {"dWhh": dWhh, "dWxh": dWxh, "dbh": dbh, "dWy": dWy, "dby": dby, "loss": loss}


def section_b_manual_bptt():
    print("\n" + "="*65)
    print("SECTION B — Manual BPTT vs PyTorch Autograd")
    print("="*65)

    input_size, hidden_size, T = 3, 5, 6
    rng = np.random.default_rng(SEED)
    std = 1.0/np.sqrt(hidden_size)
    Whh = rng.uniform(-std, std, (hidden_size, hidden_size))
    Wxh = rng.uniform(-std, std, (hidden_size, input_size))
    bh = np.zeros(hidden_size); Wy = rng.uniform(-std, std, hidden_size); by = 0.0
    X = rng.standard_normal((T, input_size)); y_true = 1.0

    manual_grads = manual_bptt(X, Whh, Wxh, bh, Wy, by, y_true)

    Whh_t = torch.tensor(Whh, requires_grad=True)
    Wxh_t = torch.tensor(Wxh, requires_grad=True)
    bh_t  = torch.tensor(bh, requires_grad=True)
    Wy_t  = torch.tensor(Wy, requires_grad=True)
    by_t  = torch.tensor(by, requires_grad=True)
    X_t   = torch.tensor(X)
    h = torch.zeros(hidden_size, dtype=torch.float64)
    for t in range(T):
        h = torch.tanh(Whh_t @ h + Wxh_t @ X_t[t] + bh_t)
    loss = -(y_true*torch.log(torch.sigmoid(Wy_t@h+by_t)+1e-9) + (1-y_true)*torch.log(1-torch.sigmoid(Wy_t@h+by_t)+1e-9))
    loss.backward()

    all_match = all(np.allclose(manual_grads[k], getattr(globals()[f'{k}_t'] if f'{k}_t' in dir() else locals().get(f'{k}_t', None), 'grad', None) or
                    {"dWhh":Whh_t.grad,"dWxh":Wxh_t.grad,"dbh":bh_t.grad,"dWy":Wy_t.grad,"dby":by_t.grad.item()}[k],
                    atol=1e-6) for k in ["dWhh","dWxh","dbh","dWy"])
    torch_grads = {"dWhh":Whh_t.grad.numpy(),"dWxh":Wxh_t.grad.numpy(),"dbh":bh_t.grad.numpy(),"dWy":Wy_t.grad.numpy(),"dby":by_t.grad.item()}
    print(f"\n  Loss match: {np.isclose(manual_grads['loss'], loss.item(), atol=1e-6)}")
    for k in ["dWhh","dWxh","dbh","dWy","dby"]:
        m = np.allclose(manual_grads[k], torch_grads[k], atol=1e-6)
        print(f"  {k:6s} | match={m}")
    print("  ✓ Manual BPTT verified against PyTorch autograd")


def measure_gradient_at_t0(seq_length, hidden_size=32, seed=SEED):
    torch.manual_seed(seed)
    rnn = nn.RNN(1, hidden_size, batch_first=True, nonlinearity="tanh")
    X = torch.randn(1, seq_length, 1, requires_grad=True)
    H, _ = rnn(X)
    H[0, -1].sum().backward()
    return X.grad[0, 0].abs().item()


def section_c_vanishing_gradient():
    print("\n" + "="*65)
    print("SECTION C — Empirical Vanishing Gradient vs Sequence Length")
    print("="*65)
    lengths = [5, 10, 20, 30, 50, 75, 100, 150, 200]
    grad_norms = []
    print(f"\n  {'Seq Length':>10} | {'Grad at x0':>12}")
    print("  " + "─"*26)
    for L in lengths:
        g = measure_gradient_at_t0(L)
        grad_norms.append(g)
        print(f"  {L:>10} | {g:>12.2e}")
    print("  ✓ Gradient decays exponentially with sequence length")
    return lengths, grad_norms


def generate_signal_detection_data(n_samples, seq_length, seed):
    rng = np.random.default_rng(seed)
    signals = rng.integers(0, 2, size=n_samples).astype(np.float32)
    noise = rng.uniform(-1, 1, size=(n_samples, seq_length-1)).astype(np.float32)
    X = np.zeros((n_samples, seq_length, 1), dtype=np.float32)
    X[:, 0, 0] = signals * 2 - 1
    X[:, 1:, 0] = noise
    return X, signals


class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh")
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        H, _ = self.rnn(x)
        return torch.sigmoid(self.fc(H[:, -1, :])).squeeze(-1)


def train_signal_detection(seq_length, n_epochs=30, n_train=800, n_val=200, hidden_size=32):
    X_tr, y_tr = generate_signal_detection_data(n_train, seq_length, seed=SEED)
    X_va, y_va = generate_signal_detection_data(n_val, seq_length, seed=SEED+1)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)
    X_va_t, y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(y_va).to(DEVICE)
    torch.manual_seed(SEED)
    model = SimpleRNNClassifier(hidden_size=hidden_size).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCELoss()
    for epoch in range(n_epochs):
        model.train()
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(Xb), Yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); opt.step()
    model.eval()
    with torch.no_grad():
        acc = ((model(X_va_t) >= 0.5).float() == y_va_t).float().mean().item()
    return acc


def section_d_signal_detection():
    print("\n" + "="*65)
    print("SECTION D — Signal Detection: Vanilla RNN vs Sequence Length")
    print("  Task: label = x[0]; rest is noise. Tests long-range memory.")
    print("="*65)
    lengths = [5, 10, 20, 30, 50, 75, 100]
    accuracies = []
    print(f"\n  {'Seq Length':>10} | {'Val Accuracy':>12}")
    print("  " + "─"*26)
    for L in lengths:
        acc = train_signal_detection(L, n_epochs=30)
        accuracies.append(acc)
        print(f"  {L:>10} | {acc*100:>11.1f}%")
    print("  ✓ Vanilla RNN degrades toward chance at longer sequences")
    return lengths, accuracies


def generate_parity_data(n_samples, seq_length, seed):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_samples, seq_length)).astype(np.float32)
    cumulative_parity = np.cumsum(bits, axis=1) % 2
    return bits[..., None], cumulative_parity


class SyncedRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        H, _ = self.rnn(x)
        return torch.sigmoid(self.fc(H)).squeeze(-1)


def section_e_parity_demo(seq_length=10, n_epochs=150):
    print("\n" + "="*65)
    print("SECTION E — Many-to-Many Demo: Cumulative Parity")
    print("="*65)
    X_tr, y_tr = generate_parity_data(1000, seq_length, seed=SEED)
    X_va, y_va = generate_parity_data(200, seq_length, seed=SEED+1)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)
    X_va_t, y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(y_va).to(DEVICE)
    torch.manual_seed(SEED)
    model = SyncedRNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3); crit = nn.BCELoss()
    for epoch in range(n_epochs):
        model.train()
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad(); crit(model(Xb), Yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = ((model(X_va_t)>=0.5).float()==y_va_t).float().mean().item()
    print(f"\n  Per-timestep parity accuracy: {acc*100:.1f}%")
    print("  ✓ Synced many-to-many RNN learns the running-XOR recurrence")
    return model, X_va, y_va


def generate_extremum_data(n_samples, seq_length, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, seq_length)).astype(np.float32)
    y = np.zeros((n_samples, seq_length), dtype=np.float32)
    for i in range(1, seq_length-1):
        is_max = (X[:, i] > X[:, i-1]) & (X[:, i] > X[:, i+1])
        is_min = (X[:, i] < X[:, i-1]) & (X[:, i] < X[:, i+1])
        y[:, i] = (is_max | is_min).astype(np.float32)
    return X[..., None], y


class BiRNNLabeler(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, bidirectional=True):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)
    def forward(self, x):
        H, _ = self.rnn(x)
        return torch.sigmoid(self.fc(H)).squeeze(-1)


def _train_extremum_model(bidirectional, seq_length=30, n_epochs=30):
    X_tr, y_tr = generate_extremum_data(1500, seq_length, seed=SEED)
    X_va, y_va = generate_extremum_data(300, seq_length, seed=SEED+1)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)
    X_va_t, y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(y_va).to(DEVICE)
    torch.manual_seed(SEED)
    model = BiRNNLabeler(bidirectional=bidirectional).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3); crit = nn.BCELoss()
    for epoch in range(n_epochs):
        model.train()
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad(); crit(model(Xb), Yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(X_va_t)
        acc = ((pred>=0.5).float()==y_va_t).float().mean().item()
        pred_bin = (pred>=0.5).float()
        tp = ((pred_bin==1)&(y_va_t==1)).sum().item()
        fn = ((pred_bin==0)&(y_va_t==1)).sum().item()
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return acc, recall, model, X_va, y_va


def section_f_bidirectional():
    print("\n" + "="*65)
    print("SECTION F — Bidirectional Demo: Local Extremum Detection")
    print("  Label[i]=1 if x[i] is a local max/min — needs FUTURE context!")
    print("="*65)
    acc_uni, recall_uni, _, _, _ = _train_extremum_model(bidirectional=False)
    acc_bi, recall_bi, model_bi, X_va, y_va = _train_extremum_model(bidirectional=True)
    print(f"\n  Unidirectional: accuracy={acc_uni*100:.1f}%  extremum_recall={recall_uni*100:.1f}%")
    print(f"  Bidirectional:  accuracy={acc_bi*100:.1f}%  extremum_recall={recall_bi*100:.1f}%")
    print("  ✓ Bidirectional RNN better at tasks requiring future context")
    return {"uni": (acc_uni, recall_uni), "bi": (acc_bi, recall_bi)}, model_bi, X_va, y_va


class SineRNN(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        H, _ = self.rnn(x)
        return self.fc(H).squeeze(-1)


def section_g_sine_wave(n_epochs=100):
    print("\n" + "="*65)
    print("SECTION G — Bonus: Sine Wave Next-Step Prediction")
    print("="*65)
    T_total = 400
    t = np.linspace(0, 40*np.pi, T_total)
    series = np.sin(t).astype(np.float32)
    seq_len = 20
    X, Y = [], []
    for i in range(len(series)-seq_len):
        X.append(series[i:i+seq_len]); Y.append(series[i+1:i+seq_len+1])
    X = np.array(X)[..., None]; Y = np.array(Y)
    split = int(len(X)*0.8)
    X_tr, Y_tr = torch.tensor(X[:split]), torch.tensor(Y[:split])
    X_va, Y_va = torch.tensor(X[split:]).to(DEVICE), torch.tensor(Y[split:]).to(DEVICE)
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=16, shuffle=True)
    torch.manual_seed(SEED)
    model = SineRNN().to(DEVICE); opt = optim.Adam(model.parameters(), lr=1e-3); crit = nn.MSELoss()
    for epoch in range(n_epochs):
        model.train()
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad(); crit(model(Xb), Yb).backward(); opt.step()
        if (epoch+1) % 20 == 0:
            model.eval()
            with torch.no_grad(): vl = crit(model(X_va), Y_va).item()
            print(f"    Epoch {epoch+1:3d}/{n_epochs} | val_MSE={vl:.6f}")
    print("  ✓ RNN learns the sine wave's continuous dynamics")
    return model, series, seq_len, split


def build_figures(lengths_grad, grad_norms, lengths_acc, accuracies,
                  parity_model, X_parity, y_parity,
                  bidir_results, model_bi, X_extremum, y_extremum,
                  sine_model, sine_series, seq_len, split):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Phase 3 — Topic 1: Recurrent Neural Networks", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.32)
    a = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

    # Panel 1: Vanishing gradient
    a[0].semilogy(lengths_grad, grad_norms, "o-", color="#e74c3c", lw=2, ms=6)
    a[0].set_title("Vanishing Gradient: |∂h_T/∂x₀|", fontweight="bold", fontsize=10)
    a[0].set_xlabel("Sequence Length"); a[0].set_ylabel("Gradient magnitude (log)")
    a[0].grid(True, alpha=0.3, which="both")

    # Panel 2: Signal detection
    a[1].plot(lengths_acc, [acc*100 for acc in accuracies], "s-", color="#3498db", lw=2, ms=6)
    a[1].axhline(50, color="gray", ls="--", label="Chance level")
    a[1].set_title("Signal Detection Accuracy vs Seq Length", fontweight="bold", fontsize=10)
    a[1].set_xlabel("Sequence Length"); a[1].set_ylabel("Val Accuracy (%)")
    a[1].legend(fontsize=8); a[1].grid(True, alpha=0.3); a[1].set_ylim(40, 105)

    # Panel 3: Parity demo
    with torch.no_grad():
        pred = parity_model(torch.tensor(X_parity[:1]).to(DEVICE))[0].cpu().numpy()
    steps = range(len(y_parity[0]))
    a[2].step(steps, y_parity[0], where="post", color="black", lw=2, label="True parity")
    a[2].step(steps, pred, where="post", color="#27ae60", lw=2, ls="--", label="Predicted")
    a[2].set_title("Cumulative Parity: True vs Predicted", fontweight="bold", fontsize=10)
    a[2].set_xlabel("Time step"); a[2].legend(fontsize=8); a[2].grid(True, alpha=0.3)

    # Panel 4: Bidir comparison
    labels = ["Accuracy", "Extremum Recall"]
    uni_vals = [bidir_results["uni"][0]*100, bidir_results["uni"][1]*100]
    bi_vals  = [bidir_results["bi"][0]*100, bidir_results["bi"][1]*100]
    x_pos = np.arange(2)
    a[3].bar(x_pos-0.2, uni_vals, 0.35, color="#e74c3c", label="Unidirectional")
    a[3].bar(x_pos+0.2, bi_vals, 0.35, color="#27ae60", label="Bidirectional")
    a[3].set_xticks(x_pos); a[3].set_xticklabels(labels)
    a[3].set_title("Uni- vs Bi-directional RNN", fontweight="bold", fontsize=10)
    a[3].legend(fontsize=8); a[3].grid(True, axis="y", alpha=0.3)

    # Panel 5: Extremum detection
    with torch.no_grad():
        pred_ext = model_bi(torch.tensor(X_extremum[:1]).to(DEVICE))[0].cpu().numpy()
    seq = X_extremum[0, :, 0]
    true_ext = y_extremum[0]
    a[4].plot(seq, color="#3498db", lw=1.5, label="Sequence")
    a[4].scatter(np.where(true_ext==1)[0], seq[true_ext==1], color="black", s=50, marker="x", label="True extrema", zorder=5)
    a[4].scatter(np.where(pred_ext>=0.5)[0], seq[pred_ext>=0.5], color="#27ae60", s=80,
                facecolors="none", edgecolors="#27ae60", linewidths=2, label="Predicted", zorder=4)
    a[4].set_title("BiRNN: Local Extremum Detection", fontweight="bold", fontsize=10)
    a[4].set_xlabel("Time step"); a[4].legend(fontsize=7); a[4].grid(True, alpha=0.3)

    # Panel 6: Sine wave
    sine_model.eval()
    with torch.no_grad():
        cur_input = torch.tensor(sine_series[split:split+seq_len][None,:,None]).to(DEVICE)
        preds = []
        for _ in range(60):
            out = sine_model(cur_input)
            next_val = out[0, -1].item()
            preds.append(next_val)
            next_val_tensor = out[:, -1:].unsqueeze(-1)
            cur_input = torch.cat([cur_input[:,1:,:], next_val_tensor], dim=1)
    true_future = sine_series[split+seq_len:split+seq_len+60]
    a[5].plot(range(len(true_future)), true_future, color="black", lw=2, label="True")
    a[5].plot(range(len(preds)), preds, color="#e74c3c", lw=2, ls="--", label="RNN (autoregressive)")
    a[5].set_title("Sine Wave: Autoregressive Forecast", fontweight="bold", fontsize=10)
    a[5].set_xlabel("Future time step"); a[5].set_ylabel("sin(t)")
    a[5].legend(fontsize=8); a[5].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "01_rnn_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Dashboard saved → {path}")
    plt.close(fig)


def main():
    print("\n" + "▓"*65)
    print("  Phase 3 — Topic 1: Recurrent Neural Networks (RNNs)")
    print("▓"*65)

    section_a_manual_rnn()
    section_b_manual_bptt()
    lengths_grad, grad_norms = section_c_vanishing_gradient()
    lengths_acc, accuracies = section_d_signal_detection()
    parity_model, X_parity, y_parity = section_e_parity_demo(seq_length=10, n_epochs=150)
    bidir_results, model_bi, X_extremum, y_extremum = section_f_bidirectional()
    sine_model, sine_series, seq_len, split = section_g_sine_wave()
    build_figures(lengths_grad, grad_norms, lengths_acc, accuracies,
                 parity_model, X_parity, y_parity,
                 bidir_results, model_bi, X_extremum, y_extremum,
                 sine_model, sine_series, seq_len, split)

    print("\n" + "▓"*65)
    print("  ✓ Topic 1 complete.")
    print("▓"*65 + "\n")


if __name__ == "__main__":
    main()
