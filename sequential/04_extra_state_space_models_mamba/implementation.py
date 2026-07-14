"""
Phase 3 -- Topic 4 (Extra): State Space Models -- S4 & Mamba
================================================================
Repository : deep-learning-mastery/phase-3-sequential/04-extra-state-space-models-mamba/
File       : implementation.py

Sections:
  A | Continuous-time SSM + ZOH discretization -- verified against fine-grained Euler simulation
  B | Linear SSM -- Recurrent form (from scratch)
  C | Linear SSM -- Convolutional form -- verified EXACTLY equivalent to recurrent form (the S4 insight)
  D | HiPPO matrix construction & visualization
  E | Selective SSM (Mamba-style S6) -- input-dependent Delta,B,C via sequential scan
  F | Benchmark: Selective SSM vs RNN vs LSTM vs GRU on Signal Detection (ties back to Topics 1-2)
  G | Visualization dashboard
"""

import os, time, warnings
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
# SECTION A -- CONTINUOUS-TIME SSM + ZOH DISCRETIZATION
# =============================================================================

def zoh_discretize(a: float, b: float, delta: float):
    """
    Zero-order hold discretization for a SCALAR (single diagonal entry) SSM:
      a_bar = exp(delta*a)
      b_bar = (1/a)(a_bar - 1)*b     (when a!=0)
    """
    a_bar = np.exp(delta * a)
    b_bar = (1.0/a) * (a_bar - 1.0) * b if abs(a) > 1e-8 else delta * b
    return a_bar, b_bar


def simulate_continuous_euler(a, b, c, u_signal, dt_fine, delta_coarse):
    """
    Ground-truth continuous simulation via fine-grained forward Euler
    integration (many small steps per coarse delta), used to VALIDATE that
    our discrete ZOH recurrence matches the true continuous-time dynamics.
    """
    steps_per_delta = int(round(delta_coarse / dt_fine))
    x = 0.0
    y_coarse = []
    for u_k in u_signal:
        for _ in range(steps_per_delta):
            x = x + dt_fine * (a*x + b*u_k)     # forward Euler: x += dt * x'(t)
        y_coarse.append(c * x)
    return np.array(y_coarse)


def simulate_discrete_recurrence(a_bar, b_bar, c, u_signal):
    """Discrete ZOH recurrence: x_k = a_bar*x_{k-1} + b_bar*u_k,  y_k = c*x_k"""
    x = 0.0
    y = []
    for u_k in u_signal:
        x = a_bar*x + b_bar*u_k
        y.append(c*x)
    return np.array(y)


def section_a_discretization():
    print("\n" + "="*65)
    print("SECTION A -- Continuous SSM + ZOH Discretization vs Fine Euler Simulation")
    print("="*65)

    a, b, c = -2.0, 1.0, 1.0    # stable (a<0), scalar SSM
    delta = 0.1                  # coarse discretization step
    rng = np.random.default_rng(SEED)
    u_signal = rng.uniform(-1, 1, size=30)

    a_bar, b_bar = zoh_discretize(a, b, delta)
    print(f"\n  Continuous params: a={a}, b={b}, c={c}, delta={delta}")
    print(f"  ZOH discretized:   a_bar={a_bar:.6f}, b_bar={b_bar:.6f}")

    y_discrete = simulate_discrete_recurrence(a_bar, b_bar, c, u_signal)
    y_continuous = simulate_continuous_euler(a, b, c, u_signal, dt_fine=delta/1000, delta_coarse=delta)

    max_abs_err = np.max(np.abs(y_discrete - y_continuous))
    rel_err = max_abs_err / (np.max(np.abs(y_continuous)) + 1e-9)
    print(f"\n  Max |discrete - fine_euler| error: {max_abs_err:.2e}")
    print(f"  Relative error: {rel_err*100:.4f}%")
    print("  OK: ZOH discretization matches fine-grained continuous simulation (exact for LTI systems)")
    assert rel_err < 0.01

    return u_signal, y_discrete, y_continuous, (a, b, c, delta)


# =============================================================================
# SECTION B & C -- RECURRENT vs CONVOLUTIONAL VIEW (the key S4 insight)
# =============================================================================

def linear_ssm_recurrent(a_bar: np.ndarray, b_bar: np.ndarray, c: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Multi-dimensional diagonal-state LTI SSM, RECURRENT computation.
    a_bar, b_bar, c: (N,) diagonal state params.  u: (L,) scalar input sequence.
    Returns: y (L,) scalar output sequence.
    """
    N = len(a_bar)
    x = np.zeros(N)
    y = np.zeros(len(u))
    for k in range(len(u)):
        x = a_bar * x + b_bar * u[k]     # elementwise on N independent diagonal channels
        y[k] = np.sum(c * x)              # readout: sum over N state dims
    return y


def build_conv_kernel(a_bar: np.ndarray, b_bar: np.ndarray, c: np.ndarray, L: int) -> np.ndarray:
    """
    Build the SSM's convolution kernel K_j = C . A_bar^j . B_bar  for j=0,...,L-1  (theory.md sec 5).
    """
    N = len(a_bar)
    K = np.zeros(L)
    a_power = np.ones(N)    # A_bar^0 = I (diagonal ones)
    for j in range(L):
        K[j] = np.sum(c * a_power * b_bar)
        a_power = a_power * a_bar    # A_bar^(j+1) = A_bar^j * A_bar  (elementwise for diagonal)
    return K


def causal_convolve(u: np.ndarray, K: np.ndarray) -> np.ndarray:
    """y_k = sum_{j=0}^{k} K_j * u_{k-j}   (causal convolution, no future leakage)"""
    L = len(u)
    y = np.zeros(L)
    for k in range(L):
        for j in range(min(k+1, len(K))):
            y[k] += K[j] * u[k-j]
    return y


def section_bc_recurrent_vs_convolutional():
    print("\n" + "="*65)
    print("SECTION B/C -- Recurrent vs Convolutional View: The Key S4 Insight")
    print("="*65)

    N = 4    # state dimension
    L = 25   # sequence length
    rng = np.random.default_rng(SEED)

    a_diag = -rng.uniform(0.5, 3.0, size=N)
    b_diag = rng.standard_normal(N)
    c_diag = rng.standard_normal(N)
    delta = 0.2

    a_bar = np.exp(delta * a_diag)
    b_bar = np.array([zoh_discretize(a_diag[i], b_diag[i], delta)[1] for i in range(N)])

    u = rng.standard_normal(L)

    print(f"\n  State dimension N={N}, sequence length L={L}")

    y_recurrent = linear_ssm_recurrent(a_bar, b_bar, c_diag, u)

    K = build_conv_kernel(a_bar, b_bar, c_diag, L)
    y_conv = causal_convolve(u, K)

    match = np.allclose(y_recurrent, y_conv, atol=1e-8)
    max_err = np.max(np.abs(y_recurrent - y_conv))
    print(f"\n  Recurrent output (first 5): {y_recurrent[:5].round(4)}")
    print(f"  Convolutional output (first 5): {y_conv[:5].round(4)}")
    print(f"  Max absolute difference: {max_err:.2e}")
    print(f"  EXACT match: {match}")
    assert match

    print("\n  OK: Recurrent and Convolutional views produce IDENTICAL outputs")
    print("      (train via fast parallel convolution; infer via cheap O(1)-per-step recurrence)")

    return u, K, y_recurrent, y_conv


# =============================================================================
# SECTION D -- HIPPO MATRIX CONSTRUCTION
# =============================================================================

def build_hippo_legs_matrix(N: int) -> np.ndarray:
    """
    HiPPO-LegS matrix (Gu et al. 2020), theory.md sec 6:
      A[n,k] = -sqrt(2n+1)*sqrt(2k+1)  if n>k
               -(n+1)                   if n=k
               0                        if n<k
    """
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -np.sqrt(2*n+1) * np.sqrt(2*k+1)
            elif n == k:
                A[n, k] = -(n+1)
    return A


def section_d_hippo():
    print("\n" + "="*65)
    print("SECTION D -- HiPPO Matrix Construction")
    print("="*65)

    N = 8
    A_hippo = build_hippo_legs_matrix(N)
    A_random = np.random.default_rng(SEED).standard_normal((N, N)) * 0.5

    eig_hippo = np.linalg.eigvals(A_hippo)
    eig_random = np.linalg.eigvals(A_random)

    print(f"\n  HiPPO-LegS matrix (N={N}):")
    print(f"  {A_hippo.round(2)}")
    print(f"\n  All HiPPO eigenvalues have negative real part (stable): "
          f"{np.all(eig_hippo.real < 0)}")
    print(f"  Random matrix eigenvalues have negative real part: "
          f"{np.all(eig_random.real < 0)}  (not guaranteed -- could be unstable!)")

    print("\n  OK: HiPPO provides a STRUCTURED, provably-stable initialization")
    print("      (vs. random init, which offers no such guarantee)")

    return A_hippo, A_random, eig_hippo, eig_random


# =============================================================================
# SECTION E -- SELECTIVE SSM (MAMBA-STYLE S6)
# =============================================================================

class SelectiveSSM(nn.Module):
    """
    Simplified Mamba-style selective SSM layer (per-channel diagonal A,
    sequential scan implementation).

    Delta_t, B_t, C_t are INPUT-DEPENDENT (computed via small linear
    projections from the current input u_t) -- this is Mamba's key
    departure from S4's fixed (time-invariant) B, C (theory.md sec 8).

    A remains a FIXED, learned parameter (per the Mamba paper), parametrized
    as -exp(A_log) to guarantee negative real parts (stability) regardless
    of the underlying learnable parameter's sign.
    """
    def __init__(self, input_dim, state_dim=16):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim

        init_A = -torch.arange(1, state_dim+1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(init_A.abs()).unsqueeze(0).repeat(input_dim, 1))  # (D,N)

        self.delta_proj = nn.Linear(input_dim, input_dim)
        self.B_proj = nn.Linear(input_dim, state_dim)
        self.C_proj = nn.Linear(input_dim, state_dim)

        self.D = nn.Parameter(torch.ones(input_dim))    # skip connection

    def forward(self, u):
        """u: (batch, L, input_dim) -> y: (batch, L, input_dim)"""
        batch, L, D = u.shape
        N = self.state_dim
        A = -torch.exp(self.A_log)                     # (D,N), guaranteed negative

        delta = F.softplus(self.delta_proj(u))          # (batch,L,D) input-dependent step size
        B_t = self.B_proj(u)                              # (batch,L,N) input-dependent
        C_t = self.C_proj(u)                              # (batch,L,N) input-dependent

        h = torch.zeros(batch, D, N, device=u.device)
        outputs = []
        for t in range(L):
            delta_t = delta[:, t, :]                        # (batch,D)
            A_bar_t = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))    # (batch,D,N)
            B_bar_t = delta_t.unsqueeze(-1) * B_t[:, t, :].unsqueeze(1)     # (batch,D,N) simplified disc.

            h = A_bar_t * h + B_bar_t * u[:, t, :].unsqueeze(-1)             # selective recurrence
            y_t = torch.einsum("bdn,bn->bd", h, C_t[:, t, :]) + self.D * u[:, t, :]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)    # (batch, L, D)


def section_e_selective_ssm_demo():
    print("\n" + "="*65)
    print("SECTION E -- Selective SSM (Mamba-style) Sanity Check")
    print("="*65)

    torch.manual_seed(SEED)
    model = SelectiveSSM(input_dim=3, state_dim=8)
    u = torch.randn(2, 10, 3)    # (batch=2, L=10, D=3)
    y = model(u)

    print(f"\n  Input shape:  {tuple(u.shape)}")
    print(f"  Output shape: {tuple(y.shape)}")
    assert y.shape == u.shape

    loss = y.sum()
    loss.backward()
    grad_norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
    print(f"\n  Gradient check -- all parameters received non-zero gradients:")
    for name, gnorm in grad_norms.items():
        print(f"    {name:16s} | grad_norm={gnorm:.4f}")
    assert all(g > 0 for g in grad_norms.values())

    print("\n  OK: Selective SSM forward pass and backprop work correctly")
    return model


# =============================================================================
# SECTION F -- BENCHMARK: SELECTIVE SSM vs RNN vs LSTM vs GRU
# =============================================================================

def generate_signal_detection_data(n_samples, seq_length, seed):
    rng = np.random.default_rng(seed)
    signals = rng.integers(0, 2, size=n_samples).astype(np.float32)
    noise = rng.uniform(-1, 1, size=(n_samples, seq_length-1)).astype(np.float32)
    X = np.zeros((n_samples, seq_length, 1), dtype=np.float32)
    X[:, 0, 0] = signals * 2 - 1
    X[:, 1:, 0] = noise
    return X, signals


class SSMClassifier(nn.Module):
    """Many-to-one classifier built on the SelectiveSSM layer."""
    def __init__(self, input_dim=1, state_dim=16, hidden_dim=16):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ssm = SelectiveSSM(hidden_dim, state_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_proj(x)
        y = self.ssm(h)
        return torch.sigmoid(self.fc(y[:, -1, :])).squeeze(-1)


class RNNBaselineClassifier(nn.Module):
    def __init__(self, cell_type, input_size=1, hidden_size=16):
        super().__init__()
        if cell_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif cell_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.rnn(x)
        H = out[0]
        return torch.sigmoid(self.fc(H[:, -1, :])).squeeze(-1)


def train_classifier(model, seq_length, n_epochs=30, n_train=800, n_val=200):
    X_tr, y_tr = generate_signal_detection_data(n_train, seq_length, seed=SEED)
    X_va, y_va = generate_signal_detection_data(n_val, seq_length, seed=SEED+1)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)
    X_va_t, y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(y_va).to(DEVICE)

    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        model.train()
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.BCELoss()(model(Xb), Yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        return ((model(X_va_t) >= 0.5).float() == y_va_t).float().mean().item()


def section_f_benchmark():
    print("\n" + "="*65)
    print("SECTION F -- Benchmark: Selective SSM vs RNN vs LSTM vs GRU")
    print("  Signal Detection task (same as Topics 1-2) -- direct architecture comparison")
    print("="*65)

    lengths = [50, 100, 150]
    results = {"RNN": [], "LSTM": [], "GRU": [], "SSM": []}

    print(f"\n  {'Length':>8} | {'RNN':>8} | {'LSTM':>8} | {'GRU':>8} | {'SSM':>8}")
    print("  " + "-"*52)
    for L in lengths:
        row = {}
        for name in ["RNN", "LSTM", "GRU"]:
            torch.manual_seed(SEED)
            model = RNNBaselineClassifier(name)
            acc = train_classifier(model, L, n_epochs=30)
            results[name].append(acc); row[name] = acc

        torch.manual_seed(SEED)
        ssm_model = SSMClassifier()
        acc_ssm = train_classifier(ssm_model, L, n_epochs=30)
        results["SSM"].append(acc_ssm); row["SSM"] = acc_ssm

        print(f"  {L:>8} | {row['RNN']*100:>7.1f}% | {row['LSTM']*100:>7.1f}% | "
              f"{row['GRU']*100:>7.1f}% | {row['SSM']*100:>7.1f}%")

    print("\n  OK: Comparison across all 4 sequential architectures on identical task")
    return lengths, results


# =============================================================================
# SECTION G -- VISUALIZATION
# =============================================================================

def build_figures(u_signal, y_discrete, y_continuous, u_conv, K, y_recurrent, y_conv,
                  A_hippo, eig_hippo, eig_random, bench_lengths, bench_results):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Phase 3 -- Topic 4 (Extra): State Space Models -- S4 & Mamba",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.32)
    a = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

    a[0].plot(y_continuous, "o-", color="black", lw=1.5, ms=4, label="Fine Euler (continuous)")
    a[0].plot(y_discrete, "x--", color="#e74c3c", lw=1.5, ms=6, label="ZOH discrete recurrence")
    a[0].set_title("ZOH Discretization vs Continuous Simulation", fontweight="bold", fontsize=10)
    a[0].set_xlabel("Time step k"); a[0].set_ylabel("y(k)")
    a[0].legend(fontsize=8); a[0].grid(True, alpha=0.3)

    a[1].plot(y_recurrent, "o-", color="#3498db", lw=2, ms=5, label="Recurrent")
    a[1].plot(y_conv, "x--", color="#e67e22", lw=1.5, ms=7, label="Convolutional")
    a[1].set_title("Recurrent vs Convolutional View (EXACT match)", fontweight="bold", fontsize=10)
    a[1].set_xlabel("Time step k"); a[1].set_ylabel("y(k)")
    a[1].legend(fontsize=8); a[1].grid(True, alpha=0.3)

    a[2].stem(K)
    a[2].set_title("SSM Convolution Kernel K_j = C.A_bar^j.B_bar", fontweight="bold", fontsize=10)
    a[2].set_xlabel("Lag j"); a[2].set_ylabel("Kernel value")
    a[2].grid(True, alpha=0.3)

    im = a[3].imshow(A_hippo, cmap="RdBu_r", vmin=-np.abs(A_hippo).max(), vmax=np.abs(A_hippo).max())
    a[3].set_title("HiPPO-LegS Matrix Structure", fontweight="bold", fontsize=10)
    a[3].set_xlabel("k"); a[3].set_ylabel("n")
    plt.colorbar(im, ax=a[3], fraction=0.046)

    a[4].scatter(eig_hippo.real, eig_hippo.imag, color="#27ae60", s=60, label="HiPPO", zorder=3)
    a[4].scatter(eig_random.real, eig_random.imag, color="#e74c3c", s=60, label="Random init", zorder=3)
    a[4].axvline(0, color="gray", ls="--", lw=1)
    a[4].set_title("Eigenvalues: HiPPO vs Random Init", fontweight="bold", fontsize=10)
    a[4].set_xlabel("Real part"); a[4].set_ylabel("Imaginary part")
    a[4].legend(fontsize=8); a[4].grid(True, alpha=0.3)
    a[4].text(0.02, 0.02, "Stable region\n(Re<0)", transform=a[4].transAxes, fontsize=7, color="gray")

    colors = {"RNN":"#e74c3c","LSTM":"#3498db","GRU":"#27ae60","SSM":"#9b59b6"}
    width = 0.2; x_pos = np.arange(len(bench_lengths))
    for i, name in enumerate(["RNN","LSTM","GRU","SSM"]):
        a[5].bar(x_pos+(i-1.5)*width, [v*100 for v in bench_results[name]], width,
                color=colors[name], label=name)
    a[5].axhline(50, color="gray", ls="--", lw=1, label="Chance")
    a[5].set_xticks(x_pos); a[5].set_xticklabels(bench_lengths)
    a[5].set_title("Signal Detection: All 4 Architectures", fontweight="bold", fontsize=10)
    a[5].set_xlabel("Sequence Length"); a[5].set_ylabel("Val Accuracy (%)")
    a[5].legend(fontsize=7); a[5].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "04_ssm_mamba_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Dashboard saved -> {path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#"*65)
    print("  Phase 3 -- Topic 4 (Extra): State Space Models -- S4 & Mamba")
    print("#"*65)

    u_signal, y_discrete, y_continuous, params = section_a_discretization()
    u_conv, K, y_recurrent, y_conv = section_bc_recurrent_vs_convolutional()
    A_hippo, A_random, eig_hippo, eig_random = section_d_hippo()
    section_e_selective_ssm_demo()
    bench_lengths, bench_results = section_f_benchmark()

    build_figures(u_signal, y_discrete, y_continuous, u_conv, K, y_recurrent, y_conv,
                 A_hippo, eig_hippo, eig_random, bench_lengths, bench_results)

    print("\n" + "#"*65)
    print("  DONE: Topic 4 complete. Phase 3 -- Sequential Modeling is now FULLY complete.")
    print("#"*65 + "\n")


if __name__ == "__main__":
    main()
