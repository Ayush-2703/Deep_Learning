"""
Phase 6 - Topic 3: MLOps - Quantization, ONNX Export, and Serving
CPU-only, synthetic tabular classification data, PyTorch + ONNX + ONNX Runtime.

Run: python3 implementation.py
Produces: outputs/training_curve.png, outputs/size_comparison.png,
          outputs/latency_comparison.png, outputs/prediction_agreement.png
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(5)
np.random.seed(5)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# 1. Synthetic tabular binary classification dataset
# ---------------------------------------------------------------------------
N_SAMPLES = 4000
N_FEATURES = 20

def make_classification_data(n=N_SAMPLES, d=N_FEATURES):
    # two Gaussian blobs in a d-dim space, class-informative on first 5 dims,
    # remaining dims are pure noise (nuisance features) -- realistic tabular setup
    n_pos = n // 2
    n_neg = n - n_pos
    informative_dim = 5
    mean_pos = np.concatenate([np.full(informative_dim, 1.2), np.zeros(d - informative_dim)])
    mean_neg = np.concatenate([np.full(informative_dim, -1.2), np.zeros(d - informative_dim)])
    X_pos = np.random.normal(mean_pos, 1.0, size=(n_pos, d))
    X_neg = np.random.normal(mean_neg, 1.0, size=(n_neg, d))
    X = np.concatenate([X_pos, X_neg], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
    idx = np.random.permutation(n)
    return X[idx], y[idx]

X, y = make_classification_data()
n_train = int(0.8 * len(X))
X_train, X_val = torch.tensor(X[:n_train]), torch.tensor(X[n_train:])
y_train, y_val = torch.tensor(y[:n_train]), torch.tensor(y[n_train:])
print(f"Synthetic tabular dataset: {X.shape}, {n_train} train / {len(X) - n_train} val, "
      f"{N_FEATURES} features (5 informative + {N_FEATURES-5} noise)")

# ---------------------------------------------------------------------------
# 2. Model: small MLP classifier (the "production model" for this exercise)
# ---------------------------------------------------------------------------
class Classifier(nn.Module):
    def __init__(self, in_dim=N_FEATURES, hidden=128, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


model = Classifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------------------
# 3. Train the base FP32 model
# ---------------------------------------------------------------------------
EPOCHS = 60
BATCH = 128
n = X_train.size(0)
train_losses, val_accs = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(n)
    epoch_losses = []
    for i in range(0, n - BATCH, BATCH):
        xb, yb = X_train[perm[i:i + BATCH]], y_train[perm[i:i + BATCH]]
        optimizer.zero_grad()
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_acc = (val_out.argmax(dim=1) == y_val).float().mean().item()
    train_losses.append(np.mean(epoch_losses))
    val_accs.append(val_acc)

    if epoch % 15 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | train_loss={train_losses[-1]:.4f} | val_acc={val_acc:.4f}")

final_val_acc = val_accs[-1]
print(f"\nFinal FP32 model val accuracy: {final_val_acc:.4f}")

plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="train loss")
plt.twinx_ax = plt.gca().twinx()
plt.twinx_ax.plot(val_accs, color="green", label="val accuracy")
plt.title("Base FP32 model training")
plt.gca().set_xlabel("epoch"); plt.gca().set_ylabel("loss")
plt.twinx_ax.set_ylabel("val accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 4. Save FP32 model, apply dynamic quantization, save quantized model
# ---------------------------------------------------------------------------
model.eval()
fp32_path = os.path.join(OUT_DIR, "model_fp32.pt")
torch.save(model.state_dict(), fp32_path)

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
quant_path = os.path.join(OUT_DIR, "model_quantized.pt")
torch.save(quantized_model.state_dict(), quant_path)

fp32_size = os.path.getsize(fp32_path) / 1024  # KB
quant_size = os.path.getsize(quant_path) / 1024  # KB
print(f"\nFP32 model size on disk:      {fp32_size:.2f} KB")
print(f"Quantized model size on disk: {quant_size:.2f} KB")
print(f"Size reduction: {(1 - quant_size / fp32_size) * 100:.1f}%")

# ---------------------------------------------------------------------------
# 5. Export FP32 model to ONNX
# ---------------------------------------------------------------------------
onnx_path = os.path.join(OUT_DIR, "model.onnx")
dummy_input = torch.randn(1, N_FEATURES)
torch.onnx.export(
    model, dummy_input, onnx_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=18,  # opset 13 triggered a failed fallback version-conversion attempt in this
    # torch/onnxscript version (the new dynamo-based exporter natively targets opset 18); using 18
    # directly avoids that noisy, ultimately-harmless conversion failure trace.
)
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)  # validates the exported graph is well-formed
print(f"\nONNX export successful, graph passed onnx.checker.check_model validation.")
# NOTE: this torch/onnxscript version's exporter stores large weight tensors in a separate
# external-data file (model.onnx.data) rather than embedding them in the .onnx file itself.
# Measuring only the .onnx file understates the true on-disk footprint by ~12x for this model --
# caught by checking `ls` output during development. Sum both files for an honest total.
onnx_data_path = onnx_path + ".data"
onnx_main_size = os.path.getsize(onnx_path) / 1024
onnx_data_size = os.path.getsize(onnx_data_path) / 1024 if os.path.exists(onnx_data_path) else 0.0
onnx_size = onnx_main_size + onnx_data_size
print(f"ONNX model size on disk: {onnx_size:.2f} KB "
      f"({onnx_main_size:.2f} KB graph + {onnx_data_size:.2f} KB external weight data)")

ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# ---------------------------------------------------------------------------
# 6. Correctness check: do FP32 PyTorch, quantized PyTorch, and ONNX Runtime agree?
# ---------------------------------------------------------------------------
model.eval()
quantized_model.eval()
with torch.no_grad():
    fp32_out = model(X_val)
    fp32_preds = fp32_out.argmax(dim=1).numpy()

    quant_out = quantized_model(X_val)
    quant_preds = quant_out.argmax(dim=1).numpy()

onnx_out = ort_session.run(None, {"input": X_val.numpy()})[0]
onnx_preds = onnx_out.argmax(axis=1)

agree_fp32_quant = (fp32_preds == quant_preds).mean()
agree_fp32_onnx = (fp32_preds == onnx_preds).mean()
max_logit_diff_onnx = np.abs(fp32_out.numpy() - onnx_out).max()

print(f"\nPrediction agreement FP32 vs Quantized: {agree_fp32_quant:.2%}")
print(f"Prediction agreement FP32 vs ONNX Runtime: {agree_fp32_onnx:.2%}")
print(f"Max absolute logit difference (FP32 vs ONNX): {max_logit_diff_onnx:.6f}")
if agree_fp32_onnx < 0.999:
    print("NOTE: ONNX predictions do not perfectly match PyTorch -- reporting honestly "
          "(expected to be extremely close but not always bit-identical due to floating point "
          "operation ordering differences between backends).")
else:
    print("NOTE: ONNX Runtime predictions match PyTorch FP32 predictions exactly on this validation set.")

if agree_fp32_quant < 0.95:
    print(f"NOTE: quantization changed a meaningful fraction of predictions "
          f"({(1-agree_fp32_quant)*100:.1f}%) -- reporting honestly; INT8 quantization is lossy "
          f"and can flip predictions near the decision boundary.")
else:
    print("NOTE: quantized model predictions closely match FP32 on this validation set "
          "(a small number of near-boundary flips is expected and acceptable).")

# ---------------------------------------------------------------------------
# 7. Latency comparison: FP32 vs Quantized vs ONNX Runtime
# ---------------------------------------------------------------------------
def benchmark_pytorch(m, x, n_runs=200):
    m.eval()
    with torch.no_grad():
        for _ in range(10):  # warmup
            m(x)
        start = time.perf_counter()
        for _ in range(n_runs):
            m(x)
        elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms per run

def benchmark_onnx(session, x_np, n_runs=200):
    for _ in range(10):
        session.run(None, {"input": x_np})
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {"input": x_np})
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000

bench_batch = X_val[:32]
bench_batch_np = bench_batch.numpy()

fp32_latency = benchmark_pytorch(model, bench_batch)
quant_latency = benchmark_pytorch(quantized_model, bench_batch)
onnx_latency = benchmark_onnx(ort_session, bench_batch_np)

print(f"\nLatency (batch=32, avg over 200 runs):")
print(f"  FP32 PyTorch:      {fp32_latency:.4f} ms")
print(f"  Quantized PyTorch: {quant_latency:.4f} ms")
print(f"  ONNX Runtime:      {onnx_latency:.4f} ms")
fastest = min([("FP32", fp32_latency), ("Quantized", quant_latency), ("ONNX", onnx_latency)], key=lambda p: p[1])
print(f"NOTE: fastest backend on this small CPU model/batch size was '{fastest[0]}' "
      f"({fastest[1]:.4f} ms) -- reported as measured, not assumed. Quantization's latency benefit "
      f"is not guaranteed for small models per theory.md section 6.")

# ---------------------------------------------------------------------------
# 8. Visualizations
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4.5))
sizes = [fp32_size, quant_size, onnx_size]
labels = ["FP32\n(PyTorch)", "Quantized INT8\n(PyTorch)", "ONNX\n(FP32)"]
bars = plt.bar(labels, sizes, color=["tab:blue", "tab:orange", "tab:green"])
plt.ylabel("Size on disk (KB)")
plt.title("Model size comparison")
for bar, size in zip(bars, sizes):
    plt.text(bar.get_x() + bar.get_width() / 2, size + 0.5, f"{size:.1f} KB", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "size_comparison.png"), dpi=110)
plt.close()

plt.figure(figsize=(7, 4.5))
latencies = [fp32_latency, quant_latency, onnx_latency]
bars = plt.bar(labels, latencies, color=["tab:blue", "tab:orange", "tab:green"])
plt.ylabel("Latency per batch of 32 (ms)")
plt.title("Inference latency comparison (measured, CPU)")
for bar, lat in zip(bars, latencies):
    plt.text(bar.get_x() + bar.get_width() / 2, lat + 0.001, f"{lat:.4f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "latency_comparison.png"), dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
categories = ["FP32 vs Quantized", "FP32 vs ONNX"]
agreements = [agree_fp32_quant * 100, agree_fp32_onnx * 100]
bars = ax.bar(categories, agreements, color=["tab:red", "tab:purple"])
ax.set_ylabel("Prediction agreement (%)")
ax.set_ylim(min(90, min(agreements) - 2), 100.5)
ax.set_title("Prediction agreement vs. original FP32 model")
ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
for bar, agr in zip(bars, agreements):
    ax.text(bar.get_x() + bar.get_width() / 2, agr + 0.05, f"{agr:.2f}%", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "prediction_agreement.png"), dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 9. Minimal "serving" simulation: a plain function wrapping the ONNX session
# ---------------------------------------------------------------------------
def serve_predict(feature_vector):
    """Simulates a production inference endpoint: takes a raw feature vector,
    returns a class prediction + confidence. Uses the ONNX Runtime session,
    the framework-neutral artifact meant for deployment per theory.md section 4."""
    x = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    logits = ort_session.run(None, {"input": x})[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    pred_class = int(probs.argmax(axis=1)[0])
    confidence = float(probs.max(axis=1)[0])
    return {"predicted_class": pred_class, "confidence": round(confidence, 4)}

sample_input = X_val[0].numpy().tolist()
serve_result = serve_predict(sample_input)
true_label = int(y_val[0].item())
print(f"\nServing simulation: serve_predict(sample) -> {serve_result} (true label: {true_label})")

print("\nSaved: training_curve.png, size_comparison.png, latency_comparison.png, prediction_agreement.png")
print("Topic 3 (MLOps: Quantization/ONNX/Serving) run complete.")
