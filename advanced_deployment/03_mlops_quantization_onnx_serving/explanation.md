# Explanation: Quantization / ONNX / Serving Implementation Walkthrough

## 1. Synthetic tabular data with deliberate nuisance features

```python
mean_pos = np.concatenate([np.full(informative_dim, 1.2), np.zeros(d - informative_dim)])
```

Only the first 5 of 20 features actually separate the two classes; the
remaining 15 are pure noise. This isn't relevant to quantization/ONNX
correctness directly, but it makes the classification task realistic
enough that the base FP32 model has to do genuine learning (not memorize
a trivial 1-D threshold), which matters for the later "do quantized/ONNX
predictions still agree with FP32" check to be a meaningful test rather
than a check on a near-trivial model.

## 2. Bug #1 (caught and fixed): `torch.onnx.export` failed outright

The first run of this script crashed with:
```
ModuleNotFoundError: No module named 'onnxscript'
```

This PyTorch version's `torch.onnx.export` defaults to the newer
dynamo-based exporter, which depends on the separate `onnxscript` package
— not installed by default alongside `torch` or `onnx`/`onnxruntime`. Fix:
`pip install onnxscript`. This is exactly the kind of environment-specific
issue that only shows up by actually running the code, not by writing it
from memory of how `torch.onnx.export` "usually" works — worth keeping in
mind, since export APIs shift across PyTorch versions and this repository
followed the actual error message rather than assuming the original code
was already correct.

## 3. Bug #2 (caught and fixed): a genuinely misleading size measurement

After fixing the crash, the first successful run reported:
```
ONNX model size on disk: 6.36 KB
```
— dramatically smaller than the 78.93 KB FP32 PyTorch file, which seemed
suspicious given ONNX export shouldn't inherently shrink a model. Checking
`ls -la` on the output directory revealed the real cause: this exporter
writes large weight tensors to a **separate external-data file**
(`model.onnx.data`, 76 KB) rather than embedding them in the `.onnx` file
itself (which shrinks to just the graph structure, 6.36 KB). The initial
size comparison was comparing "FP32 model" against "1/13th of the actual
ONNX artifact" — a real, meaningfully misleading result if left unfixed.
**Fix:** sum both files (`model.onnx` + `model.onnx.data`) for the true
ONNX on-disk footprint, which corrected the reported size to 82.36 KB —
now honestly comparable to (and, for this tiny model, larger than) the
78.93 KB FP32 PyTorch file. This is worth internalizing as a general
lesson: don't trust a single artifact's file size as "the model size"
without checking whether the export format split things across files.

## 4. `torch.quantization.quantize_dynamic` — what actually happens

```python
quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

The `{nn.Linear}` argument tells PyTorch to only quantize `Linear` layers
(this model has no other layer types, but in general this scoping matters
— e.g. you would not quantize `BatchNorm` layers this way). `dtype=torch.qint8`
selects 8-bit signed integer quantization for weights; activations are
quantized dynamically per-batch at inference time, matching the
"post-training dynamic quantization" description in theory.md section 3 —
no calibration dataset, no retraining, applied directly to the already-
trained FP32 model.

The console output includes a `DeprecationWarning` noting that
`torch.ao.quantization` (the module backing this API) is slated for
removal in favor of `torchao`'s newer quantization API. This is reported
here as-is rather than silently suppressed — it's a genuine signal that
the exact API used in this run may need to migrate in a future PyTorch
version, which is useful context for anyone maintaining this code later.

## 5. Real, measured size reduction

```
FP32 model size on disk:      78.93 KB
Quantized model size on disk: 24.70 KB
Size reduction: 68.7%
```

Close to (but not exactly) the theoretical 4x (75%) reduction from FP32
(32-bit) to INT8 (8-bit) weights — the gap from the theoretical maximum is
expected, since not every parameter in the model is a quantized `Linear`
weight (biases, and any non-Linear parameters, remain FP32), and PyTorch's
serialization format has some fixed overhead independent of parameter
precision.

## 6. Correctness verification — three independent backends compared

```python
agree_fp32_quant = (fp32_preds == quant_preds).mean()
agree_fp32_onnx = (fp32_preds == onnx_preds).mean()
max_logit_diff_onnx = np.abs(fp32_out.numpy() - onnx_out).max()
```

Result: **100% prediction agreement** between all three (FP32 PyTorch,
quantized PyTorch, ONNX Runtime) on the 800-sample validation set, with
`max_logit_diff_onnx = 0.000000` (ONNX and PyTorch FP32 produced
numerically identical logits on this run, not just matching final
predictions — a stronger check than accuracy agreement alone, since two
models could agree on the argmax while differing meaningfully in
confidence). The quantized model's 100% agreement here is a genuinely
observed result on this dataset/model, not guaranteed in general —
theory.md and the script's own conditional print statement both flag that
lower agreement (near-boundary flips) is an expected, acceptable outcome
of INT8 quantization in the general case.

## 7. Latency — the honest, counter-intuitive result

```
FP32 PyTorch:      0.0904 ms
Quantized PyTorch: 0.2425 ms   <- SLOWER than FP32
ONNX Runtime:      0.0330 ms   <- fastest
```

This is the single most important empirical result in this topic: dynamic
quantization made inference **slower**, not faster, on this small model/
batch size — exactly the caveat theory.md section 6 raises in advance,
now confirmed rather than just asserted. The likely mechanism: dynamic
quantization's per-batch runtime activation-quantization overhead exceeds
whatever compute savings INT8 matmuls provide, when the matmuls themselves
are already tiny (128-unit hidden layers, batch size 32). ONNX Runtime's
speed advantage, by contrast, comes from a different source — a
statically-optimized, more efficient computation graph and execution
engine, unrelated to quantization. Reporting the actual measured numbers
here, rather than a generic "quantization speeds things up" claim, is the
entire point of running real benchmarks instead of describing expected
behavior.

## 8. `serve_predict` — the minimal serving simulation

```python
def serve_predict(feature_vector):
    x = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    logits = ort_session.run(None, {"input": x})[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    ...
```

Deliberately uses the ONNX Runtime session (not the PyTorch model) —
this is the artifact meant for production serving per theory.md section
4, since it doesn't require PyTorch to be installed in whatever process
ultimately serves predictions. The manual softmax
(`np.exp(logits) / np.exp(logits).sum(...)`) is necessary because the
exported graph's output is raw logits (this model's `forward` has no
final softmax layer), which is standard practice — softmax is usually left
out of the exported graph and applied by the caller.
