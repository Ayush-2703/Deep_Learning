# MLOps: Quantization, ONNX Export, and Serving

## 1. Why this topic exists — the gap between "trained" and "deployed"

Every prior topic in this repository ends at a trained PyTorch model living
in Python process memory. Production deployment adds a distinct set of
concerns: making the model smaller and faster (quantization), portable
across frameworks/languages (ONNX export), and callable as a stable service
(serving). None of these change what the model *predicts* in principle —
they change how cheaply and portably it can be run.

## 2. Quantization — reducing numeric precision

Neural networks are normally trained and stored in 32-bit floating point
(FP32). Quantization converts weights (and sometimes activations) to lower
precision, most commonly 8-bit integers (INT8):

```
x_int8 = round( x_fp32 / scale ) + zero_point,   clipped to [-128, 127]
x_fp32_approx = (x_int8 - zero_point) * scale
```

`scale` and `zero_point` are chosen (per-tensor or per-channel) so the
INT8 range covers the FP32 tensor's actual value range as precisely as
possible. This is inherently lossy — quantization trades a small amount of
numeric precision for a ~4x reduction in model size (32 bits -> 8 bits) and
often faster inference on hardware with efficient INT8 kernels.

## 3. Post-training dynamic quantization (used in this implementation)

PyTorch's `torch.quantization.quantize_dynamic` quantizes weights to INT8
ahead of time (offline), but converts activations to INT8 dynamically at
inference time, per batch. This requires no calibration dataset and no
retraining — the simplest quantization mode, at the cost of somewhat less
aggressive speedup than static quantization (which quantizes activations
ahead of time using a calibration pass). It's applied here specifically to
`nn.Linear` layers, which is where dynamic quantization gives the most
benefit (memory-bandwidth-bound matrix multiplies).

## 4. ONNX — a framework-neutral model format

ONNX (Open Neural Network Exchange) represents a trained model's
computation graph (layers, weights, operations) in a standardized format
that isn't tied to PyTorch. Once exported, the same `.onnx` file can be run
by ONNX Runtime, or converted to run in TensorRT, CoreML, mobile runtimes,
etc., without needing PyTorch installed at all in the serving environment
— a common real-world constraint (e.g. a lightweight container image with
no Python/PyTorch dependency).

```
PyTorch model (.pt) --torch.onnx.export()--> model.onnx --onnxruntime--> prediction
```

Export requires tracing the model with example input (`torch.onnx.export`
runs a forward pass and records the resulting operation graph), so dynamic
control flow that depends on tensor *values* (not just shapes) can be a
source of export bugs — worth checking explicitly, not assuming it "just
works."

## 5. What this implementation actually measures

Rather than asserting quantization/ONNX export "work" in the abstract, this
topic trains one concrete model (a small classifier on synthetic tabular
data reused conceptually from Phase 1's regression/classification setup),
then empirically measures, before vs. after quantization and ONNX export:

- **Model file size on disk** (expect INT8 substantially smaller than FP32)
- **Prediction agreement** between original FP32 PyTorch, quantized
  PyTorch, and ONNX Runtime outputs (expect close but not bit-identical
  agreement — quantization is lossy by construction)
- **Inference latency** (CPU wall-clock time per batch), across all three
  versions

## 6. Honest expectations for a small CPU model

Quantization's speed benefits are most pronounced on large models and on
hardware/runtimes with optimized INT8 kernels. For a small MLP on CPU, the
*size* reduction should be clearly visible and reliable, but the *latency*
improvement may be modest or even absent — dynamic quantization has
overhead (the runtime int8-conversion of activations) that can offset gains
on a network this small. This implementation reports whichever result
actually occurs, rather than assuming quantization must make things faster
in every case.
