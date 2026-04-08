# Test Templates and Debugging Guide

## Audit Existing Tests (Refactoring Only)

When refactoring an existing op (e.g., `.cpp` → `.su`), **do NOT assume existing tests are adequate**. Before writing new tests, audit existing tests against these criteria:

1. **Assert method**: Does the test use `torch.testing.assert_close` with explicit tolerances? If it uses `self.assertEqual` without tolerances for float ops, flag it as needing upgrade.
2. **Dtype coverage**: Does the test cover all dtypes the kernel supports? If the refactoring adds new dtype support (e.g., adding `float16`), a test for that dtype MUST be added.
3. **C++ vs Triton capability mismatch**: If the C++ kernel has narrower capability than the Triton backend (e.g., C++ only supports last-dim softmax, Triton supports all dims), verify that existing tests will still pass when Triton is disabled. Tests that exercise unsupported C++ paths should be decorated with `@skipIfUseSipuTritonKernels()` or the test should conditionally skip unsupported cases.
4. **Edge cases**: Check if empty tensor, non-contiguous input, and shape variation are tested. If not, add them.

Output a gap table:
```
Existing test gaps:
| Gap | Current | Required | Action |
|---|---|---|---|
| float16 not tested | no test | add test | Add test_softmax_float16 |
| assertEqual used | self.assertEqual | assert_close | Upgrade (optional, pre-existing) |
```

For **pre-existing issues** (not introduced by this change): flag them but mark as optional fixes. For **gaps directly caused by the change** (e.g., new dtype added without test): these are mandatory.

## Hard Rules

1. **SIPU results MUST match CPU reference** — CPU is the golden implementation.
2. **Use `torch.testing.assert_close`** — never `torch.equal` for floating-point.
3. **Test at least `torch.float32` and `torch.bfloat16`.**
4. **If new dtype support is added, test EVERY new dtype.** E.g., adding `float16` support requires a `float16` test.
5. **Do NOT test backward gradients** — this is an inference backend. Skip backward tests entirely unless the op is explicitly a training-only op.

## Forward Test Template

> **Note:** For ops that have upstream PyTorch test infrastructure (most unary/binary ufuncs), prefer using the project's parametrized test framework (`instantiate_device_type_tests` + `@ops` decorator) as in `test/test_unary_ufuncs.py`. The standalone template below is for custom or non-standard ops.

```python
import torch
import torch_sipu
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMyOp(TestCase):

    def test_forward_float32(self):
        x_cpu = torch.randn(128, 256, dtype=torch.float32)
        x_sipu = x_cpu.clone().to("sipu")

        out_cpu = torch.<op>(x_cpu)
        out_sipu = torch.<op>(x_sipu)

        torch.testing.assert_close(out_sipu.cpu(), out_cpu, rtol=1e-5, atol=1e-5)

    def test_forward_bfloat16(self):
        x_cpu = torch.randn(128, 256, dtype=torch.bfloat16)
        x_sipu = x_cpu.clone().to("sipu")

        out_cpu = torch.<op>(x_cpu)
        out_sipu = torch.<op>(x_sipu)

        torch.testing.assert_close(out_sipu.cpu(), out_cpu, rtol=1e-2, atol=1e-2)

    def test_broadcast(self):
        a_cpu = torch.randn(16, 256, dtype=torch.bfloat16)
        b_cpu = torch.randn(1, 256, dtype=torch.bfloat16)
        a_sipu, b_sipu = a_cpu.clone().to("sipu"), b_cpu.clone().to("sipu")

        out_cpu = torch.<op>(a_cpu, b_cpu)
        out_sipu = torch.<op>(a_sipu, b_sipu)

        torch.testing.assert_close(out_sipu.cpu(), out_cpu, rtol=1e-3, atol=1e-3)

    def test_empty_tensor(self):
        x_cpu = torch.randn(0, 128, dtype=torch.float32)
        x_sipu = x_cpu.clone().to("sipu")
        out_sipu = torch.<op>(x_sipu)
        out_cpu = torch.<op>(x_cpu)
        self.assertEqual(out_sipu.shape, out_cpu.shape)

    def test_non_contiguous(self):
        x_cpu = torch.randn(64, 128, dtype=torch.float32).t()
        x_sipu = x_cpu.clone().to("sipu")
        assert not x_sipu.is_contiguous()
        torch.testing.assert_close(
            torch.<op>(x_sipu).cpu(), torch.<op>(x_cpu), rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    run_tests()
```

## Backward Gradient Test Template

> **Inference Backend: Skip this section.** Do not write backward tests. This template is retained for reference only (e.g., if a future training use-case is added).

```python
# NOT needed for inference backend — do not add this test
def test_backward(self):
    x_cpu = torch.randn(64, 128, dtype=torch.float32, requires_grad=True)
    x_sipu = x_cpu.clone().detach().to("sipu").requires_grad_(True)

    out_cpu = torch.<op>(x_cpu)
    out_sipu = torch.<op>(x_sipu)

    grad = torch.randn_like(out_cpu)
    out_cpu.backward(grad)
    out_sipu.backward(grad.to("sipu"))

    torch.testing.assert_close(x_sipu.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)
```

## Tolerance Guidelines

| dtype | rtol | atol |
|---|---|---|
| `float32` | 1e-5 | 1e-5 |
| `bfloat16` | 1e-2 | 1e-2 |
| `float16` | 1e-3 | 1e-3 |
| `int32/int64` | 0 | 0 |

Widen tolerances only with justification. Document the reason in a comment.

## Debugging Test Failures

When `assert_close` fails, use these techniques to diagnose the root cause.

**Print the error distribution:**
```python
diff = (out_sipu.cpu() - out_cpu).abs()
print(f"max error:  {diff.max().item():.8g}")
print(f"mean error: {diff.mean().item():.8g}")
idx = diff.argmax()
idx_nd = torch.unravel_index(idx, diff.shape)
print(f"max error at: {tuple(int(i) for i in idx_nd)}")
print(f"  cpu:  {out_cpu[idx_nd].item():.8g}")
print(f"  sipu: {out_sipu.cpu()[idx_nd].item():.8g}")
```

**Use cosine similarity for matmul / large tensors:**
```python
def cosine_sim(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
sim = cosine_sim(out_sipu.cpu(), out_cpu)
assert sim > 0.9999, f"cosine similarity too low: {sim}"
```

**Use `TRITON_KERNEL_VERIFY=1` for runtime diagnosis (Triton ops only):**

This activates the `@sipu_verify` decorator, which automatically compares SIPU results against CPU reference at runtime. The decorator uses tolerances set by `func_default_rtol`/`func_default_atol` in the decorator call, overridable via env vars:

```bash
TRITON_KERNEL_VERIFY=1 python examples/run_my_op.py
# Override thresholds:
TRITON_KERNEL_VERIFY=1 SIPU_VERIFY_RTOL=1e-3 SIPU_VERIFY_ATOL=1e-3 python examples/run_my_op.py
```

**Isolate backend differences:**
```bash
# C++ kernel backend (default)
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=sipu pytest test/test_my_op.py -v
# Triton backend
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=sipu PYTORCH_TEST_WITH_SIPU_TRITON_OPS=1 pytest test/test_my_op.py -v
# AI backend
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=sipu PYTORCH_TEST_WITH_SIPU_TRITON_OPS=1 TORCH_SIPU_USE_AI_BACKEND=1 pytest test/test_my_op.py -v
```

**Common mismatch causes:**

| Symptom | Likely cause |
|---|---|
| Large errors only in bf16/fp16 | Intermediate computation not promoted to float32 |
| Errors at specific indices, rest exact | Out-of-bounds access or padding issue |
| Consistent small bias across all elements | Reduction order difference — may be acceptable, document and widen tolerance |
| NaN in SIPU output, not in CPU | Division by zero, log of negative, or uninitialized memory |
| Correct values, wrong shape | Missing or incorrect `remove_padding` call |
| Test passes alone, fails in batch | Shared state or stream synchronization issue |
