# Dispatch Mechanism Decision Guide

PyTorch's dispatch system routes operations to device-specific implementations. Different operator types use **fundamentally different dispatch mechanisms** in C++. Choosing the wrong mechanism will result in compilation errors or runtime dispatch failures.

## Dispatch Flow

```
User Code: tensor.add(other)
    ↓
PyTorch Dispatcher (dispatch key = PrivateUse1)
    ↓
┌──────────────────────────────────────────────┐
│ Runtime backend selection:                   │
│  ├─ Triton backend (default):                │
│  │   torch.library.impl("add.Tensor", ...)   │
│  │   → Python Triton kernel                  │
│  └─ C++ backend (fallback):                  │
│      native_functions.yaml dispatch           │
│      ├─ DispatchStub → .su kernel            │
│      ├─ TORCH_SIPU_IMPL_FUNC → .cpp kernel  │
│      └─ use_native_impl → PyTorch native     │
└──────────────────────────────────────────────┘
```

When the Triton backend is active (default), it **overrides** C++ kernels via `torch.library.impl()` at higher priority. C++ kernels serve as the fallback when Triton is disabled.

## C++ Dispatch Mechanism Decision Tree

There are **three** C++ dispatch mechanisms. The choice depends on the operator category:

| Mechanism | When to use | Typical ops | Key pattern |
|---|---|---|---|
| **DispatchStub** (`REGISTER_PRIVATEUSE1_DISPATCH`) | Element-wise, reduction, comparison ops that use `TensorIterator` | unary (cos, neg, sigmoid), binary (add, mul, div), comparison (eq, lt), reduction (sum, mean, argmax), fill, scatter/gather | Kernel function takes `TensorIteratorBase&`, registered via `REGISTER_PRIVATEUSE1_DISPATCH(<op>_stub, &func)` in `.su` file |
| **Structured Kernel** (`TORCH_SIPU_IMPL_FUNC`) | Complex ops needing custom shape inference or output setup | mm, bmm, cat, topk, softmax, triu/tril | Two-part: `TORCH_META_FUNC` (shapes) + `TORCH_SIPU_IMPL_FUNC` (computation) in `.cpp` file |
| **use_native_impl: True** | Ops where PyTorch's generic implementation works on SIPU | view, reshape, unfold, as_strided, permute | Only a YAML entry needed — no kernel code |

## Decision Rules

1. **Does the op use `TensorIterator` in upstream PyTorch?** (Check the CUDA kernel — if it calls `gpu_kernel(iter, ...)` or uses `TensorIterator`, it's a DispatchStub op.)
   → **Yes**: Use `REGISTER_PRIVATEUSE1_DISPATCH` in a `.su` file.

2. **Does the op need custom output shape/stride setup?** (Check if upstream has `TORCH_META_FUNC(<op>)` defining output shapes.)
   → **Yes**: Use `TORCH_SIPU_IMPL_FUNC` in a `.cpp` file, paired with the meta function.

3. **Is the op purely metadata (no device computation)?** (E.g., view, reshape, unfold — only changes tensor metadata, no data movement.)
   → **Yes**: Use `use_native_impl: True` in YAML. No kernel needed.

4. **Is the op a custom/extension op not in upstream ATen?**
   → **Yes**: Use `ext_native_functions.yaml` with `Meta` dispatch key.

## Stub Header Reference (for DispatchStub pattern)

When using `REGISTER_PRIVATEUSE1_DISPATCH`, you MUST include the header that **declares** the stub:

| Op category | Stub header | Example stubs |
|---|---|---|
| Unary ops | `<ATen/native/UnaryOps.h>` | `sigmoid_stub`, `rsqrt_stub`, `bitwise_not_stub`, `logical_not_stub` |
| Binary ops | `<ATen/native/BinaryOps.h>` | `add_stub`, `sub_stub`, `mul_stub`, `div_true_stub` |
| Comparison ops | `<ATen/native/TensorCompare.h>` | `where_kernel` |
| Reduce ops | `<ATen/native/ReduceOps.h>` or `<ATen/native/ReduceAllOps.h>` | `sum_stub`, `mean_stub`, `argmax_stub`, `any_stub`, `all_stub` |
| Fill/copy ops | `<ATen/native/Fill.h>` | `fill_stub` |
| Index ops | `<ATen/native/IndexKernel.h>` | `index_put_stub` |
| Scatter/gather | `<ATen/native/ScatterGatherChecks.h>` | `gather_stub`, `scatter_stub` |
| Sort ops | `<ATen/native/Sorting.h>` | `sort_stub`, `topk_stub` |
| Activation ops | `<ATen/native/Activation.h>` | `silu_stub` |

> **Tip:** To find the correct stub header for any op, search upstream PyTorch: `grep -r "<op>_stub" aten/src/ATen/native/*.h`

## native_functions.yaml Entry Patterns

There are three YAML entry patterns, each mapping to a different dispatch behavior:

**Pattern 1: Bare listing** — for DispatchStub ops where the stub name follows convention:
```yaml
# The stub is auto-discovered via REGISTER_PRIVATEUSE1_DISPATCH in .su file
- add.out
- cos.out
- cat.out
```

**Pattern 2: Explicit SIPU dispatch** — when function name differs from convention or needs routing:
```yaml
- op: eq.Scalar_out
  dispatch:
    SIPU: eq_Scalar_out
  use_native_impl: True

- op: index_select
  dispatch:
    SIPU: index_select
```

**Pattern 3: Native implementation only** — no custom kernel, use PyTorch's generic code:
```yaml
- op: view
  use_native_impl: True

- op: _reshape_alias
  use_native_impl: True
```

## Operator Category Quick Reference

> **Note:** `.su` files can use **either** DispatchStub or TORCH_SIPU_IMPL_FUNC — the file extension does NOT determine the dispatch mechanism. The choice depends on how the op is structured in upstream PyTorch. Always check upstream first.

| Category | C++ dispatch | Template | Stub header | YAML pattern |
|---|---|---|---|---|
| Unary (cos, neg, sigmoid) | DispatchStub | Option A `.su` | `UnaryOps.h` | Bare or explicit |
| Binary — add, sub | TORCH_SIPU_IMPL_FUNC | Option A `.su` (with IMPL_FUNC) | N/A (structured) | Bare |
| Binary — mul, div, floor_div | DispatchStub | Option A `.su` | `BinaryOps.h` | Bare or explicit |
| Comparison (eq, lt, ge) | DispatchStub | Option A `.su` | `TensorCompare.h` | Explicit SIPU |
| Reduction (sum, mean, argmax) | DispatchStub | Option A `.su` | `ReduceOps.h` | Mixed |
| Index (gather, scatter, index_select) | DispatchStub | Option A `.su` | `IndexKernel.h` | Explicit SIPU |
| Sort (sort) | DispatchStub | Option A `.su` | `Sorting.h` | Explicit SIPU |
| Sort (topk) | TORCH_SIPU_IMPL_FUNC | Option C `.cpp` | `Sorting.h` | Bare |
| Triangular (triu, tril) | TORCH_SIPU_IMPL_FUNC | Option B `.su` | N/A | Explicit SIPU |
| Activation (softmax) | TORCH_SIPU_IMPL_FUNC | Option B `.su` | N/A | Explicit SIPU |
| BLAS (mm, bmm) | TORCH_SIPU_IMPL_FUNC | Option C `.cpp` | N/A | Bare |
| Shape (view, reshape) | use_native_impl | None | N/A | native_impl: True |
| Memory (copy_, resize_) | Special | `.su`/`.cpp` | N/A | `device_guard: False` |
| Extension (_scaled_grouped_mm) | Meta dispatch | `.cpp` | N/A | `ext_native_functions.yaml` |

## Dispatch Key Selection (Triton/AI Backend)

Most ops use `PrivateUse1`. Only use `AutogradPrivateUse1` for the rare cases listed:

| Dispatch key | When to use | Current examples |
|---|---|---|
| `PrivateUse1` | 99% of ops — element-wise, reduction, BLAS, etc. | `add.Tensor`, `cos`, `mm`, `gather` |
| `AutogradPrivateUse1` | Ops that need custom gradient handling through dtype conversion | `to.dtype`, `type_as` |

```python
# Standard (vast majority):
_sipu_lib_aten.impl("cos", cos, dispatch_key="PrivateUse1")

# Autograd (rare — only for dtype/type conversion ops):
_sipu_lib_aten.impl("to.dtype", to_dtype, dispatch_key="AutogradPrivateUse1")
```
