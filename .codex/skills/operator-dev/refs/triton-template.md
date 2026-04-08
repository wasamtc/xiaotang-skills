# Triton Backend Implementation Template

Create `torch_sipu/backends/sipu_triton_kernels/ops/<op>.py`. Use the **modern decorator stack** pattern.

## Decorator Stack (outer → inner)

The **recommended** (modern) decorator stack for new Triton ops is:

```python
@sipu_verify()              # 1. Runtime verification against CPU (outermost)
@cpu_fallback(...)          # 2. Automatic CPU fallback for unsupported dtypes/errors
@triton_preprocess(CONFIG)  # 3. Tensor preprocessing: contiguous, aligned, dtype-promoted (innermost)
def my_op(input, *, out=None):
    ...
```

> **Note:** Not all existing ops use the full stack. Some older ops use only `@sipu_verify` (e.g., `silu.py`), others use `@cpu_fallback` + `@triton_preprocess` without `@sipu_verify` (e.g., `reciprocal.py`). When writing **new** ops, always use the full stack. When modifying existing ops, follow the existing pattern for that op unless the task is to modernize it.

## Complete Template (Unary Op)

```python
import torch
import triton
import triton.language as tl

from .preprocessing_framework import triton_preprocess, UNARY_FLOAT_OP_CONFIG
from .utils import cpu_fallback, precheck_supported_dtypes
from .verify_decorator import sipu_verify


@triton.jit
def <op>_kernel(x_ptr, output_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)

    # Promote to float32 for computation accuracy
    in_dtype = x.dtype
    x_fp32 = x.to(tl.float32)
    y_fp32 = tl.<op>(x_fp32)     # ... computation ...
    y = y_fp32.to(in_dtype)

    tl.store(output_ptr + offs, y, mask=mask)


@sipu_verify()
@cpu_fallback(
    aten_op="<op>",
    precheck=precheck_supported_dtypes(
        torch.float32, torch.bfloat16, torch.float16, torch.float64,
        torch.int8, torch.int16, torch.int32, torch.int64,
    ),
)
@triton_preprocess(UNARY_FLOAT_OP_CONFIG)
def <op>(self: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if self.numel() == 0:
        return out

    numel = self.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(numel), 512)
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    <op>_kernel[grid](self, out, numel, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
    return out
```

## Preprocessing Configs (choose one)

| Config | Use for | What it does |
|---|---|---|
| `UNARY_OP_CONFIG` | Unary ops (bitwise, etc.) | contiguous + 64-bit→32-bit conversion |
| `UNARY_FLOAT_OP_CONFIG` | Unary float ops (cos, sin, exp) | above + dtype promotion + int→float |
| `BINARY_OP_CONFIG` | Binary ops (add, mul) | above + broadcast + scalar handling |
| `COMPARISON_OP_CONFIG` | Comparison ops (eq, lt, ge) | BINARY + output dtype=bool |
| `TERNARY_OP_CONFIG` | Ternary ops (where, clamp) | broadcast all 3 + dtype promotion |
| `REDUCTION_OP_CONFIG` | Reduction ops (sum, mean) | contiguous only (minimal) |
| `NULLARY_OP_CONFIG` | Nullary ops (full, zeros) | no preprocessing |

The `@triton_preprocess` decorator handles: contiguity, 1024-byte alignment, broadcasting, dtype promotion, 64-bit→32-bit conversion (Triton limitation), and memory overlap checks. You do NOT need to call `convert_to_contiguous_and_aligned` manually when using this decorator.

## `@cpu_fallback` Decorator

Provides automatic fallback to CPU ATen when:
1. Triton backend is globally disabled
2. `precheck` function returns `False` (e.g., unsupported dtype)
3. Kernel raises `FallbackRequested` or any exception

```python
@cpu_fallback(
    aten_op="<op>",  # ATen function name used for fallback (calls torch.ops.aten.<op>)
    precheck=precheck_supported_dtypes(torch.float32, torch.bfloat16, ...),
)
```

Use `precheck_supported_dtypes(...)` to list supported dtypes. For ops where only specific argument positions need checking, use `precheck_supported_dtypes_indices((0, 1), torch.float32, ...)`.

To trigger fallback from inside a kernel, call `request_fallback("reason")`.

## Inplace Operations

For inplace ops (e.g., `add_`, `fill_`), use configs with `is_inplace=True` or `BINARY_OP_CONFIG_WITH_OVERLAP_CHECK` which includes memory overlap detection. The preprocessing framework will handle the first input being both input and output.

## Export and Registration

**Export** — add to `torch_sipu/backends/sipu_triton_kernels/ops/__init__.py`:
```python
from .<op> import <op_func>
```

**Register** — add to `_prefer_triton_kernels()` in `torch_sipu/backends/sipu_triton_kernels/__init__.py`:
```python
_sipu_lib_aten.impl("<aten_schema>", <op_func>, dispatch_key="PrivateUse1")
# Register ALL overloads the op uses — see schema name table below
```

The ATen schema name must match PyTorch's op registry. The dispatch key is always `"PrivateUse1"`.

**Common schema name patterns** (register ALL that apply):

| Op type | Schema names to register | Example |
|---|---|---|
| Unary (trig, etc.) | `"<op>"`, `"<op>.out"` | `"cos"`, `"cos.out"` |
| Unary (with tensor overload) | `"<op>"`, `"<op>.Tensor"`, `"<op>_.Tensor"` | `"neg"`, `"neg.Tensor"`, `"neg_.Tensor"` |
| Binary | `"<op>.Tensor"`, `"<op>.out"`, `"<op>_.Tensor"` | `"add.Tensor"`, `"add.out"`, `"add_.Tensor"` |
| Binary (with scalar) | above + `"<op>.Scalar"`, `"<op>.Scalar_out"` | `"eq.Scalar"`, `"eq.Scalar_out"` |

To discover all required schema names for an op:
```bash
python -c "import torch; print(torch._C._dispatch_dump('aten::<op>'))"
```
