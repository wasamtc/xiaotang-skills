---
name: findop
description: >-
  Investigate a PyTorch operator's implementation details: dispatch mechanism,
  structured vs unstructured, supported dtypes, CPU scalar/vector/CUDA kernels,
  and implementation notes for porting. Only use when the user explicitly types
  "/findop". Do NOT use automatically or proactively.
---

# PyTorch Operator Investigation

## Trigger

ONLY activate when the user explicitly types `/findop <op_name>`. Do NOT trigger automatically.

`<op_name>` is the ATen operator name (e.g. `add`, `softmax`, `topk`, `silu`, `index_select`).

## Target Directory

PyTorch source: `/share_data/tangcong/project/pytorch_v2.7.1`

Key paths:
- `aten/src/ATen/native/native_functions.yaml` — op schema & dispatch table
- `aten/src/ATen/native/` — high-level op implementation (.cpp)
- `aten/src/ATen/native/cpu/` — CPU kernel implementations
- `aten/src/ATen/native/cuda/` — CUDA kernel implementations
- `aten/src/ATen/native/sparse/` — sparse implementations
- `aten/src/ATen/native/quantized/` — quantized implementations
- `torch/` — Python-level entry points

## Workflow

### Step 1: Locate the Op in native_functions.yaml

Search for the operator's schema entry:

```bash
grep -n "<op_name>" /share_data/tangcong/project/pytorch_v2.7.1/aten/src/ATen/native/native_functions.yaml
```

Read the matched section (typically 5-20 lines per op) to extract:
- **func signature**: `aten::<op>.<overload>(...) -> ...`
- **variants**: `function`, `method`
- **dispatch table**: which backends have custom implementations
- **structured / structured_delegate**: whether the op uses structured kernels
- **tags**: `pointwise`, `core`, etc.

Present a summary table:

```
Op: aten::<op>
Signature: <func line>
Variants: function / method
Structured: Yes (structured_delegate: <base>) / No
Tags: [pointwise, ...]
```

### Step 2: Determine Structured vs Unstructured

**Structured ops** have one of:
- `structured: True` — this IS the base structured kernel
- `structured_delegate: <op>.out` — delegates to a structured out variant

**Unstructured ops** have neither field.

If structured, explain:
1. The `meta()` function computes output shape/dtype without data
2. The `impl()` function runs the actual computation
3. Structured ops share shape-checking logic across backends

Report:
```
Dispatch type: Structured kernel (base: <op>.out)
  — meta() at: <path>
  — impl() at: <path per backend>
```
OR
```
Dispatch type: Unstructured (traditional dispatch)
  — Each backend registers its own full implementation
```

### Step 3: Identify Supported dtypes

#### 3.1 Check dispatch table in native_functions.yaml

The `dispatch:` section shows which backends have implementations:
- `CPU` / `CUDA` / `SparseCPU` / `SparseCUDA` / `QuantizedCPU` etc.
- `CompositeImplicitAutograd` — auto-decomposes, no per-backend kernel
- `CompositeExplicitAutograd` — explicit composite, also no per-backend kernel

#### 3.2 Find dtype constraints in CPU kernel

Search for dtype dispatch macros in the CPU implementation:

```bash
grep -n "AT_DISPATCH_\w*TYPES" <cpu_kernel_file>
```

Common patterns:
- `AT_DISPATCH_ALL_TYPES` — int8..int64, float, double
- `AT_DISPATCH_ALL_TYPES_AND` — above + specified extras (BFloat16, Half, etc.)
- `AT_DISPATCH_FLOATING_TYPES` — float, double only
- `AT_DISPATCH_FLOATING_TYPES_AND` — float, double + extras
- `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES` — float, double, cfloat, cdouble
- `AT_DISPATCH_COMPLEX_TYPES` — cfloat, cdouble only
- `AT_DISPATCH_INTEGRAL_TYPES` — uint8, int8, int16, int32, int64

#### 3.3 Find dtype constraints in CUDA kernel

Same macro search in the CUDA file. CUDA may support fewer or more types than CPU.

Present a comparison table:

```
| dtype       | CPU | CUDA |
|-------------|-----|------|
| float32     |  Y  |  Y   |
| float64     |  Y  |  Y   |
| bfloat16    |  ?  |  ?   |
| float16     |  ?  |  ?   |
| int8        |  ?  |  ?   |
| int16       |  ?  |  ?   |
| int32       |  ?  |  ?   |
| int64       |  ?  |  ?   |
| bool        |  ?  |  ?   |
| complex64   |  ?  |  ?   |
| complex128  |  ?  |  ?   |
```

### Step 4: Trace the Dispatch Path

Show the full dispatch chain from Python call to kernel execution.

#### 4.1 Python entry point

Find the Python binding:
```bash
grep -rn "def <op_name>" /share_data/tangcong/project/pytorch_v2.7.1/torch/_refs/
grep -rn "<op_name>" /share_data/tangcong/project/pytorch_v2.7.1/torch/functional.py
```

#### 4.2 Native function dispatch

From `native_functions.yaml`, determine:
- If `dispatch:` has `CPU: <func>` → registered to CPU dispatch key
- If `dispatch:` has `CUDA: <func>` → registered to CUDA dispatch key
- If `CompositeImplicitAutograd` → decomposes into other ops
- If uses a DispatchStub → `DECLARE_DISPATCH` + `DEFINE_DISPATCH` + `REGISTER_DISPATCH`

#### 4.3 DispatchStub pattern (if applicable)

Many ops use DispatchStub for CPU/CUDA:

```
<Op>.cpp (declares stub) → DECLARE_DISPATCH(<fn_type>, <stub_name>)
cpu/<Op>Kernel.cpp → REGISTER_DISPATCH(<stub_name>, &<cpu_impl>)
cuda/<Op>Kernel.cu → REGISTER_DISPATCH(<stub_name>, &<cuda_impl>)
```

Search for the stub:
```bash
grep -rn "DECLARE_DISPATCH.*<op>" /share_data/tangcong/project/pytorch_v2.7.1/aten/src/ATen/native/
grep -rn "REGISTER_DISPATCH.*<op>" /share_data/tangcong/project/pytorch_v2.7.1/aten/src/ATen/native/
```

Present the full path:

```
torch.<op>(tensor)
  → aten::<op> (C++ dispatcher)
    → [CPU] <file.cpp>:<line> → DispatchStub → cpu/<file>Kernel.cpp:<kernel_func>
    → [CUDA] <file.cpp>:<line> → DispatchStub → cuda/<file>Kernel.cu:<kernel_func>
```

### Step 5: Analyze CPU Implementation

Read the CPU kernel file and identify:

#### 5.1 Scalar path
- The innermost computation without vectorization
- Usually in a lambda or template function processing single elements

#### 5.2 Vectorized path (Vec256/Vectorized)
- Uses `at::vec::Vectorized<scalar_t>` for SIMD
- Look for `at::vec::map`, `at::vec::map2`, or explicit Vectorized operations
- Note which vec ops are used (e.g., `vec.exp()`, `vec.log()`, custom formulas)

#### 5.3 TensorIterator usage
- Does the op use `TensorIterator` / `TensorIteratorConfig`?
- What kind: `unary_op`, `binary_op`, `reduce_op`, `nullary_op`?
- Any special config: `check_mem_overlap`, `allow_cpu_scalars`, etc.

Present:
```
CPU Implementation: <file>:<line_range>
  TensorIterator: Yes/No (type: unary_op/binary_op/reduce_op/...)
  Scalar kernel: <brief description of scalar computation>
  Vectorized kernel: <brief description, which Vectorized ops used>
  Special handling: <any edge cases, special dtype paths, etc.>
```

### Step 6: Analyze CUDA Implementation

Read the CUDA kernel file and identify:

#### 6.1 Launch pattern
- `gpu_kernel` (element-wise via TensorIterator)
- `gpu_reduce_kernel` (reduction)
- Custom kernel launch (`<<<blocks, threads>>>`)

#### 6.2 Kernel structure
- Element-wise functor / lambda
- Shared memory usage
- Block/thread configuration
- Multi-pass algorithms

#### 6.3 Special optimizations
- Vectorized loads (`float4`, `__half2`)
- Warp-level primitives (`__shfl_down_sync`, etc.)
- cuBLAS/cuDNN/cuSOLVER calls

Present:
```
CUDA Implementation: <file>:<line_range>
  Launch pattern: gpu_kernel / gpu_reduce_kernel / custom<<<>>>
  Kernel type: element-wise functor / block reduction / ...
  Vectorized loads: Yes/No
  Shared memory: Yes/No
  Library calls: cuBLAS / cuDNN / none
  Special: <any notable optimizations>
```

### Step 7: Implementation Notes for SIPU Porting

Based on the analysis above, provide concrete guidance for implementing this op in torch_sipu:

#### 7.1 Recommended approach

Map PyTorch's implementation pattern to SIPU's infrastructure:

| PyTorch pattern | SIPU equivalent |
|---|---|
| TensorIterator + unary/binary functor | `Loops.suh` / `VecLoops.suh` / `TileLoops.suh` |
| TensorIterator + reduction | `Reduce.suh` (vectorized_reduction) |
| Vectorized<scalar_t> | `VectorizedM1` (Vec.suh) |
| Custom CUDA kernel | `parallel_for` + `VectorizedM1` (Parallel.suh) |
| cuBLAS/cuDNN call | sikernel library or Triton backend |

#### 7.2 Classification

State the operator category per the operator-dev skill:

```
Category: E1/E2/C/R1/R2/M/S/X
Recommended path: PATH-A / PATH-A-REDUCE / PATH-B / PATH-C / Triton
```

#### 7.3 Key considerations

Check and report on each:

1. **CompositeImplicitAutograd?** — If yes, the op auto-decomposes; do NOT register unless there's a performance reason.
2. **Structured kernel?** — If yes, must implement `TORCH_SIPU_IMPL_FUNC` pattern.
3. **dtype coverage** — Which dtypes MUST be supported (at minimum float32 + bfloat16)?
4. **Precision pitfalls** — Any intermediate computation that requires float32 accumulation for bf16/fp16?
5. **In-place / out variants** — Does the op have `.out()` or `_()` variants that need separate registration?
6. **Broadcasting** — Does TensorIterator handle it, or is manual handling needed?
7. **Memory layout** — Any contiguity requirements or non-contiguous fast paths?
8. **Edge cases** — Empty tensors, 0-dim tensors, scalar inputs, negative dimensions?

### Step 8: Final Summary

Output a concise report combining all findings:

```
═══════════════════════════════════════════════════════
  PyTorch Operator Investigation Report: aten::<op>
═══════════════════════════════════════════════════════

1. Schema
   <func signature from native_functions.yaml>

2. Dispatch Type
   Structured / Unstructured
   CompositeImplicitAutograd: Yes/No

3. Supported dtypes
   CPU:  [list]
   CUDA: [list]

4. Dispatch Path
   Python → C++ → [CPU] <path>
                → [CUDA] <path>

5. CPU Kernel
   File: <path>
   Pattern: TensorIterator + scalar/vec
   Vectorized ops: <list>

6. CUDA Kernel
   File: <path>
   Pattern: gpu_kernel / custom
   Key optimizations: <list>

7. SIPU Implementation Recommendation
   Category: <E1/E2/C/R1/R2/M/S/X>
   Path: <PATH-A/B/C/Triton>
   Key notes:
   - <note 1>
   - <note 2>
   - ...
═══════════════════════════════════════════════════════
```

## Rules

- Use the Read tool to read source files — do NOT guess implementations
- For files over 500 lines, use Grep to locate relevant sections first, then read specific line ranges
- Always verify claims by reading actual source code
- If an op has multiple overloads, investigate ALL of them
- If the op is CompositeImplicitAutograd, still show the decomposition to understand what sub-ops are used
- Present dtype info based on actual `AT_DISPATCH_*` macros, not assumptions
