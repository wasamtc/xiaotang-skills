# Performance Review Guide (MANDATORY for C++ Kernels)

After implementing the kernel, **before testing**, verify that the implementation uses the appropriate performance level. Scalar-only implementations leave 2-100x performance on the table.

## Checklist

1. **Option A ops (TensorIterator)**: Does the kernel implement the full Tile → RVV → Scalar cascade? Check against the Performance Primitive Selection Guide below.
   - FAIL if only `sipu_kernel()` (scalar) is used when the dtypes support Tile/RVV.

2. **Option B ops (parallel_for + custom kernel)**: Does the device kernel use `VectorizedM1<scalar_t>` for the hot loop?
   - The device kernel MUST have a **vectorized path + scalar tail** pattern:
     ```cpp
     int64_t d = 0;
     // Vectorized path
     if (dim_size >= vec_size) {
         for (d = 0; d <= dim_size - vec_size; d += vec_size) {
             Vec data = Vec::loadu(in + d);
             // ... vectorized computation ...
             data.store(out + d);
         }
     }
     // Scalar tail for remaining elements
     for (; d < dim_size; ++d) {
         out[d] = /* scalar computation */ in[d];
     }
     ```
   - FAIL if the hot loop is entirely scalar when `VectorizedM1` operations exist for the needed computation.
   - Check `VectorizedM1` available methods: `neg()`, `sigmoid()`, `rsqrt()`, `reciprocal()`, `sin()`, `cos()`, `silu()`, `pow()`, `maximum()`, `minimum()`, `operator+`, `operator-`, `operator*`, `operator/`, `loadu()`, `store()`.
   - For operations NOT in `VectorizedM1` (e.g., `exp`): use the **store-compute-reload** pattern with scalar `fast_exp()`:
     ```cpp
     // Store vector to array, apply scalar function, reload
     scalar_t temp[vec_size];
     data_vec.store(temp);
     for (int j = 0; j < vec_size; ++j) {
         temp[j] = static_cast<scalar_t>(fast_exp(static_cast<double>(temp[j])));
     }
     Vec result = Vec::loadu(temp);
     ```
   - **CRITICAL — Vectorized/Scalar Tail Consistency**: The scalar tail MUST use **identical computation and precision** as the vectorized path for the same operation. Any divergence is a correctness bug. Specifically:
     - Same precision: if vectorized path computes `Vec * inv_sum_vec` (in `scalar_t`), the scalar tail must also use `scalar_t` (e.g., `out[d] *= inv_sum`), NOT `float32` (e.g., `static_cast<float>(out[d]) * (1.0f / sum)`)
     - Same formula: if vectorized uses `x * inv_sum`, tail must use `x * inv_sum`, NOT `x / sum`
     - Same function: if vectorized uses `fast_exp()`, tail must use `fast_exp()`, NOT `std::exp()`
     - **Test**: if `dim_size` is a multiple of `vec_size`, only the vectorized path runs. If not, the tail also runs. Results must be identical regardless.

3. **Precision promotion**: For bf16/fp16 inputs, intermediate accumulation (sum, max reduction across elements) MUST be done in float32. This is NOT optional — it's a correctness requirement that also affects performance (hardware float32 ALUs are typically faster than bf16 emulation for accumulation).
   - **Vectorized compute in scalar_t** (load/store, element-wise ops like max, multiply) + **scalar accumulation in float32** (sum, final normalization) is the correct pattern.
   - Do NOT cast every single element to float32 for simple operations that VectorizedM1 can handle natively.

4. **Numerically-sensitive operator full-precision check**: For operators that are **numerically sensitive** (softmax, log_softmax, layernorm, groupnorm, batchnorm, rmsnorm, cross_entropy_loss, nll_loss, kl_div_loss, cosine_similarity, and other loss/normalization functions), the standard precision promotion rule (item 3) is **insufficient**. These ops require **ALL intermediate arithmetic** — not just accumulation — to be computed in float32 when the input is fp16/bf16.
   - **Why**: In numerically-sensitive ops, even non-accumulation steps like subtraction (`x - max`) or division (`x / sum`) can amplify precision errors. For example, in softmax with fp16 inputs, computing `exp(x - max)` where the subtraction is done in fp16 precision loses significant bits, causing large output errors (atol > 0.01).
   - **Required pattern**: Use `using opmath_t = at::opmath_type<scalar_t>` and cast inputs to `opmath_t` at the beginning of the computation. All intermediate values (max, subtraction, exp, sum, division/multiplication by inverse) must remain in `opmath_t`. Only cast back to `scalar_t` at the final store.
   - **Vectorized path**: When the vectorized path operates on `VectorizedM1<scalar_t>`, ensure that precision-critical steps (subtraction from max, exp computation, normalization) either use float32 vectors or go through a store-compute-reload pattern in float32. The key invariant is: **no intermediate result that feeds into exp() or division should be computed in fp16/bf16 precision**.
   - **Test tolerance guideline**: If your fp16 test requires `atol > 0.01` or `rtol > 5e-3` to pass, this is a strong signal that intermediate precision is insufficient — investigate and fix the precision chain before widening tolerances.
   - **Operators covered**: softmax, log_softmax, layernorm, groupnorm, batchnorm, rmsnorm, cross_entropy_loss, nll_loss, kl_div_loss, cosine_similarity, and any op involving `exp()`, `log()`, or `div()` applied to reduced/normalized values. (Same list as the opening of this item — cross-check if adding a new op to either list.)

5. **[Inference] Fusion opportunity check**: Could this op's computation be fused with an adjacent op to eliminate a kernel launch or intermediate tensor allocation?
   - Flag it in the MR description if yes — even if not fused in this MR, it should be tracked.
   - Typical opportunities: norm→activation, matmul→scale, reduce→normalize (e.g., softmax steps).

6. **[Inference] FP8 dtype coverage**: If the op is used downstream of a quantized layer (e.g., after `act_quant` which outputs `float8_e4m3fn`), does it need to accept FP8 inputs?
   - Check whether the Triton kernel handles `tl.float8e4m3fn` input dtype or requires dequantization upstream.
   - If FP8 input is expected, add `torch.float8_e4m3fn` to the dtype test matrix.

7. **[Inference] Memory alignment and stride constraints**:
   - Triton ops: inputs must be 1024-byte aligned. Verify `@triton_preprocess` handles this, or call `convert_to_contiguous_and_aligned(alignment=1024)` explicitly.
   - For 2D ops that assume row-major layout: assert `tensor.stride(-1) == 1` at the entry point (see `softmax.py` for example).
   - C++ `.su` ops: vectorized loops with scalar tails handle any alignment — no manual padding needed (padding is the old `.cpp` pattern being phased out).

## Performance Impact Reference

| Implementation | Relative Speed (vs scalar) | Example |
|---|---|---|
| Scalar-only loop | 1x (baseline) | `for (d=0; d<N; d++) out[d] = f(in[d]);` |
| VectorizedM1 (RVV) | 2-10x | `Vec::loadu → compute → store` with scalar tail |
| Tiled (TileCore) | 10-100x | `sipu_kernel_tile` (Option A only) |

A scalar-only softmax on a 256-element row processes 1 element/cycle. VectorizedM1 processes 32 float32 elements/cycle (or 64 bf16/fp16). This is a **32-64x difference** on the hot path.

## Output

After review, state the performance level:
```
Performance: RVV vectorized (VectorizedM1) with scalar tail — OK
Performance: Scalar only — NEEDS OPTIMIZATION (add VectorizedM1 path)
Performance: Tile + RVV + Scalar cascade — OK
```

---

## Performance Primitive Selection Guide

The SIPU hardware provides three execution paths with different performance characteristics. **Always implement the fastest available path first, with fallbacks for unsupported types.**

### Performance Hierarchy

| Path | Loop Function | Speed | Supported Types | Grain Size |
|---|---|---|---|---|
| **Tile** (TileCore hardware) | `sipu_kernel_tile()` | Fastest (10-100x vs scalar) | `float`, `Half`, `BFloat16` only | 4096 |
| **RVV** (RISC-V Vector) | `sipu_kernel_vec()` | Fast (2-10x vs scalar) | `float`, `Half`, `BFloat16`, `int8-32`, `Bool` | 512 |
| **Scalar** | `sipu_kernel()` | Baseline | All types | — |

### Decision Tree: Which Loop Function + Dispatch Macro?

```
Is this an element-wise / pointwise op using TensorIterator?
├─ YES → Use the multi-path pattern (Tile → RVV → Scalar):
│   ├─ Path 1: if (isIterAllTileSupportedTypes(iter))
│   │     → _AT_DISPATCH_TILE_TYPES(dtype, ...) + sipu_kernel_tile(...)
│   ├─ Path 2: else if (isIterAllRvvSupportedTypes(iter))
│   │     → _AT_DISPATCH_RVV_TYPES(dtype, ...) + sipu_kernel_vec(...)
│   └─ Path 3: else
│         → AT_DISPATCH_ALL_TYPES(dtype, ...) + sipu_kernel(...)
│
├─ Is it a binary op with scalar input (e.g., x + 5)?
│   ├─ Is it symmetric (f(a,b)==f(b,a))? → use opmath_symmetric_*_with_scalars*() variants
│   └─ Is it non-symmetric (f(a,b)!=f(b,a))? → use sipu_kernel_with_scalars*() variants
│
└─ NO (complex data traversal like softmax, triu/tril) →
    Use Option B: parallel_for + VectorizedM1 (manual vectorization)
```

### Correct Pairings (MUST match)

Each loop function MUST be paired with the correct dispatch macro and type guard:

| Loop Function | Dispatch Macro | Type Guard | Vector/Tile Type |
|---|---|---|---|
| `sipu_kernel_tile()` | `_AT_DISPATCH_TILE_TYPES` | `isIterAllTileSupportedTypes(iter)` | `Tiled<scalar_t, 512>` |
| `sipu_kernel_vec()` | `_AT_DISPATCH_RVV_TYPES` or `AT_DISPATCH_ALL_TYPES` | `isIterAllRvvSupportedTypes(iter)` | `VectorizedM1<scalar_t>` |
| `sipu_kernel()` | `AT_DISPATCH_ALL_TYPES*` | None (always safe) | N/A (scalar only) |

### Unified Functions (Simplified Alternative)

For simpler code, these unified functions **auto-select** the best path at runtime:

| Manual 3-path | Unified Equivalent | Use Case |
|---|---|---|
| `sipu_kernel_tile` + `sipu_kernel_vec` + `sipu_kernel` | `sipu_kernel_unified()` | Unary / binary without scalar |
| `*_with_scalars_tile` + `*_with_scalars_rvv` + `*_with_scalars` | `opmath_symmetric_sipu_kernel_with_scalars_unified()` | Symmetric binary with scalar |

Unified functions are defined in `TileLoops.suh`. They call `isIterAllTileSupportedTypes` internally and dispatch to the best path. Use them when the 3-path boilerplate is not needed.

### Binary Op Variant Selection

| Variant Family | When to Use | Example Ops |
|---|---|---|
| `opmath_symmetric_*_with_scalars*()` | `f(a,b) == f(b,a)`, supports scalar input | add, mul, max, min |
| `sipu_kernel_with_scalars*()` | `f(a,b) != f(b,a)`, supports scalar input | sub, div |
| Base `sipu_kernel*()` | No scalar input | unary ops (cos, neg, sigmoid) |

### AT_DISPATCH Macro Reference

| Macro | Defined In | Types Dispatched | Use With |
|---|---|---|---|
| `_AT_DISPATCH_TILE_TYPES` | `SIPUKernelHelper.h` | float, Half, BFloat16 | `sipu_kernel_tile` |
| `_AT_DISPATCH_RVV_TYPES` | `SIPUKernelHelper.h` | float, Half, BFloat16, int8-32 | `sipu_kernel_vec` |
| `_AT_DISPATCH_RVV_TYPES_WITH_BOOL` | `SIPUKernelHelper.h` | Above + Bool | `sipu_kernel_vec` |
| `AT_DISPATCH_ALL_TYPES` | PyTorch upstream | All standard types | `sipu_kernel` (scalar) |
| `AT_DISPATCH_FLOATING_TYPES_AND2` | PyTorch upstream | float, double + 2 extra | Option B `parallel_for` |
| `AT_DISPATCH_REDUCED_FLOATING_TYPES` | PyTorch upstream | Half, BFloat16 only | Precision promotion paths |

**Legacy macros (avoid in new code):**
| Macro | Notes |
|---|---|
| `AT_DISPATCH_SIPU_FLOATING_AND_COMPLEX_TYPES_AND2` | Old `.cpp` pattern with CPU fallback — use standard `AT_DISPATCH_*` in `.su` |
| `AT_DISPATCH_SIPU_ALL_TYPES_AND3` | Same — legacy `.cpp` |

### Anti-Patterns (Common Mistakes)

| Anti-Pattern | Why It's Wrong | Correct |
|---|---|---|
| Using only `sipu_kernel()` when types support Tile | Leaves 10-100x performance on the table | Add Tile + RVV paths with type guards |
| `_AT_DISPATCH_TILE_TYPES` without `isIterAllTileSupportedTypes` guard | Tile dispatch macro only handles 3 types — other types won't match and won't execute | Always guard with `isIterAllTileSupportedTypes(iter)` |
| `AT_DISPATCH_ALL_TYPES` with `sipu_kernel_tile` | Int types are NOT tile-supported — runtime crash | Use `_AT_DISPATCH_TILE_TYPES` with tile loop |
| Missing scalar fallback path | Some types (int64, double, complex) won't dispatch to any path | Always include scalar `sipu_kernel` as final fallback |
| Using `VectorizedM1` in tile lambda | Wrong vector type for tile path | Use `Tiled<scalar_t, 512>` for tile path |
| Using legacy `AT_DISPATCH_SIPU_*` macros in `.su` files | Legacy macros include CPU fallback logic not needed in `.su` | Use standard PyTorch `AT_DISPATCH_*` macros |

### Vector/Tile Type Reference

| Type | Path | Element Count (float32) | Element Count (bf16/fp16) |
|---|---|---|---|
| `Tiled<scalar_t, 512>` | Tile | 512 | 512 |
| `VectorizedM1<scalar_t>` | RVV | 32 | 64 |
| `VectorizedM2<scalar_t>` | RVV | 64 | 128 |
| `VectorizedM4<scalar_t>` | RVV | 128 | 256 |
| `VectorizedM8<scalar_t>` | RVV | 256 | 512 |

Most ops use `VectorizedM1` for RVV path and `Tiled<scalar_t, 512>` for Tile path. Use larger vector sizes only when the algorithm benefits from wider SIMD (e.g., reductions).
