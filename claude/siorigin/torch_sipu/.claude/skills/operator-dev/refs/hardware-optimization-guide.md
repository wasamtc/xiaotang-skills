# Hardware Optimization Decision Guide

This guide embeds SIPU hardware architecture knowledge and provides a structured decision engine for choosing the optimal implementation strategy. Inspired by KernelMem's Machine-Check approach — deterministic rules first, not guesswork.

---

## 1. SIPU Hardware Architecture Summary

### 1.1 Chip Topology

```
Chip (1 device)
├── PEG 0 (uSIPU) — 16 GB DRAM
│   ├── PEC 0 — 512 KB SRAM
│   │   ├── PE 0 (2 RV Cores + 1 Tile Core) → Thread 0
│   │   └── PE 1 (2 RV Cores + 1 Tile Core) → Thread 1
│   ├── PEC 1 — 512 KB SRAM
│   ├── PEC 2 — 512 KB SRAM
│   └── PEC 3 — 512 KB SRAM
└── PEG 1 (uSIPU) — 16 GB DRAM
    ├── PEC 4 — PEC 7 (same structure)
```

| Level | Count | Code Mapping | Threads |
|-------|-------|-------------|---------|
| Chip | 1 | Device | — |
| PEG | 2 | Cluster | — |
| PEC | 8 (4/PEG) | Block | `num_blocks()=16` |
| PE | 16 (2/PEC) | Thread | `num_threads()=2` |
| **Total** | **32 threads** | | `SIPU_MAX_TOTAL_THREAD=32` |

### 1.2 PE Internal Architecture

Each PE contains:

**2 RV Cores** (RISC-V + RVV 1.0):
- VLEN = 1024 bit per core
- LMUL m1~m8 (32~256 float32 elements per instruction)
- Control flow, address calculation, scalar/vector compute
- FP mode switching via CSR (BF16/FP16 share hardware, switched by register)

**1 Tile Core** with 5 functional units:
- **TMAC** — Matrix Multiply-Accumulate (MMA): bf16×bf16→fp32, fp16×fp16→fp32, int8×int8→int32
- **TALU** — Tile ALU: element-wise arithmetic (+, -, ×, ÷, compare) on entire tiles
- **TSFU** — Tile Special Function Unit: sigmoid, rsqrt, reciprocal, silu, sin, cos on entire tiles
- **TLSU** — Tile Load/Store Unit: DRAM/SRAM ↔ Tile Registers
- **TDTE** — Data Transport Engine: async DMA with linear↔tiled format conversion

**Tile Registers**: ~160 per PE, each 1024 Bytes (1KB), total ~160 KB/PE

### 1.3 Storage Hierarchy

```
Level              Latency       Capacity            Access Scope
─────────────────  ──────────    ──────────────      ──────────────
Tile Registers     0 cycle       ~160 KB/PE          PE-local (Tile Core)
RVV Vector Regs    0 cycle       ~32 KB/PE           PE-local (RV Core)
SRAM (__shared__)  ~few cycles   512 KB/PEC          PEC-local (2 PEs share)
DRAM               ~100s cycles  16 GB/PEG           PEG-local (NUMA across PEGs)
```

### 1.4 Key Hardware Constraints

| Constraint | Value | Impact |
|-----------|-------|--------|
| Tile types | float, bf16, fp16 only | int/bool ops cannot use Tile path |
| SRAM per PEC | 512 KB | Limits working set for producer-consumer |
| Tile register count | ~160/PE | Limits tile-level parallelism |
| RVV VLEN | 1024 bit | M1 = 32 float, 64 bf16, 128 int8 |
| BF16/FP16 CSR switch | Shared hardware | Must call `set_fp_mode()` before half-precision RVV ops |
| `__syncblocks()` | Chip-wide barrier | Only in cooperative kernels |

---

## 2. Optimization Decision Engine

### 2.1 Step 1: Classify the Operator

Determine the operator category — this drives all subsequent decisions:

| Category | Examples | Key Characteristic |
|----------|----------|-------------------|
| **E1: Element-wise Unary** | neg, sigmoid, silu, rsqrt, abs, exp, log | 1 input → 1 output, per-element |
| **E2: Element-wise Binary** | add, mul, sub, div, pow, maximum | 2 inputs → 1 output, per-element |
| **C: Comparison** | eq, ne, gt, ge, lt, le | 2 inputs → bool output |
| **R1: Simple Reduction** | sum, prod, any, all, max, min | tensor → scalar/smaller tensor |
| **R2: Compound Reduction** | softmax, log_softmax, layernorm, rmsnorm | Multi-pass: reduce → normalize |
| **M: Matrix** | mm, bmm, addmm, attention | GEMM-class, data-parallel |
| **S: Structural** | cat, stack, index_select, topk, sort | Non-element-wise data movement |
| **X: Custom SIPU** | mm_t2t, _mx_mm, flash_attention | SIPU-specific ops using sikernel |

### 2.2 Step 2: Select Execution Path

Based on the category, select the primary execution strategy:

```
Category → Execution Path Decision
─────────────────────────────────────────────────────────────────────
E1, E2, C  → PATH-A: TensorIterator + Tile→RVV→Scalar cascade
             Use: sipu_kernel_tile / sipu_kernel_unified
             File: .su with REGISTER_PRIVATEUSE1_DISPATCH

R1         → PATH-A-REDUCE: TensorIterator + Reduce.suh
             Use: binary_kernel_reduce_vec_sipu
             File: .su with REGISTER_PRIVATEUSE1_DISPATCH

R2         → PATH-B: TORCH_SIPU_IMPL_FUNC + parallel_for + VectorizedM1
             Use: manual loop with vectorized+scalar-tail pattern
             File: .su with TORCH_SIPU_IMPL_FUNC

M          → PATH-C: Host-only .cpp calling sikernel library
             Or: Triton kernel with appropriate BLOCK_SIZE/num_warps
             File: .cpp or Triton .py

S          → PATH-B or PATH-C: depends on data access pattern
             If regular stride access → PATH-B with parallel_for
             If irregular access → PATH-C or scalar implementation

X          → PATH-C: sikernel library calls
             File: .cpp with TORCH_SIPU_IMPL_FUNC
```

### 2.3 Step 3: Determine Precision Strategy

```
Input dtype?
├── float32
│   └── Compute in float32 (no promotion needed)
│
├── bfloat16 / float16
│   ├── Is the op numerically sensitive? (softmax, layernorm, rmsnorm,
│   │   log_softmax, cross_entropy, kl_div, cosine_similarity, or
│   │   any op involving exp/log/div on reduced values)
│   │   ├── YES → ALL intermediates in float32
│   │   │         Cast input to opmath_t at start, cast back at final store only
│   │   │         Vec path: M1→M2 widen, compute in float M2, narrow at store
│   │   └── NO  → Standard M1 storage / M2 compute pattern
│   │             Accumulations (sum, max) in float32
│   │             Element-wise ops (mul, neg) can stay in scalar_t
│   │
│   └── For Tile path: hardware handles precision internally (use directly)
│
├── int8 / int16 / int32 / uint8 / uint16 / uint32
│   └── RVV path only (no Tile support). VectorizedM1 directly.
│
├── bool
│   └── RVV path only. VectorizedM1<bool>.
│
├── float64 / complex / int64
│   └── Scalar only (no RVV or Tile support). sipu_kernel().
│
└── float8_e4m3fn / float8_e5m2
    └── Triton kernel path (if downstream of quantized layer)
        Add to dtype test matrix if applicable.
```

### 2.4 Step 4: Select Parallel Strategy

For PATH-B ops (compound reduction, structural ops with parallel_for):

```
Parallelism Decision
────────────────────────────────────────────────
Outer loop (batch/row dimension):
  → parallel_for with grain_size

Inner loop (reduction/feature dimension):
  → VectorizedM1 vectorized + scalar tail

grain_size selection:
  ├── dim_size < 128        → grain_size = 1 (maximize parallelism)
  ├── dim_size 128-1024     → grain_size = 1
  └── dim_size > 1024       → grain_size = 1 (let runtime decide)
  NOTE: grain_size=1 is usually optimal; runtime scheduler handles load balancing
```

### 2.5 Step 5: Triton-Specific Parameters (if Triton path)

```
BLOCK_SIZE selection:
  ├── Small elements (< 1K)    → BLOCK_SIZE = 1024
  ├── Medium elements (1K-64K) → BLOCK_SIZE = 1024 or 2048
  └── Large elements (> 64K)   → BLOCK_SIZE = 2048

num_warps selection:
  ├── Simple element-wise ops  → num_warps = 2
  ├── Reduction ops            → num_warps = 4
  └── Matrix ops               → num_warps = 4

num_ctas selection:
  ├── Default                  → num_ctas = 4
  └── Large batch size         → num_ctas = 8
```

---

## 3. Implementation Pattern Library

### 3.1 Pattern: Element-wise Unary with Full Cascade (PATH-A)

**When:** E1 category ops (sigmoid, silu, neg, rsqrt, abs, ...)

```cpp
void my_op_kernel_sipu(TensorIteratorBase& iter) {
    ScalarType dtype = iter.dtype(1);

    if (at::isReducedFloatingType(dtype)) {
        // BF16/FP16: promote to float
        AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "my_op_reduced", [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            sipu_kernel_tile(iter,
                [=] SIPU_LAMBDA(scalar_t a) -> scalar_t {
                    return static_cast<scalar_t>(f(static_cast<opmath_t>(a)));
                },
                [=] SIPU_LAMBDA(vec::VectorizedM1<scalar_t> a) -> vec::VectorizedM1<scalar_t> {
                    vec::VectorizedM2<opmath_t> wide(a);
                    wide = wide.my_op();
                    return vec::VectorizedM1<scalar_t>(wide);
                },
                [=] SIPU_LAMBDA(vec::Tiled<scalar_t, 512> a) -> vec::Tiled<scalar_t, 512> {
                    return a.my_op();
                });
        });
    } else if (vec::isTileSupportedType(dtype)) {
        // float: direct Tile
        AT_DISPATCH_FLOATING_TYPES(dtype, "my_op_tile", [&]() {
            sipu_kernel_tile(iter,
                [=] SIPU_LAMBDA(scalar_t a) -> scalar_t { return f(a); },
                [=] SIPU_LAMBDA(vec::VectorizedM1<scalar_t> a) -> vec::VectorizedM1<scalar_t> {
                    return a.my_op();
                },
                [=] SIPU_LAMBDA(vec::Tiled<scalar_t, 512> a) -> vec::Tiled<scalar_t, 512> {
                    return a.my_op();
                });
        });
    } else if (vec::isRvSupportedType(dtype)) {
        // int types: RVV only
        AT_DISPATCH_ALL_TYPES(dtype, "my_op_rvv", [&]() {
            sipu_kernel_vec(iter,
                [=] SIPU_LAMBDA(scalar_t a) -> scalar_t { return f(a); },
                [=] SIPU_LAMBDA(vec::VectorizedM1<scalar_t> a) -> vec::VectorizedM1<scalar_t> {
                    return a.my_op();
                });
        });
    } else {
        // double, complex, etc: scalar
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX(dtype, "my_op_scalar", [&]() {
            sipu_kernel(iter, [=] SIPU_LAMBDA(scalar_t a) -> scalar_t { return f(a); });
        });
    }
}
REGISTER_PRIVATEUSE1_DISPATCH(my_op_stub, &my_op_kernel_sipu);
```

### 3.2 Pattern: Symmetric Binary with Unified (PATH-A)

**When:** E2 category, symmetric ops (add, mul, max, min)

```cpp
void mul_kernel_sipu(TensorIteratorBase& iter) {
    auto dtype = iter.common_dtype();
    if (vec::isIterAllRvvSupportedTypes(iter)) {
        AT_DISPATCH_ALL_TYPES_AND3_RVV(kHalf, kBFloat16, kBool, dtype, "mul", [&] {
            using opmath_t = at::opmath_type<scalar_t>;
            opmath_symmetric_sipu_kernel_with_scalars_unified(iter,
                [=] SIPU_LAMBDA(opmath_t a, opmath_t b) -> opmath_t { return a * b; },
                [=] SIPU_LAMBDA(vec::VectorizedM2<opmath_t> a, vec::VectorizedM2<opmath_t> b)
                    -> vec::VectorizedM2<opmath_t> { return a * b; },
                [=] SIPU_LAMBDA(vec::TiledM2<opmath_t> a, vec::TiledM2<opmath_t> b)
                    -> vec::TiledM2<opmath_t> { return a * b; });
        });
    } else {
        AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, dtype, "mul", [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            opmath_symmetric_sipu_kernel_with_scalars<scalar_t>(iter, MulFunctor<opmath_t>());
        });
    }
}
```

### 3.3 Pattern: Compound Reduction with VectorizedM1 (PATH-B)

**When:** R2 category ops (softmax, layernorm, rmsnorm)

```cpp
TORCH_SIPU_IMPL_FUNC(my_norm_out)
(const Tensor& self, ..., const Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, dtype, "my_norm", [&] {
        using Vec = vec::VectorizedM1<scalar_t>;
        constexpr int vec_size = Vec::size();
        using opmath_t = at::opmath_type<scalar_t>;

        at::parallel_for(0, outer_size, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; i++) {
                scalar_t* row = data + i * dim_size;

                // === Pass 1: Vectorized reduce (e.g., find max) ===
                opmath_t max_val = -std::numeric_limits<opmath_t>::infinity();
                int64_t d = 0;
                for (; d <= dim_size - vec_size; d += vec_size) {
                    Vec data_vec = Vec::loadu(row + d);
                    // ... vectorized reduce ...
                }
                for (; d < dim_size; d++) {
                    max_val = std::max(max_val, static_cast<opmath_t>(row[d]));
                }

                // === Pass 2: Vectorized transform (e.g., exp(x - max)) ===
                d = 0;
                for (; d <= dim_size - vec_size; d += vec_size) {
                    Vec data_vec = Vec::loadu(row + d);
                    // ... vectorized compute ...
                    data_vec.store(out_row + d);
                }
                for (; d < dim_size; d++) {
                    out_row[d] = /* scalar compute, SAME formula as vec path */;
                }
            }
        });
    });
}
```

**Critical:** Vectorized path and scalar tail MUST use identical computation and precision.

### 3.4 Pattern: Simple Reduction (PATH-A-REDUCE)

**When:** R1 category ops (sum, prod, any, all)

```cpp
void or_kernel_sipu(TensorIterator& iter) {
    using Vec = vec::VectorizedM1<bool>;
    binary_kernel_reduce_vec_sipu(
        iter,
        [=](bool a, bool b) -> bool { return a || b; },
        [=](Vec a, Vec b) -> Vec { return a | b; },
        /*identity=*/false
    );
}
REGISTER_PRIVATEUSE1_DISPATCH(or_stub, &or_kernel_sipu);
```

---

## 4. Similar-Op Reference Matching

When implementing a new op, find the most similar existing implementation:

### Find Similar Ops

| Your new op type | Look at these existing implementations |
|-----------------|---------------------------------------|
| Unary activation (gelu, mish, ...) | `silu` in UnaryOpsKernel.su |
| Unary math (acos, asin, ...) | `rsqrt` in UnaryOpsKernel.su |
| Binary arithmetic (fmod, remainder) | `mul` or `add` in BinaryOpsKernel.su |
| Binary with alpha (addcmul, addcdiv) | `add` with alpha in Add.su |
| Comparison op | `ge_kernel_sipu` in CompareKernel.su |
| Reduction (mean, std) | `sum` in ReduceOps.su, `softmax` for multi-pass |
| Normalization (batchnorm, groupnorm) | `layernorm` or `rmsnorm` |
| Index/gather/scatter | `index_select` or existing structural ops |

### How to Find Implementations

```bash
# Find C++ kernel for an op
grep -r "REGISTER_PRIVATEUSE1_DISPATCH.*<op>" torch_sipu/csrc/
grep -r "TORCH_SIPU_IMPL_FUNC.*<op>" torch_sipu/csrc/

# Find Triton kernel for an op
grep -r "def <op>" torch_sipu/backends/sipu_triton_kernels/ops/
```

---

## 5. Performance Validation Checklist

After implementation, verify against these rules (deterministic checks, not subjective):

### 5.1 Execution Path Check

| Rule | Check | FAIL Condition |
|------|-------|----------------|
| P1 | Does the op implement the correct execution path per §2.2? | Category E1/E2/C uses scalar-only when Tile/RVV available |
| P2 | For PATH-A: is full Tile→RVV→Scalar cascade present? | Missing Tile path when dtypes support it |
| P3 | For PATH-B: does hot loop use VectorizedM1? | Scalar-only loop when Vec methods exist |
| P4 | For PATH-B: is scalar tail present AND consistent? | Missing tail, or tail uses different formula/precision |

### 5.2 Precision Check

| Rule | Check | FAIL Condition |
|------|-------|----------------|
| PR1 | BF16/FP16 inputs → float32 accumulation? | sum/max reduction in half precision |
| PR2 | Numerically-sensitive op → ALL intermediates in float32? | `exp(x-max)` where subtraction is in fp16 |
| PR3 | RVV M1→M2 widening used for bf16/fp16 compute? | Direct VectorizedM1<BFloat16> arithmetic |
| PR4 | fp16/bf16 test tolerance ≤ atol=0.01, rtol=5e-3? | Wider tolerance = investigate precision chain |

### 5.3 Correctness Check

| Rule | Check | FAIL Condition |
|------|-------|----------------|
| C1 | SIPU_LAMBDA on all device lambdas? | Missing → compile error or wrong results |
| C2 | Scalar values broadcast via splat() in vec lambdas? | Direct scalar-vector arithmetic |
| C3 | isIterAll* guard (not is*SupportedType)? | Only checks common_dtype, misses mixed types |
| C4 | Correct dispatch macro paired with loop function? | `_AT_DISPATCH_TILE_TYPES` with `sipu_kernel_vec` |

### 5.4 Triton-Specific Check

| Rule | Check | FAIL Condition |
|------|-------|----------------|
| T1 | @cpu_fallback with correct precheck dtypes? | Unsupported dtype crashes instead of falling back |
| T2 | @sipu_verify with reasonable tolerance? | Missing verification decorator |
| T3 | All variants registered (op, op.Tensor, op_)? | Missing dispatch entries |
| T4 | 1024-byte alignment handled? | @triton_preprocess or manual alignment |

---

## 6. Performance Impact Reference

| Implementation Level | Speed vs Scalar | Hardware Unit |
|---------------------|----------------|---------------|
| Scalar loop | 1x (baseline) | RV Core scalar ALU |
| VectorizedM1 (RVV) | 2-10x | RV Core RVV unit (32 float/cycle) |
| VectorizedM8 (RVV) | 8-40x | RV Core RVV unit (256 float/cycle) |
| Tiled (Tile Core) | 10-100x | TALU/TSFU (1024B tile/op) |
| TMAC (matrix multiply) | 100-1000x | TMAC unit (systolic array) |

**Example:** Softmax on 256-element row:
- Scalar: 256 cycles (1 element/cycle)
- VectorizedM1: 8 cycles (32 float/cycle)
- **32x speedup** just from vectorization
