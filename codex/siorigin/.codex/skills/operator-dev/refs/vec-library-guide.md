# Vec Library Reference — SIPU Vectorization API

This is the self-contained reference for the torch_sipu vectorization library (`Vec.suh`, `Tile.suh`, `VecLoops.suh`, `TileLoops.suh`). It covers all entry functions, vector/tile types, precision patterns, and common pitfalls needed for C++ kernel development.

---

## 1. Three Execution Paths

SIPU provides three compute paths, from fastest to slowest:

| Path | Mechanism | Speed | Supported Types |
|------|-----------|-------|-----------------|
| **Tile** | DMA + hardware Tile engine (TMAC/TALU/TSFU) | 10-100x vs scalar | float, BFloat16, Half |
| **RVV** | RISC-V Vector Extension (VLEN=1024 bit) | 2-10x vs scalar | float, BF16, Half, int8/16/32, uint8/16/32, bool |
| **Scalar** | Element-by-element | 1x baseline | All types (including complex, double) |

**Core principle:** Tile first → RVV fallback → Scalar last resort.

---

## 2. Entry Function Decision Tree

When writing an element-wise op, select the entry function based on:

```
What type of op?
│
├─ Element-wise unary (sigmoid, silu, neg, ...)
│   ├─ Best performance (Tile) → sipu_kernel_tile(iter, scalar_op, vec_op, tile_op)
│   ├─ RVV only              → sipu_kernel_vec(iter, scalar_op, vec_op)
│   └─ Scalar fallback        → sipu_kernel(iter, scalar_op)
│
├─ Element-wise binary (add, mul, sub, ...)
│   ├─ Has scalar operand (tensor + 3.0)?
│   │   ├─ Symmetric (f(a,b)==f(b,a), e.g. add/mul)?
│   │   │   └─ opmath_symmetric_sipu_kernel_with_scalars_unified(iter, s, v, t)
│   │   └─ Non-symmetric (e.g. sub/div/pow)?
│   │       └─ sipu_kernel_with_scalars_unified(iter, s, v, t)
│   └─ Both operands are tensors → same as unary (sipu_kernel_tile/vec/kernel)
│
├─ Reduction (sum, max, any, all)
│   └─ binary_kernel_reduce_vec_sipu(iter, scalar_op, vec_op, identity)
│       (from Reduce.suh, NOT the element-wise entry functions)
│
└─ Auto-select path (recommended for new ops)
    ├─ No scalar operand → sipu_kernel_unified(iter, s, v, t)
    └─ Has scalar operand → sipu_kernel_with_scalars_unified(iter, s, v, t)
```

### Symmetric vs Non-Symmetric

- **Symmetric** (`f(a,b) == f(b,a)`): `add`, `mul`, `bitwise_and/or`, `eq`, `ne`
- **Non-symmetric**: `sub`, `div`, `pow`, `gt`, `lt`

Symmetric ops use `opmath_symmetric_*` variants that auto-optimize scalar operand position.

---

## 3. Entry Function Signatures

### 3.1 Scalar Entry (fallback)

```cpp
sipu_kernel(iter, [=] SIPU_LAMBDA(scalar_t a) -> scalar_t { return f(a); });
```

### 3.2 RVV Entry

```cpp
sipu_kernel_vec(
    iter,
    [=] SIPU_LAMBDA(scalar_t a) -> scalar_t { ... },                         // scalar fallback
    [=] SIPU_LAMBDA(VectorizedM1<scalar_t> a) -> VectorizedM1<scalar_t> { ... }  // RVV vec
);
```

### 3.3 Tile Entry (three lambdas)

```cpp
sipu_kernel_tile(
    iter,
    [=] SIPU_LAMBDA(scalar_t a) -> scalar_t { ... },                              // scalar
    [=] SIPU_LAMBDA(VectorizedM1<scalar_t> a) -> VectorizedM1<scalar_t> { ... },  // RVV
    [=] SIPU_LAMBDA(Tiled<scalar_t, 512> a) -> Tiled<scalar_t, 512> { ... }       // Tile
);
```

### 3.4 Unified Entry (auto-selects Tile or RVV)

```cpp
sipu_kernel_unified(
    iter,
    [=] SIPU_LAMBDA(scalar_t a) -> scalar_t { ... },
    [=] SIPU_LAMBDA(VectorizedM1<scalar_t> a) -> VectorizedM1<scalar_t> { ... },
    [=] SIPU_LAMBDA(Tiled<scalar_t, 512> a) -> Tiled<scalar_t, 512> { ... }
);
```

### 3.5 With-Scalars Variants (binary ops)

```cpp
// Non-symmetric (sub, div, pow)
sipu_kernel_with_scalars_unified(
    iter,
    [=] SIPU_LAMBDA(opmath_t a, opmath_t b) -> opmath_t { return a - b; },
    [=] SIPU_LAMBDA(VectorizedM2<opmath_t> a, VectorizedM2<opmath_t> b) -> VectorizedM2<opmath_t> { return a - b; },
    [=] SIPU_LAMBDA(TiledM2<opmath_t> a, TiledM2<opmath_t> b) -> TiledM2<opmath_t> { return a - b; }
);

// Symmetric (add, mul)
opmath_symmetric_sipu_kernel_with_scalars_unified(
    iter,
    [=] SIPU_LAMBDA(opmath_t a, opmath_t b) -> opmath_t { return a * b; },
    [=] SIPU_LAMBDA(VectorizedM2<opmath_t> a, VectorizedM2<opmath_t> b) -> VectorizedM2<opmath_t> { return a * b; },
    [=] SIPU_LAMBDA(TiledM2<opmath_t> a, TiledM2<opmath_t> b) -> TiledM2<opmath_t> { return a * b; }
);
```

---

## 4. VectorizedM1/M2 API

`Vectorized<T, vec_size>` wraps RVV intrinsics. Common aliases:

```cpp
using vec = at::vec::sipu;
vec::VectorizedM1<float>           // LMUL=1, 32 floats
vec::VectorizedM2<float>           // LMUL=2, 64 floats
vec::VectorizedM1<c10::BFloat16>   // LMUL=1, 64 bf16 elements
```

### 4.1 Static Methods

```cpp
auto v = VectorizedM1<float>::loadu(ptr, vl);       // Load vl elements
v.store(ptr, vl);                                     // Store vl elements
auto v = VectorizedM1<float>::splat(&scalar_val);    // Broadcast scalar
auto v = VectorizedM1<float>(scalar_val);            // Constructor broadcast
size_t vl = VectorizedM1<float>::vsetvl(remaining);  // Set vector length
size_t max_vl = VectorizedM1<float>::vsetvlmax();    // Max vector length
VectorizedM1<c10::BFloat16>::set_fp_mode();          // Set FP mode for bf16
constexpr size_t sz = VectorizedM1<float>::size();   // Compile-time size (32)
```

### 4.2 Arithmetic

```cpp
auto c = a + b;    // add
auto c = a - b;    // subtract (NOTE: some types use a + b.neg() instead)
auto c = a * b;    // multiply
auto c = a / b;    // divide
auto c = a & b;    // bitwise AND
auto c = a | b;    // bitwise OR
auto c = ~a;       // bitwise NOT
auto c = !a;       // logical NOT
```

### 4.3 Math Functions

```cpp
a.neg()           // -x
a.abs()           // |x|
a.sigmoid()       // 1/(1+exp(-x))
a.silu()          // x * sigmoid(x)
a.rsqrt()         // 1/sqrt(x)
a.reciprocal()    // 1/x
a.sin()           // sin(x)
a.cos()           // cos(x)
a.exp()           // exp(x)
a.log()           // log(x)
a.pow(b)          // x^y
a.cumsum()        // cumulative sum
a.minimum(b)      // element-wise min
a.maximum(b)      // element-wise max
a.floor_divide(b) // floor division
```

### 4.4 Comparisons

Two styles with different return types:

```cpp
// Operator style → returns CompareVec<scalar_t> (bool vector, for blendv/where)
auto mask = (a >= b);
auto mask = (a > b);

// Member function style → returns same-type Vectorized (mask vector)
auto mask = a.ge(b);
auto mask = a.gt(b);
auto mask = a.eq(b);
auto mask = a.ne(b);
```

### 4.5 Conditional Selection

```cpp
auto result = vec::blendv(mask, a, b);  // mask ? a : b
```

### 4.6 Fast Scalar Math (for scalar lambdas)

```cpp
vec::fast_exp(x)       // fast exponential
vec::fast_sqrt(x)      // fast square root
vec::fast_sigmoid(x)   // fast sigmoid
vec::fast_silu(x)      // fast SiLU
```

### 4.7 Type Conversion Constructors

```cpp
// BF16 M1 → float M2 (widen, for low→high precision compute)
VectorizedM2<float> wide(bf16_m1_vec);

// float M2 → BF16 M1 (narrow, for write-back)
VectorizedM1<c10::BFloat16> narrow(float_m2_vec);

// Half M1 → float M2
VectorizedM2<float> wide(half_m1_vec);
```

These map to single RVV widening/narrowing instructions — near-zero cost.

---

## 5. Tiled API

`Tiled<T, tile_size>` wraps Tile Core operations. API mirrors Vectorized:

```cpp
vec::Tiled<float, 512>           // 512 floats
vec::TiledM1<float>              // = Tiled<float, 256>
vec::TiledM2<float>              // = Tiled<float, 512> (most common)
vec::TiledM2<c10::BFloat16>      // = Tiled<BFloat16, 1024>
```

### Methods

```cpp
auto t = Tiled<float, 512>::loadu(ptr, offset_byte);
t.store(ptr, offset_byte);
auto t = Tiled<float, 512>::splat(&scalar_val);

// Arithmetic (→ TALU)
auto c = a + b;  auto c = a * b;  auto c = a / b;

// Special functions (→ TSFU)
a.sigmoid()  a.silu()  a.rsqrt()  a.reciprocal()  a.sin()  a.cos()  a.neg()
```

---

## 6. Type Guards

Use these to decide which path to take:

```cpp
vec::isTileSupportedType(dtype)           // float, BF16, Half
vec::isRvSupportedType(dtype)             // float, BF16, Half, int8/16/32, uint8/16/32, bool

vec::isIterAllTileSupportedTypes(iter)    // ALL operands support Tile?
vec::isIterAllRvvSupportedTypes(iter)     // ALL operands support RVV?
```

**Always use `isIterAll*` (not `is*SupportedType`)** — binary ops may have mixed input types.

### Typical Pattern

```cpp
void my_kernel_sipu(TensorIteratorBase& iter) {
    auto dtype = iter.common_dtype();
    if (vec::isIterAllTileSupportedTypes(iter)) {
        // Tile path
    } else if (vec::isIterAllRvvSupportedTypes(iter)) {
        // RVV path
    } else {
        // Scalar fallback
    }
}
```

---

## 7. BF16/FP16 Precision Pattern: M1 Storage, M2 Compute

Low-precision types MUST be promoted to float32 for computation. Standard pattern:

```cpp
AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "my_op_reduced", [&]() {
    using opmath_t = at::opmath_type<scalar_t>;  // BF16/Half → float

    sipu_kernel_tile(
        iter,
        // Scalar: manual cast
        [=] SIPU_LAMBDA(scalar_t a) -> scalar_t {
            opmath_t val = static_cast<opmath_t>(a);
            opmath_t result = /* float precision compute */;
            return static_cast<scalar_t>(result);
        },
        // RVV: M1 → M2 widen → compute → M1 narrow
        [=] SIPU_LAMBDA(vec::VectorizedM1<scalar_t> a) -> vec::VectorizedM1<scalar_t> {
            vec::VectorizedM2<opmath_t> a_wide(a);       // BF16 M1 → float M2 (1 instruction)
            a_wide = a_wide.sigmoid();                     // float precision compute
            return vec::VectorizedM1<scalar_t>(a_wide);   // float M2 → BF16 M1 (1 instruction)
        },
        // Tile: hardware handles precision internally
        [=] SIPU_LAMBDA(vec::Tiled<scalar_t, 512> a) -> vec::Tiled<scalar_t, 512> {
            return a.sigmoid();
        });
});
```

### Why LMUL Doubles

BF16 is 16-bit, float is 32-bit. Same register width (1024 bit):
- BF16 M1 = 64 elements
- float M1 = 32 elements

To keep the same element count (64), float needs M2. Hence **BF16 M1 widens to float M2**.

---

## 8. Reduction Ops

Reduction ops use `Reduce.suh`, NOT the element-wise entry functions:

```cpp
#include "torch_sipu/csrc/aten/native/sipu/Reduce.suh"

binary_kernel_reduce_vec_sipu(
    iter,
    [=](scalar_t a, scalar_t b) -> scalar_t { return a + b; },         // scalar reduce
    [=](VectorizedM1<scalar_t> a, VectorizedM1<scalar_t> b)
        -> VectorizedM1<scalar_t> { return a + b; },                     // vector reduce
    /*identity=*/static_cast<scalar_t>(0)                                // initial value
);
```

### Reduce Helpers

```cpp
scalar_t result = vec_reduce_all(vec_binary_op, accumulated_vec, size);
scalar_t result = reduce_all(vec_binary_op, data_ptr, num_elements);
```

---

## 9. Header Include Guide

```cpp
// Required
#include "torch_sipu/csrc/aten/native/sipu/Vec.suh"        // Vectorized types

// By path
#include "torch_sipu/csrc/aten/native/sipu/Loops.suh"       // sipu_kernel (scalar)
#include "torch_sipu/csrc/aten/native/sipu/VecLoops.suh"    // sipu_kernel_vec (RVV)
#include "torch_sipu/csrc/aten/native/sipu/TileLoops.suh"   // sipu_kernel_tile/unified (Tile)
#include "torch_sipu/csrc/aten/native/sipu/Tile.suh"        // Tiled types

// Optional
#include "torch_sipu/csrc/aten/native/sipu/Reduce.suh"      // reduction ops
#include "torch_sipu/csrc/aten/native/sipu/aten_fallback.h"  // CPU fallback
#include "torch_sipu/csrc/aten/TensorIteratorBridge.h"       // type promotion bridge
```

---

## 10. Common Mistakes

### 10.1 Missing SIPU_LAMBDA

```cpp
// WRONG: no SIPU_LAMBDA
sipu_kernel_vec(iter, [=](float a) { ... }, [=](VectorizedM1<float> a) { ... });

// CORRECT
sipu_kernel_vec(iter,
    [=] SIPU_LAMBDA(float a) { ... },
    [=] SIPU_LAMBDA(VectorizedM1<float> a) { ... });
```

### 10.2 Scalar-Vector Type Mismatch

```cpp
// WRONG: scalar * vector
[=] SIPU_LAMBDA(VectorizedM2<float> a) { return a * alpha_val; }

// CORRECT: splat first
[=] SIPU_LAMBDA(VectorizedM2<float> a) {
    auto alpha_vec = VectorizedM2<float>::splat(&alpha_val);
    return a * alpha_vec;
}
```

### 10.3 BF16/FP16 Without Precision Promotion

```cpp
// WRONG: direct BF16 compute (severe precision loss)
[=] SIPU_LAMBDA(VectorizedM1<BFloat16> a) { return a.sigmoid(); }

// CORRECT: widen → compute → narrow
[=] SIPU_LAMBDA(VectorizedM1<BFloat16> a) -> VectorizedM1<BFloat16> {
    VectorizedM2<float> wide(a);
    wide = wide.sigmoid();
    return VectorizedM1<BFloat16>(wide);
}
```

### 10.4 Wrong Tile Size

```cpp
// float uses 512 (M2)
[=] SIPU_LAMBDA(vec::Tiled<float, 512> a) -> vec::Tiled<float, 512> { ... }

// BF16/Half also uses 512
[=] SIPU_LAMBDA(vec::Tiled<c10::BFloat16, 512> a) -> vec::Tiled<c10::BFloat16, 512> { ... }

// Or use aliases: vec::TiledM2<float>, vec::TiledM2<c10::BFloat16>
```

### 10.5 Using is*SupportedType Instead of isIterAll*

```cpp
// INSUFFICIENT: only checks common_dtype, but inputs may differ
if (vec::isRvSupportedType(iter.common_dtype())) { ... }

// CORRECT: checks ALL operand types
if (vec::isIterAllRvvSupportedTypes(iter)) { ... }
```

---

## 11. Quick Reference Tables

### Entry Function Quick Reference

| Need | Function |
|------|----------|
| Unary, best perf | `sipu_kernel_tile(iter, s, v, t)` |
| Unary, RVV only | `sipu_kernel_vec(iter, s, v)` |
| Unary, fallback | `sipu_kernel(iter, s)` |
| Binary, auto-select | `sipu_kernel_unified(iter, s, v, t)` |
| Binary + scalar, symmetric | `opmath_symmetric_..._with_scalars_unified(iter, s, v, t)` |
| Binary + scalar, non-symmetric | `sipu_kernel_with_scalars_unified(iter, s, v, t)` |
| Reduction | `binary_kernel_reduce_vec_sipu(iter, s, v, identity)` |

### Vectorized Type Quick Reference

| Data Type | Storage | Compute | Elements |
|-----------|---------|---------|----------|
| float | VectorizedM1 | VectorizedM2 | M1=32, M2=64 |
| BFloat16 | VectorizedM1 | VectorizedM2 (widened to float) | M1=64 |
| Half | VectorizedM1 | VectorizedM2 (widened to float) | M1=64 |
| int32 | VectorizedM1 | VectorizedM1 | M1=32 |
| int16 | VectorizedM1 | VectorizedM1 | M1=64 |
| int8/uint8 | VectorizedM1 | VectorizedM1 | M1=128 |
| bool | VectorizedM1 | VectorizedM1 | M1=128 |

### Lambda Parameter Type Quick Reference

| Entry Function | Scalar Lambda | Vec Lambda | Tile Lambda |
|----------------|---------------|------------|-------------|
| `sipu_kernel` | `(scalar_t) → scalar_t` | — | — |
| `sipu_kernel_vec` | `(scalar_t) → scalar_t` | `(VecM1<T>) → VecM1<T>` | — |
| `sipu_kernel_tile` | `(scalar_t) → scalar_t` | `(VecM1<T>) → VecM1<T>` | `(Tiled<T,512>) → Tiled<T,512>` |
| `*_with_scalars_rvv` | `(opmath_t, opmath_t) → opmath_t` | `(VecM2<T>, VecM2<T>) → VecM2<T>` | — |
| `*_with_scalars_tile` | `(opmath_t, opmath_t) → opmath_t` | `(VecM2<T>, VecM2<T>) → VecM2<T>` | `(TiledM2<T>, TiledM2<T>) → TiledM2<T>` |
