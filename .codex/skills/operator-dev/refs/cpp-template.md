# C++ Kernel Implementation Templates

## Option A: Joint Compilation `.su` (Recommended for element-wise / reduction ops)

Create `torch_sipu/csrc/aten/native/sipu/<Op>.su`:

```cpp
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>          // stub declarations (use BinaryOps.h, CompareOps.h, ReduceOps.h for other op types)
#include <c10/core/ScalarType.h>
#include <sikernel.h>

#include "torch_sipu/csrc/aten/TensorIteratorBridge.h"              // TensorIteratorBridge::Guard
#include "torch_sipu/csrc/aten/native/sipu/Loops.suh"              // scalar loops
#include "torch_sipu/csrc/aten/native/sipu/VecLoops.suh"           // vectorized M1/M2 loops
#include "torch_sipu/csrc/aten/native/sipu/TileLoops.suh"          // tiled M2 loops
#include "torch_sipu/csrc/aten/native/sipu/Vec.suh"                // vector utilities
#include "torch_sipu/csrc/aten/native/sipu/Tile.suh"               // tile utilities
#include "torch_sipu/csrc/aten/native/sipu/aten_fallback.h"        // sipu_call_fallback_fn
#include "torch_sipu/csrc/aten/native/sipu/thread_constants.h"     // thread constants

namespace at::native {
namespace {

void <op>_kernel_sipu(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();

  // Path 1: TILE (highest performance, hardware tiled execution)
  if (vec::isIterAllTileSupportedTypes(iter)) {
    _AT_DISPATCH_TILE_TYPES(dtype, "<op>_sipu_tile", [&] {
      sipu_kernel_tile(
          iter,
          [=] SIPU_LAMBDA(scalar_t a) -> scalar_t { /* scalar fallback */ },
          [=] SIPU_LAMBDA(vec::VectorizedM1<scalar_t> a) -> vec::VectorizedM1<scalar_t> { /* vec */ },
          [=] SIPU_LAMBDA(vec::Tiled<scalar_t, 512> a) -> vec::Tiled<scalar_t, 512> { /* tile */ });
    });
  }
  // Path 2: RVV (RISC-V Vector, for types not supported by TILE)
  else if (vec::isIterAllRvvSupportedTypes(iter)) {
    AT_DISPATCH_ALL_TYPES(dtype, "<op>_sipu_rvv", [&] {
      sipu_kernel_vec(
          iter,
          [=] SIPU_LAMBDA(scalar_t a) -> scalar_t { /* scalar */ },
          [=] SIPU_LAMBDA(vec::VectorizedM1<scalar_t> a) -> vec::VectorizedM1<scalar_t> { /* vec */ });
    });
  }
  // Path 3: Scalar fallback
  else {
    AT_DISPATCH_ALL_TYPES(dtype, "<op>_sipu_scalar", [&] {
      sipu_kernel(iter, [=] SIPU_LAMBDA(scalar_t a) -> scalar_t { /* scalar only */ });
    });
  }
}

} // namespace

REGISTER_PRIVATEUSE1_DISPATCH(<op>_stub, &<op>_kernel_sipu)
} // namespace at::native
```

**Key `.su` rules:**
- Use `SIPU_LAMBDA` (not regular lambda) for device-side code.
- Always provide at least the scalar fallback path.
- Not all ops require all 3 dispatch paths. Some ops use only 2 paths (e.g., RVV + scalar without TILE). Match the existing pattern in similar ops.
- For reduced floating types (bf16/fp16), use `AT_DISPATCH_REDUCED_FLOATING_TYPES` with `opmath_type<scalar_t>` promotion.
- `.su` files are auto-discovered by CMake — no `CMakeLists.txt` changes needed.

## Option B: Structured Kernel `.su` (For complex ops with `parallel_for`)

Use this when the op uses `TORCH_SIPU_IMPL_FUNC` (structured kernel pattern) but needs joint compilation for device-side execution. This pattern is for ops that do **NOT** use `TensorIterator` — they manage data access and parallelism directly via `sipu::parallel_for` and custom device functions.

**When to use:** Softmax, triu/tril, or any op where you need custom data traversal patterns (not element-wise) with device-side execution.

**CRITICAL: Vectorization is mandatory.** The device kernel MUST use `VectorizedM1<scalar_t>` for the hot loop (vectorized path + scalar tail pattern). A scalar-only implementation is 2-10x slower and will be rejected in review. See the performance guide for details.

> **Note:** If your structured kernel op CAN use `TensorIterator` internally (like `add_out`), use Option A's loop infrastructure with `TORCH_SIPU_IMPL_FUNC` instead of `REGISTER_PRIVATEUSE1_DISPATCH`. The key difference is the entry point macro, not the loop machinery.

Create `torch_sipu/csrc/aten/native/sipu/<Op>.su`:

```cpp
#include <ATen/Dispatch.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <sikernel.h>

#include "torch_sipu/csrc/aten/SIPUNativeFunctions.h"
#include "torch_sipu/csrc/aten/native/sipu/Parallel.suh"       // sipu::parallel_for
#include "torch_sipu/csrc/aten/native/sipu/Vec.suh"            // vec::VectorizedM1<scalar_t>
// Optional — include if your kernel needs special math functions:
// #include "torch_sipu/csrc/aten/native/sipu/intrinsics/rvCore/MathUtils.suh"  // vec::fast_exp, etc.

namespace at::native {

namespace vec = at::vec::sipu;

namespace {

// Step 1: Device kernel — operates on a range of work items
// Mark C10_HOST_DEVICE so it compiles for both host and SIPU device.
template <typename scalar_t>
C10_HOST_DEVICE void <op>_device_kernel(
    const scalar_t* input_data,
    scalar_t* output_data,
    int64_t dim_size,
    int64_t begin,
    int64_t end) {
  using Vec = vec::VectorizedM1<scalar_t>;
  constexpr int vec_size = Vec::size();

  for (int64_t i = begin; i < end; ++i) {
    const scalar_t* in = input_data + i * dim_size;
    scalar_t* out = output_data + i * dim_size;

    // Vectorized path
    int64_t d = 0;
    if (dim_size >= vec_size) {
      for (d = 0; d <= dim_size - vec_size; d += vec_size) {
        Vec data = Vec::loadu(in + d);
        // ... computation using Vec operations ...
        data.store(out + d);
      }
    }
    // Scalar tail
    for (; d < dim_size; ++d) {
      out[d] = /* scalar computation on */ in[d];
    }
  }
}

// Step 2: Host dispatch — distributes work via parallel_for
template <typename scalar_t>
void <op>_host_dispatch(
    const scalar_t* input_data,
    scalar_t* output_data,
    int64_t outer_size,
    int64_t dim_size) {
  sipu::parallel_for(
      0,
      outer_size,
      1, // grain_size (0 = auto, 1 = per-row)
      [=] SIPU_LAMBDA(int64_t begin, int64_t end) {
        <op>_device_kernel<scalar_t>(
            input_data, output_data, dim_size, begin, end);
      });
}

// Step 3: Main implementation — validates inputs and dispatches by dtype
void <op>_impl(
    const at::Tensor& input,
    /* op-specific args */
    const Tensor& output) {
  if (input.numel() == 0) {
    return;
  }

  auto input_ = input.contiguous();

  // Calculate outer/inner/dim sizes as needed
  int64_t outer_size = /* ... */;
  int64_t dim_size = /* ... */;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input_.scalar_type(),
      "<op>_sipu",
      [&] {
        const scalar_t* in = input_.data_ptr<scalar_t>();
        scalar_t* out = output.data_ptr<scalar_t>();
        <op>_host_dispatch<scalar_t>(in, out, outer_size, dim_size);
      });
}

} // anonymous namespace

// Step 4: Structured kernel entry point
TORCH_SIPU_IMPL_FUNC(<op>_out)
(const at::Tensor& self,
 /* op-specific args matching the ATen schema */
 const at::Tensor& out) {
  <op>_impl(self, /* args, */ out);
}

} // namespace at::native
```

**Key differences from Option A:**
- Uses `TORCH_SIPU_IMPL_FUNC` (structured kernel) instead of `REGISTER_PRIVATEUSE1_DISPATCH` (dispatch stub).
- Uses `sipu::parallel_for` + `SIPU_LAMBDA` for explicit work distribution instead of `TensorIterator`-based loops.
- Device kernels are marked `C10_HOST_DEVICE` and use `vec::VectorizedM1<scalar_t>` directly (not `sipu_kernel_tile`/`sipu_kernel_vec`).
- Requires `Parallel.suh` and `Vec.suh` headers; does NOT need `Loops.suh`/`VecLoops.suh`/`TileLoops.suh`.
- For math functions (exp, log, etc.), include `intrinsics/rvCore/MathUtils.suh` which provides `vec::fast_exp()` and similar.

**Existing examples:**
- `TriangularOps.su` — triu/tril with `parallel_for` + row-level parallelism
- `SoftMax.su` — softmax with vectorized max/exp/normalize + `parallel_for`
- `Add.su` — uses `TORCH_SIPU_IMPL_FUNC` but with `TensorIterator` internally (hybrid of Option A + B)

**Register in `native_functions.yaml`** with explicit dispatch:
```yaml
- op: <op>.out
  dispatch:
    SIPU: <op>_out
```

## Option C: Host-only `.cpp` (For ops calling sikernel library)

Create `torch_sipu/csrc/aten/native/sipu/<Op>.cpp`:

```cpp
#include <ATen/core/Tensor.h>
#include <sikernel.h>
#include "torch_sipu/csrc/aten/SIPUNativeFunctions.h"

namespace at::native {
TORCH_SIPU_IMPL_FUNC(<op>_out)
(const at::Tensor& self, const at::Tensor& out) {
    auto input = self.contiguous();
    switch (input.scalar_type()) {
        case at::ScalarType::Float:
            <op>_f32(out.data_ptr(), input.data_ptr(), ...);
            break;
        case at::ScalarType::BFloat16:
            <op>_bf16(out.data_ptr(), input.data_ptr(), ...);
            break;
        default:
            TORCH_CHECK(0, "Unsupported dtype for <op>: ", input.scalar_type());
    }
}
} // namespace at::native
```

## Register in `native_functions.yaml`

```yaml
- <op>.out
```
Or with explicit dispatch:
```yaml
- op: <op>.out
  dispatch:
    SIPU: <op>_out
  use_native_impl: True
```
