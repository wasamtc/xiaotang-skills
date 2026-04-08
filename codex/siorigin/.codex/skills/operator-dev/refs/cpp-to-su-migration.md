# Refactoring from `.cpp` to `.su` (Joint Compilation)

When the task is to refactor an existing Option C (host-only `.cpp` calling sikernel) to Option A or B (`.su` with inline device code), follow this migration workflow.

## Pre-flight Checks

1. **CompositeImplicitAutograd check** (Step 1.0): Verify the op is NOT `CompositeImplicitAutograd`. If it is, the correct action is to **remove** the registration from `native_functions.yaml` and delete the `.cpp` file — not refactor to `.su`. Exception: performance-critical composite ops (like `rms_norm`, `layer_norm`) may justify keeping a dedicated kernel for fusion benefits (single kernel launch vs multiple decomposed launches).

2. **Check for Triton backend**: If the op also has a Triton backend implementation in `sipu_triton_kernels/ops/`, the Triton kernel overrides the C++ kernel when active. The refactored `.su` serves as fallback when Triton is disabled. Test the C++ path with `@skipIfUseSipuTritonKernels()`.

3. **Identify the target pattern**:
   - If the op uses `TensorIterator` in upstream PyTorch → **Option A** (DispatchStub `.su`)
   - If the op needs custom data traversal (softmax, triu/tril) → **Option B** (Structured kernel `.su`)

## Migration Steps

1. **Analyze the existing `.cpp`** — identify:
   - Which sikernel functions it calls (e.g., `::rms_norm<opsipu_t>(...)`, `::softmax_f32(...)`)
   - What data preparation it does (padding, reshaping, contiguous)
   - Which dtypes it supports
   - The entry point macro (usually `TORCH_SIPU_IMPL_FUNC`)

2. **Create the new `.su` file** using Option A or B template:
   - Replace sikernel calls with inline device kernels using `VectorizedM1`, `parallel_for`, etc.
   - Remove padding hacks — vectorized loops with scalar tails handle variable sizes naturally
   - Keep the same entry point macro (`TORCH_SIPU_IMPL_FUNC`)
   - Consider adding dtype support (e.g., add `float16` if the old `.cpp` only supported `float32` + `bfloat16`)

3. **Update `native_functions.yaml`** — change bare listing to explicit dispatch:
   ```yaml
   # Before (bare listing):
   - _softmax.out

   # After (explicit dispatch):
   - op: _softmax.out
     dispatch:
       SIPU: _softmax_out
   ```

4. **Delete the old `.cpp` file** — the `.su` file replaces it entirely. CMake auto-discovers `.su` files, no build changes needed.

5. **Check C++ vs Triton capability parity** — if the Triton backend supports more cases than the C++ kernel (e.g., Triton softmax supports all dims but C++ only supports last-dim), ensure existing tests won't fail when Triton is disabled. Options:
   - Add `@skipIfUseSipuTritonKernels()` to test cases that exercise functionality only available in the Triton backend.
   - Or make the C++ kernel match Triton's capability coverage.
   Flag any parity gaps in the Step 5 risk assessment.

6. **Run tests** — verify both C++ and Triton paths (if applicable):
   ```bash
   # C++ kernel path (Triton disabled)
   CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=sipu pytest test/test_sipu.py -v -k "test_<op>"
   # Triton path (if Triton backend exists)
   CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=sipu PYTORCH_TEST_WITH_SIPU_TRITON_OPS=1 pytest test/test_sipu.py -v -k "test_<op>"
   ```

## Common `.cpp` → `.su` Mappings

| `.cpp` pattern (old) | `.su` replacement (new) |
|---|---|
| `::sikernel_func<opsipu_t>(ptr, ...)` | Inline device kernel with `parallel_for` + `VectorizedM1` |
| `AT_DISPATCH_SIPU_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, ...)` | `AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, ...)` |
| Manual padding: `padded_N = ((N + 511) / 512) * 512` | Vectorized loop with scalar tail (handles any size) |
| `at::narrow(padded, dim, 0, N).copy_(input)` | Direct `data_ptr` access on contiguous tensor |
| `update_operand_tensor_info(op, std::move(tensor))` | Direct computation — no `TensorIterator` manipulation needed |
| `sipu_call_fallback_fn<ATEN_OP(op), scalar_t>::call(...)` | Not needed if all required dtypes are handled inline |

## Refactoring Example: SoftMax

Before (`SoftMax.cpp` — 91 lines, Option C):
```cpp
TORCH_SIPU_IMPL_FUNC(_softmax_out)(...) {
    // ... AT_DISPATCH_SIPU_FLOATING_AND_COMPLEX_TYPES_AND2 ...
    ::softmax_f32(out_ptr, in_ptr, M, N);  // calls sikernel
}
```

After (`SoftMax.su` — Option B with vectorization):
```cpp
// Inline device kernel with vectorized max/exp/normalize
template <typename scalar_t>
C10_HOST_DEVICE void softmax_lastdim_kernel(..., int64_t begin, int64_t end) {
    using Vec = vec::VectorizedM1<scalar_t>;
    constexpr int vec_size = Vec::size();

    for (int64_t i = begin; i < end; ++i) {
        // Step 1: Vectorized max reduction
        Vec max_vec = Vec::loadu(in);
        for (d = vec_size; d <= dim_size - vec_size; d += vec_size)
            max_vec = max_vec.maximum(Vec::loadu(in + d));
        // reduce max_vec to scalar, then handle tail...

        // Step 2: Vectorized exp(x-max) + float32 sum accumulation
        float sum = 0.0f;  // NOTE: accumulate in float32 for precision!
        Vec max_bcast(max_input);
        for (d = 0; d <= dim_size - vec_size; d += vec_size) {
            Vec data = Vec::loadu(in + d);
            // store-compute-reload for exp (no Vec::exp available)
            scalar_t temp[vec_size];
            data.store(temp);
            for (int j = 0; j < vec_size; ++j) {
                temp[j] = static_cast<scalar_t>(fast_exp(double(temp[j] - max_input)));
                sum += static_cast<float>(temp[j]);  // float32 accumulation
            }
            Vec::loadu(temp).store(out + d);
        }
        // handle tail...

        // Step 3: Vectorized normalize
        scalar_t inv_sum = scalar_t(1) / scalar_t(sum);
        Vec inv_vec(inv_sum);
        for (d = 0; d <= dim_size - vec_size; d += vec_size)
            (Vec::loadu(out + d) * inv_vec).store(out + d);
        // handle tail...
    }
}
```

Key improvements: no sikernel dependency, no padding hacks, added `float16` support, **vectorized computation with VectorizedM1**, float32 accumulation for precision.
