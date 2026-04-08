---
name: pr-review
description: Structured code review checklist for torch_sipu changes. Use when reviewing code, a PR, a diff, or auditing operator implementation.
---

# torch_sipu PR Review Workflow

You are reviewing changes in the torch_sipu repository — a PyTorch device backend extension for SIPU hardware. You MUST evaluate every change against the checklist below, then output a structured review report.

---

## Procedure

1. **Read every changed file.** Do not review from summaries alone.
2. **Identify the change category** — new operator, operator modification, bug fix, refactor, infra/build change, or test-only change.
3. **Run through each checklist section.** Mark each item PASS, FAIL, WARN, or N/A.
4. **Output the review report** in the format specified at the end of this document.

---

## Checklist A: Correctness (Blocking)

Items in this section are merge-blockers. A FAIL here means the change must not be merged.

### A1. SIPU vs CPU Reference Test Exists

Every new or modified operator MUST have a test that does:

```python
torch.testing.assert_close(out_sipu.cpu(), out_cpu, rtol=..., atol=...)
```

**How to check:**
- Look in `test/test_*.py` for a test that creates input on CPU, copies to SIPU, runs the op on both, and asserts close.
- The reference must be computed on CPU, not on SIPU.
- `torch.testing.assert_close` must be used — not `torch.equal`, not `assertEqual` without tolerances for floating-point.

**Verdict:**
- PASS: Test exists with proper assert_close and tolerances.
- FAIL: No test, or test uses `torch.equal` / missing tolerances for float ops.
- N/A: Change does not touch operator behavior (e.g., build-only change).

### A2. Backward Gradient Coverage

> **Note:** torch_sipu is a **pure inference backend**. Standard compute ops should NOT register `AutogradPrivateUse1` and should NOT have backward tests. This check is N/A for most ops. Only apply it when an op explicitly registers `AutogradPrivateUse1` (rare — e.g., `to.dtype`, `type_as`, `_index_put_impl_`).

If the operator is differentiable and is registered with `AutogradPrivateUse1`, there must be a gradient test.

**How to check:**
- Look for a test that calls `.backward()` on both CPU and SIPU outputs with the same upstream gradient and compares `.grad` tensors.
- The SIPU tensor must be created via `x.clone().detach().to("sipu").requires_grad_(True)`.

**Verdict:**
- PASS: Gradient test exists and compares correctly.
- FAIL: Op registered with AutogradPrivateUse1 but no gradient test.
- WARN: Op is not differentiable — confirm `AutogradPrivateUse1` is intentionally omitted.
- N/A: Op does not register `AutogradPrivateUse1` (expected for inference backend ops).

### A3. Numerical Tolerance Appropriateness

**How to check against these baselines:**

| dtype | rtol | atol |
|---|---|---|
| `float32` | 1e-5 | 1e-5 |
| `bfloat16` | 1e-2 | 1e-2 |
| `float16` | 1e-3 | 1e-3 |
| `int32/int64` | 0 | 0 |

**Verdict:**
- PASS: Tolerances match or are tighter than baselines.
- WARN: Tolerances wider than baselines — check if a comment explains why.
- FAIL: No tolerances specified, or absurdly wide (e.g., `rtol=1` for float32).

### A4. Computation Correctness in Kernel

For Triton kernels in `torch_sipu/backends/sipu_triton_kernels/ops/`:

- Does the kernel promote `bfloat16`/`float16` to `float32` before computation and cast back before store? This is required to avoid precision loss.
- Does the kernel handle edge cases: zero-length inputs, single-element tensors?
- For C++ kernels in `torch_sipu/csrc/aten/native/sipu/`: does it call `.contiguous()` on input before taking `data_ptr()`?

**Verdict:**
- PASS: Precision promotion is correct and edges handled.
- FAIL: bf16/fp16 inputs computed directly without promotion to fp32.
- WARN: Edge cases not explicitly handled but may be acceptable.

### A5. Vectorized Path / Scalar Tail Consistency (Blocking)

For C++ kernels that use the **vectorized path + scalar tail** pattern (common in Option B ops with `VectorizedM1`), verify that **both paths compute identical results for the same operation**. Any divergence between the two paths is a correctness bug that produces different outputs depending on whether `dim_size` is a multiple of `vec_size`.

**How to check:**

For each operation that appears in both the vectorized loop and the scalar tail loop, compare:
1. **Same precision**: If the vectorized path computes in `scalar_t` (e.g., `Vec * inv_sum_vec`), the scalar tail must also compute in `scalar_t` (e.g., `output[d] *= inv_sum`), NOT in a different precision like `float32`.
2. **Same formula**: If the vectorized path uses `x * inv_sum`, the scalar tail must also use `x * inv_sum`, NOT `x / sum` (which may differ due to floating-point rounding).
3. **Same function**: If the vectorized path uses `fast_exp()`, the scalar tail must also use `fast_exp()`, NOT `std::exp()`.

**Common anti-pattern (real bug found in softmax):**
```cpp
// Vectorized path: multiply in scalar_t
scalar_t inv_sum = static_cast<scalar_t>(1.0f / sum);
Vec inv_sum_vec(inv_sum);
for (d = 0; d <= dim_size - vec_size; d += vec_size) {
    Vec out_vec = Vec::loadu(output_data + d);
    (out_vec * inv_sum_vec).store(output_data + d);  // scalar_t precision
}
// BUG: scalar tail uses float32 precision — different result!
for (; d < dim_size; ++d) {
    output_data[d] = static_cast<scalar_t>(
        static_cast<float>(output_data[d]) * (1.0f / sum));  // float32 precision
}
```

**Correct version:**
```cpp
// Scalar tail matches vectorized path exactly
for (; d < dim_size; ++d) {
    output_data[d] *= inv_sum;  // same scalar_t precision as vectorized path
}
```

**Verdict:**
- PASS: All operations in vectorized and scalar tail paths use identical computation and precision.
- FAIL: Any operation differs in precision, formula, or function between the two paths. This is a **blocking correctness issue**.

---

## Checklist B: Integration (Blocking)

### B1. Dispatcher Registration Complete

For Triton backend ops, check `torch_sipu/backends/sipu_triton_kernels/__init__.py`:
- Is the op registered via `_sipu_lib_aten.impl("<schema>", <func>, dispatch_key="PrivateUse1")`?
- Are all required overloads registered? Common overloads:
  - `"<op>"` — base
  - `"<op>.out"` — out variant
  - `"<op>.Tensor"` — tensor variant
  - `"<op>_"` or `"<op>_.Tensor"` — in-place variant

For C++ kernel ops, check `torch_sipu/csrc/aten/native/native_functions.yaml`:
- Is the op listed under `supported:`?
- Does the `dispatch: SIPU: <func>` mapping point to the correct function name?

For AI backend ops (if applicable), check `torch_sipu/backends/AI/__init__.py`.

**Verdict:**
- PASS: All required overloads registered, schema names match PyTorch's ATen registry.
- FAIL: Missing registration for an overload that the op uses (e.g., `.out` variant present but in-place `_` variant missing).
- WARN: Only a subset of overloads registered — verify if intentional.

### B2. Python API and Kernel Signature Consistency

- Does the Python wrapper function signature match the ATen schema it registers for?
- For Triton ops: does the `@sipu_verify` decorator have an entry in the `pytorch_map` inside `verify_decorator.py`? If not, runtime verification will silently skip reference comparison.
- For C++ ops: does the `TORCH_SIPU_IMPL_FUNC` signature match `native_functions.yaml`?

**Verdict:**
- PASS: Signatures are consistent.
- FAIL: Signature mismatch (e.g., extra/missing arguments compared to ATen schema).

### B3. Export Chain Complete

For Triton ops, verify the full chain:
1. Kernel function defined in `torch_sipu/backends/sipu_triton_kernels/ops/<op>.py`
2. Exported in `torch_sipu/backends/sipu_triton_kernels/ops/__init__.py`
3. Imported and registered in `torch_sipu/backends/sipu_triton_kernels/__init__.py`

A missing link at any point means the op will silently not be available.

**Verdict:**
- PASS: Full chain verified.
- FAIL: A link is missing.

### B4. No Dual Registration Conflict

Check that the same ATen schema is not registered by both:
- The Triton backend AND the C++ kernel backend simultaneously (unless the Triton backend intentionally overrides).
- Two different functions for the same dispatch key.

**Verdict:**
- PASS: No conflict.
- FAIL: Same schema registered twice with different implementations, no override mechanism.

### B5. CompositeImplicitAutograd Ops Not Over-Registered

Check whether any newly registered op (in `native_functions.yaml` or `__init__.py`) is a `CompositeImplicitAutograd` op in upstream PyTorch.

`CompositeImplicitAutograd` ops are automatically decomposed into primitive ATen ops by PyTorch's dispatcher. Explicitly registering them with a SIPU dispatch key **overrides** this decomposition and forces the project to maintain a dedicated kernel — which is unnecessary if all constituent primitive ops already have SIPU implementations.

**How to check:**
- Look up the op name in PyTorch's upstream `native_functions.yaml`.
- If it has `dispatch: CompositeImplicitAutograd` or no explicit dispatch key, it is composite.
- Common examples: `rms_norm`, `dropout`, `layer_norm`, `group_norm`.
- If a composite op is being registered in `native_functions.yaml` without an explicit `dispatch: SIPU: ...` mapping (bare listing like `- rms_norm`), it will still trigger codegen to generate a SIPU dispatch entry, overriding the composite decomposition.

**Verdict:**
- PASS: No composite ops are unnecessarily registered; or the registration is intentional for performance with a comment explaining why a custom kernel is preferred over decomposition.
- FAIL: A `CompositeImplicitAutograd` op is registered without justification, creating unnecessary maintenance burden.
- N/A: No new op registrations in this change.

---

## Checklist C: Edge Case Coverage (Non-blocking but Important)

### C1. Dtype Coverage

**Required minimum:** `torch.float32` and `torch.bfloat16`.
**Recommended additions:** `torch.float16`, `torch.int32`, `torch.int64` (where applicable).

**Verdict:**
- PASS: At least float32 + bfloat16 tested.
- WARN: Only one dtype tested.
- FAIL: No dtype variation in tests.

### C2. Shape Variation

Tests should exercise:
- Typical shapes (e.g., `(128, 256)`)
- Non-power-of-2 shapes (e.g., `(7, 13)`)
- Single-row / single-column (e.g., `(1, N)`, `(N, 1)`)
- Higher-rank tensors (e.g., 3D, 4D) if the op supports them.

**Verdict:**
- PASS: Multiple shape variations tested.
- WARN: Only one shape tested.

### C3. Non-Contiguous Input

Does the test exercise non-contiguous tensors (e.g., via `.t()`, `.permute()`, slicing)?

Many kernels assume contiguous input. If the wrapper calls `.contiguous()`, test that non-contiguous input still produces correct results (the wrapper should handle it).

**Verdict:**
- PASS: Non-contiguous input tested.
- WARN: Not tested — check if kernel calls `.contiguous()` internally.

### C4. Broadcasting (Binary Ops)

For binary ops, does the test cover:
- `(M, N) op (1, N)` — row broadcast
- `(M, N) op scalar` — scalar broadcast
- `(M, N) op (M, 1)` — column broadcast

**Verdict:**
- PASS: Broadcasting tested.
- N/A: Unary op.

### C5. Empty Tensor

Does the test exercise `torch.randn(0, N)` or similar zero-element inputs?

**Verdict:**
- PASS: Empty tensor tested.
- WARN: Not tested — may crash at runtime.

---

## Checklist D: Performance (Non-blocking)

### D1. Unnecessary Device-to-Device Copies

Scan for patterns that copy data to CPU and back:

```python
# ANTI-PATTERN: round-trip through CPU
tensor = tensor.cpu().reshape(...).to("sipu")
out = out.to("cpu").to(type).to("sipu").view(shape)
```

These patterns serialize execution and destroy performance. The reshape/view should be done on-device when possible.

**How to check:**
- Search for `.cpu()` followed by `.to("sipu")` or `.to(DEVICE)` in the same function.
- Search for `.to("cpu").to(...)` chains.

**Verdict:**
- PASS: No unnecessary round-trips.
- WARN: Round-trip exists but may be intentional (e.g., known SIPU limitation for reshape). Must have a comment explaining why.
- FAIL: Gratuitous CPU round-trip with no justification.

### D2. Unnecessary `.contiguous()` Calls

`convert_to_contiguous_and_aligned` in `ops/utils.py` already calls `.contiguous()`. If the wrapper also calls `.contiguous()` before passing to this utility, the data is copied twice.

**Verdict:**
- PASS: No redundant contiguous calls.
- WARN: Double `.contiguous()` detected.

### D3. Memory Allocation in Hot Path

Check for allocations that could be hoisted:
- `torch.empty_like()` or `torch.zeros_like()` inside a loop.
- Temporary tensors created per kernel launch that could be pre-allocated.

**Verdict:**
- PASS: No unnecessary allocations.
- WARN: Allocation in a suspected hot path.

### D4. Memory Leak Risk

For C++ code, check:
- Are raw `data_ptr()` pointers used without corresponding lifetime management?
- Is `at::Tensor` used to hold all device memory (correct), or are manual allocations (`malloc`/`new`) used (risky)?
- Are there SIPU-side resources (streams, events) that need cleanup?

For Python code, check:
- Are tensors captured in closures or global state that could prevent garbage collection?
- Are there circular references involving SIPU tensors?

**Verdict:**
- PASS: No leak risk identified.
- WARN: Potential risk — flag for author to verify.
- FAIL: Obvious leak (e.g., raw allocation without deallocation in error path).

### D5. Kernel Launch Overhead

For Triton ops:
- Is `grid` computed correctly? Over-launching (grid much larger than needed) wastes resources.
- Is `BLOCK_SIZE` reasonable? Too small = excessive launches. Too large = wasted memory.

**Verdict:**
- PASS: Grid and block size are reasonable.
- WARN: Potential inefficiency.

### D5a. C++ Kernel Vectorization Level (Blocking for New/Refactored Kernels)

For C++ kernel code (`.su` files), check whether the implementation uses the appropriate vectorization level. A scalar-only kernel leaves 2-100x performance on the table.

**How to check:**

1. **Option A ops (DispatchStub / TensorIterator)**: Does the kernel implement the multi-path cascade?
   - **Full cascade**: `sipu_kernel_tile` (Tile) → `sipu_kernel_vec` (RVV/VectorizedM1) → `sipu_kernel` (scalar fallback)
   - **Minimum acceptable**: `sipu_kernel_vec` (RVV) + `sipu_kernel` (scalar fallback)
   - **FAIL**: Only `sipu_kernel()` (scalar) when the dtypes support Tile/RVV paths

2. **Option B ops (parallel_for + custom kernel)**: Does the device kernel use `VectorizedM1<scalar_t>` for the hot loop?
   - Look for the **vectorized path + scalar tail** pattern:
     ```cpp
     int64_t d = 0;
     for (d = 0; d <= size - vec_size; d += vec_size) {
         Vec data = Vec::loadu(ptr + d);
         // ... vectorized computation ...
         data.store(out + d);
     }
     for (; d < size; ++d) { /* scalar tail */ }
     ```
   - For math functions not in VectorizedM1 (e.g., `exp`), the **store-compute-reload** pattern is acceptable:
     ```cpp
     scalar_t temp[vec_size];
     data_vec.store(temp);
     for (int j = 0; j < vec_size; ++j) { temp[j] = fast_exp(temp[j]); }
     Vec::loadu(temp).store(out + d);
     ```
   - **FAIL**: Hot loop is entirely scalar when `VectorizedM1` operations exist for the needed computation

3. **VectorizedM1 available methods** (check if vectorization is possible):
   `loadu()`, `store()`, `neg()`, `sigmoid()`, `rsqrt()`, `reciprocal()`, `sin()`, `cos()`, `silu()`, `pow()`, `maximum()`, `minimum()`, `operator+`, `operator-`, `operator*`, `operator/`

**Performance impact reference:**

| Implementation | Relative Speed |
|---|---|
| Scalar-only | 1x (baseline) |
| VectorizedM1 (RVV) | 2-10x |
| Tiled (TileCore) | 10-100x |

**Verdict:**
- PASS: Kernel uses vectorization appropriate for the op type (Tile/RVV cascade for Option A, VectorizedM1 for Option B).
- FAIL: Scalar-only implementation when vectorization is feasible. This is a **blocking issue** for new or refactored kernels.
- WARN: Vectorization exists but could be improved (e.g., only RVV when Tile is possible for supported dtypes).
- N/A: No C++ kernel changes, or the kernel operates on non-contiguous strided data where vectorization is not feasible.

### D6. C++ Parameter Passing (Value vs Reference)

For C++ code (`.cpp`, `.h`, `.su`, `.suh`), check function signatures for unnecessary copies of large objects:

**Anti-patterns to flag:**
```cpp
// BAD: copies the entire string/vector/tensor
void foo(std::string handle);
void bar(std::vector<int64_t> sizes);
void baz(at::Tensor input);              // unless ownership transfer is intended

// GOOD: const reference for read-only access
void foo(const std::string& handle);
void bar(at::IntArrayRef sizes);          // PyTorch's lightweight array view
void baz(const at::Tensor& input);
```

**Types that should almost always be passed by `const &`:**
- `at::Tensor` / `c10::TensorBase`
- `std::string`
- `std::vector<T>`
- `c10::optional<at::Tensor>`
- `at::TensorList`

**Types that are fine to pass by value** (small/trivial):
- `int64_t`, `size_t`, `bool`, `double`, `c10::DeviceIndex`
- `c10::ScalarType`, `c10::DeviceType`
- `at::IntArrayRef` (already a lightweight view)
- `c10::optional<ScalarType>` (small trivial type)
- Raw pointers (`void*`, `scalar_t*`)

**Verdict:**
- PASS: Large objects passed by const reference or moved appropriately.
- WARN: Large object passed by value — may be intentional for ownership transfer, but needs justification.
- N/A: No C++ changes in this diff.

### D7. Lambda Capture and Passing

Check lambda captures and parameter passing in C++ code. There are **three aspects** to review: capture mode, captured content, and how the lambda is passed through call chains.

#### D7a. Capture Mode — Device vs Host Context

Apply **different rules** based on execution context:

**Device kernels (`SIPU_LAMBDA` / `__host__ __device__`):**
- MUST use `[=]` (capture by value) — device code cannot access host stack references. Using `[&]` in a device kernel **will crash at runtime**.
- This is the reason all `parallel_for` / `run_task_kernel` lambdas in the codebase use `[=]`.

**Host-side code (e.g., `AT_DISPATCH_*` macros, `AT_WRAP` callbacks):**
- Should use `[&]` to avoid unnecessary copies.
- `AT_DISPATCH_*` and `AT_WRAP` macros expand to immediately invoke the lambda in-place — no copy occurs, so `[&]` is correct and efficient.
- For lambdas stored beyond the current scope (e.g., callbacks, `std::function`), value capture or `shared_ptr` is required.
- Verify that captured references outlive the lambda's execution (no dangling references).

**Nested pattern (common in torch_sipu):**
```cpp
// CORRECT: outer HOST lambda uses [&], inner DEVICE lambda uses [=]
AT_DISPATCH_FLOATING_TYPES(..., [&] {            // HOST: [&] OK
    const scalar_t* input_data = input_.data_ptr<scalar_t>();
    sipu::parallel_for(0, N, 1,
        [=] SIPU_LAMBDA(int64_t begin, int64_t end) {  // DEVICE: [=] required
            // use input_data (pointer, captured by value)
        });
});
```

**How to check:**
- Search for `[&]` near `SIPU_LAMBDA`, `<<<grid, block>>>` — will crash at runtime.
- Search for `[=]` in non-kernel host-only contexts — unnecessary copy but not incorrect.

#### D7b. Captured Content — Lightweight Requirement for Device Lambdas

Device lambdas are copied from host to device via kernel parameters. Captured content **must be lightweight**:

**Safe to capture by value in device lambda** (trivially copyable, small):
- Raw pointers (`scalar_t*`, `void*`, `char*`) — 8 bytes
- Scalars (`int64_t`, `double`, `bool`, `dim3`) — 4-8 bytes
- Small structs/functors (e.g., `BinaryFunctor` with a few scalars)

**MUST NOT capture by value in device lambda** (large or non-trivial):
- `at::Tensor` — contains refcount, metadata, allocator pointer; copying to device is incorrect
- `std::string`, `std::vector<T>` — heap-allocated, pointer invalid on device
- `std::shared_ptr`, `std::unique_ptr` — ownership semantics break across host/device

**Correct pattern**: extract raw pointers/scalars on the host side, then capture only those:
```cpp
// GOOD: capture pointers and scalars, not Tensor objects
const scalar_t* input_data = input_.data_ptr<scalar_t>();
scalar_t* output_data = output.data_ptr<scalar_t>();
int64_t dim_size = input_.size(dim_);
sipu::parallel_for(0, N, 1,
    [=] SIPU_LAMBDA(int64_t begin, int64_t end) {
        // use input_data, output_data, dim_size — all lightweight
    });

// BAD: capturing Tensor by value into device lambda
sipu::parallel_for(0, N, 1,
    [=, input_] SIPU_LAMBDA(int64_t begin, int64_t end) {
        // input_ is an at::Tensor — BAD
    });
```

#### D7c. Lambda Passing Through Call Chains — Value vs Forward

When a lambda is passed through helper function call chains, check whether unnecessary copies accumulate.

**torch_sipu's `parallel_for` → `invoke_parallel` → `run_task_kernel` chain:**

```
parallel_for(F f)          — copy 1: caller → f parameter
  invoke_parallel(func_t f)  — copy 2: f → f parameter
    [=] captures f             — copy 3: f → device lambda member
      <<<grid, block>>>(lambda)  — copy 4: host → device (unavoidable)
```

This creates up to **4 copies** of the lambda object. In theory, `parallel_for` and `invoke_parallel` could use `F&&` + `std::forward<F>(f)` to eliminate copies 1-2. However:

**When this matters:**
- The lambda captures large objects (vectors, strings, non-trivial functors)
- The function chain is deep (3+ levels of template forwarding)

**When this does NOT matter (current codebase):**
- Lambda captures only pointers and scalars (~32-40 bytes total)
- `run_task_kernel(__global__)` **must** take the functor by value (kernel parameter, host→device copy is unavoidable)
- The device lambda `[=]` capture is also unavoidable (cannot use `[&]` in device code)
- Copying 32 bytes 4 times is negligible vs kernel launch overhead

**What to flag:**
- If a lambda capturing large objects (>128 bytes, or non-trivially-copyable types) is passed by value through 2+ function template layers, WARN about missing perfect forwarding.
- If the receiving function stores the lambda (not just forwards it), `std::move` should be used at the storage point.
- Do NOT flag the standard `parallel_for` → `invoke_parallel` → `run_task_kernel` chain for lambdas that only capture pointers and scalars.

**Verdict:**
- PASS: Capture mode matches context; captured content is lightweight for device; lambda passing is appropriate for the captured data size.
- FAIL: `[&]` used in device kernel; large object (Tensor/string/vector) captured by value in device lambda.
- WARN: `[=]` used in host-only context; or lambda with large captures (>128 bytes) passed by value through multiple template layers without forwarding.

### D8. Move Semantics

Check that `std::move` is used correctly and not missing at key points:

**Where `std::move` should be used:**
```cpp
// Constructor initializer lists with non-trivial members
MyClass(std::string name, std::function<void()> cb)
    : name_(std::move(name)), callback_(std::move(cb)) {}

// Passing to emplace_back / push_back
observers_.emplace_back(std::move(observer));

// Transferring Tensor ownership in update/exchange patterns
update_operand_tensor_info(op, std::move(reshaped_tensor));
```

**Where `std::move` should NOT be used:**
```cpp
// Don't move from const references
void foo(const Tensor& t) { bar(std::move(t)); }  // BAD: move from const is a copy

// Don't move return values (defeats NRVO)
Tensor make() { Tensor t = ...; return std::move(t); }  // BAD: prevents copy elision

// Don't use after move
auto x = std::move(y);
y.size();  // BAD: use-after-move
```

**Verdict:**
- PASS: Move semantics applied correctly at ownership transfer points.
- WARN: Missing `std::move` where ownership is clearly being transferred (e.g., constructors storing parameters).
- FAIL: `std::move` from const reference, use-after-move, or `std::move` on return value preventing NRVO.

### D9. Unnecessary Tensor Copies

For C++ code, check for Tensor copy operations that could be avoided:

**Anti-patterns:**
```cpp
// BAD: unnecessary copy when only reading
at::Tensor input = self;           // copies refcount, but still worth avoiding in hot path
at::Tensor input = self.clone();   // full data copy — only needed if modifying

// GOOD: const reference
const at::Tensor& input = self;

// BAD: double contiguous
auto a = self.contiguous();
auto b = a.contiguous();          // redundant — already contiguous

// BAD: contiguous then clone
auto t = self.contiguous().clone();  // unnecessary if contiguous() already made a copy
```

**Legitimate Tensor copy patterns (do NOT flag):**
```cpp
// contiguous() creates a copy only when needed — this is correct
auto input_ = input.contiguous();

// clone() + resize() when creating a new owned tensor from a view
Tensor narrow_tensor = at::narrow(output, 1, 0, N).clone();
narrow_tensor.resize_(original_shape);
```

**Verdict:**
- PASS: No unnecessary Tensor copies.
- WARN: Potential redundant copy — author should verify if copy is needed.

---

## Checklist E: Code Hygiene (Non-blocking)

### E1. Minimal Change Principle

- Does the diff contain only changes necessary for the stated task?
- Are there formatting-only changes to unmodified lines?
- Are there drive-by refactors, comment additions, or import reordering in unrelated code?

**Verdict:**
- PASS: Changes are focused.
- WARN: Minor unrelated changes present.
- FAIL: Significant unrelated refactoring mixed in.

### E2. Debug Artifacts Removed

- Are there leftover `print()` statements? (Note: `[TRITON INFO]` prints in some existing ops are pre-existing; do not flag those unless the change adds new ones.)
- Are there commented-out code blocks that should be removed?
- Are there hardcoded file paths or test-only hacks?

**Verdict:**
- PASS: Clean.
- WARN: Debug prints or commented code present.

### E3. License Header on New Files

Every **new** source file in the diff must have the Apache v2.0 license header:
```
Copyright (c) <YEAR> SiOrigin Co. Ltd.
SPDX-License-Identifier: Apache-2.0
```

**How to check:**
- Look for newly created files in the diff (mode `100644` new file).
- Verify the header exists at the top with correct comment syntax for the language.
- `<YEAR>` should be the file's creation year.
- Files originating from other open-source projects should keep their original headers — do NOT flag missing SiOrigin header on those.

**Verdict:**
- PASS: All new files have the Apache v2.0 header.
- FAIL: New SiOrigin-authored file missing the license header.
- N/A: No new files in this diff.

### E4. Commit Message Format

Per `development_requirements.md`, commits should follow:
```
<type>(<scope>): <description> [jira#S1SW-XXXX]
```

Types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`.

**Verdict:**
- PASS: Follows convention.
- WARN: Minor deviation.

---

## Output Format

After completing the checklist, output the review report in this exact structure:

```markdown
## PR Review: <brief description of the change>

### Summary
<1-3 sentences describing what the change does>

### Verdict: APPROVE / REQUEST CHANGES / COMMENT

### Blocking Issues
<List items with FAIL verdict from Sections A and B, or "None">
- [ ] **A1 FAIL**: <description>
- [ ] **B1 FAIL**: <description>

### Warnings
<List items with WARN verdict, or "None">
- **C3 WARN**: <description>
- **D1 WARN**: <description>

### Passed
<Summarize passed items by section>
- Correctness: A1 ✓, A2 N/A, A3 ✓, A4 ✓, A5 ✓
- Integration: B1 ✓, B2 ✓, B3 ✓, B4 ✓, B5 ✓
- Edge cases: C1 ✓, C2 WARN, C3 ✓, C4 N/A, C5 WARN
- Performance: D1 ✓, D2 ✓, D3 ✓, D4 ✓, D5 ✓, D5a ✓, D6 ✓, D7 ✓, D8 ✓, D9 ✓
- Hygiene: E1 ✓, E2 ✓, E3 ✓, E4 ✓

### Suggested Actions
<Numbered list of concrete actions for the author>
1. Add bfloat16 test case in test/test_xxx.py.
2. Register the `.out` variant in __init__.py line 195.
3. ...

### Risk Assessment
- **Regression risk**: <Low/Medium/High> — <why>
- **Dtype gaps**: <which dtypes are NOT covered>
- **Known limitations**: <any shape/dim restrictions>
```

### Verdict Decision Rules

- **REQUEST CHANGES**: Any FAIL in Section A (Correctness), Section B (Integration), or D5a (Vectorization for new/refactored kernels).
- **COMMENT**: No FAILs, but multiple WARNs that warrant discussion.
- **APPROVE**: No FAILs and at most minor WARNs.
