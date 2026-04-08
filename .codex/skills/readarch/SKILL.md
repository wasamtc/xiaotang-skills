---
name: readarch
description: >-
  Read SIPU architecture documentation (docs/tile_extension.md) and search
  SIPU SDK headers under /share_data/sicx_sdk/release/latest/ for low-level
  intrinsic functions, device APIs, runtime APIs, and hardware instructions.
  ONLY use when the user explicitly types /readarch in their message. Do NOT
  trigger automatically or for any other reason.
---

# /readarch — SIPU Architecture & SDK Intrinsic Lookup

## Trigger

This skill activates **only** when the user explicitly includes `/readarch` in their message. Never invoke this skill implicitly.

## Part 1: Architecture Reference

Read `docs/tile_extension.md` in the project root to understand the SIPU instruction set architecture.

## Part 2: SDK Header Lookup

## SDK Root

```
/share_data/sicx_sdk/release/latest/
```

## Directory Map

| Category | Path (relative to SDK root) | Contents |
|---|---|---|
| **Tile intrinsics** | `bin/nds64le-elf-newlib-v5d/lib/clang/19/include/siorigin_tile.h` | SIPU tile-extension builtins (`__builtin_siorigin_*`) |
| **Tile inter-core** | `bin/nds64le-elf-newlib-v5d/lib/clang/19/include/siorigin_tile_inter.h` | Inter-tile intrinsics |
| **Tile wrappers** | `bin/nds64le-elf-newlib-v5d/lib/clang/19/include/siorigin_wrappers/` | Derived types, tmv, tcvt, atomic, union helpers |
| **Device builtins** | `bin/nds64le-elf-newlib-v5d/lib/clang/19/include/__clang_siorigin_*.h` | Builtin vars, sync, device functions, runtime wrapper |
| **RVV intrinsics** | `bin/nds64le-elf-newlib-v5d/lib/clang/19/include/riscv_vector.h` | RISC-V Vector Extension intrinsics |
| **RISC-V extensions** | `bin/nds64le-elf-newlib-v5d/lib/clang/19/include/riscv_*.h` | bitmanip, crypto, ntlh |
| **Device APIs** | `include/sipu_dev_apis/` | High-level device-side API umbrella |
| **TDTE (tacp)** | `include/sipu_dev_apis/tdte/tacp.h` | Tile DTE / async copy |
| **Tsync** | `include/sipu_dev_apis/tsync/tsync.h` | Tile synchronization primitives |
| **Mbarrier** | `include/sipu_dev_apis/mbarrier/mbarrier.h` | Memory barrier APIs |
| **Device utils** | `include/sipu_dev_apis/utils/` | tcsr, timer, pointer, cvt_mode, print, libc, kill |
| **Cooperative groups** | `include/sipu_dev_apis/cooperative_groups.h` | Thread block cooperation |
| **Data types** | `include/sipu_dev_apis/dtype/union.h` | SIPU data type union |
| **Runtime API** | `include/sipurt/sipu_runtime_api.h` | sipuMalloc, sipuMemcpy, sipuLaunchKernel, etc. |
| **Runtime types** | `include/sipurt/` | builtin_types, vector_types, driver_types, fp16 |
| **Host API** | `include/sipu.h` | Top-level host-side include |
| **Driver API** | `include/api/sipu_driver_api.h` | Low-level driver functions |
| **Siformat** | `include/siformat/` | Soft-float, BF16, FP8, MX, SIMD, matrix format lib |
| **Hardware defs** | `include/cmodel_define.h` | PEC/PE/Core counts, TCSR addresses, HW constants |
| **MMIO registers** | `include/mmio/` | SOC address maps and register definitions |

## Workflow

When the user sends a message containing `/readarch`:

1. **Parse the query** — extract what function, instruction, or API they are looking for (e.g. "tacp", "vfadd", "sipuMalloc", "mbarrier", "tcsr", "tile intrinsic for multiply").

2. **Locate candidates** — use `Grep` or `Glob` to search under the SDK root:
   - For **compiler intrinsics / builtins**: search `bin/nds64le-elf-newlib-v5d/lib/clang/19/include/`
   - For **device-side APIs** (tdte, tsync, mbarrier, cooperative_groups, utils): search `include/sipu_dev_apis/`
   - For **runtime API** (sipuMalloc, sipuMemcpy, streams, events): search `include/sipurt/`
   - For **host API** (context, memory, kernel launch): search `include/sipu*.h`
   - For **driver API**: search `include/api/`
   - For **data formats** (bf16, fp8, softfloat, mx): search `include/siformat/`
   - For **hardware constants / registers**: search `include/cmodel_define.h` and `include/mmio/`
   - If unsure, search broadly: `rg -l "<pattern>" /share_data/sicx_sdk/release/latest/include/`

3. **Read the relevant header** — use `Read` to display the function signature, doc comment, and any associated type definitions.

4. **Present results** — show:
   - File path
   - Function/macro signature
   - Parameter descriptions (from comments if available)
   - Usage notes or constraints
   - Related functions in the same header if helpful

5. **If not found** — report which directories were searched and suggest the user check with `rg` manually or consult a different SDK version.

## Examples

**User**: `/readarch tacp`
→ Search `include/sipu_dev_apis/tdte/tacp.h`, read and present the TDTE async copy APIs.

**User**: `/readarch vfadd`
→ Search `bin/.../riscv_vector.h` and `siorigin_tile.h` for vector float add intrinsics.

**User**: `/readarch sipuMemcpy`
→ Search `include/sipurt/sipu_runtime_api.h` for the memcpy runtime function.

**User**: `/readarch bf16 convert`
→ Search `include/siformat/siformat_bf16.h` and `siorigin_wrappers/tcvt.h`.

**User**: `/readarch mbarrier`
→ Read `include/sipu_dev_apis/mbarrier/mbarrier.h`.

**User**: `/readarch TCSR_ADDR`
→ Search `include/cmodel_define.h` for TCSR address constants.
