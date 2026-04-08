# torch_sipu — Project Context

torch_sipu is a PyTorch device backend extension for SIPU hardware. It registers as `PrivateUse1`, enabling `tensor.to("sipu")` like CUDA uses `tensor.to("cuda")`.

## Key Facts

- **C++ standard**: C++20, GCC >= 9.3
- **PyTorch version**: >= 2.7.1, < 2.7.2  <!-- Hard constraint: ABI compatibility with SIPU SDK build -->
- **Dual backend**: Triton (default) and AI (`TORCH_SIPU_USE_AI_BACKEND=1`)
- **Device code**: Joint compilation `.su` files — host (`-O0`/`-O3` per build type) and device (always `-O3`)
- **Git workflow**: Branch from `develop`, branch name `<username>/<feature>`, squash commits, MR with CI labels

## Repository Layout

```
torch_sipu/
├── torch_sipu/                     # Main Python package
│   ├── __init__.py                 # Device registration, autoload entry
│   ├── csrc/                       # C++ source
│   │   ├── aten/native/sipu/       #   ATen op implementations (.cpp, .su, .suh)
│   │   │   ├── native_functions.yaml   # C++ dispatch registration
│   │   │   └── ext_native_functions.yaml
│   │   ├── aten/sipu/              #   Device runtime (context, allocators, events)
│   │   └── c10/                    #   C10 extensions and header interceptions
│   ├── backends/
│   │   ├── sipu_triton_kernels/    # Triton backend (DEFAULT)
│   │   │   ├── __init__.py         #   Dispatcher registration (_prefer_triton_kernels)
│   │   │   └── ops/                #   One Python file per op
│   │   └── AI/                     # AI backend (opt-in)
│   ├── sipu/                       # torch.sipu module (streams, memory, tensor)
│   ├── _inductor/                  # TorchInductor backend
│   ├── _dynamo/                    # TorchDynamo support
│   ├── distributed/                # c10d backend
│   └── testing/_internal/          # Test utilities (triton_utils, common_utils)
├── test/                           # Unit tests (pytest)
├── examples/                       # Runnable demos
├── third_party/                    # sikernel submodule
├── codegen/                        # Dispatch stub generation
├── CMakeLists.txt                  # Main CMake (C++20, SIPU language)
├── Makefile                        # Dev targets (install-dev, lint, clean)
├── setup.py                        # Python build entry
├── run_test.sh                     # Test runner (-k kernel, -t triton, -p pytorch, etc.)
├── development_requirements.md     # Parameter naming, MR spec, C++ conventions
└── CODING_GUIDELINES.md            # Code style, lint setup
```

## Dispatch Architecture

Operators register at three layers; Triton backend can override C++ kernel:

| Layer | Registration | Dispatch key |
|---|---|---|
| C++ kernel | `native_functions.yaml` + `TORCH_SIPU_IMPL_FUNC` | `PrivateUse1` |
| Triton backend | `sipu_triton_kernels/__init__.py` + `torch.library` | `PrivateUse1` (overrides C++) |
| AI backend | `AI/__init__.py` + `torch.library` | `PrivateUse1` (overrides C++) |

`CompositeImplicitAutograd` ops auto-decompose — do NOT register them unless a custom kernel is needed.

## Build Commands

```bash
make install-all-dev      # Build kernels + extension (debug)
make install-dev           # Extension only (skip kernel rebuild)
make install-all-release   # Release build
make clean-all             # Clean everything
```

## Test Commands

```bash
./run_test.sh -k           # C++ kernel tests
./run_test.sh -t           # Triton tests
./run_test.sh -p           # Upstream PyTorch tests
./run_test.sh -l           # Lint only

# Single file
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=sipu pytest test/test_xxx.py -v

# With runtime verification
TRITON_KERNEL_VERIFY=1 python examples/run_xxx.py
```

## Lint

```bash
make setup-lint            # First time setup
make lint                  # Check all files
lintrunner -a --all-files  # Auto-fix
```

## Commit Message Format

```
<type>(<scope>): jira#S1SW-XXXX <description>
```

**`jira#` goes after the colon, before the description — never at the end.**

Types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`

Examples:
```
feat(aten): jira#S1SW-1311 integrate runtime apis
fix(ops): jira#S1SW-2046 add to_tiled_o to derive output tile shape
refactor(aten): jira#S1SW-1901 refactor triu_and_tril op using joint compilation
chore(docker): jira#S1SW-2060 replace ubuntu apt sources with aliyun mirror
```

For hotfixes and simple updates (typos, docs), Jira number is optional.

## License Header (MANDATORY for New Files)

All **new** source files created in this project MUST include the Apache v2.0 license header. Use the file's creation year (not the current year).

**C++/C/SU/SUH** (`.cpp`, `.h`, `.su`, `.suh`):
```cpp
// Copyright (c) <YEAR> SiOrigin Co. Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```

**Python/Shell/YAML** (`.py`, `.sh`, `.yaml`):
```python
# Copyright (c) <YEAR> SiOrigin Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

**Rules:**
- `<YEAR>` = the year the file was **created** (e.g., 2024, 2025, 2026)
- Do NOT add SiOrigin header to files originating from other open-source projects — keep their original MIT/BSD/etc. headers
- When backfilling existing files, use the original creation year

## Developer Setup (For Claude Code Skills)

The skills (`operator-dev`, `dev-workflow`, etc.) automate JIRA ticket creation and GitLab MR creation. Each developer needs to configure two files:

### 1. JIRA — `~/.jira.cfg`

Use Personal Access Token (do NOT use plaintext password basic_auth):
```ini
[jira]
url = https://jiraoffice.siorigin.com
token = <YOUR_PERSONAL_ACCESS_TOKEN>
project = S1SW
```
Generate token at: JIRA → Profile → Personal Access Tokens. Set file permission: `chmod 600 ~/.jira.cfg`.

**Install:** `pip install jira`

### 2. GitLab — `~/.python-gitlab.cfg`

```ini
[global]
default = siorigin
timeout = 30

[siorigin]
url = https://gitlabsoft.siorigin.com
private_token = <YOUR_GITLAB_PERSONAL_ACCESS_TOKEN>
api_version = 4
```
Generate token at: GitLab → Preferences → Access Tokens (scope: `api`). Set file permission: `chmod 600 ~/.python-gitlab.cfg`.

**Install:** `pip install python-gitlab`

### 3. Git Identity

```bash
git config user.name "<your_name>"
git config user.email "<your_email>@siorigin.com"
```

### 4. OpenSSL SSH Workaround (If Needed)

If `git push` fails with `OpenSSL version mismatch`, set `GIT_SSH_COMMAND` to a compatible SSH binary:
```bash
GIT_SSH_COMMAND="<path-to-compatible-ssh>" git push -u origin <branch>
```

## Key Infrastructure Files (Shared by Many Ops)

- `Reduce.suh` — reduction utilities (vectorized_reduction, binary_kernel_reduce_*)
- `Parallel.suh` — parallel execution (parallel_for, invoke_parallel)
- `Loops.suh` / `VecLoops.suh` — element-wise loop infrastructure
- `ops/utils.py` — convert_to_contiguous_and_aligned, remove_padding
- `ops/verify_decorator.py` — @sipu_verify runtime verification
