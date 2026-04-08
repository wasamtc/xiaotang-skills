---
name: dev-workflow
description: Standard development workflow for torch_sipu — branching, coding standards, commit, lint, MR, and CI. Use when committing, pushing, creating MRs, or checking development process compliance.
---

# torch_sipu Development Workflow

You are working in the torch_sipu repository. When the user asks you to commit, push, create an MR, or check code against development standards, follow this workflow.

---

## Step 0: Create JIRA Ticket (If Needed)

Before starting development, check if a JIRA ticket exists for this task. If the user provides a ticket number (e.g., `S1SW-1952`), use it. If not, **ask the user** whether to create one.

**To create a JIRA ticket automatically:**

```python
import configparser, os
from jira import JIRA

cfg = configparser.ConfigParser()
cfg.read(os.path.expanduser('~/.jira.cfg'))

jira_url = cfg.get('jira', 'url')

# Use token-based auth (required — do NOT store plaintext passwords)
if not cfg.has_option('jira', 'token'):
    raise RuntimeError("JIRA token not found in ~/.jira.cfg. Add 'token = <YOUR_PERSONAL_ACCESS_TOKEN>' under [jira].")
j = JIRA(jira_url, token_auth=cfg.get('jira', 'token'))

issue = j.create_issue(
    project=cfg.get('jira', 'project', fallback='S1SW'),
    issuetype={'name': 'Task'},  # 'Task', 'Bug', 'Story'
    summary='<descriptive summary of the task>',
    # description='<optional detailed description>',
)
print(f"Created: {issue.key} — {jira_url}/browse/{issue.key}")
```

Replace `<descriptive summary>` with a meaningful title (e.g., "Implement abs operator for SIPU" or "Refactor CI test infrastructure").

For hotfixes and simple updates (typos, docs), Jira number is optional — skip this step if the user confirms no ticket is needed.

---

## Branch Guard (MANDATORY — Always Check First)

**Before any development work**, check the current branch:

```bash
git branch --show-current
```

- If already on a feature branch (`<username>/<feature>`): proceed — no action needed.
- If on `develop` or any non-feature branch: **create a feature branch immediately**:
  ```bash
  git checkout develop && git pull origin develop
  git checkout -b <username>/<feature-name>
  git submodule update --init --recursive
  ```
- **Do NOT write code, make edits, or run builds until you are on a feature branch.**

Branch naming: `<username>/<feature>` — lowercase, hyphens, no ticket number in branch name.

### One Task Per Branch

An MR (a single development branch) should ideally only contain changes related to a **single task**. Do not mix unrelated changes.

If a new SIPU SDK version is needed, update `sipu_sdk_version.txt` and stage it.

---

## Step 1: Build Environment Setup

Before building or testing, ensure the environment is configured. All three steps are required, in order:

```bash
# 1. Activate conda env (provides Python 3.11, torch 2.7.1+cpu, clang/clang++)
conda activate pytorch

# 2. Source SIPU SDK env (sets SI_CROSS_COMPILE_PATH, SI_SDK_ROOT, CMAKE_PREFIX_PATH, etc.)
source setup_sipu_sdk_env.sh

# 3. Build
make install-dev           # Extension only (skip kernel rebuild)
# OR
make install-all-dev       # Full rebuild including sikernel (use if third_party/sikernel changed)
```

**Common build failures:**

| Error | Cause | Fix |
|---|---|---|
| `CMAKE_SIPU_COMPILER not found` / `Findsirt.cmake not found` | Forgot `source setup_sipu_sdk_env.sh` | Run step 2 |
| `Could not find compiler: clang++` | Forgot `conda activate pytorch` | Run step 1 |
| `gmake: No rule to make target 'Makefile'` | Stale build directory | `rm -rf build/` then rebuild |

Subsequent `make install-dev` runs are incremental — only modified files recompile.

---

## Step 2: Coding Standards

### 2.1 Code Comment Markers

Use these markers for issues that need follow-up. If the issue depends on another team, create a JIRA and reference it in the comment.

| Marker | Meaning | Urgency | Example |
|---|---|---|---|
| `FIXME` | Known bug or defect that needs fixing | High | `// FIXME: This may cause memory leak, need proper cleanup.` |
| `TODO` | Planned task or optimization | Medium | `// TODO: Optimize this O(n^2) algorithm to O(n log n).` |
| `NOTE` | Important explanation, no action needed | Info | `// NOTE: This function assumes input is already normalized.` |
| `HACK` | Inelegant workaround, needs refactoring | Medium | `// HACK: Workaround for upstream bug, remove when fixed.` |
| `BUG` | Confirmed defect, must fix | Critical | `// BUG: Race condition under concurrent access.` |
| `OPTIMIZE` | Performance optimization point | Low | `// OPTIMIZE: Cache this computation to avoid redundant calculations.` |

### 2.2 License Header (New Files)

All **new** source files MUST have the Apache v2.0 license header at the top. Use the correct comment syntax for the language and the file's **creation year**.

```
C++/SU/SUH:  // Copyright (c) <YEAR> SiOrigin Co. Ltd.
Python/SH:   # Copyright (c) <YEAR> SiOrigin Co. Ltd.
```

Full header template is in CLAUDE.md. Key rules:
- `<YEAR>` = year the file was created (not current year; e.g., backfilling a 2024 file uses 2024)
- Do NOT add SiOrigin header to files from other open-source projects — keep their original MIT/BSD headers
- When in doubt, check existing files in the same directory for the expected header format

### 2.3 C++ Conventions

- Use `c10::SmallVector` when vector size is <= 64 bytes (stack-allocated, faster).
- Use `AT_DISPATCH` macros for dtype handling — do NOT directly use `AT_DISPATCH_SIPU`, use the `AT_DISPATCH_SIPU_xxxx` version corresponding to the CUDA variant.
- Follow Google C++ Style Guide and CppCoreGuidelines.
- Include c10 headers with angle brackets: `#include <c10/sipu/SIPUAllocatorConfig.h>` (not relative paths).

### 2.4 Parameter Naming (C++ Kernel Interface)

Follow the names from PyTorch's wrapper functions in `build/aten/src/ATen/Register*.cpp`:

- Input/output: use original names (`self`, `other`, `out`).
- Size/stride/ndim: append `_sizes`, `_strides`, `_ndims` to the parameter name (e.g., `self_sizes`, `self_strides`, `self_ndims`).
- `ITensorListRef` inputs: add `num_<param>` to record count (e.g., `num_tensors`).

### 2.5 Testing Requirements

- **Unit tests are mandatory** for new features and bug fixes.
- Tests should at minimum pass accuracy-related UTs:
  - Unary ops: `test/test_unary_ufuncs.py` (`test_reference_numerics_*`)
  - Binary ops: `test/test_binary_ufuncs.py` (`test_reference_numerics_*`)

---

## Step 3: Pre-Commit Checks

Before committing, **always** complete these steps:

1. **Run unit tests** — verify all relevant tests pass.
2. **Run linter:**
   ```bash
   make lint
   ```
3. **Auto-fix lint errors** (if any):
   ```bash
   lintrunner -a --all-files
   ```
4. **Verify Jira reference** — for tracked tasks, ensure commit message includes `jira#S1SW-XXXX`.

### Lint Setup (First Time)

```bash
make setup-lint
```

Requires `clang-format` and `clang-tidy`.

---

## Step 4: Commit Message Format

```
<type>(<scope>): jira#S1SW-XXXX <description>
```

**`jira#` goes after the colon, before the description — never at the end.**

### Types

| Type | When to use |
|---|---|
| `feat` | New feature or operator |
| `fix` | Bug fix |
| `refactor` | Code restructuring without behavior change |
| `chore` | Build, lint, CI changes |
| `docs` | Documentation only |
| `test` | Test-only changes |

### Rules

- For tracked tasks, include Jira number as `jira#S1SW-XXXX`.
- For hotfixes and simple updates (typos, docs), Jira number is optional.
- Use **English** for commit messages and MR titles.

### Examples

```
feat(demo): Implement jira#S1SW-144 Llama2 demo for Feb. (Matmul+Softmax)
fix(aten): fix xxx bug jira#S1SW-1302
feat(runtime): integrate xxxx apis jira#S1SW-1311
refactor(aten): Implement jira#S1SW-1901 refactor triu_and_tril op using joint compilation
chore(lint): update ruff configuration
fix(docs): fix typo in README
```

---

## Step 5: Squash, Rebase & Push

### 5.1 One Commit Per Branch (MANDATORY)

A branch **must have exactly one commit** before pushing. This is enforced at review time — fragmented commits will clutter `develop` history.

**If you have multiple commits on the branch, squash them first:**

```bash
# Count commits ahead of develop
git log --oneline origin/develop..HEAD

# Squash all of them into one (replace N with the count)
git reset --soft HEAD~N
git commit -m "<type>(<scope>): jira#S1SW-XXXX <description>"
```

**When assisting the user (Claude instructions):**
- Before creating any commit, run `git log --oneline origin/develop..HEAD` to check existing commits.
- If a commit already exists on this branch, use `git reset --soft HEAD~<count>` to squash all changes together, then create a single new commit.
- **Never create a second commit to fix something that went wrong in the first commit.** Always squash.

### 5.2 Rebase onto Latest develop

The project uses **fast-forward only** (`merge_method: ff`) merging. This means:
- The branch must be a direct descendant of `develop`'s HEAD to be mergeable
- Whenever anyone merges a new MR into `develop`, GitLab will show **"Rebase required"** on your MR
- This is expected behavior — rebase before merging, not necessarily before every push

```bash
git fetch origin
git rebase origin/develop
# If conflicts arise, resolve them, then: git rebase --continue
```

### 5.3 Push

```bash
git push -u origin <branch>
# After a rebase or squash, force push is required:
git push --force
```

**OpenSSL workaround** — if encountering `OpenSSL version mismatch`:
```bash
GIT_SSH_COMMAND="<path-to-compatible-ssh>" git push --force
```

---

## Step 6: Create Merge Request (MR)

1. Push branch to GitLab, then create an MR targeting `develop`.
2. Modify the MR title to follow the commit message specification.
3. **Add the appropriate label** (required) based on the change:

   | Label | CI Tests Triggered |
   |---|---|
   | `sikernel` | Linter + torch_sipu tests with C++ kernel |
   | `triton` | Linter + torch_sipu tests with Triton backend |
   | `ai` | Linter + torch_sipu tests with AI backend |
   | `pytorch` | Linter + PyTorch upstream tests (excluding slow) |
   | `pytorch-triton` | Linter + PyTorch upstream tests with Triton backend |
   | `compiler` | Linter + TorchCompiler tests |
   | `dist` | Linter + communication library tests |
   | `lint` (or no label) | Linter only |

   Slow tests are NOT triggered via MR labels. They run automatically:
   - **nightly**: all normal suites + torch_sipu `@slowTest` cases (kernel_slow)
   - **weekly**: PyTorch upstream slow tests (>1000s each)

4. After adding the label, **post a comment `test CI`** to trigger the CI pipeline.

---

## Step 7: Close JIRA

After the MR is merged, set the JIRA issue status to **DONE**.

---

## Reference: Test Script Usage

```bash
./run_test.sh [OPTIONS]
```

| Short | Long | Description |
|-------|------|-------------|
| `-k` | `--kernel` | Run C++ kernel tests |
| `-t` | `--triton` | Run Triton kernel tests |
| `-c` | `--torchcompile` | Run torch compile tests |
| `-p` | `--pytorch` | Run PyTorch source tests |
| `-d` | `--dist` | Run distributed tests |
| `-s` | `--slowtest` | Enable slow tests |
| `-T` | `--pytorch_triton` | Run PyTorch tests with Triton ops |
| `-A` | `--ai-triton` | Run AI Triton kernel tests |
| `-S` | `--pytorch_slow` | Run PyTorch slow tests (hours) |
| `-l` | `--lint` | Run lint checks |
| `-v` | `--verbose` | Verbose output |

Examples:
```bash
./run_test.sh -k          # C++ kernel tests
./run_test.sh -t          # Triton kernel tests
./run_test.sh -l -k       # Lint + kernel tests
./run_test.sh -k -s       # Kernel tests with slow tests
```

### Kernel Development: Lazy Load

For compilation acceleration, use `tile_vector.h` (with `__tile_` prefix) instead of `siorigin_tile.h`. See `development_requirements.md` for the required SDK version and details.
