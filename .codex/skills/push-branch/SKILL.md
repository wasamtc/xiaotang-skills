---
name: push-branch
description: >-
  Push current feature branch to remote with minimal, code-only commits.
  Handles squash, rebase onto origin/develop, conflict detection, and force push.
  Use when the user says "push", "submit branch", "提交分支", "推送", or
  explicitly invokes /push-branch.
---

# Push Branch to Remote

Automate the full push workflow: stage only code files related to the current branch, squash into one commit with S1SW-tagged message, rebase onto `origin/develop`, and push.

---

## Prerequisites

- You MUST be on a **feature branch** (`<username>/<feature>`), NOT on `develop`.
- Verify with `git branch --show-current`. If on `develop`, **stop and ask the user** to switch.

---

## Step 1: Identify Changed Files

```bash
# 1. Already-committed changes on this branch (vs origin/develop)
git diff --name-only origin/develop...HEAD

# 2. Uncommitted changes (staged + unstaged + untracked)
git status --porcelain
```

`git status --porcelain` covers staged (`A`/`M`), unstaged (`M`/`D`), and untracked (`??`) files in one command.

Merge the two lists into a unified set of **all files changed on this branch** (both committed and uncommitted).

### 1.1 Filter: Code Files Only

**Exclude** all documentation, test, script, and non-code files. Remove any file matching these patterns:

| Exclude Pattern | Reason |
|---|---|
| `*.md` | Markdown documentation |
| `*.rst` | reStructuredText documentation |
| `*.txt` (except `requirements*.txt`, `sipu_sdk_version.txt`) | Plain text docs |
| `*.pdf`, `*.docx`, `*.pptx` | Binary documents |
| `*.png`, `*.jpg`, `*.jpeg`, `*.gif`, `*.svg`, `*.ico` | Images |
| `docs/**` | Documentation directory |
| `*.log` | Log files |
| `.env`, `credentials.*`, `*.key`, `*.pem` | Secrets |
| `test/**` | Test directory — all test files (**see whitelist below**) |
| `test_*.py`, `*_test.py` | Test files anywhere in the tree |
| `tests/**` | Test directory (alternate naming) |
| `**/test/**`, `**/tests/**` | Nested test directories |
| `conftest.py` | Pytest config/fixtures |
| `examples/**` | Example/demo scripts |
| `scripts/**` | Utility scripts |
| `*.sh` | Shell scripts (run_test.sh, etc.) |
| `run_test.sh` | Test runner script |
| `Makefile` | Build/script file (not source code) |
| `benchmarks/**` | Benchmark scripts |

### 1.1.1 Whitelist: Operator-Integration Test Files (MUST Commit)

The following test infrastructure files are **part of operator integration** (see `docs/新加算子接入测试.md`). When a new operator is added, these files carry registration and CI-critical count assertions. They **override the general test exclusion** and MUST be committed if modified:

| Whitelisted File | Role |
|---|---|
| `test/test_unary_ufuncs.py` | Unary op supported list (`current_supported_unary_ufuncs`) |
| `test/test_binary_ufuncs.py` | Binary op supported list (`current_supported_binary_ufuncs`) |
| `test_runner/catalog/pytorch.py` | Op registration + suite builder (`register_*_op`) |
| `test_runner/catalog/registry.py` | Registration infrastructure (if modified) |
| `test/test_catalog.py` | Catalog self-check case count assertions |
| `test/test_profiles.py` | MR/weekly profile golden case count assertions |

**Rule:** When evaluating changed files, check the whitelist **before** applying the exclude rules. If a file is in the whitelist and has been modified on this branch, **always include it** in the commit regardless of the `test/**` exclusion.

**Include** only production source code and build-config files:

| Include Pattern | Examples |
|---|---|
| `*.py` (excluding test/script patterns above) | Python source in `torch_sipu/`, `codegen/` |
| `*.cpp`, `*.h`, `*.su`, `*.suh` | C++/SIPU source |
| `*.yaml`, `*.yml` (in source tree) | Registration YAML (`native_functions.yaml`, etc.) |
| `*.cmake`, `CMakeLists.txt` | CMake build config |
| `*.json` (in source tree) | Config files |
| `*.toml`, `*.cfg`, `*.ini` | Config files |
| `setup.py`, `setup.cfg`, `pyproject.toml` | Package config |
| `sipu_sdk_version.txt` | SDK version |
| `requirements*.txt` | Dependencies |

### 1.2 Filter: Minimal Scope

From the filtered code files, further narrow down:

1. **Only files that are actually modified** — do not add files with whitespace-only or formatting-only changes.
2. **Skip generated files** in `build/`, `dist/`, `*.egg-info/`, `__pycache__/`.
3. **Skip submodule changes** in `third_party/` unless the user explicitly confirms.

### 1.3 Show the File List and Confirm

Before staging, **print the filtered file list** and ask the user for confirmation:

```
Files to commit (N files):
  M  torch_sipu/csrc/aten/native/sipu/NewOp.su
  M  torch_sipu/csrc/aten/native/native_functions.yaml
  M  test/test_unary_ufuncs.py          [whitelisted: op integration]
  M  test_runner/catalog/pytorch.py     [whitelisted: op integration]
  M  test/test_catalog.py               [whitelisted: op integration]
  M  test/test_profiles.py              [whitelisted: op integration]

Excluded (documentation):
  M  docs/ops/new_op.md
  M  README.md

Excluded (test/script):
  A  test/test_new_op.py
  M  examples/run_new_op.py
  M  run_test.sh

Proceed? [Y/n]
```

Wait for user confirmation before continuing.

---

## Step 2: Stage and Commit

### 2.1 Determine Commit Message

1. Run `git log --oneline origin/develop..HEAD` to find existing commits on this branch.
2. If existing commits exist, **reuse the most recent commit message** (adjusting if needed).
3. If no commits exist yet, derive the message from:
   - The **branch name** (e.g., `tangcong/native-batch-norm` → "implement native_batch_norm")
   - The **type of change** (feat/fix/refactor/chore/test)
4. The commit message MUST follow the project format:
   ```
   <type>(<scope>): jira#S1SW-XXXX <description>
   ```
5. If no S1SW Jira ticket number is found in existing commits or branch name, **ask the user** for the ticket number. If the user says no ticket is needed, omit `jira#S1SW-XXXX` but keep the `<type>(<scope>): <description>` format.

### 2.2 Stage Files

```bash
git add <file1> <file2> ...
```

Only add the files from the confirmed list in Step 1.3. **Never use `git add .` or `git add -A`.**

### 2.3 Squash into One Commit

```bash
# Count existing commits on this branch
COMMIT_COUNT=$(git rev-list --count origin/develop..HEAD)

if [ "$COMMIT_COUNT" -gt 0 ]; then
    # Squash all existing commits + staged changes into one
    git reset --soft $(git merge-base origin/develop HEAD)
    git commit -m "<type>(<scope>): jira#S1SW-XXXX <description>"
else
    # First commit on this branch
    git commit -m "<type>(<scope>): jira#S1SW-XXXX <description>"
fi
```

After this step, verify there is exactly **one commit** ahead of `origin/develop`:

```bash
git log --oneline origin/develop..HEAD
```

If more than one commit shows, repeat squash.

---

## Step 3: Fetch and Rebase onto origin/develop

```bash
git fetch origin develop
git rebase origin/develop
```

### 3.1 Conflict Handling

If `git rebase` reports conflicts:

1. Run `git diff --name-only --diff-filter=U` to list conflicting files.
2. For each conflicting file, run `git diff <file>` to inspect the conflict.

**Decision matrix:**

| Conflict Type | Action |
|---|---|
| **Full conflict** — the file's entire content is different between ours and theirs (e.g., both sides rewrote the same section completely) | Run `git rebase --abort`, **stop**, and report to the user: "Rebase aborted due to full conflict in `<file>`. Manual resolution needed." |
| **Partial conflict** — only some hunks conflict, and the intent of both sides is compatible | Resolve by keeping both changes where possible (prefer `origin/develop` for unrelated changes, prefer ours for the feature code). Then `git add <resolved_file> && git rebase --continue`. |
| **Trivial conflict** — import ordering, whitespace, adjacent-line additions | Auto-resolve, prefer the version that includes both sides' additions. Then `git add <resolved_file> && git rebase --continue`. |

**Rules:**

- If **any** file has a full conflict → abort the entire rebase and stop.
- After resolution, verify the build files still look correct.
- After `git rebase --continue`, verify with `git log --oneline origin/develop..HEAD` that there is still exactly one commit.

---

## Step 4: Push

```bash
git push --force-with-lease -u origin $(git branch --show-current)
```

Use `--force-with-lease` (not `--force`) for safety — it prevents overwriting others' work on the same branch.

### 4.1 OpenSSL Workaround

If push fails with `OpenSSL version mismatch`, retry with:

```bash
GIT_SSH_COMMAND="/usr/bin/ssh" git push --force-with-lease -u origin $(git branch --show-current)
```

If this also fails, try the system SSH:

```bash
GIT_SSH_COMMAND="$(which ssh)" git push --force-with-lease -u origin $(git branch --show-current)
```

### 4.2 Verify

After push succeeds, run:

```bash
git log --oneline origin/develop..HEAD
git status
```

Report to the user:
- Branch name
- Commit hash and message
- Number of files changed
- Remote URL

---

## Summary Checklist

```
Push Branch Workflow:
- [ ] Step 1: Identify changed files (code only, minimal scope)
- [ ] Step 1.3: Show file list and get user confirmation
- [ ] Step 2.1: Determine commit message (S1SW format)
- [ ] Step 2.2: Stage confirmed files only
- [ ] Step 2.3: Squash to exactly one commit
- [ ] Step 3: Fetch origin/develop and rebase
- [ ] Step 3.1: Handle conflicts (abort on full conflict)
- [ ] Step 4: Push with --force-with-lease
- [ ] Step 4.2: Verify and report result
```

---

## Error Recovery

| Error | Recovery |
|---|---|
| Not on feature branch | Stop. Ask user to create/switch to a feature branch. |
| No changed files | Stop. Nothing to commit. |
| Rebase full conflict | `git rebase --abort`. Report conflicting files to user. |
| Push rejected (non-OpenSSL) | Check if remote branch has new commits. Ask user. |
| Push OpenSSL error | Retry with `GIT_SSH_COMMAND` workaround. |
