# operator-dev Skill

`operator-dev` is the AI Skill for operator development in the torch_sipu project, covering the full workflow from JIRA ticket to MR merge.

## How to Use

In a Claude Code session, type `/operator-dev`, then describe your task in natural language:

```
/operator-dev implement aten::abs operator
/operator-dev refactor SoftMax.cpp to .su joint compilation
/operator-dev my MR was rejected, reviewer said: [paste review comments]
```

## Applicable Scenarios

| Scenario | Example |
|---|---|
| New operator | "implement cos operator for Triton backend" |
| Modify existing operator | "add float16 support for sigmoid" |
| Refactor operator (.cpp -> .su) | "refactor RmsNorm.cpp to joint compilation" |
| Fix operator bug | "fix softmax precision issue with bfloat16" |
| Infrastructure change | "optimize vectorized_reduction in Reduce.suh" |
| Handle MR review feedback | "reviewer says my accumulation precision is wrong" |

**Not applicable**: pure distributed communication (use `distributed-dev`), torch.compile related (use `torch-compile-dev`).

## Workflow Overview

The skill automatically detects the scenario:

- **Scenario A (New development)**: Step 0 (JIRA + branch) -> Step 1 (locate files) -> Step 2 (minimal modification check) -> Step 3 (implement) -> Step 4 (build) -> Step 5 (performance check) -> Step 6 (test) -> Step 7 (commit + MR)
- **Scenario B (Handle review feedback)**: Jump directly to Step 8, no new JIRA or branch creation

See [SKILL.md](./SKILL.md) for detailed workflow.

## Environment Setup

The automation features (JIRA ticket creation, GitLab MR creation) require the following configuration:

### JIRA -- `~/.jira.cfg`

```ini
[jira]
url = https://jiraoffice.siorigin.com
token = <YOUR_PERSONAL_ACCESS_TOKEN>
project = S1SW
```

Token: JIRA -> Profile icon (top right) -> Profile -> Personal Access Tokens -> Create new Token

`chmod 600 ~/.jira.cfg` / `pip install jira`

### GitLab -- `~/.python-gitlab.cfg`

```ini
[global]
default = siorigin
timeout = 30

[siorigin]
url = https://gitlabsoft.siorigin.com
private_token = <YOUR_GITLAB_PERSONAL_ACCESS_TOKEN>
api_version = 4
```

Token: GitLab -> Preferences -> Access Tokens -> Create new Token (scope: `api`)

`pip install python-gitlab`

### SSH (if git push fails)

If `git push` reports `OpenSSL version mismatch`:

```bash
GIT_SSH_COMMAND="<path-to-compatible-ssh>" git push -u origin <branch>
```

## FAQ

**Q: The op is `CompositeImplicitAutograd`, do I need to implement it?**
Generally no, PyTorch will auto-decompose. But it may be worth implementing a fused kernel for performance-critical inference paths.

**Q: What is the relationship between Triton backend and C++ backend?**
Triton is active by default (higher priority), C++ serves as fallback. Triton changes (Python) don't require compilation; C++ changes require `make install-dev`.

**Q: `.su` vs `.cpp`?**
New development should use `.su` (joint compilation). Only use `.cpp` when calling sikernel library functions.

**Q: Do I need to implement backward?**
No, torch_sipu is a pure inference backend.
