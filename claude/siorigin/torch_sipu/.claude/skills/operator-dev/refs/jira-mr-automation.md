# JIRA Ticket & MR Automation Scripts

## 1. Create JIRA Ticket

**Prerequisites:** `pip install jira` and `~/.jira.cfg` configured:

```ini
[jira]
url = https://jiraoffice.siorigin.com
token = <YOUR_PERSONAL_ACCESS_TOKEN>
project = S1SW
```

**Script:**

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
    summary='<descriptive summary of the operator task>',
    # description='<optional detailed description>',
)
print(f"Created: {issue.key} — {jira_url}/browse/{issue.key}")
```

Replace `<descriptive summary>` with a meaningful title (e.g., "Implement abs operator for SIPU").

---

## 2. Create Merge Request (MR) — Automated

**Prerequisites:** `pip install python-gitlab` and `~/.python-gitlab.cfg` configured:

```ini
[global]
default = siorigin
timeout = 30

[siorigin]
url = https://gitlabsoft.siorigin.com
private_token = <YOUR_GITLAB_PERSONAL_ACCESS_TOKEN>
api_version = 4
```

### CI Label Reference

| Label | CI Tests Triggered |
|---|---|
| `sikernel` | Linter + all C++ kernel tests |
| `triton` | Linter + all Triton kernel tests |
| `ai-tri` | Linter + AI backend + Triton tests |
| `compiler` | Linter + TorchCompiler tests |
| `dist` | Linter + communication library tests |
| `torchtest` | Linter + PyTorch unittest (excluding slow tests) |
| `slowtest` | Slow tests only |
| `torchtest-slow` | Linter + PyTorch unittest (slow tests only) |
| `aten` | Linter + ATen op tests |
| `lint` (or no label) | Linter only |

### MR Creation Script (with JIRA link)

```python
import configparser, os
import gitlab

# Read JIRA config for linking tickets
jira_cfg = configparser.ConfigParser()
jira_cfg.read(os.path.expanduser('~/.jira.cfg'))

gl = gitlab.Gitlab.from_config('siorigin')  # reads ~/.python-gitlab.cfg [siorigin] section
gl.auth()

project = gl.projects.get('algo/framework/torch_sipu')

# Check if MR already exists
mrs = project.mergerequests.list(source_branch='<branch>', state='opened')
if mrs:
    print(f"MR already exists: {mrs[0].web_url}")
else:
    # Build description with JIRA link (if ticket exists)
    description = '<summary of changes>'
    jira_key = '<S1SW-XXXX or empty>'  # from Step 0.0a
    if jira_key:
        jira_url = jira_cfg.get('jira', 'url', fallback='')
        description += f'\n\nJIRA: {jira_url}/browse/{jira_key}'

    mr = project.mergerequests.create({
        'source_branch': '<branch>',
        'target_branch': 'develop',
        'title': '<commit message first line>',
        'description': description,
        'labels': ['<ci_label>'],
        'remove_source_branch': True,
    })
    print(f"MR created: {mr.web_url}")
    # Trigger CI
    mr.notes.create({'body': 'test CI'})
    print("Commented 'test CI' to trigger CI")
```

Replace `<branch>`, `<commit message first line>`, `<summary>`, `<ci_label>`, and `<S1SW-XXXX>` with actual values.

After MR is merged, set the JIRA issue status to **DONE** (if applicable).

---

## 3. Reply to MR Review Comments — GitLab API

```python
import gitlab, configparser, os

cfg = configparser.ConfigParser()
cfg.read(os.path.expanduser('~/.python-gitlab.cfg'))
gl = gitlab.Gitlab.from_config('siorigin')

project = gl.projects.get('algo/framework/torch_sipu')
mr = project.mergerequests.get(<MR_IID>)

# Reply to a specific discussion thread
discussion = mr.discussions.get('<discussion_id>')
discussion.notes.create({'body': 'Fixed, see commit <hash>'})
```
