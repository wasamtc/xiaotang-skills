# Handling MR Review Feedback

**Trigger:** Use this workflow when an MR has been submitted and a colleague has left review comments (request changes / inline comments).

Do NOT re-run Step 0 (no new JIRA, no new branch). Stay on the existing feature branch.

---

## 1. Collect and Parse Review Comments

Ask the user to paste all review comments. For each comment, extract:
- **File + line** (if inline)
- **What the reviewer said** (verbatim or paraphrased)

Then classify each comment into one of:

| Type | Description | Fix approach |
|---|---|---|
| **Implementation bug** | Logic error, wrong algorithm, incorrect behavior | Go to Step 3 (relevant sub-section) to fix |
| **Missing / wrong test** | Test coverage gap, wrong tolerance, missing edge case | Go to Step 6 to fix |
| **Performance** | Not vectorized, unnecessary memory allocation, slow path | Go to Step 5 to fix |
| **Style / naming** | Variable names, formatting, comment style | Fix in place — no need to revisit full workflow |
| **Design concern** | Architectural issue, wrong dispatch path chosen | Re-read Step 1.1 / 1.1a and discuss with user before changing |
| **Ambiguous** | Comment is unclear | See §2 below |

Present the classified list to the user before making any changes:

```
Comment 1 (SoftMax.su:42, @zhang_san): "Should use float32 accumulation here"
  → Type: Implementation bug → Fix in Step 3.2

Comment 2 (test_softmax.py, @li_si): "Missing half precision test"
  → Type: Missing test → Fix in Step 6

Comment 3 (SoftMax.su:10, @zhang_san): "Variable name inv is not clear enough"
  → Type: Style → Fix in place
```

---

## 2. Handle Ambiguous Comments

If a comment is unclear (e.g., "there's an issue here" without specifics), **do not guess**. Ask the user:
- "What do you think this comment means?"
- Or suggest asking the reviewer for clarification before making changes.

Making the wrong fix wastes a round-trip. Clarifying first is always cheaper.

---

## 3. Make the Fixes

Fix each comment in the order: **bugs first → tests → performance → style → design**.

For each fix:
- Make a **targeted change** — do not refactor surrounding code unless the reviewer explicitly asked for it
- After fixing, state: "Fixed comment N: [what was changed]"

---

## 4. Commit the Fixes

**Do NOT amend the original commit** — that rewrites history and makes the reviewer's diff harder to follow.

Add a fixup commit per logical group of fixes:

```bash
git add <changed files>
git commit -m "fix(aten): address review comments for <op> jira#S1SW-XXXX

- fix float32 accumulation in softmax kernel (SoftMax.su:42)
- add half precision test case (test_softmax.py)
- rename inv → inv_sum for clarity"
```

---

## 5. Reply to Reviewer on MR

After pushing, reply to each inline comment on the MR. For each:
- **Fixed**: "Fixed, see commit <hash>"
- **Won't fix (with reason)**: "Keeping as-is, reason: XXX"
- **Need clarification**: "Could you clarify the expected behavior?"

For auto-reply via GitLab API, see `refs/jira-mr-automation.md` §3.

---

## 6. Re-trigger CI

After pushing the fixup commit:

```bash
# Push to the same branch
GIT_SSH_COMMAND="<path-to-compatible-ssh>" git push origin <branch>
```

Then on the MR, comment `test CI` to re-trigger the CI pipeline.
