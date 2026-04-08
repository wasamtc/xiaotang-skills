---
name: read-torch
description: >-
  Read PyTorch source code and documentation from the project directory.
  Only use when the user explicitly types "/readtorch". Do NOT use automatically
  or proactively.
---

# Read Torch Project Files

## Trigger

ONLY activate when the user explicitly types `/readtorch`. Do NOT trigger automatically based on context or conversation topic.

## Target Directory

`/share_data/tangcong/project/pytorch_v2.7.1`

## Workflow

### Step 1: Explore directory structure

Use the Shell tool to list the top-level directory structure:

```bash
ls -la /share_data/tangcong/project/pytorch/
```

Present the directory tree to the user in a clear format.

### Step 2: Ask the user what to explore

After showing the top-level structure, ask the user:
- Which subdirectory or area they want to explore
- What they are looking for (a specific module, function, concept, etc.)

Do NOT read any files until the user specifies what they want.

### Step 3: Drill down into the chosen area

Use Glob and Shell tools to list the contents of the user's chosen subdirectory. Show the file list and ask the user which files to read.

### Step 4: Read selected files

- Use the Read tool to read only the files the user selects
- For large files (>500 lines), read in chunks and summarize first, then read specific sections on request
- Do not limit by file type — read any file format (.py, .cpp, .h, .md, .yaml, .txt, .rst, etc.)

### Step 5: Repeat

After presenting file contents, ask if the user wants to:
- Explore another directory
- Read more files
- Search for specific content within the project

## Rules

- NEVER read files without user confirmation
- NEVER limit file types — all files are valid targets
- For files over 500 lines, show a summary (line count, key sections) first and let the user decide which parts to read
- Always show the user what's available before reading
- Use Grep to search for specific symbols or patterns when the user asks
