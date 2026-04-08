---
name: thesis-blind-review
description: Review a degree thesis as a blind-review expert using the governing rules in `review/盲审规则.pdf` and the supporting materials under `documents/`. Use when Codex needs to assess the actual thesis content in this repository, explicitly exclude the format-reference file `徐子才毕业论文.pdf` from the review target unless the user says otherwise, produce a clear blind-review conclusion, list major problems and revision suggestions, and save the written review report under `results/`.
---

# Thesis Blind Review

## Overview

Review the thesis in this repository from the perspective of a blind-review expert. Read the blind-review rules first, inspect the thesis and supporting materials, then write a structured review report with a conclusion and revision suggestions into `results/`.

## Review Inputs

- Treat `review/盲审规则.pdf` as the governing standard. Extract the actual review dimensions, mandatory checks, scoring bands, and conclusion labels from that PDF before drafting the report.
- Treat the actual thesis manuscript in the repository as the review target. Prefer the thesis sources that belong to this project, such as `HNU_DoctoralThesis_Latex/` and its `chapters/`, or any manuscript file the user explicitly identifies as the real thesis.
- Treat `/share_data/tangcong/project/paper/my_graduate/徐子才毕业论文.pdf` only as a formatting or layout reference. Do not review it as the thesis content and do not use its technical content as evidence for the blind-review conclusion unless the user explicitly instructs otherwise.
- Treat `documents/` as supporting evidence. Use those files to verify technical background, implementation claims, experimental settings, terminology, and feasibility statements made by the thesis.
- Treat `results/` as the output directory. Write the final report to `results/盲审审阅意见.md` unless the user asks for a different file name.

## Workflow

1. Inventory the candidate thesis files and confirm which manuscript is the main review target. Exclude `徐子才毕业论文.pdf` by default because it is a format reference rather than the thesis under review.
2. Read `review/盲审规则.pdf` first and extract the review framework from it. Do not invent rule labels or conclusion categories if the PDF defines them explicitly.
3. Read the real thesis manuscript in full or by chapter. Prefer project-owned source materials. When PDF text extraction is poor or no thesis PDF is provided, read the LaTeX chapters and figures in `HNU_DoctoralThesis_Latex/`.
4. Cross-check the thesis against the materials in `documents/` and any relevant files already present in `results/` when those materials support or contradict thesis claims.
5. Evaluate the thesis against the dimensions required by the blind-review rules. If the PDF does not provide a full dimension list, assess at least topic significance, literature review, method rigor, experiment validity, innovation, completeness, and writing quality.
6. Distinguish critical defects from secondary polish issues. Prioritize findings that affect the blind-review conclusion.
7. Draft a professional blind-review report with a clear conclusion, major problems, and actionable revision suggestions.
8. Save the final report as Markdown under `results/`.

## Evidence Rules

- Cite concrete evidence with file paths, chapter names, section names, figure or table identifiers, or nearby headings whenever possible.
- Label any unverified judgment as an inference.
- Do not fabricate rule clauses, page numbers, references, experimental results, or missing evidence.
- Treat unsupported claims as findings. If the thesis asserts something important without proof, say so directly.
- Maintain a blind-review perspective. Focus on quality, compliance, rigor, completeness, and presentation rather than author identity.

## Output Structure

Use the following structure unless the user requests a different format:

### 审阅对象

- Main thesis file(s) reviewed
- Supporting materials consulted

### 盲审结论

- Use the conclusion labels defined in `review/盲审规则.pdf`
- State the overall judgment in one concise paragraph

### 主要问题

- List the most important issues first
- Explain why each issue affects the review result

### 修改意见

- Give concrete, executable revision suggestions
- Keep suggestions specific to the thesis content and structure

### 可保留优点

- Summarize the main strengths briefly

### 结论依据

- Summarize the rule items and evidence that led to the conclusion

## Output Quality Bar

- Make the conclusion explicit, not implied.
- Keep the tone professional and review-like.
- Prefer concise, high-signal prose over generic praise or filler.
- Ensure the revision suggestions are specific, prioritized, and non-duplicative.
- Save the final content to `results/盲审审阅意见.md`.
