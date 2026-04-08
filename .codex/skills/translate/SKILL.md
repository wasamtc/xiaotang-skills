---
name: translate
description: Translate English or mixed-language articles, blog posts, documentation pages, newsletters, and long-form web content into polished Chinese Markdown while preserving structure and images as much as possible. Use when Codex needs to convert an existing article or webpage into Chinese, keep key terms in bilingual form on first mention, retain code blocks/tables/lists/media links, and add author, source URL, and an AI translation note at the beginning.
---

# Translate

Translate source articles into readable Chinese Markdown without losing the source structure, media, or technical precision.

## Workflow

1. Identify the source material, original title, author name, original link, and the original content structure before translating.
2. Preserve the document hierarchy. Keep headings, lists, tables, quotes, code blocks, math, callouts, and separators in a Markdown-friendly form.
3. Preserve images whenever possible.
4. Translate paragraph by paragraph instead of rewriting from scratch.
5. Keep technical meaning exact. Prefer direct translation over paraphrase unless the source wording is unnatural in Chinese.
6. Add a short metadata block at the top with the original author, original link, and a clear AI translation note.
7. Output the final result as Markdown only.

## Required Output Rules

- Use Markdown as the final output format.
- Add a front section before the translated title and body.
- Include these items in the front section:
  - `原作者：...`
  - `原文链接：...`
  - `说明：本文由 AI 翻译，尽量保持原意与原始结构；术语在首次出现时提供中英文对照。`
- If the author or link is missing from the source, write `未知` and state that the source material did not provide it.

## Terminology Rules

- Keep key terms in both Chinese and English on first mention.
- Use the format `中文（English）` when Chinese is the main sentence language.
- Reuse the shorter Chinese-only form after the first mention unless keeping English is clearer.
- Do not forcibly translate proper nouns, product names, library names, protocol names, paper titles, company names, or API field names. Keep the original term and add Chinese only when it improves clarity.
- Keep terminology consistent across the whole article.
- If the source already uses an accepted Chinese translation, keep that translation and append the English term on first mention.

## Image Preservation Rules

- Preserve images, figures, diagrams, and captions whenever possible.
- If the source is already in Markdown, keep the original image syntax and path unchanged.
- If the source uses HTML images and the image cannot be cleanly converted, keep the HTML image block instead of dropping it.
- Keep nearby captions or explanatory text with the image.
- Do not invent missing image URLs, alt text, or captions.
- If an image is unavailable in the provided source, leave a short placeholder note instead of silently removing it.

## Formatting Rules

- Preserve the original heading depth and section order.
- Preserve inline code, fenced code blocks, tables, links, footnotes, and blockquotes.
- Do not translate code, CLI commands, file paths, environment variables, URLs, or API payload keys unless the source explicitly explains them in prose.
- Translate link text, but keep the underlying URL unchanged.
- Preserve emphasis such as bold and italics where it carries meaning.
- Avoid adding summaries, commentary, or opinions that do not exist in the source.

## Quality Bar

- Prefer faithful translation over aggressive localization.
- Make Chinese read naturally, but do not omit technical nuance.
- If a sentence is ambiguous, resolve it conservatively and keep the original term visible when needed.
- If part of the source is unreadable or missing, mark that explicitly instead of guessing.

## Output Template

Use the template in [references/translation-output-template.md](references/translation-output-template.md) unless the user requests another layout.
