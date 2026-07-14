# Dataset Pre-Training Workspace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Local, GPU-first pipeline that turns a PDF corpus into training data.

- Convert a PDF dataset into PNG pages
- OCR PNG pages with [Surya OCR](https://github.com/datalab-to/surya)
- Summarize OCR text with a local Gemma 3 model ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))

## Folder structure

- `DATASET/`: Input PDFs (example: `Release_1/`).
- `outputs/[timestamp]_[dataset]/`: everything produced from one PDF dataset,
  self-contained in a single folder:
  - `[slug]-page-[n].png` — the converted pages.
  - `[timestamp]_[dataset]-OCR.csv` — OCR text per page (Surya OCR).
  - `[timestamp]_[dataset]-SUMMARIES.csv` — per-page summary (local Gemma 3).

Keeping the PNGs and their CSVs in the same timestamped folder means each
`outputs/[timestamp]_[dataset]/` is a complete, portable unit for that run, and
re-running a step against the same folder resumes instead of colliding with a
different run of a similarly-named dataset.

`outputs/` is where all generated PNGs and CSVs go, and none of it is
committed to git — only `outputs/README.md` is tracked; everything else in
that folder is git-ignored.


## Run Files

```bat
uv_setup.bat     :: create/sync local venv, install CUDA torch, validate CUDA
exec_1.bat       :: Step 1 - Convert PDF dataset to PNG pages (resumable)
exec_2.bat       :: Step 2 - OCR PNG pages with Surya OCR (resumable)
exec_3.bat       :: Step 3 - Summarize OCR with local Gemma 3 (resumable)
main.bat         :: interactive menu covering all pipeline steps
```

Each `exec_N.bat` is a standalone, double-clickable entry point for one
pipeline step: it bootstraps the env via `uv_setup.bat`, then runs the
matching script under `scripts/`. `main.bat` still covers the full menu,
including steps that don't have an `exec_N.bat` yet.

All operational batch and Python scripts are now under `scripts/`.


## Ideas: more pre-training transformations for LLaVA LoRA training data

Today this repo produces exactly one training pair per page: OCR text →
summary, and the fine-tuning side trains LLaVA *text-only* — the PNG image
is extracted but never actually fed to the vision tower. That leaves real
headroom, using data this repo already produces or could produce with one
more step:

- **Use the image the model already has a vision tower for.** Replace or
  augment the text-only pairs with `(page image, instruction) → summary`
  pairs so LoRA training actually exercises LLaVA's vision encoder instead
  of discarding the PNG after OCR. This is the most direct way to use what's
  already sitting in `outputs/[timestamp]_[dataset]/`.
- **Image-grounded layout/description pairs.** Generate a second target per
  page describing visual structure — tables, stamps, redaction blocks,
  handwritten annotations, letterhead — as a `(image) → layout description`
  pair, teaching the model to reason about page structure, not just prose
  content.
- **OCR-noise correction pairs.** The pipeline deliberately preserves OCR
  garbling (`(newline)` markers, misreads) as training signal for the
  summarizer; the inverse task — noisy OCR text → cleaned text — is a cheap
  auxiliary pair to add from the same CSVs, with no new extraction step.
- **Structured extraction pairs.** Classification markings (CONFIDENTIAL,
  SECRET), dates, document type (cable/memo/report), sender/recipient — a
  lightweight regex or LLM pass over the existing OCR CSV could produce
  `(OCR text) → {field: value, ...}` pairs, giving the LoRA a structured
  extraction skill alongside free-text summarization.
- **Synthetic QA pairs.** Gemma 3 is already in the pipeline generating
  summaries; the same call could also produce 2-3 question/answer pairs per
  page ("who is mentioned", "what date", "what's being requested"),
  expanding the adapter from pure summarization into instruction-following
  QA over documents.
- **Document-level (multi-page) summaries.** Several source PDFs run to
  hundreds of pages; today's granularity is strictly per-page. Rolling
  per-page summaries up into a document-level summary (or feeding several
  consecutive pages' OCR text as one longer-context training example) would
  add a long-context summarization signal the current data doesn't cover.
- **Visual region detection.** The pipeline used to run YOLO object
  detection over the PNGs (since removed) — reviving a lighter version of
  that, scoped to document-relevant classes (stamp, table, photo, signature,
  redaction bar) rather than general objects, would give bounding-box
  training signal to pair with the images.

None of this is implemented yet — this is a list of directions to evaluate,
not a roadmap.


## License

Licensed under the [MIT License](./LICENSE)