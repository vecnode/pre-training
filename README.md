# Dataset Pre-Training Workspace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Local, GPU-first pipeline that turns a PDF corpus into training data and fine-tunes a LLaVA 1.5 7B LoRA adapter (OCR text → summary), served over a FastAPI inference endpoint.

- Convert a PDF dataset into PNG pages
- OCR PNG pages with Surya OCR
- Summarize OCR text with a local Gemma 3 model (no Ollama/HTTP hop)
- Fine-tune a LLaVA 1.5 7B LoRA adapter on (OCR text → summary) pairs
- Serve the adapter locally through a FastAPI inference server (`deploy/`)

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


## License

Licensed under the [MIT License](./LICENSE)