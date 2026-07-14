# AGENTS.md — pre-training repo

Guidance for AI coding agents (and humans) working in this repository. This file
is the canonical agent guide; `CLAUDE.md` defers to it.

## What this repository is

A local, GPU-first **dataset pre-training pipeline**. It turns a PDF corpus
into training data:

1. Convert PDFs → PNG pages (`scripts/convert_pdf_to_png.*`), written to
   `outputs/[timestamp]_[dataset]/`.
2. OCR the PNGs → `[timestamp]_[dataset]-OCR.csv` in that same folder
   (`scripts/ocr_detection_png.py`, [Surya OCR](https://github.com/datalab-to/surya)).
3. Summarize OCR → `[timestamp]_[dataset]-SUMMARIES.csv` in that same folder
   (`scripts/summarize_ocr_gemma.py`, a local Gemma 3 model — no Ollama/HTTP
   hop; default `unsloth/gemma-3-4b-it` is an ungated mirror, no `HF_TOKEN`
   needed) — this is the training target.

Fine-tuning (LLaVA LoRA training) and serving live in the separate
[`fine-tuning`](https://github.com/vecnode/fine-tuning) repo, which trains on
the OCR/SUMMARIES CSVs this repo produces — this repo does not train or serve
a model itself.

## Environment & commands

GPU-first (NVIDIA + CUDA). Uses **uv** for the Python env; torch/torchvision are
pinned to the CUDA 12.8 wheels in `pyproject.toml` — never let `uv sync` pull the
CPU-only wheel (it silently disables GPU inference).

```bat
uv_setup.bat                :: create/sync .venv, install CUDA torch, validate CUDA
exec_1.bat                  :: Step 1 - Convert PDF dataset to PNG pages
exec_2.bat                  :: Step 2 - OCR PNG pages with Surya OCR
exec_3.bat                  :: Step 3 - Summarize OCR with local Gemma 3
main.bat                    :: interactive menu for all pipeline steps
```

Each `exec_N.bat` at the project root bootstraps the env via `uv_setup.bat`
then calls the matching `scripts/*.bat` wrapper — a one-to-one, double-click
entry point per pipeline step. `main.bat` remains as a menu covering every
step, including ones that don't have an `exec_N.bat` yet.

Pipeline steps are individual `scripts/*.bat` wrappers that call `scripts/*.py`
through the local `.venv`. Run them via `exec_N.bat`, `main.bat`, or directly with
`.venv\Scripts\python.exe scripts\<name>.py`.

## Conventions & guardrails

- **uv only** for deps; respect the pinned CUDA index in `pyproject.toml`. Don't
  add deps that drag in a CPU torch.
- **Batch scripts** follow the existing style: `@echo off`,
  `setlocal EnableExtensions`, resolve `SCRIPT_DIR`/`PROJECT_DIR` from `%~dp0`,
  call `uv_setup.bat` before doing work, `exit /b 1` on failure. Keep the
  `vecnode 2026` copyright header.
- **OCR input format is load-bearing:** feed OCR exactly as the pipeline emits it
  (keep `(newline)` markers and garbled spellings — that's the training
  distribution consumed downstream by the fine-tuning repo).
- **Never commit large artifacts** (already in `.gitignore`): `*.pdf`, `*.png`,
  `*.csv`, `*.pt`, `*.safetensors`, `.venv/`.
