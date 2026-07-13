# CLAUDE.md

Guidance for Claude Code in this repository. The full agent/contributor guide is
in `AGENTS.md` — read it first.

@AGENTS.md

## Claude-specific quick reference

- This repo is a **local GPU pipeline** that turns a PDF corpus into training
  data and fine-tunes a **LLaVA 1.5 7B LoRA** adapter (OCR text → summary), plus
  a FastAPI inference server in `deploy/`.
- It is **app #2** of a three-app system, started/queried over HTTP by the
  **metaagent** C++ controller (`vecnode/metaagent`). metaagent never imports
  this code — keep the `deploy/` HTTP contract (`/api/summarize`, `/api/health`,
  `/api/model-info`) and `deploy/deploy.bat` stable for it.
- **Env:** uv only, GPU-first. `uv_setup.bat` sets up `.venv` with CUDA torch;
  each pipeline step also has a standalone `exec_N.bat` at the project root
  (currently `exec_1.bat` = convert PDFs, `exec_2.bat` = OCR) that bootstraps
  the env itself and can be double-clicked directly; `main.bat` runs the
  interactive menu for all steps; `deploy\deploy.bat [host] [port]` serves the
  model (default `:8008`).
- **Two weight sources, don't confuse them:** the **LoRA adapter** (~40 MB,
  `training/runs/llava15_lora/final_adapter/`) is trained locally and only ever
  copied, never downloaded — if it's missing that's a local-artifact problem.
  The **base model** (~14 GB, `llava-hf/llava-1.5-7b-hf`) is public and fetched
  automatically from Hugging Face Hub on first server start via
  `ensure_base_model_cached()` in `deploy/infer.py`, into `training/hf_cache/`.
  Both dirs are git-ignored.
- **Don't** let `uv sync` resolve CPU-only torch (respect the pinned CUDA index
  in `pyproject.toml`), change `infer.py`'s `INSTRUCTION`/truncation without
  matching the trainer, or commit weights/CSVs/PDFs/PNGs (all git-ignored).
- Deployment details: `deploy/README.md`.
