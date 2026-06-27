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
- **Env:** uv only, GPU-first. `uv_bootstrap.bat` sets up `.venv` with CUDA
  torch; `main.bat` runs the pipeline menu; `deploy\deploy.bat [host] [port]`
  serves the model (default `:8008`).
- **Don't** let `uv sync` resolve CPU-only torch (respect the pinned CUDA index
  in `pyproject.toml`), change `infer.py`'s `INSTRUCTION`/truncation without
  matching the trainer, or commit weights/CSVs/PDFs/PNGs (all git-ignored).
- Deployment details: `deploy/README.md`.
