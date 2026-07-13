# AGENTS.md — pre-training repo

Guidance for AI coding agents (and humans) working in this repository. This file
is the canonical agent guide; `CLAUDE.md` defers to it.

## What this repository is

A local, GPU-first **dataset pre-training pipeline** plus the **trained LoRA
adapter** and its inference server. It turns a PDF corpus into training data and
fine-tunes a vision-language model:

1. Convert PDFs → PNG pages (`scripts/convert_pdf_to_png.*`).
2. OCR the PNGs → `output/*_OCR.csv` (`scripts/ocr_detection_png.py`, [Baidu Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)).
3. YOLO object detection → `output/*_OBJS.csv` (`scripts/object_detection_png.py`, ultralytics).
4. Summarize OCR via Ollama (gemma) → `output/*_SUMMARIES.csv` (the training target).
5. Train a **LLaVA 1.5 7B LoRA** adapter on (OCR text → summary) pairs (`training/`).
6. Serve inference from `deploy/` (FastAPI).

### Role in the larger system

This is **app #2 of a three-app system**, controlled by **metaagent** (the C++
agent controller, repo `vecnode/metaagent`). metaagent does **not** import this
code — it talks to the deployed server over HTTP and can start it remotely:

- Inference: metaagent `POST /api/adapter/summarize` → this server's `POST /api/summarize`.
- Health: metaagent `GET /api/adapter/status` → this server's `/api/health` + `/api/model-info`.
- Launch: metaagent `POST /api/adapter/launch` runs `deploy/deploy.bat` (configured via `METAAGENT_ADAPTER_DIR` + `METAAGENT_ADAPTER_LAUNCH_CMD`).

The third app is `vecnode/media-player-cpp` (openFrameworks player). This repo
has no direct link to it.

## Environment & commands

GPU-first (NVIDIA + CUDA). Uses **uv** for the Python env; torch/torchvision are
pinned to the CUDA 12.8 wheels in `pyproject.toml` — never let `uv sync` pull the
CPU-only wheel (it silently disables GPU inference).

```bat
uv_bootstrap.bat            :: create/sync .venv, install CUDA torch, validate CUDA
main.bat                    :: interactive menu for the pipeline steps (1-5)
```

Pipeline steps are individual `scripts/*.bat` wrappers that call `scripts/*.py`
through the local `.venv`. Run them via `main.bat` or directly with
`.venv\Scripts\python.exe scripts\<name>.py`.

### Inference server (`deploy/`)

```bat
deploy\deploy.bat [host] [port]      :: bootstrap + serve (defaults 127.0.0.1 8008)
```

or directly: `..\.venv\Scripts\python.exe deploy\app.py --port 8008`.

| Method | Endpoint | Purpose |
| ------ | -------- | ------- |
| GET  | `/`                | Web UI: dataset browser + 3-way (OCR / reference / live) comparison |
| GET  | `/api/health`      | `{status, device}` |
| GET  | `/api/model-info`  | params, dtype, device, VRAM, mode (adapter vs fused) |
| GET  | `/api/weights`     | per-tensor shape/dtype/mean/std/L2 (`?filter=&limit=`) |
| GET  | `/api/rows`, `/api/row/{i}`, `/api/image/{i}` | dataset browsing |
| POST | `/api/summarize`   | `{"ocr_text": "...", "max_new_tokens"?: n}` → `{summary, elapsed_ms, input_chars}` |

The model loads once at startup; GPU generation is serialized with a lock.
Prefer the fused model: `python merge_adapter.py --out merged_model` writes a
self-contained ~14 GB model that `app.py`/`infer.py` auto-detect and prefer.

### Two different weight sources — do not confuse them

| Weight | What it is | Where it comes from | Where it lives |
| ------ | ---------- | -------------------- | --------------- |
| **LoRA adapter** (~40 MB) | The model **you trained** on this machine (`training/train_llava15_lora.py`) | Never downloaded — it must already exist locally, or be copied in from wherever training ran | `training/runs/llava15_lora/final_adapter/` (git-ignored) |
| **Base model** (~14 GB) | Public, untrained `llava-hf/llava-1.5-7b-hf` | Fetched from Hugging Face Hub the first time the adapter server starts, via `ensure_base_model_cached()` in `deploy/infer.py` (`huggingface_hub.snapshot_download`) | `training/hf_cache/` (git-ignored) |

`ensure_base_model_cached()` runs before `LlavaForConditionalGeneration.from_pretrained(base_id, ...)` in `Summarizer.__init__` (adapter-mode only — a fused model doesn't need the public base weights at all). It logs what it's doing and raises a clear `RuntimeError` (not a raw traceback) on network/disk failure. Subsequent runs are a fast cache check — `snapshot_download` compares hub metadata before deciding what to fetch, so nothing re-downloads once cached.

If `training/runs/llava15_lora/final_adapter/` is missing, `Summarizer` raises `FileNotFoundError` immediately — that's a **local artifact problem**, never a network/download problem. Copy the adapter in (or point `--adapter-dir` at it) rather than expecting a fetch.

## Conventions & guardrails

- **uv only** for deps; respect the pinned CUDA index in `pyproject.toml`. Don't
  add deps that drag in a CPU torch.
- **Batch scripts** follow the existing style: `@echo off`,
  `setlocal EnableExtensions`, resolve `SCRIPT_DIR`/`PROJECT_DIR` from `%~dp0`,
  call `uv_bootstrap.bat` before doing work, `exit /b 1` on failure. Keep the
  `vecnode 2026` copyright header.
- **Inference must match training:** the `INSTRUCTION` prompt, `max_length=2048`,
  and head+tail truncation in `infer.py` mirror the trainer. Changing them
  degrades quality — keep them in sync if you touch one.
- **OCR input format is load-bearing:** feed OCR exactly as the pipeline emits it
  (keep `(newline)` markers and garbled spellings — that's the training
  distribution). Greedy decoding (`do_sample=False`) makes output deterministic.
- **Never commit large artifacts** (already in `.gitignore`): `*.pdf`, `*.png`,
  `*.csv`, `*.pt`, `*.safetensors`, `.venv/`, `training/runs/`, `training/hf_cache/`,
  `deploy/merged_model/`. The adapter (~40 MB) ships by copying it alongside the
  code (e.g. into a distribution); the base model (~14 GB) is never shipped —
  it's fetched from Hugging Face Hub on first run (see "Two different weight
  sources" above).
- **Text-only at inference:** despite the LLaVA base, the page image is not used
  for generation — you don't need the PNGs to run `/api/summarize`.

See `deploy/README.md` for the full deployment + weight-inspection guide.
