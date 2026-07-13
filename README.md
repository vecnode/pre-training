# Dataset Pre-Training Workspace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Local, GPU-first pipeline that turns a PDF corpus into training data and fine-tunes a LLaVA 1.5 7B LoRA adapter (OCR text → summary), served over a FastAPI inference endpoint.

- Convert a PDF dataset into PNG pages
- OCR PNG pages with Baidu Unlimited-OCR
- Detect objects on PNG pages with YOLO (ultralytics)
- Summarize OCR text via the Ollama API (gemma)
- Fine-tune a LLaVA 1.5 7B LoRA adapter on (OCR text → summary) pairs
- Serve the adapter locally through a FastAPI inference server (`deploy/`)

## Folder structure

- `DATASET/`: Input PDFs (example: `Release_1/`).
- `outputs/[timestamp]_[dataset]/`: PNG pages from one conversion run, named `[slug]-page-[n].png`.
- `output/`: Generated CSVs (`DATASET_1_OCR.csv`, `DATASET_OBJS.csv`, `DATASET_SUMMARIES.csv`).


## Run Files

```bat
uv_setup.bat     :: create/sync local venv, install CUDA torch, validate CUDA
exec_1.bat       :: Step 1 - Convert PDF dataset to PNG pages
exec_2.bat       :: Step 2 - OCR PNG pages with Baidu Unlimited-OCR
main.bat         :: interactive menu covering all pipeline steps
```

Each `exec_N.bat` is a standalone, double-clickable entry point for one
pipeline step: it bootstraps the env via `uv_setup.bat`, then runs the
matching script under `scripts/`. `main.bat` still covers the full menu,
including steps that don't have an `exec_N.bat` yet.

All operational batch and Python scripts are now under `scripts/`.


## License

Licensed under the [MIT License](./LICENSE)