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
- `DATASET_PNG/`: Generated PNG pages (example: `Release_1_PNG/`).
- `output/`: Generated outputs (`DATASET_1_OCR.csv`, `DATASET_OBJS.csv`, `DATASET_SUMMARIES.csv`).


## Run Files

```bat
uv_bootstrap.bat
main.bat
```

All operational batch and Python scripts are now under `scripts/`.


## License

Licensed under the [MIT License](./LICENSE)