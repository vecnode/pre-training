# LLaVA 1.5 7B LoRA test (first epoch)

This folder contains a first-start training smoke test for:

- model: `llava-hf/llava-1.5-7b-hf`
- data source: `../output/Release_1_OCR.csv` + `../output/Release_1_SUMMARIES.csv`
- images: paths read from CSV (root-relative, e.g. `Release_1_PNG\\...`)

The commands below are intended to run from inside `training/` and reuse the existing root virtual environment (`../.venv`).

## 1) Install training-only packages in existing .venv

```bash
uv pip install --python ../.venv/Scripts/python.exe transformers peft accelerate datasets sentencepiece
```

## 2) Build JSONL training file from CSV files

Quick proof set (faster):

```bash
uv run --python ../.venv/Scripts/python.exe python build_llava15_dataset.py --root .. --max-samples 512
```

Full set:

```bash
uv run --python ../.venv/Scripts/python.exe python build_llava15_dataset.py --root ..
```

Output file is created at:

- `training/data/llava15_train.jsonl`

## 3) Run first-epoch LoRA training

Quick run:

```bash
uv run --python ../.venv/Scripts/python.exe python train_llava15_lora_smoke.py --dataset-jsonl data/llava15_train.jsonl --output-dir runs/llava15_lora_smoke --num-epochs 1 --max-samples 256
```

Larger run:

```bash
uv run --python ../.venv/Scripts/python.exe python train_llava15_lora_smoke.py --dataset-jsonl data/llava15_train.jsonl --output-dir runs/llava15_lora_smoke --num-epochs 1
```

The trainer prints train and eval metrics (including loss) so you can confirm the model is training.

## Files

- `build_llava15_dataset.py`: joins OCR + summaries CSVs and resolves image paths
- `train_llava15_lora_smoke.py`: runs LoRA smoke training for one epoch and saves adapter
