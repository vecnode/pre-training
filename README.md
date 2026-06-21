# Dataset Pre-Training Workspace

Structured first-stage pre-training pipeline:
1. Convert PDF dataset to PNG pages.
2. Run OCR over PNG pages.
3. Run YOLO object detection over PNG pages.
4. Summarize OCR CSV content via Ollama HTTP API.

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


## Dataset Output Types

- Default write mode (`OCR` and `OBJ RECOG`): append/resume by image section (`--no-resume` disables resume and rewrites processing flow).
- OCR output file: `output/DATASET_1_OCR.csv`.
- OCR row structure: one image per row with columns `image`, `full_path`, `status`, `reason`, `method`, `confidence`, `text`.
- OCR write mode: append/resume by image row (`--no-resume` disables resume and rewrites processing flow).
- `status`/`reason` record whether OCR passed and which preprocessing variant won, or why it failed.
- Object recognition output file: `output/DATASET_OBJS.csv`.
- OBJ row structure: one object per row with `image`, `full_path`, `object_id`, `label`, `confidence`, `x`, `y`, `width`, `height`.
- Summaries output file: `output/DATASET_SUMMARIES.csv` generated from `DATASET_OCR.csv` via Ollama HTTP API.

## License

Licensed under the [MIT License](./LICENSE)