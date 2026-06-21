# Dataset Pre-Training Workspace

Structured first-stage pre-training pipeline:
1. Convert PDF dataset to PNG pages.
2. Run OCR over PNG pages.
3. Run YOLO object detection over PNG pages.

## Folder structure

- `DATASET/`: Input PDFs (example: `Release_1/`).
- `DATASET_PNG/`: Generated PNG pages (example: `Release_1_PNG/`).
- `output/`: Generated markdown outputs (`DATASET_OCR.md`, `DATASET_YOLO.md`).


## Run Files

```bat
uv_bootstrap.bat
main.bat
```

All operational batch and Python scripts are now under `scripts/`.

## License

Licensed under the [MIT License](./LICENSE)