# LLaVA 1.5 7B LoRA training

Model: `llava-hf/llava-1.5-7b-hf`  
Data: `../output/` CSVs + `../Release_1_PNG/` images  
All downloads, cache, checkpoints, and adapters stay in `training/`.

## 1) Install deps (once)

```bash
uv pip install --python ../.venv/Scripts/python.exe transformers peft accelerate sentencepiece
```

## 2) Build JSONL dataset

```bash
../.venv/Scripts/python.exe build_llava15_dataset.py --root ..
```

Output: `data/llava15_train.jsonl` with fields: `image_path`, `prompt`, `summary`.

## 3) Evaluated smoke test (must pass before full run)

```bash
../.venv/Scripts/python.exe train_llava15_lora_smoke.py --max-samples 256
```

Expected: train loss decreases and `eval_loss` is numeric (not `nan`).

## 4) Full training with checkpoints

```bash
../.venv/Scripts/python.exe train_llava15_lora.py --num-epochs 2
```

Continue:

```bash
../.venv/Scripts/python.exe train_llava15_lora.py --output-dir runs/llava15_lora --resume-from-checkpoint last --num-epochs 2
```


One-epoch run (recommended first full pass):

```bash
../.venv/Scripts/python.exe train_llava15_lora.py --num-epochs 1 --output-dir runs/llava15_lora
```

Defaults: `save_strategy=epoch`, `eval_strategy=epoch`, and rolling checkpoint retention.

Useful variants:

```bash
# Save/eval by steps instead of epoch
../.venv/Scripts/python.exe train_llava15_lora.py --save-strategy steps --save-steps 250 --eval-strategy steps --eval-steps 250

# Resume from latest checkpoint in output dir
../.venv/Scripts/python.exe train_llava15_lora.py --resume-from-checkpoint last

# Resume from explicit checkpoint path
../.venv/Scripts/python.exe train_llava15_lora.py --resume-from-checkpoint runs/llava15_lora/checkpoint-500
```

## Default training setup

- LoRA on `q_proj` and `v_proj` only (~0.14% trainable params)
- Batch size `1`, grad accumulation `4`
- Learning rate `2e-4`
- FP16 enabled when CUDA is available
- Max sequence length `1024`

## Output layout

```text
training/
  data/llava15_train.jsonl
  hf_cache/
  runs/llava15_lora/checkpoint-*/
  runs/llava15_lora/latest_checkpoint.txt
  runs/llava15_lora/resume_command.txt
  runs/llava15_lora/training_tracker.json
  runs/llava15_lora/run_summary.json
  runs/llava15_lora/final_adapter/
```

Resume after stopping:

```bash
../.venv/Scripts/python.exe train_llava15_lora.py --output-dir runs/llava15_lora --resume-from-checkpoint last
```
