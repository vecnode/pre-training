# LLaVA 1.5 7B LoRA training

Model: `llava-hf/llava-1.5-7b-hf`  
Data: `../output/` CSVs + `../<dataset>_PNG/` images  
All downloads, cache, checkpoints, and adapters stay in `training/`.

### 1) Install deps (once)

```bash
../uv_bootstrap.bat
```

This installs project dependencies (including `transformers`, `peft`, `accelerate`, `sentencepiece`) through `uv sync`.

### 2) Build JSONL dataset

```bash
../.venv/Scripts/python.exe build_llava15_dataset.py --root ..
```

Output: `data/llava15_train.jsonl` with fields: `image_path`, `prompt`, `summary`.

### 3) Evaluated test (must pass before full run)

```bash
../.venv/Scripts/python.exe train_llava15_lora_smoke.py --max-samples 256
```

Expected: train loss decreases and `eval_loss` is numeric (not `nan`).

### Training View

The code trains a frozen LLaVA 1.5 base model with LoRA adapters attached to the attention projections (`q_proj` and `v_proj`). Only the adapter weights update during backpropagation; the base model weights stay fixed.

```mermaid
flowchart TB
	subgraph Input
		I1[Image page]
		T1[OCR prompt]
	end

	subgraph FrozenBase["Frozen LLaVA 1.5 base model"]
		E1[Vision encoder]
		C1[Multimodal projector]
		B1[Transformer block]
		A1[Self-attention]
		F1[MLP / feed-forward]
		N1[LayerNorm]
	end

	subgraph Trainable["Trainable LoRA adapters"]
		L1[LoRA on q_proj]
		L2[LoRA on v_proj]
	end

	O1[Assistant summary]
	G1[Loss]
	U1[Backprop updates adapter weights only]

	I1 --> E1 --> C1 --> B1 --> A1 --> F1 --> O1
	T1 --> B1
	A1 --- L1
	A1 --- L2
	L1 -. low-rank update .-> A1
	L2 -. low-rank update .-> A1
	O1 --> G1 --> U1
	U1 -. frozen .-> FrozenBase
	U1 --> Trainable
```

### 4) Full training with checkpoints

```bash
../.venv/Scripts/python.exe train_llava15_lora.py --num-epochs 2
```

Resume for one more epoch:

```bash
../.venv/Scripts/python.exe train_llava15_lora.py --output-dir runs/llava15_lora --resume-from-checkpoint last --extra-epochs 1
```

Fresh one-epoch run:

```bash
../.venv/Scripts/python.exe train_llava15_lora.py --num-epochs 1 --output-dir runs/llava15_lora
```

Checkpoint files are written under `runs/llava15_lora/`, including `latest_checkpoint.txt`, `resume_command.txt`, and `final_adapter/`.
