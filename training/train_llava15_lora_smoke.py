from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments


class JsonlDataset(Dataset):
    def __init__(self, path: Path, records: list[dict[str, str]]):
        self.path = path
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.records[index]


class LlavaCollator:
    def __init__(self, processor: AutoProcessor, max_length: int = 1024):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        images = []
        texts = []

        for item in features:
            image = Image.open(item["image_path"]).convert("RGB")
            images.append(image)

            # LLaVA 1.5 chat-style instruction with image token.
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": item["prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["target"]}],
                },
            ]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=False)
            texts.append(text)

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA 1.5 7B LoRA smoke training (1 epoch)")
    parser.add_argument("--dataset-jsonl", type=Path, default=Path("data/llava15_train.jsonl"), help="Path to JSONL created by build_llava15_dataset.py")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/llava15_lora_smoke"), help="Where to save LoRA adapter")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="Hugging Face model id")
    parser.add_argument("--num-epochs", type=float, default=1.0, help="Epoch count for smoke run")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick checks (0 = all)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=1024, help="Max tokenized sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                continue
            if not item.get("image_path") or not item.get("prompt") or not item.get("target"):
                continue
            records.append(item)
    return records


def main() -> int:
    args = parse_args()

    data_path = args.dataset_jsonl.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset JSONL not found: {data_path}")

    records = read_jsonl(data_path)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    if len(records) < 10:
        raise RuntimeError("Need at least 10 records for a useful smoke run.")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    n_val = max(1, int(len(records) * args.val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    print(f"Loaded records: total={len(records)} train={len(train_records)} val={len(val_records)}")
    print(f"Model: {args.model_id}")

    processor = AutoProcessor.from_pretrained(args.model_id)

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = JsonlDataset(data_path, train_records)
    val_ds = JsonlDataset(data_path, val_records)
    collator = LlavaCollator(processor=processor, max_length=args.max_length)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=5,
        eval_steps=20,
        evaluation_strategy="steps",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    print("Train metrics:")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")

    eval_metrics = trainer.evaluate()
    print("Eval metrics:")
    for key in sorted(eval_metrics):
        print(f"  {key}: {eval_metrics[key]}")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved LoRA adapter and processor to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
