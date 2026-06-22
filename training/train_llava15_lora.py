from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path

# Keep all Hugging Face artifacts inside training/
_TRAINING_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(_TRAINING_DIR / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_TRAINING_DIR / "hf_cache" / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_TRAINING_DIR / "hf_cache" / "datasets"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from PIL import Image
from PIL import ImageDraw
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class ProgressCallback(TrainerCallback):
    """Readable progress logs with elapsed time and VRAM."""

    def __init__(self, total_steps: int, num_epochs: float) -> None:
        self.total_steps = total_steps
        self.num_epochs = num_epochs
        self._t0: float = 0.0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw) -> None:
        import time

        self._t0 = time.time()
        vram = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        print("\n" + "=" * 70)
        print(f"  Training started  |  steps={self.total_steps}  epochs={self.num_epochs}")
        print(f"  VRAM reserved at start: {vram:.1f} GB")
        print("=" * 70 + "\n")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict | None = None, **kw) -> None:
        import time

        if logs is None or "loss" not in logs:
            return

        elapsed = time.time() - self._t0
        step = state.global_step
        frac = step / max(self.total_steps, 1)
        eta = (elapsed / max(frac, 1e-6)) * (1 - frac) if frac > 0 else 0
        vram = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0

        epoch_str = f"{logs.get('epoch', 0):.3f}/{self.num_epochs}"
        loss_str = f"{logs['loss']:.4f}"
        grad_str = f"{logs.get('grad_norm', 'n/a')}"
        lr_str = f"{logs.get('learning_rate', 0):.2e}"
        elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"
        eta_str = f"{int(eta // 60)}m{int(eta % 60):02d}s"

        print(
            f"  step {step:>4}/{self.total_steps}"
            f"  epoch {epoch_str}"
            f"  loss {loss_str}"
            f"  grad {grad_str}"
            f"  lr {lr_str}"
            f"  vram {vram:.1f}GB"
            f"  elapsed {elapsed_str}"
            f"  ETA {eta_str}"
        )

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict | None = None, **kw) -> None:
        import time

        elapsed = time.time() - self._t0
        if metrics:
            eloss = metrics.get("eval_loss", "n/a")
            print(f"\n  [eval]  epoch {state.epoch:.3f}  eval_loss {eloss}  elapsed {int(elapsed // 60)}m{int(elapsed % 60):02d}s\n")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw) -> None:
        import time

        elapsed = time.time() - self._t0
        print("\n" + "=" * 70)
        print(f"  Training finished  |  total {int(elapsed // 60)}m{int(elapsed % 60):02d}s")
        print("=" * 70 + "\n")


class CheckpointTrackerCallback(TrainerCallback):
    """Writes small tracker files so training can always be resumed safely."""

    def __init__(self, output_dir: Path, launch_cmd: str):
        self.output_dir = output_dir
        self.launch_cmd = launch_cmd

    def _write_tracking_files(self, state: TrainerState) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        latest_ckpt = _find_last_checkpoint(self.output_dir)
        latest_ckpt_path = str(latest_ckpt) if latest_ckpt else ""

        latest_ckpt_file = self.output_dir / "latest_checkpoint.txt"
        latest_ckpt_file.write_text(latest_ckpt_path + "\n", encoding="utf-8")

        resume_cmd_file = self.output_dir / "resume_command.txt"
        if latest_ckpt:
            resume_cmd = (
                "../.venv/Scripts/python.exe train_llava15_lora.py "
                f"--output-dir {self.output_dir} --resume-from-checkpoint \"{latest_ckpt}\""
            )
        else:
            resume_cmd = (
                "../.venv/Scripts/python.exe train_llava15_lora.py "
                f"--output-dir {self.output_dir} --resume-from-checkpoint last"
            )
        resume_cmd_file.write_text(resume_cmd + "\n", encoding="utf-8")

        tracker = {
            "updated_unix": int(time.time()),
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
            "is_world_process_zero": state.is_world_process_zero,
            "latest_checkpoint": latest_ckpt_path,
            "launch_command": self.launch_cmd,
        }
        tracker_file = self.output_dir / "training_tracker.json"
        tracker_file.write_text(json.dumps(tracker, indent=2), encoding="utf-8")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw) -> None:
        self._write_tracking_files(state)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw) -> None:
        self._write_tracking_files(state)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw) -> None:
        self._write_tracking_files(state)


class JsonlDataset(Dataset):
    def __init__(self, records: list[dict[str, str]]):
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
        prompt_lengths: list[int] = []

        for item in features:
            image = Image.open(item["image_path"]).convert("RGB")
            images.append(image)

            user_prefix = f"USER: <image>\n{item['prompt']} ASSISTANT: "
            full_text = user_prefix + item["summary"] + self.processor.tokenizer.eos_token
            texts.append(full_text)

            prefix_ids = self.processor.tokenizer(user_prefix, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prefix_ids))

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

        for i, plen in enumerate(prompt_lengths):
            seq_len = int(batch["attention_mask"][i].sum().item())
            if seq_len <= 0:
                continue

            cutoff = min(plen, seq_len - 1)
            labels[i, :cutoff] = -100

            # Keep at least one target token so eval loss cannot become NaN.
            if torch.all(labels[i, :seq_len] == -100):
                last_idx = seq_len - 1
                labels[i, last_idx] = batch["input_ids"][i, last_idx]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA 1.5 7B full LoRA training")
    parser.add_argument("--dataset-jsonl", type=Path, default=_TRAINING_DIR / "data" / "llava15_train.jsonl", help="Path to JSONL created by build_llava15_dataset.py")
    parser.add_argument("--output-dir", type=Path, default=_TRAINING_DIR / "runs" / "llava15_lora", help="Where checkpoints and final adapter are saved")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="Hugging Face model id")
    parser.add_argument("--num-epochs", type=float, default=2.0, help="Number of epochs for full run")
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
    parser.add_argument("--save-strategy", choices=["epoch", "steps"], default="epoch", help="Checkpoint schedule")
    parser.add_argument("--save-steps", type=int, default=250, help="Checkpoint interval when --save-strategy steps")
    parser.add_argument("--save-total-limit", type=int, default=4, help="Max number of checkpoints to keep")
    parser.add_argument("--eval-strategy", choices=["epoch", "steps"], default="epoch", help="Evaluation schedule")
    parser.add_argument("--eval-steps", type=int, default=250, help="Eval interval when --eval-strategy steps")
    parser.add_argument("--resume-from-checkpoint", default="", help="Checkpoint path to resume from, or 'last' for newest checkpoint in output dir")
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
            if not item.get("image_path") or not item.get("prompt") or not item.get("summary"):
                continue
            records.append(item)
    return records


def _find_last_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    checkpoints = []
    for p in output_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if m:
            checkpoints.append((int(m.group(1)), p))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda t: t[0])
    return checkpoints[-1][1]


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metric_series(log_history: list[dict]) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {
        "train_loss": [],
        "eval_loss": [],
        "grad_norm": [],
        "learning_rate": [],
    }

    fallback_step = 0.0
    for row in log_history:
        step = _to_float(row.get("step"))
        if step is None:
            step = _to_float(row.get("global_step"))
        if step is None:
            fallback_step += 1.0
            step = fallback_step

        loss = _to_float(row.get("loss"))
        if loss is not None and _to_float(row.get("eval_loss")) is None:
            series["train_loss"].append((step, loss))

        eval_loss = _to_float(row.get("eval_loss"))
        if eval_loss is not None:
            series["eval_loss"].append((step, eval_loss))

        grad_norm = _to_float(row.get("grad_norm"))
        if grad_norm is not None:
            series["grad_norm"].append((step, grad_norm))

        learning_rate = _to_float(row.get("learning_rate"))
        if learning_rate is not None:
            series["learning_rate"].append((step, learning_rate))

    return series


def _draw_line_chart(
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    lines: dict[str, list[tuple[float, float]]],
) -> bool:
    active = {k: v for k, v in lines.items() if v}
    if not active:
        return False

    width, height = 1280, 720
    left, right, top, bottom = 90, 40, 70, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x = [x for pts in active.values() for x, _ in pts]
    all_y = [y for pts in active.values() for _, y in pts]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        pad = 1.0 if y_min == 0 else abs(y_min) * 0.1
        y_min -= pad
        y_max += pad

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
    ]

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    for i in range(6):
        yy = top + i * (plot_h / 5)
        draw.line([(left, yy), (left + plot_w, yy)], fill=(230, 230, 230), width=1)
        y_val = y_max - i * (y_max - y_min) / 5
        draw.text((10, yy - 8), f"{y_val:.4g}", fill=(80, 80, 80))

    for i in range(6):
        xx = left + i * (plot_w / 5)
        draw.line([(xx, top), (xx, top + plot_h)], fill=(240, 240, 240), width=1)
        x_val = x_min + i * (x_max - x_min) / 5
        draw.text((xx - 15, top + plot_h + 12), f"{x_val:.4g}", fill=(80, 80, 80))

    draw.line([(left, top), (left, top + plot_h)], fill=(50, 50, 50), width=2)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=(50, 50, 50), width=2)

    for idx, (name, points) in enumerate(active.items()):
        color = palette[idx % len(palette)]
        scaled = [(sx(x), sy(y)) for x, y in points]
        if len(scaled) == 1:
            px, py = scaled[0]
            draw.ellipse([(px - 3, py - 3), (px + 3, py + 3)], fill=color)
        else:
            draw.line(scaled, fill=color, width=3)

        ly = top + 8 + idx * 20
        lx = left + plot_w - 220
        draw.rectangle([(lx, ly), (lx + 14, ly + 10)], fill=color)
        draw.text((lx + 20, ly - 2), name, fill=(20, 20, 20))

    draw.text((left, 18), title, fill=(20, 20, 20))
    draw.text((left + plot_w // 2 - 20, height - 36), x_label, fill=(30, 30, 30))
    draw.text((12, top - 30), y_label, fill=(30, 30, 30))

    img.save(out_path)
    return True


def _save_history_csv(output_dir: Path, log_history: list[dict]) -> Path:
    csv_path = output_dir / "training_log_history.csv"
    keys = sorted({k for row in log_history for k in row.keys()})
    if not keys:
        keys = ["step"]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in log_history:
            writer.writerow({k: row.get(k, "") for k in keys})
    return csv_path


def _save_training_graphs(output_dir: Path, trainer: Trainer, train_metrics: dict, eval_metrics: dict) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_history = list(trainer.state.log_history or [])

    (output_dir / "training_log_history.json").write_text(
        json.dumps(log_history, indent=2),
        encoding="utf-8",
    )
    saved: list[Path] = [_save_history_csv(output_dir, log_history)]

    series = _extract_metric_series(log_history)

    loss_png = output_dir / "loss_curve.png"
    if _draw_line_chart(
        out_path=loss_png,
        title="Training and Evaluation Loss",
        x_label="Step",
        y_label="Loss",
        lines={
            "train_loss": series["train_loss"],
            "eval_loss": series["eval_loss"],
        },
    ):
        saved.append(loss_png)

    grad_png = output_dir / "grad_norm_curve.png"
    if _draw_line_chart(
        out_path=grad_png,
        title="Gradient Norm",
        x_label="Step",
        y_label="Grad Norm",
        lines={"grad_norm": series["grad_norm"]},
    ):
        saved.append(grad_png)

    lr_png = output_dir / "learning_rate_curve.png"
    if _draw_line_chart(
        out_path=lr_png,
        title="Learning Rate Schedule",
        x_label="Step",
        y_label="Learning Rate",
        lines={"learning_rate": series["learning_rate"]},
    ):
        saved.append(lr_png)

    metric_summary = {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    metric_json = output_dir / "metrics_summary.json"
    metric_json.write_text(json.dumps(metric_summary, indent=2), encoding="utf-8")
    saved.append(metric_json)
    return saved


def main() -> int:
    args = parse_args()

    data_path = args.dataset_jsonl.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset JSONL not found: {data_path}")

    records = read_jsonl(data_path)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    if len(records) < 10:
        raise RuntimeError("Need at least 10 records for training.")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    n_val = max(1, int(len(records) * args.val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded records: total={len(records)} train={len(train_records)} val={len(val_records)}")
    print(f"Model: {args.model_id}")
    print(f"Output: {output_dir}")
    print(f"HF cache: {os.environ['HF_HOME']}\n")

    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"

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

    train_ds = JsonlDataset(train_records)
    val_ds = JsonlDataset(val_records)
    collator = LlavaCollator(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        disable_tqdm=False,
        log_level="warning",
        load_best_model_at_end=False,
    )

    total_steps = int((len(train_records) / args.batch_size / args.grad_accum) * args.num_epochs)
    print(f"Total optimiser steps: {total_steps}  (batch={args.batch_size} x accum={args.grad_accum} x epochs={args.num_epochs})")

    launch_cmd = " ".join([
        "../.venv/Scripts/python.exe",
        "train_llava15_lora.py",
        f"--dataset-jsonl {data_path}",
        f"--output-dir {output_dir}",
        f"--num-epochs {args.num_epochs}",
        f"--save-strategy {args.save_strategy}",
        f"--eval-strategy {args.eval_strategy}",
    ])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[
            ProgressCallback(total_steps=total_steps, num_epochs=args.num_epochs),
            CheckpointTrackerCallback(output_dir=output_dir, launch_cmd=launch_cmd),
        ],
    )

    resume_checkpoint: str | None = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "last":
            last_ckpt = _find_last_checkpoint(output_dir)
            if last_ckpt is None:
                raise RuntimeError(f"No checkpoint-* directories found in {output_dir}")
            resume_checkpoint = str(last_ckpt)
        else:
            resume_checkpoint = str(Path(args.resume_from_checkpoint).resolve())
        print(f"Resuming from checkpoint: {resume_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_metrics = train_result.metrics
    print("Train metrics:")
    for key in sorted(train_metrics):
        print(f"  {key}: {train_metrics[key]}")

    eval_metrics = trainer.evaluate()
    print("Eval metrics:")
    for key in sorted(eval_metrics):
        print(f"  {key}: {eval_metrics[key]}")

    graph_files = _save_training_graphs(
        output_dir=output_dir,
        trainer=trainer,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
    )
    print("Saved training graphs and logs:")
    for p in graph_files:
        print(f"  {p}")

    summary = {
        "dataset_jsonl": str(data_path),
        "output_dir": str(output_dir),
        "num_epochs": args.num_epochs,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    run_summary_file = output_dir / "run_summary.json"
    run_summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    final_dir = output_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Saved LoRA adapter and processor to: {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
