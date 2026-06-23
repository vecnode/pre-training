from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# Keep all Hugging Face artifacts inside training/
_TRAINING_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(_TRAINING_DIR / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_TRAINING_DIR / "hf_cache" / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_TRAINING_DIR / "hf_cache" / "datasets"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate summaries with trained LLaVA LoRA adapter")
    parser.add_argument("--adapter-dir", type=Path, default=_TRAINING_DIR / "runs" / "llava15_lora" / "final_adapter", help="Path to trained LoRA adapter directory")
    parser.add_argument("--model-id", default="", help="Base model id (auto-detected from adapter config when omitted)")
    parser.add_argument("--ocr-csv", type=Path, required=True, help="OCR CSV input (e.g., ../output/Release_1_OCR.csv)")
    parser.add_argument("--reference-csv", type=Path, default=None, help="Optional reference summaries CSV (e.g., ../output/Release_1_SUMMARIES.csv)")
    parser.add_argument("--image-root", type=Path, default=_TRAINING_DIR.parent, help="Root folder used to resolve relative image paths")
    parser.add_argument("--out-csv", type=Path, default=_TRAINING_DIR / "runs" / "llava15_lora" / "generated_epoch3.csv", help="Output CSV with generated predictions")
    parser.add_argument("--out-metrics", type=Path, default=None, help="Optional output metrics JSON path")
    parser.add_argument("--max-rows", type=int, default=100, help="Maximum rows to generate (0 = all rows)")
    parser.add_argument("--max-ocr-chars", type=int, default=4000, help="Max OCR chars injected into the prompt")
    parser.add_argument("--max-new-tokens", type=int, default=180, help="Max generated tokens per sample")
    return parser.parse_args()


def normalize_image_key(value: str) -> str:
    key = (value or "").strip().replace("\\", "/")
    while "//" in key:
        key = key.replace("//", "/")
    return key


def resolve_image_path(image_root: Path, image_key: str, full_path: str) -> Path | None:
    if full_path:
        p = Path(full_path)
        if p.exists():
            return p

    key = normalize_image_key(image_key)
    if not key:
        return None

    p = Path(key)
    if p.is_absolute() and p.exists():
        return p

    candidate = image_root / key
    if candidate.exists():
        return candidate

    return None


def read_base_model_id(adapter_dir: Path, fallback: str) -> str:
    if fallback:
        return fallback

    cfg = adapter_dir / "adapter_config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            value = (data.get("base_model_name_or_path") or "").strip()
            if value:
                return value
        except json.JSONDecodeError:
            pass

    return "llava-hf/llava-1.5-7b-hf"


def load_reference_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}

    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalize_image_key(row.get("image") or row.get("image_key") or "")
            summary = (row.get("summary") or "").strip()
            status = (row.get("status") or "ok").strip().lower()
            if not key or not summary or status != "ok":
                continue
            out[key] = summary
    return out


def token_f1(pred: str, ref: str) -> float:
    p = pred.lower().split()
    r = ref.lower().split()
    if not p or not r:
        return 0.0

    counts: dict[str, int] = {}
    for tok in r:
        counts[tok] = counts.get(tok, 0) + 1

    overlap = 0
    for tok in p:
        cur = counts.get(tok, 0)
        if cur > 0:
            overlap += 1
            counts[tok] = cur - 1

    precision = overlap / len(p)
    recall = overlap / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_assistant_text(decoded: str) -> str:
    marker = "ASSISTANT:"
    if marker in decoded:
        return decoded.split(marker, 1)[1].strip()
    return decoded.strip()


def main() -> int:
    args = parse_args()

    adapter_dir = args.adapter_dir.resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    ocr_csv = args.ocr_csv.resolve()
    if not ocr_csv.exists():
        raise FileNotFoundError(f"OCR CSV not found: {ocr_csv}")

    image_root = args.image_root.resolve()
    out_csv = args.out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_metrics = args.out_metrics.resolve() if args.out_metrics else out_csv.with_name(out_csv.stem + "_metrics.json")

    base_model_id = read_base_model_id(adapter_dir, args.model_id)
    print(f"Adapter: {adapter_dir}")
    print(f"Base model: {base_model_id}")
    print(f"OCR CSV: {ocr_csv}")
    print(f"Output CSV: {out_csv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(adapter_dir)
    model = LlavaForConditionalGeneration.from_pretrained(base_model_id, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.to(device)
    model.eval()

    reference_map = load_reference_map(args.reference_csv.resolve() if args.reference_csv else None)

    rows_written = 0
    skipped_bad_ocr = 0
    skipped_missing_image = 0
    with_refs = 0
    f1_sum = 0.0

    with ocr_csv.open("r", encoding="utf-8", newline="") as src, out_csv.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(
            dst,
            fieldnames=[
                "row_id",
                "image_key",
                "image_path",
                "prediction",
                "reference",
                "token_f1",
                "ocr_chars",
            ],
        )
        writer.writeheader()

        for row in reader:
            if args.max_rows > 0 and rows_written >= args.max_rows:
                break

            image_key = normalize_image_key(row.get("image") or "")
            ocr_text = (row.get("text") or "").strip()
            ocr_status = (row.get("status") or "").strip().lower()
            full_path = (row.get("full_path") or "").strip()

            if (not ocr_text) or ocr_text == "(no text detected)" or (ocr_status in {"error", "empty", "legacy"}):
                skipped_bad_ocr += 1
                continue

            image_path = resolve_image_path(image_root, image_key, full_path)
            if image_path is None:
                skipped_missing_image += 1
                continue

            prompt = (
                "Summarize this scanned document page in one concise paragraph. "
                "Focus on key entities, dates, events, and any UAP-related content if present.\n\n"
                f"OCR text:\n{ocr_text[: args.max_ocr_chars]}"
            )

            user_text = f"USER: <image>\n{prompt} ASSISTANT: "
            image = Image.open(image_path).convert("RGB")
            batch = processor(text=user_text, images=image, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.inference_mode():
                output_ids = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            decoded = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred = extract_assistant_text(decoded)

            ref = reference_map.get(image_key, "")
            f1 = token_f1(pred, ref) if ref else 0.0
            if ref:
                with_refs += 1
                f1_sum += f1

            rows_written += 1
            writer.writerow(
                {
                    "row_id": rows_written,
                    "image_key": image_key,
                    "image_path": str(image_path),
                    "prediction": pred,
                    "reference": ref,
                    "token_f1": f"{f1:.4f}" if ref else "",
                    "ocr_chars": len(ocr_text),
                }
            )

            if rows_written % 10 == 0:
                print(f"Generated {rows_written} rows...")

    metrics = {
        "adapter_dir": str(adapter_dir),
        "base_model_id": base_model_id,
        "ocr_csv": str(ocr_csv),
        "output_csv": str(out_csv),
        "rows_written": rows_written,
        "skipped_bad_ocr": skipped_bad_ocr,
        "skipped_missing_image": skipped_missing_image,
        "rows_with_reference": with_refs,
        "avg_token_f1": (f1_sum / with_refs) if with_refs else None,
    }
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nGeneration complete.")
    print(f"Predictions CSV: {out_csv}")
    print(f"Metrics JSON: {out_metrics}")
    if with_refs:
        print(f"Average token F1 vs reference summaries: {metrics['avg_token_f1']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
