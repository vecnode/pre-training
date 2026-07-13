"""Extract text from PNG images in Release_1_PNG using Baidu Unlimited-OCR."""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Release_1_PNG"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "DATASET_1_OCR.csv"
CSV_FIELDS = ["image", "full_path", "status", "reason", "method", "confidence", "text"]
DEFAULT_MODEL_ID = "baidu/Unlimited-OCR"
DEFAULT_PROMPT = "<image>Free OCR. "
DEFAULT_MAX_LENGTH = 8192
DEFAULT_NO_REPEAT_NGRAM_SIZE = 35
DEFAULT_NGRAM_WINDOW = 128

# gundam crops a high-res image into tiles (better for dense text pages);
# base resizes the whole page to one square (faster, coarser).
IMAGE_MODE_CONFIGS = {
    "gundam": dict(base_size=1024, image_size=640, crop_mode=True),
    "base": dict(base_size=1024, image_size=1024, crop_mode=False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR PNG images to output/DATASET_1_OCR.csv")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing PNG images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV output file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N pending images (0 = all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess images even if already present in the output file",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local path for Unlimited-OCR",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt sent with each image, e.g. '<image>Free OCR. ' or '<image>document parsing.'",
    )
    parser.add_argument(
        "--image-mode",
        choices=list(IMAGE_MODE_CONFIGS),
        default="gundam",
        help="gundam: tiled crops for dense pages (recommended). base: single resized page.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Max generated sequence length (model+prompt+output tokens)",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=DEFAULT_NO_REPEAT_NGRAM_SIZE,
        help="Sliding-window no-repeat ngram size (0 disables)",
    )
    parser.add_argument(
        "--ngram-window",
        type=int,
        default=DEFAULT_NGRAM_WINDOW,
        help="Sliding-window size for the no-repeat ngram check",
    )
    return parser.parse_args()


def load_processed_paths(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    processed: set[str] = set()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return processed
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            reason = (row.get("reason") or "").strip().lower()
            # Allow reruns to recover pages that previously failed due runtime/config errors.
            if status == "error":
                continue
            if status == "empty" and "ocr error" in reason:
                continue
            full_path = (row.get("full_path") or "").strip()
            image = (row.get("image") or "").strip()
            if image:
                processed.add(image.replace("/", "\\"))
                processed.add(Path(image).name)
            if full_path:
                processed.add(full_path)
                processed.add(Path(full_path).name)
    return processed


def relative_display_path(image_path: Path, image_dir: Path) -> str:
    try:
        rel = image_path.relative_to(image_dir.parent)
    except ValueError:
        rel = image_path.name
    return rel.as_posix().replace("/", "\\")


def format_row(
    image_path: Path,
    image_dir: Path,
    status: str,
    reason: str,
    method: str,
    confidence: float,
    text: str,
) -> dict[str, str]:
    rel = relative_display_path(image_path, image_dir)
    full = str(image_path.resolve())
    raw = text.strip() if text.strip() else "(no text detected)"
    body = " (newline) ".join(part.strip() for part in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n") if part.strip())
    if not body:
        body = "(no text detected)"
    return {
        "image": rel,
        "full_path": full,
        "status": status,
        "reason": reason,
        "method": method,
        "confidence": f"{confidence:.4f}" if confidence else "0.0000",
        "text": body,
    }


def ensure_header(output_path: Path, image_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()


def normalize_existing_csv(output_path: Path) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        return

    if all(field in fieldnames for field in CSV_FIELDS):
        return

    upgraded_rows: list[dict[str, str]] = []
    for row in rows:
        image = (row.get("image") or "").strip()
        full_path = (row.get("full_path") or "").strip()
        text = (row.get("text") or "").strip()
        upgraded_rows.append(
            {
                "image": image,
                "full_path": full_path,
                "status": "ok" if text and text != "(no text detected)" else "legacy",
                "reason": "legacy markdown conversion",
                "method": "legacy",
                "confidence": "0.0000",
                "text": text or "(no text detected)",
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(upgraded_rows)


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(image_dir.glob("*.png"), key=lambda p: p.name.lower())


def count_pdfs(image_dir: Path) -> int:
    return sum(1 for _ in image_dir.glob("*.pdf"))


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    return tokenizer, model


def run_ocr(
    tokenizer,
    model,
    image_path: Path,
    scratch_dir: Path,
    prompt: str,
    image_mode: str,
    max_length: int,
    no_repeat_ngram_size: int,
    ngram_window: int,
) -> tuple[str, str, float, str, str]:
    mode_cfg = IMAGE_MODE_CONFIGS[image_mode]
    try:
        outputs = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(scratch_dir),
            base_size=mode_cfg["base_size"],
            image_size=mode_cfg["image_size"],
            crop_mode=mode_cfg["crop_mode"],
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            ngram_window=ngram_window,
            eval_mode=True,
        )
    except Exception as exc:
        return "error", str(exc), 0.0, "", f"[OCR error: {exc}]"

    text = (outputs or "").strip()
    if not text:
        return "empty", "no text detected", 0.0, image_mode, ""
    return "ok", image_mode, 0.0, image_mode, text


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    output_path = args.output.resolve()

    if not image_dir.is_dir():
        print(f"Image directory not found: {image_dir}", file=sys.stderr)
        return 1

    images = collect_images(image_dir)
    if not images:
        pdf_count = count_pdfs(image_dir)
        if pdf_count > 0:
            print(
                f"No PNG files found in {image_dir}. Found {pdf_count} PDF(s); run convert_pdf_to_png first.",
                file=sys.stderr,
            )
        else:
            print(f"No PNG files found in {image_dir}", file=sys.stderr)
        return 1

    processed = set() if args.no_resume else load_processed_paths(output_path)
    pending: list[Path] = []
    for image_path in images:
        rel = relative_display_path(image_path, image_dir)
        full = str(image_path)
        if rel in processed or full in processed or image_path.name in processed:
            continue
        pending.append(image_path)

    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"Found {len(images)} PNG(s); pending {len(pending)}", flush=True)
    if not pending:
        print("Nothing to do.", flush=True)
        return 0

    if not torch.cuda.is_available():
        print(
            "CUDA is not available in the current Python environment. "
            "Unlimited-OCR requires a GPU. Run uv_bootstrap.bat to install CUDA-enabled PyTorch.",
            file=sys.stderr,
        )
        return 1

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}", flush=True)
    print(f"Loading Unlimited-OCR ({args.model_id}), image_mode={args.image_mode}...", flush=True)
    tokenizer, model = load_model(args.model_id)

    normalize_existing_csv(output_path)
    ensure_header(output_path, image_dir)
    started = time.time()
    done = 0

    with tempfile.TemporaryDirectory(prefix="unlimited_ocr_") as scratch:
        scratch_dir = Path(scratch)
        with output_path.open("a", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
            for image_path in pending:
                t0 = time.time()
                status, reason, confidence, method, text = run_ocr(
                    tokenizer,
                    model,
                    image_path,
                    scratch_dir,
                    args.prompt,
                    args.image_mode,
                    args.max_length,
                    args.no_repeat_ngram_size,
                    args.ngram_window,
                )
                writer.writerow(format_row(image_path, image_dir, status, reason, method, confidence, text))
                out.flush()
                done += 1
                elapsed = time.time() - t0
                print(
                    f"[{done}/{len(pending)}] {image_path.name} ({elapsed:.1f}s) {status}: {reason}",
                    flush=True,
                )

    total = time.time() - started
    print(f"Done. Wrote {done} row(s) to {output_path} in {total:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
