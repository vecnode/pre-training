"""Extract text from PNG images in Release_1_PNG using EasyOCR."""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Release_1_PNG"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "DATASET_1_OCR.csv"
CSV_FIELDS = ["image", "full_path", "status", "reason", "method", "confidence", "text"]
DEFAULT_MAX_SIDE = 720
DEFAULT_PROFILE = "fast"


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
        "--lang",
        default="en",
        help="EasyOCR language code(s), comma-separated",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N pending images (0 = all)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU inference",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess images even if already present in the output file",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_MAX_SIDE,
        help="Resize the longest image side to this many pixels before OCR (0 disables resizing)",
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality"],
        default=DEFAULT_PROFILE,
        help="OCR speed/quality profile: fast (recommended), balanced, or quality",
    )
    parser.add_argument(
        "--detect-network",
        choices=["dbnet18", "craft"],
        default="dbnet18",
        help="EasyOCR detector network. dbnet18 is faster; craft may improve difficult pages",
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


def resize_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def preprocess_variants(image_path: Path, max_side: int, profile: str) -> list[tuple[str, np.ndarray]]:
    with Image.open(image_path) as image:
        rgb = resize_to_max_side(image.convert("RGB"), max_side)
        gray = ImageOps.grayscale(rgb)
        enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
        sharpened = enhanced.filter(ImageFilter.SHARPEN)
        autocontrast = ImageOps.autocontrast(gray)
        threshold = autocontrast.point(lambda value: 255 if value > 165 else 0)

        suffix = f"@{max_side}" if max_side > 0 else ""
        if profile == "fast":
            return [
                (f"gray{suffix}", np.array(gray)),
                (f"rgb{suffix}", np.array(rgb)),
            ]
        if profile == "balanced":
            return [
                (f"gray{suffix}", np.array(gray)),
                (f"rgb{suffix}", np.array(rgb)),
                (f"contrast{suffix}", np.array(enhanced)),
            ]
        return [
            (f"rgb{suffix}", np.array(rgb)),
            (f"gray{suffix}", np.array(gray)),
            (f"contrast{suffix}", np.array(enhanced)),
            (f"sharpen{suffix}", np.array(sharpened)),
            (f"threshold{suffix}", np.array(threshold)),
        ]


def flatten_ocr_items(items: object) -> list[tuple[str, float]]:
    if isinstance(items, tuple):
        if len(items) >= 3 and isinstance(items[1], str):
            try:
                confidence = float(items[2])
            except (TypeError, ValueError):
                confidence = 0.0
            return [(items[1], confidence)]
        flattened: list[tuple[str, float]] = []
        for item in items:
            flattened.extend(flatten_ocr_items(item))
        return flattened
    if isinstance(items, list):
        flattened = []
        for item in items:
            flattened.extend(flatten_ocr_items(item))
        return flattened
    if isinstance(items, str):
        return [(items, 0.0)]
    return []


def is_dbnet_extension_error(message: str) -> bool:
    lowered = message.lower()
    markers = [
        "deform_conv_cuda",
        "deform_pool_cuda",
        "dbnet",
        "where', 'cl'",
        "input type is cuda",
    ]
    return any(marker in lowered for marker in markers)


def score_ocr(lines: object) -> tuple[str, float]:
    flattened = flatten_ocr_items(lines)
    text = "\n\n".join(piece.strip() for piece, _ in flattened if piece and piece.strip()).strip()
    if not text:
        return "", 0.0
    confidences = [float(conf) for piece, conf in flattened if piece and piece.strip()]
    confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    return text, confidence


def choose_best_ocr(
    reader: easyocr.Reader, image_path: Path, max_side: int, profile: str
) -> tuple[str, str, float, str, str]:
    best_text = ""
    best_reason = "no text detected"
    best_confidence = 0.0
    best_method = ""

    for method, array in preprocess_variants(image_path, max_side, profile):
        try:
            lines = reader.readtext(array, detail=1, paragraph=True, decoder="greedy", beamWidth=1)
        except Exception as exc:
            if is_dbnet_extension_error(str(exc)):
                raise RuntimeError(str(exc)) from exc
            if not best_reason or best_reason == "no text detected":
                best_reason = f"{method}: OCR error: {exc}"
            continue

        text, confidence = score_ocr(lines)
        if not text:
            continue
        if confidence > best_confidence or (confidence == best_confidence and len(text) > len(best_text)):
            best_text = text
            best_confidence = confidence
            best_method = method
            best_reason = method

    if best_text:
        return "ok", best_reason, best_confidence, best_method, best_text
    return "empty", best_reason, 0.0, "", ""


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


def create_reader(langs: list[str], use_gpu: bool, detect_network: str) -> easyocr.Reader:
    return easyocr.Reader(
        langs,
        gpu=use_gpu,
        verbose=False,
        detect_network=detect_network,
    )


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

    if not args.no_gpu and not torch.cuda.is_available():
        print(
            "CUDA is not available in the current Python environment. "
            "Run uv_bootstrap.bat to install CUDA-enabled PyTorch, or use --no-gpu.",
            file=sys.stderr,
        )
        return 1

    use_gpu = not args.no_gpu
    active_use_gpu = use_gpu
    active_detect_network = args.detect_network
    if active_detect_network == "dbnet18" and shutil.which("cl") is None:
        print(
            "MSVC compiler (cl.exe) not found. Switching detector to craft for stability.",
            flush=True,
        )
        active_detect_network = "craft"
    device_name = torch.cuda.get_device_name(0) if use_gpu else "cpu"
    langs = [part.strip() for part in args.lang.split(",") if part.strip()]
    print(f"Device: {device_name} (cuda={use_gpu})", flush=True)
    print(
        f"Loading EasyOCR ({', '.join(langs)}), gpu={active_use_gpu}, profile={args.profile}, detector={active_detect_network}...",
        flush=True,
    )
    reader = create_reader(langs, active_use_gpu, active_detect_network)

    normalize_existing_csv(output_path)
    ensure_header(output_path, image_dir)
    started = time.time()
    done = 0

    with output_path.open("a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        for image_path in pending:
            t0 = time.time()
            status = "error"
            reason = "unknown"
            method = ""
            confidence = 0.0
            text = ""

            for _ in range(3):
                try:
                    status, reason, confidence, method, text = choose_best_ocr(
                        reader, image_path, args.max_side, args.profile
                    )
                    break
                except Exception as exc:
                    msg = str(exc)
                    if is_dbnet_extension_error(msg) and active_detect_network == "dbnet18":
                        print(
                            "DBNet extension unavailable. Switching detector to craft and retrying current image...",
                            flush=True,
                        )
                        active_detect_network = "craft"
                        reader = create_reader(langs, active_use_gpu, active_detect_network)
                        continue
                    if is_dbnet_extension_error(msg) and active_use_gpu:
                        print(
                            "Detector failed on GPU. Switching EasyOCR to CPU and retrying current image...",
                            flush=True,
                        )
                        active_use_gpu = False
                        reader = create_reader(langs, active_use_gpu, active_detect_network)
                        continue
                    status = "error"
                    reason = msg
                    method = ""
                    confidence = 0.0
                    text = f"[OCR error: {msg}]"
                    break
            writer.writerow(format_row(image_path, image_dir, status, reason, method, confidence, text))
            out.flush()
            done += 1
            elapsed = time.time() - t0
            print(
                f"[{done}/{len(pending)}] {image_path.name} ({elapsed:.1f}s) {status}: {reason}",
                flush=True,
            )

    total = time.time() - started
    print(f"Done. Wrote {done} row(s) to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())