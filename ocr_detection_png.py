"""Extract text from PNG images in Release_1_PNG using EasyOCR."""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import easyocr
import torch

DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent / "Release_1_PNG"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "OCR_TEXT.md"
SECTION_RE = re.compile(
    r"^## `(.+?)`\s*\r?\n\s*Full path: `(.+?)`",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR PNG images to output/OCR_TEXT.md")
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
        help="Markdown output file",
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
    return parser.parse_args()


def load_processed_paths(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    text = output_path.read_text(encoding="utf-8", errors="replace")
    processed: set[str] = set()
    for rel_path, full_path in SECTION_RE.findall(text):
        processed.add(rel_path.replace("/", "\\"))
        processed.add(full_path)
        processed.add(Path(full_path).name)
    return processed


def relative_display_path(image_path: Path, image_dir: Path) -> str:
    try:
        rel = image_path.relative_to(image_dir.parent)
    except ValueError:
        rel = image_path.name
    return rel.as_posix().replace("/", "\\")


def format_section(image_path: Path, image_dir: Path, text: str) -> str:
    rel = relative_display_path(image_path, image_dir)
    full = str(image_path.resolve())
    body = text.strip() if text.strip() else "(no text detected)"
    return (
        f"## `{rel}`\n\n"
        f"Full path: `{full}`\n\n"
        f"```text\n{body}\n```\n\n---\n\n"
    )


def ensure_header(output_path: Path, image_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    header = (
        "# OCR Text Extraction\n\n"
        f"Source folder: `{image_dir.resolve()}`\n\n"
        f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        "---\n\n"
    )
    output_path.write_text(header, encoding="utf-8")


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(image_dir.glob("*.png"), key=lambda p: p.name.lower())


def run_ocr(reader: easyocr.Reader, image_path: Path) -> str:
    lines = reader.readtext(
        str(image_path),
        detail=0,
        paragraph=True,
    )
    if not lines:
        return ""
    return "\n\n".join(str(line).strip() for line in lines if str(line).strip())


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    output_path = args.output.resolve()

    if not image_dir.is_dir():
        print(f"Image directory not found: {image_dir}", file=sys.stderr)
        return 1

    images = collect_images(image_dir)
    if not images:
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
    device_name = torch.cuda.get_device_name(0) if use_gpu else "cpu"
    langs = [part.strip() for part in args.lang.split(",") if part.strip()]
    print(f"Device: {device_name} (cuda={use_gpu})", flush=True)
    print(f"Loading EasyOCR ({', '.join(langs)}), gpu={use_gpu}...", flush=True)
    reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)

    ensure_header(output_path, image_dir)
    started = time.time()
    done = 0

    with output_path.open("a", encoding="utf-8") as out:
        for image_path in pending:
            t0 = time.time()
            try:
                text = run_ocr(reader, image_path)
            except Exception as exc:
                text = f"[OCR error: {exc}]"
            section = format_section(image_path, image_dir, text)
            out.write(section)
            out.flush()
            done += 1
            elapsed = time.time() - t0
            print(
                f"[{done}/{len(pending)}] {image_path.name} ({elapsed:.1f}s)",
                flush=True,
            )

    total = time.time() - started
    footer = (
        f"\nCompleted: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        f"Processed this run: {done}\n\n"
        f"Elapsed: {total / 60:.1f} min\n"
    )
    with output_path.open("a", encoding="utf-8") as out:
        out.write(footer)

    print(f"Done. Wrote {done} section(s) to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())