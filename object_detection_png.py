"""Detect objects, text regions, and colors in PNG images for mask reproduction."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import easyocr
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from ultralytics import YOLO

DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent / "Release_1_PNG"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "OBJS_TEXT.md"
DEFAULT_YOLO_MODEL = Path(__file__).resolve().parent / "models" / "yolov8n.pt"
SECTION_RE = re.compile(r"^## `(.+?)`", re.MULTILINE)
SCHEMA_VERSION = 1


def slim_region(region: dict[str, object]) -> dict[str, object]:
    text = region.get("text") or region.get("label") or ""
    slim: dict[str, object] = {
        "id": region.get("id"),
        "type": region.get("type"),
        "text": text,
        "confidence": region.get("confidence"),
        "bbox": region.get("bbox"),
    }
    return slim


def slim_object(obj: dict[str, object]) -> dict[str, object]:
    return {
        "id": obj.get("id"),
        "type": obj.get("type", "object"),
        "label": obj.get("label"),
        "confidence": obj.get("confidence"),
        "bbox": obj.get("bbox"),
    }


def slim_payload(payload: dict[str, object]) -> dict[str, object]:
    text_regions = payload.get("text_regions") or []
    objects = payload.get("objects") or []
    return {
        "schema_version": payload.get("schema_version", SCHEMA_VERSION),
        "image": payload.get("image", {}),
        "text_regions": [slim_region(r) for r in text_regions if isinstance(r, dict)],
        "objects": [slim_object(o) for o in objects if isinstance(o, dict)],
        "counts": payload.get("counts", {}),
        "error": payload.get("error"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects/text/colors in PNG images to output/OBJS_TEXT.md",
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--text-conf", type=float, default=0.2)
    parser.add_argument(
        "--with-colors",
        action="store_true",
        help="Compute per-region KMeans colors (slow; omitted from compact output)",
    )
    parser.add_argument("--palette-colors", type=int, default=0)
    parser.add_argument("--region-colors", type=int, default=0)
    parser.add_argument(
        "--max-side",
        type=int,
        default=1600,
        help="Max image side for OCR/YOLO (0=full resolution). Bboxes are mapped to original pixels.",
    )
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    return parser.parse_args()


def load_processed_paths(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    text = output_path.read_text(encoding="utf-8", errors="replace")
    processed: set[str] = set()
    for rel_path in SECTION_RE.findall(text):
        normalized = rel_path.replace("/", "\\")
        processed.add(normalized)
        processed.add(Path(normalized).name)
    return processed


def relative_display_path(image_path: Path, image_dir: Path) -> str:
    try:
        rel = image_path.relative_to(image_dir.parent)
    except ValueError:
        rel = image_path.name
    return rel.as_posix().replace("/", "\\")


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(image_dir.glob("*.png"), key=lambda p: p.name.lower())


def load_rgb_array(image_path: Path) -> tuple[np.ndarray, int, int]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    return np.array(image), width, height


def downscale_for_detection(
    arr: np.ndarray, max_side: int
) -> tuple[np.ndarray, float]:
    """Return (working image, scale) where scale maps detection coords -> original pixels."""
    height, width = arr.shape[:2]
    longest = max(width, height)
    if max_side <= 0 or longest <= max_side:
        return arr, 1.0
    scale = max_side / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = np.array(Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR))
    return resized, 1.0 / scale


def scale_bbox_to_original(bbox: dict[str, int], coord_scale: float) -> dict[str, int]:
    if coord_scale == 1.0:
        return bbox
    return {
        "x": int(round(bbox["x"] * coord_scale)),
        "y": int(round(bbox["y"] * coord_scale)),
        "width": max(1, int(round(bbox["width"] * coord_scale))),
        "height": max(1, int(round(bbox["height"] * coord_scale))),
    }


def clamp_bbox(x: int, y: int, w: int, h: int, width: int, height: int) -> dict[str, int]:
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return {"x": x, "y": y, "width": w, "height": h}


def polygon_to_bbox(polygon: list[list[float]], width: int, height: int) -> dict[str, int]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x1 = int(round(min(xs)))
    y1 = int(round(min(ys)))
    x2 = int(round(max(xs)))
    y2 = int(round(max(ys)))
    return clamp_bbox(x1, y1, max(1, x2 - x1), max(1, y2 - y1), width, height)


def round_polygon(polygon: list[list[float]]) -> list[list[int]]:
    return [[int(round(x)), int(round(y))] for x, y in polygon]


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def sample_region_pixels(arr: np.ndarray, bbox: dict[str, int]) -> np.ndarray:
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    crop = arr[y : y + h, x : x + w]
    if crop.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return crop.reshape(-1, 3)


def dominant_colors(pixels: np.ndarray, k: int) -> list[dict[str, object]]:
    if pixels.size == 0:
        return []
    if len(pixels) > 3000:
        idx = np.random.default_rng(0).choice(len(pixels), 3000, replace=False)
        pixels = pixels[idx]
    cluster_count = min(k, len(pixels))
    if cluster_count <= 0:
        return []
    model = KMeans(n_clusters=cluster_count, n_init=3, random_state=0)
    labels = model.fit_predict(pixels)
    counts = np.bincount(labels, minlength=cluster_count)
    total = counts.sum()
    colors: list[dict[str, object]] = []
    for center, count in zip(model.cluster_centers_, counts):
        rgb = tuple(int(v) for v in np.round(center))
        colors.append(
            {
                "rgb": list(rgb),
                "hex": rgb_to_hex(rgb),
                "percentage": round(float(count) * 100.0 / total, 2),
            }
        )
    colors.sort(key=lambda c: c["percentage"], reverse=True)
    return colors


def detect_text_regions(
    reader: easyocr.Reader,
    work_arr: np.ndarray,
    full_arr: np.ndarray,
    width: int,
    height: int,
    min_conf: float,
    region_colors: int,
    coord_scale: float,
) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    results = reader.readtext(work_arr, paragraph=False)
    for idx, (polygon, text, confidence) in enumerate(results, start=1):
        if float(confidence) < min_conf:
            continue
        bbox = scale_bbox_to_original(
            polygon_to_bbox(polygon, work_arr.shape[1], work_arr.shape[0]), coord_scale
        )
        bbox = clamp_bbox(bbox["x"], bbox["y"], bbox["width"], bbox["height"], width, height)
        entry: dict[str, object] = {
            "id": idx,
            "type": "text",
            "text": str(text).strip(),
            "confidence": round(float(confidence), 4),
            "bbox": bbox,
        }
        if region_colors > 0:
            entry["colors"] = dominant_colors(
                sample_region_pixels(full_arr, bbox), region_colors
            )
        regions.append(entry)
    return regions


def detect_objects(
    model: YOLO,
    work_arr: np.ndarray,
    full_arr: np.ndarray,
    width: int,
    height: int,
    conf: float,
    region_colors: int,
    use_gpu: bool,
    coord_scale: float,
    imgsz: int,
) -> list[dict[str, object]]:
    device = 0 if use_gpu else "cpu"
    result = model(
        work_arr,
        conf=conf,
        verbose=False,
        device=device,
        half=use_gpu,
        imgsz=imgsz,
    )[0]
    objects: list[dict[str, object]] = []
    if result.boxes is None:
        return objects
    names = result.names
    for idx, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        bbox = clamp_bbox(
            x1, y1, max(1, x2 - x1), max(1, y2 - y1), work_arr.shape[1], work_arr.shape[0]
        )
        bbox = scale_bbox_to_original(bbox, coord_scale)
        bbox = clamp_bbox(bbox["x"], bbox["y"], bbox["width"], bbox["height"], width, height)
        label = names[int(box.cls)]
        entry: dict[str, object] = {
            "id": idx,
            "type": "object",
            "label": label,
            "confidence": round(float(box.conf), 4),
            "bbox": bbox,
        }
        if region_colors > 0:
            entry["colors"] = dominant_colors(
                sample_region_pixels(full_arr, bbox), region_colors
            )
        objects.append(entry)
    return objects


def ensure_minimum_regions(
    text_regions: list[dict[str, object]],
    objects: list[dict[str, object]],
    width: int,
    height: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Guarantee at least one bbox per image for the media player."""
    if text_regions or objects:
        return text_regions, objects

    margin = 0.08
    x = int(width * margin)
    y = int(height * margin)
    w = max(1, int(width * (1.0 - 2.0 * margin)))
    h = max(1, int(height * (1.0 - 2.0 * margin)))
    bbox = clamp_bbox(x, y, w, h, width, height)
    text_regions = [
        {
            "id": 1,
            "type": "fallback",
            "text": "page",
            "confidence": 1.0,
            "bbox": bbox,
        }
    ]
    return text_regions, objects


def analyze_image(
    reader: easyocr.Reader,
    yolo: YOLO,
    image_path: Path,
    args: argparse.Namespace,
    use_gpu: bool,
) -> dict[str, object]:
    arr, width, height = load_rgb_array(image_path)
    work_arr, coord_scale = downscale_for_detection(arr, args.max_side)
    region_colors = args.region_colors if args.with_colors else 0
    text_regions = detect_text_regions(
        reader, work_arr, arr, width, height, args.text_conf, region_colors, coord_scale
    )
    objects = detect_objects(
        yolo,
        work_arr,
        arr,
        width,
        height,
        args.yolo_conf,
        region_colors,
        use_gpu,
        coord_scale,
        args.yolo_imgsz,
    )
    text_regions, objects = ensure_minimum_regions(text_regions, objects, width, height)
    image_palette: list[dict[str, object]] = []
    if args.with_colors and args.palette_colors > 0:
        image_palette = dominant_colors(arr.reshape(-1, 3), args.palette_colors)
    return {
        "schema_version": SCHEMA_VERSION,
        "image": {
            "path": str(image_path.resolve()),
            "width": width,
            "height": height,
        },
        "text_regions": text_regions,
        "objects": objects,
        "image_palette": image_palette,
        "counts": {
            "text_regions": len(text_regions),
            "objects": len(objects),
        },
    }


def format_section(
    image_path: Path,
    image_dir: Path,
    payload: dict[str, object],
) -> str:
    rel = relative_display_path(image_path, image_dir)
    body = json.dumps(slim_payload(payload), ensure_ascii=False, separators=(",", ":"))
    return f"## `{rel}`\n```json\n{body}\n```\n\n"


def ensure_header(output_path: Path, image_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    header = (
        "# Object, Text Region, and Color Detection\n\n"
        f"Source folder: `{image_dir.resolve()}`\n\n"
        "One compact JSON block per image (`bbox` only; polygons/colors omitted).\n\n"
        f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
    )
    output_path.write_text(header, encoding="utf-8")


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    output_path = args.output.resolve()
    yolo_model_path = args.yolo_model.resolve()

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
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    langs = [part.strip() for part in args.lang.split(",") if part.strip()]
    device_name = torch.cuda.get_device_name(0) if use_gpu else "cpu"
    print(
        f"Device: {device_name} (cuda={use_gpu}, fp16_yolo={use_gpu}, max_side={args.max_side})",
        flush=True,
    )
    print(f"Loading EasyOCR ({', '.join(langs)}), gpu={use_gpu}...", flush=True)
    reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
    print(f"Loading YOLO model {yolo_model_path}...", flush=True)
    yolo = YOLO(str(yolo_model_path))
    if use_gpu:
        yolo.fuse()

    ensure_header(output_path, image_dir)
    started = time.time()
    done = 0

    with output_path.open("a", encoding="utf-8") as out:
        for image_path in pending:
            t0 = time.time()
            try:
                payload = analyze_image(reader, yolo, image_path, args, use_gpu)
            except Exception as exc:
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "image": {"path": str(image_path.resolve()), "width": 0, "height": 0},
                    "error": str(exc),
                    "text_regions": [],
                    "objects": [],
                    "image_palette": [],
                }
            section = format_section(image_path, image_dir, payload)
            out.write(section)
            out.flush()
            done += 1
            counts = payload.get("counts", {})
            elapsed = time.time() - t0
            print(
                f"[{done}/{len(pending)}] {image_path.name} "
                f"text={counts.get('text_regions', 0)} "
                f"objects={counts.get('objects', 0)} ({elapsed:.1f}s)",
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