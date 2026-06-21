"""Detect objects in PNG images and write one CSV row per detection."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Release_1_PNG"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "DATASET_OBJS.csv"
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "models" / "yolov8n.pt"
CSV_FIELDS = [
    "image",
    "full_path",
    "object_id",
    "label",
    "confidence",
    "x",
    "y",
    "width",
    "height",
    "status",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects in PNG images to output/DATASET_OBJS.csv",
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
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
    processed: set[str] = set()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            if status == "error":
                continue
            image = (row.get("image") or "").strip()
            full_path = (row.get("full_path") or "").strip()
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


def detect_objects(
    model: YOLO,
    work_arr: np.ndarray,
    width: int,
    height: int,
    conf: float,
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
            "label": label,
            "confidence": round(float(box.conf), 4),
            "bbox": bbox,
        }
        objects.append(entry)
    return objects


def analyze_image(yolo: YOLO, image_path: Path, args: argparse.Namespace, use_gpu: bool) -> dict[str, object]:
    arr, width, height = load_rgb_array(image_path)
    work_arr, coord_scale = downscale_for_detection(arr, args.max_side)
    objects = detect_objects(
        yolo,
        work_arr,
        width,
        height,
        args.yolo_conf,
        use_gpu,
        coord_scale,
        args.yolo_imgsz,
    )
    return {
        "image": {
            "relative": relative_display_path(image_path, args.image_dir.resolve()),
            "path": str(image_path.resolve()),
            "width": width,
            "height": height,
        },
        "objects": objects,
        "counts": {
            "objects": len(objects),
        },
    }


def object_row(
    image_rel: str,
    image_full: str,
    obj_id: int,
    label: str,
    confidence: float,
    bbox: dict[str, int],
    status: str,
    reason: str,
) -> dict[str, str]:
    return {
        "image": image_rel,
        "full_path": image_full,
        "object_id": str(obj_id),
        "label": label,
        "confidence": f"{confidence:.4f}" if confidence else "0.0000",
        "x": str(bbox.get("x", 0)),
        "y": str(bbox.get("y", 0)),
        "width": str(bbox.get("width", 0)),
        "height": str(bbox.get("height", 0)),
        "status": status,
        "reason": reason,
    }


def ensure_header(output_path: Path, image_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()


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
    device_name = torch.cuda.get_device_name(0) if use_gpu else "cpu"
    print(
        f"Device: {device_name} (cuda={use_gpu}, fp16_yolo={use_gpu}, max_side={args.max_side})",
        flush=True,
    )
    print(f"Loading YOLO model {yolo_model_path}...", flush=True)
    yolo = YOLO(str(yolo_model_path))
    if use_gpu:
        yolo.fuse()

    ensure_header(output_path, image_dir)
    started = time.time()
    done = 0

    with output_path.open("a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        for image_path in pending:
            t0 = time.time()
            payload: dict[str, object] = {}
            try:
                payload = analyze_image(yolo, image_path, args, use_gpu)
                image_meta = payload.get("image", {})
                image_rel = str(image_meta.get("relative", image_path.name))
                image_full = str(image_meta.get("path", str(image_path.resolve())))
                objects = payload.get("objects", [])
                if isinstance(objects, list) and objects:
                    for obj in objects:
                        bbox = obj.get("bbox") if isinstance(obj, dict) else {}
                        if not isinstance(bbox, dict):
                            bbox = {"x": 0, "y": 0, "width": 0, "height": 0}
                        writer.writerow(
                            object_row(
                                image_rel=image_rel,
                                image_full=image_full,
                                obj_id=int(obj.get("id", 0)) if isinstance(obj, dict) else 0,
                                label=str(obj.get("label", "")) if isinstance(obj, dict) else "",
                                confidence=float(obj.get("confidence", 0.0)) if isinstance(obj, dict) else 0.0,
                                bbox=bbox,
                                status="ok",
                                reason="",
                            )
                        )
                else:
                    writer.writerow(
                        object_row(
                            image_rel=image_rel,
                            image_full=image_full,
                            obj_id=0,
                            label="",
                            confidence=0.0,
                            bbox={"x": 0, "y": 0, "width": 0, "height": 0},
                            status="no_objects",
                            reason="no detections",
                        )
                    )
            except Exception as exc:
                writer.writerow(
                    object_row(
                        image_rel=relative_display_path(image_path, image_dir),
                        image_full=str(image_path.resolve()),
                        obj_id=0,
                        label="",
                        confidence=0.0,
                        bbox={"x": 0, "y": 0, "width": 0, "height": 0},
                        status="error",
                        reason=str(exc),
                    )
                )
            out.flush()
            done += 1
            counts = payload.get("counts", {}) if isinstance(payload, dict) else {}
            elapsed = time.time() - t0
            print(
                f"[{done}/{len(pending)}] {image_path.name} "
                f"objects={counts.get('objects', 0)} ({elapsed:.1f}s)",
                flush=True,
            )

    print(f"Done. Wrote detections for {done} image(s) to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())