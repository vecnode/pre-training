"""Summarize OCR CSV rows using Ollama HTTP API and write <dataset>_SUMMARIES.csv."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_OLLAMA_URL = "http://localhost:11434"
CSV_FIELDS = ["image", "summary", "status", "reason", "model"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OCR CSV rows with Ollama")
    parser.add_argument("--input", type=Path, required=True, help="Input OCR CSV path")
    parser.add_argument("--output", type=Path, required=True, help="Output summaries CSV path")
    parser.add_argument("--model", default="", help="Ollama model name (blank uses first available)")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--no-resume", action="store_true", help="Recompute existing summaries")
    return parser.parse_args()


def http_json(url: str, method: str = "GET", payload: dict[str, object] | None = None, timeout: int = 120) -> dict[str, object]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body) if body.strip() else {}


def list_ollama_models(base_url: str, timeout: int) -> list[str]:
    result = http_json(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    models = result.get("models")
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for entry in models:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.append(entry["name"])
    return names


def choose_model(requested: str, available: list[str]) -> str:
    if requested:
        if requested in available:
            return requested
        raise ValueError(f"Requested model not available in Ollama: {requested}")
    if not available:
        raise ValueError("No Ollama models available. Run `ollama pull <model>` first.")
    return available[0]


def load_processed(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    processed: set[str] = set()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image = (row.get("image") or "").strip()
            status = (row.get("status") or "").strip().lower()
            if image and status != "error":
                processed.add(image)
    return processed


def normalize_text(value: str) -> str:
    collapsed = " (newline) ".join(
        part.strip() for part in value.replace("\r\n", "\n").replace("\r", "\n").split("\n") if part.strip()
    )
    return collapsed or "(empty)"


def summarize_text(base_url: str, model: str, text: str, timeout: int) -> str:
    prompt = (
        "You are summarizing OCR text from one scanned document page. "
        "Write one concise paragraph (max 90 words) describing what the page is about, "
        "including key entities/dates if present.\n\n"
        f"OCR text:\n{text}\n"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = http_json(f"{base_url.rstrip('/')}/api/generate", method="POST", payload=payload, timeout=timeout)
    summary = response.get("response")
    if not isinstance(summary, str):
        raise RuntimeError("Ollama response did not include a text summary")
    return normalize_text(summary)


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        print(f"Input OCR CSV not found: {input_path}", file=sys.stderr)
        return 1

    try:
        available = list_ollama_models(args.ollama_url, args.timeout)
        model = choose_model(args.model.strip(), available)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to initialize Ollama model selection: {exc}", file=sys.stderr)
        return 1

    print(f"Using Ollama model: {model}", flush=True)

    processed = set() if args.no_resume else load_processed(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists() or output_path.stat().st_size == 0:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

    with input_path.open("r", encoding="utf-8", newline="") as src:
        rows = list(csv.DictReader(src))

    pending = [row for row in rows if (row.get("image") or "").strip() and (row.get("image") or "").strip() not in processed]
    print(f"Input rows: {len(rows)} | Pending summaries: {len(pending)}", flush=True)

    done = 0
    with output_path.open("a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        for row in pending:
            image = (row.get("image") or "").strip()
            text = (row.get("text") or "").strip()
            status = (row.get("status") or "").strip().lower()

            if not text or text == "(no text detected)" or status in {"empty", "error"}:
                writer.writerow(
                    {
                        "image": image,
                        "summary": "",
                        "status": "skipped",
                        "reason": f"ocr_status={status or 'unknown'}",
                        "model": model,
                    }
                )
                done += 1
                continue

            try:
                summary = summarize_text(args.ollama_url, model, text, args.timeout)
                writer.writerow(
                    {
                        "image": image,
                        "summary": summary,
                        "status": "ok",
                        "reason": "",
                        "model": model,
                    }
                )
            except Exception as exc:
                writer.writerow(
                    {
                        "image": image,
                        "summary": "",
                        "status": "error",
                        "reason": normalize_text(str(exc)),
                        "model": model,
                    }
                )
            done += 1
            if done % 10 == 0:
                print(f"[{done}/{len(pending)}] summarized", flush=True)

    print(f"Done. Wrote {done} summary row(s) to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
