"""Minimal local HTTP service for OCR -> summary inference.

Uses only the Python standard library (no FastAPI/Flask). The model loads once
at startup and is reused for every request.

Run (from deploy/):

    ../.venv/Scripts/python.exe serve.py --host 127.0.0.1 --port 8008

Call it:

    curl -s -X POST http://127.0.0.1:8008/summarize \
         -H "Content-Type: application/json" \
         -d "{\"ocr_text\": \"EONFIDENTIAt (newline) FM AMEMBASSY MOSCOW ...\"}"

    -> {"summary": "This classified report ..."}

Health check:

    curl -s http://127.0.0.1:8008/health   ->  {"status": "ok", "device": "cuda"}
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from infer import DEFAULT_ADAPTER, DEFAULT_MAX_NEW_TOKENS, Summarizer

_SUMMARIZER: Summarizer | None = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path.rstrip("/") == "/health":
            device = _SUMMARIZER.device if _SUMMARIZER else "not-loaded"
            self._send(200, {"status": "ok", "device": device})
        else:
            self._send(404, {"error": "not found", "try": "GET /health or POST /summarize"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/summarize":
            self._send(404, {"error": "not found", "try": "POST /summarize"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            ocr_text = (data.get("ocr_text") or data.get("text") or "").strip()
            if not ocr_text:
                self._send(400, {"error": "missing 'ocr_text'"})
                return
            assert _SUMMARIZER is not None
            summary = _SUMMARIZER.summarize(ocr_text)
            self._send(200, {"summary": summary})
        except Exception as exc:  # surface errors as JSON rather than a stack trace
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve the OCR -> summary adapter over HTTP")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8008)
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--merged-model", type=Path, default=None)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return p.parse_args()


def main() -> int:
    global _SUMMARIZER
    args = parse_args()
    print("Loading model (one time)...")
    _SUMMARIZER = Summarizer(
        adapter_dir=args.adapter_dir,
        merged_model_dir=args.merged_model,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Model ready on {_SUMMARIZER.device}.")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Listening on http://{args.host}:{args.port}  (POST /summarize, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
