"""Standalone real-world inference for the OCR -> summary LoRA adapter.

This module is deliberately independent of the training code: deployment should
not import anything from training/. It loads the base LLaVA 1.5 language model,
attaches the trained LoRA adapter (or loads an already-merged model), and turns
raw OCR text into a one-paragraph summary.

Usage (run from deploy/):

    # one-off
    ../.venv/Scripts/python.exe infer.py --text "EONFIDENTIAt (newline) FM AMEMBASSY ..."

    # from a file (best for long/noisy pages)
    ../.venv/Scripts/python.exe infer.py --text-file page.txt

    # interactive: paste one OCR page per line, blank line / 'quit' to exit
    ../.venv/Scripts/python.exe infer.py

    # read a single page from stdin
    echo "OCR text..." | ../.venv/Scripts/python.exe infer.py --text-file -
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_DEPLOY_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _DEPLOY_DIR.parent

# Reuse the training HF cache so the base model is not re-downloaded.
os.environ.setdefault("HF_HOME", str(_PROJECT_DIR / "training" / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ---------------------------------------------------------------------------
# These MUST stay identical to training (training/train_llava15_lora.py).
# If the instruction wording or truncation differs from training, quality drops.
# ---------------------------------------------------------------------------
INSTRUCTION = (
    "Summarize this scanned document page in one concise paragraph. "
    "Focus on key entities, dates, events, and any UAP-related content if present.\n\n"
    "OCR text:\n"
)
DEFAULT_ADAPTER = _PROJECT_DIR / "training" / "runs" / "llava15_lora" / "final_adapter"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_MAX_NEW_TOKENS = 220


def truncate_ocr_ids(ids: list[int], budget: int) -> list[int]:
    """Fit OCR token ids into `budget`, keeping the head and tail of the page."""
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return ids
    head = int(budget * 0.75)
    tail = budget - head
    if tail <= 0:
        return ids[:budget]
    return ids[:head] + ids[-tail:]


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


class Summarizer:
    """Load once, summarize many. Importable as a library.

    >>> s = Summarizer()
    >>> s.summarize("EONFIDENTIAt (newline) FM AMEMBASSY MOSCOW ...")
    'This classified report ...'
    """

    def __init__(
        self,
        adapter_dir: Path | str = DEFAULT_ADAPTER,
        base_model_id: str = "",
        merged_model_dir: Path | str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str | None = None,
    ) -> None:
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        if merged_model_dir:
            # Standalone, already-merged model (no PEFT needed at runtime).
            merged = Path(merged_model_dir).resolve()
            self.processor = AutoProcessor.from_pretrained(merged)
            model = LlavaForConditionalGeneration.from_pretrained(merged, torch_dtype=dtype)
        else:
            from peft import PeftModel

            adapter = Path(adapter_dir).resolve()
            if not adapter.exists():
                raise FileNotFoundError(
                    f"Adapter not found: {adapter}\n"
                    "training/runs/ is gitignored - copy the trained adapter here or pass --adapter-dir."
                )
            base_id = read_base_model_id(adapter, base_model_id)
            self.processor = AutoProcessor.from_pretrained(adapter)
            model = LlavaForConditionalGeneration.from_pretrained(base_id, torch_dtype=dtype)
            model = PeftModel.from_pretrained(model, str(adapter))

        self.tokenizer = self.processor.tokenizer
        self.model = model.to(self.device)
        self.model.eval()

        # Fixed wrapper; only the OCR body is re-tokenized per call.
        self.head_ids = self.tokenizer(f"USER: {INSTRUCTION}", add_special_tokens=True).input_ids
        self.suffix_ids = self.tokenizer(" ASSISTANT: ", add_special_tokens=False).input_ids

    def _build_input_ids(self, ocr_text: str) -> list[int]:
        budget = self.max_length - len(self.head_ids) - len(self.suffix_ids) - self.max_new_tokens
        ocr_ids = self.tokenizer(ocr_text, add_special_tokens=False).input_ids
        ocr_ids = truncate_ocr_ids(ocr_ids, budget)
        return self.head_ids + ocr_ids + self.suffix_ids

    def summarize(self, ocr_text: str) -> str:
        ocr_text = (ocr_text or "").strip()
        if not ocr_text:
            return ""
        ids = self._build_input_ids(ocr_text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize OCR text with the trained LoRA adapter")
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER, help="Trained LoRA adapter directory")
    p.add_argument("--merged-model", type=Path, default=None, help="Use a standalone merged model dir (skips PEFT)")
    p.add_argument("--base-model", default="", help="Base model id (auto-detected from adapter config when omitted)")
    p.add_argument("--text", default="", help="OCR text to summarize")
    p.add_argument("--text-file", default="", help="Read OCR text from this file ('-' for stdin)")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Input token budget (match training)")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max generated tokens")
    return p.parse_args()


def _load_text(args: argparse.Namespace) -> str:
    if args.text_file == "-":
        return sys.stdin.read().strip()
    if args.text_file:
        return Path(args.text_file).resolve().read_text(encoding="utf-8").strip()
    return args.text.strip()


def main() -> int:
    args = parse_args()
    summarizer = Summarizer(
        adapter_dir=args.adapter_dir,
        base_model_id=args.base_model,
        merged_model_dir=args.merged_model,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Loaded on: {summarizer.device}")

    text = _load_text(args)
    if text:
        print("\n=== Summary ===\n")
        print(summarizer.summarize(text))
        print()
        return 0

    # Interactive mode: one OCR page per line (their OCR uses literal "(newline)").
    print("\nInteractive mode. Paste one OCR page and press Enter. Blank line or 'quit' to exit.\n")
    while True:
        try:
            line = input("OCR> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in {"quit", "exit"}:
            break
        print("\n=== Summary ===\n")
        print(summarizer.summarize(line))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
