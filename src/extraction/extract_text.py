"""
extract_text.py
---------------
Extracts text from the Cyberpunk 2077 lore book using GPT-4o Vision.

Walks through each page, converts to image, sends to GPT-4o, saves output.
Safe to re-run — already-extracted pages are skipped automatically.

Usage:
    python -m src.extraction.extract_text

Output:
    data/extracted/page_001.txt ... page_153.txt
    data/extracted/full_book.txt  (combined clean text)
    data/extracted/run_stats.json (cost + extraction summary)
"""

import os
import base64
import json
import time
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI, RateLimitError, APIError
from src.config import config, get_logger

log = get_logger(__name__)

cfg           = config["extraction"]
PDF_PATH      = cfg["pdf_path"]
OUTPUT_DIR    = Path(cfg["output_dir"])
MODEL         = cfg["model"]
DPI           = cfg["dpi"]
MAX_TOKENS    = cfg["max_tokens"]
REQUEST_DELAY = cfg["request_delay"]

COMBINED_OUTPUT = OUTPUT_DIR / "full_book.txt"
STATS_OUTPUT    = OUTPUT_DIR / "run_stats.json"

SYSTEM_PROMPT = """You are extracting text from a page of 'The World of Cyberpunk 2077', a lore art book.

Extract ALL readable text from the image exactly as it appears.
- Preserve paragraph breaks with blank lines
- Preserve headings and section titles
- Skip page numbers and decorative elements
- If a page is purely an image with no text, respond with: [IMAGE ONLY]
- Do not add commentary or descriptions of images
- Output only the extracted text, nothing else"""

# GPT-4o-mini vision pricing (as of 2024)
# High detail: 85 + (tiles * 170) tokens per image
# At 150 DPI a typical page is ~1 tile = ~255 tokens
COST_PER_1K_INPUT_TOKENS  = 0.000150  # $0.150 per 1M
COST_PER_1K_OUTPUT_TOKENS = 0.000600  # $0.600 per 1M

# ── Setup ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def page_to_base64(page: fitz.Page, dpi: int = DPI) -> str:
    """Render a PDF page to a base64-encoded PNG string."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def extract_page_text(img_b64: str, retries: int = 3) -> tuple[str, int, int]:
    """
    Send a base64 image to GPT-4o Vision and return extracted text.

    Returns:
        (text, input_tokens, output_tokens)

    Retries up to `retries` times on rate limit or API errors
    with exponential backoff.
    """
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=MAX_TOKENS
            )
            text          = response.choices[0].message.content.strip()
            input_tokens  = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            return text, input_tokens, output_tokens

        except RateLimitError:
            wait = 2 ** attempt
            log.warning(f"Rate limit hit. Retrying in {wait}s... (attempt {attempt}/{retries})")
            time.sleep(wait)

        except APIError as e:
            wait = 2 ** attempt
            log.warning(f"API error: {e}. Retrying in {wait}s... (attempt {attempt}/{retries})")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {retries} retries")


def is_valid_extraction(text: str) -> bool:
    """Sanity check — flag suspiciously short or error extractions."""
    if text.startswith("[ERROR"):
        return False
    if text == "[IMAGE ONLY]":
        return True  # Valid, just no text
    if len(text) < 10:
        return False
    return True


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens  / 1000) * COST_PER_1K_INPUT_TOKENS +
        (output_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS
    )


# ── Main Extraction Loop ──────────────────────────────────────────────────────

def run():
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)

    log.info(f"PDF loaded: {total_pages} pages")
    log.info(f"Model: {MODEL} | DPI: {DPI} | Output: {OUTPUT_DIR}/")

    stats = {
        "total_pages": total_pages,
        "extracted": 0,
        "skipped": 0,
        "image_only": 0,
        "errors": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "failed_pages": []
    }

    for page_num in range(total_pages):
        output_file = OUTPUT_DIR / f"page_{page_num + 1:03d}.txt"

        # Skip already-extracted pages
        if output_file.exists():
            existing = output_file.read_text(encoding="utf-8").strip()
            if is_valid_extraction(existing):
                log.info(f"[skip] Page {page_num + 1}/{total_pages}")
                stats["skipped"] += 1
                continue

        log.info(f"[extract] Page {page_num + 1}/{total_pages}...")

        try:
            img_b64 = page_to_base64(doc[page_num])
            text, input_tok, output_tok = extract_page_text(img_b64)

            output_file.write_text(text, encoding="utf-8")

            # Update stats
            stats["extracted"] += 1
            stats["total_input_tokens"]  += input_tok
            stats["total_output_tokens"] += output_tok
            stats["estimated_cost_usd"]  += estimate_cost(input_tok, output_tok)

            if text == "[IMAGE ONLY]":
                stats["image_only"] += 1

            log.info(
                f"  done | {len(text)} chars | "
                f"{input_tok}+{output_tok} tokens | "
                f"${estimate_cost(input_tok, output_tok):.4f}"
            )

        except Exception as e:
            log.error(f"  FAILED page {page_num + 1}: {e}")
            output_file.write_text(f"[ERROR: {e}]", encoding="utf-8")
            stats["errors"] += 1
            stats["failed_pages"].append(page_num + 1)

        time.sleep(REQUEST_DELAY)

    # ── Combine all pages ─────────────────────────────────────────────────────
    log.info("Combining pages into full_book.txt...")
    pages = sorted(OUTPUT_DIR.glob("page_*.txt"))
    combined = []

    for page_file in pages:
        text = page_file.read_text(encoding="utf-8").strip()
        if text and text not in ("[IMAGE ONLY]",) and not text.startswith("[ERROR"):
            combined.append(text)

    COMBINED_OUTPUT.write_text("\n\n".join(combined), encoding="utf-8")
    total_chars = len(COMBINED_OUTPUT.read_text())

    # ── Save run stats ────────────────────────────────────────────────────────
    stats["total_chars"] = total_chars
    STATS_OUTPUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    log.info(f"{'='*60}")
    log.info(f"Extraction complete")
    log.info(f"  Pages extracted : {stats['extracted']}")
    log.info(f"  Pages skipped   : {stats['skipped']}")
    log.info(f"  Image only      : {stats['image_only']}")
    log.info(f"  Errors          : {stats['errors']}")
    log.info(f"  Total chars     : {total_chars:,}")
    log.info(f"  Total tokens    : {stats['total_input_tokens'] + stats['total_output_tokens']:,}")
    log.info(f"  Estimated cost  : ${stats['estimated_cost_usd']:.4f}")
    if stats["failed_pages"]:
        log.warning(f"  Failed pages    : {stats['failed_pages']}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    run()
