"""
convert_to_markdown.py
----------------------
Converts the extracted full_book.txt into properly structured markdown
for optimal chunking with MarkdownHeaderTextSplitter.

Sends text in chapter-sized chunks to GPT-4o-mini for structural conversion.
Safe to re-run — already-converted chapters are skipped.

Usage:
    python -m src.extraction.convert_to_markdown

Input:
    data/extracted/full_book.txt

Output:
    data/extracted/chapters/chapter_01.md ... chapter_06.md
    data/extracted/full_book.md (combined)
"""

import os
import re
import json
from pathlib import Path

from openai import OpenAI
from src.config import config, get_logger

log = get_logger(__name__)

INPUT_FILE  = Path("data/extracted/full_book.txt")
CHAPTER_DIR = Path("data/extracted/chapters")
OUTPUT_FILE = Path("data/extracted/full_book.md")
MODEL       = config["extraction"]["model"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Known book structure ──────────────────────────────────────────────────────
# These markers help us split the raw text into chapters reliably.
# Identified from reading the full extracted text.

CHAPTER_SPLITS = [
    {
        "number": 1,
        "title": "The Modern World",
        "start_marker": "CHAPTER 1",
        "sections": [
            "The Collapse",
            "The Fourth Corporate War",
            "Postwar",
            "Reunification",
            "Unification War / Metal Wars",
            "Modern Threats"
        ]
    },
    {
        "number": 2,
        "title": "Technology of Tomorrow",
        "start_marker": "CHAPTER 2",
        "sections": [
            "Cyberware",
            "Weapons",
            "Vehicles",
            "Braindance",
            "Netrunning"
        ]
    },
    {
        "number": 3,
        "title": "Night City",
        "start_marker": "CHAPTER 3",
        "sections": [
            "Watson", "Little China", "Kabuki",
            "Northside Industrial District",
            "Arasaka Waterfront",
            "Westbrook", "Japantown",
            "Charter Hill and North Oak",
            "City Center", "Corpo Plaza", "Downtown",
            "Heywood", "Wellsprings", "The Glen", "Vista del Rey",
            "Santo Domingo", "Arroyo", "Rancho Coronado",
            "Pacifica", "West Wind Estate and Coast View"
        ]
    },
    {
        "number": 4,
        "title": "A Vertical Slice of Night City Society in 2077",
        "start_marker": "CHAPTER 4",
        "sections": [
            "The Rich and Powerful",
            "Corporations",
            "Government Officials",
            "Celebrities",
            "The Struggling Middle Tier",
            "The Down-and-Out"
        ]
    },
    {
        "number": 5,
        "title": "Law and Disorder",
        "start_marker": "CHAPTER 5",
        "sections": [
            "Law Enforcement",
            "Corporate Agents", "NCPD", "NetWatch",
            "Gangs: The Bad and the Ugly",
            "Maelstrom", "Animals", "Voodoo Boys",
            "6th Street", "The Mox", "Tyger Claws",
            "Valentinos", "Scavengers",
            "Nomads: The Roving Refugees",
            "Aldecaldos", "Wraiths"
        ]
    },
    {
        "number": 6,
        "title": "Cyberpunks: Edgerunners and Mercs",
        "start_marker": "CHAPTER 6",
        "sections": [
            "An Interview with Rogue"
        ]
    }
]

CONVERSION_PROMPT = """You are converting extracted text from 'The World of Cyberpunk 2077' art book into clean, well-structured markdown.

This is Chapter {chapter_num}: {chapter_title}.

Known sections in this chapter: {sections}

Rules:
1. HEADERS: Use proper markdown hierarchy:
   - # for the chapter title (e.g., # Chapter 1: The Modern World)
   - ## for major sections (e.g., ## The Collapse)
   - ### for subsections (e.g., ### Little China)
   - #### for minor subsections if needed

2. REMOVE completely:
   - All ``` backtick/code block markers
   - Page numbers like [010], [073], 106, 107, etc.
   - "START" markers
   - "PORT: CONNECTED..." and similar decorative elements
   - "INFO NO." entries
   - "ADVERTISEMENT LINK" blocks and ad content
   - "ERROR#" entries
   - Lines that are purely image descriptions or captions (e.g., "▲ 14: HERRERA DESIGNS...")
   - Navigation text like "[Cont. on page 142]"

3. PRESERVE completely:
   - ALL lore content, history, descriptions, dialogue, and narrative text
   - Quote attributions (e.g., "—THE EDITOR", "—NIX, 2077")
   - Corporation profile data (convert to clean markdown tables or structured sections)
   - Interview Q&A format
   - Thompson's Bitter Statements (as blockquotes)

4. FORMATTING:
   - Use > blockquotes for in-world quotes and editorial notes
   - Use **bold** for emphasis where appropriate
   - Clean up ALL CAPS text to normal Title Case for headers
   - Keep ALL CAPS only for very short emphasis phrases within body text
   - Separate sections with blank lines
   - For corporation profiles, use a consistent format:
     ### Corporation Name
     - **Branches:** ...
     - **Founded:** ...
     - **Founder:** ...
     etc.

5. DO NOT:
   - Add any content that isn't in the original
   - Summarize or shorten any lore text
   - Add commentary
   - Change the meaning of any text

Output ONLY the converted markdown. No explanations."""


# ── Split raw text into chapters ──────────────────────────────────────────────

def split_into_chapters(text: str) -> dict[int, str]:
    """Split the raw book text into chapters using known markers."""
    chapters = {}

    # Find chapter boundaries
    markers = []
    for ch in CHAPTER_SPLITS:
        pattern = re.compile(re.escape(ch["start_marker"]), re.IGNORECASE)
        match = pattern.search(text)
        if match:
            markers.append((ch["number"], match.start()))
        else:
            log.warning(f"Could not find marker for Chapter {ch['number']}: '{ch['start_marker']}'")

    markers.sort(key=lambda x: x[1])

    # Also capture the front matter (before Chapter 1)
    if markers:
        front_matter = text[:markers[0][1]].strip()
        if front_matter:
            chapters[0] = front_matter

    # Extract each chapter's text
    for i, (num, start) in enumerate(markers):
        if i + 1 < len(markers):
            end = markers[i + 1][1]
        else:
            end = len(text)
        chapters[num] = text[start:end].strip()

    return chapters


# ── Convert a chapter via GPT ─────────────────────────────────────────────────

MAX_CHUNK_CHARS = 15000  # Max chars per API call to avoid timeouts


def split_large_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for \n\n
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def convert_chunk(chapter_num: int, chunk_text: str, part: int, total_parts: int) -> str:
    """Send a single chunk to GPT for markdown conversion."""
    ch_info = next((c for c in CHAPTER_SPLITS if c["number"] == chapter_num), None)

    if ch_info is None:
        title = "Front Matter"
        sections = "Credits, Table of Contents, Introduction"
    else:
        title = ch_info["title"]
        sections = ", ".join(ch_info["sections"])

    part_note = ""
    if total_parts > 1:
        part_note = f"\n\nThis is part {part}/{total_parts} of this chapter. Continue the markdown formatting consistently."

    prompt = CONVERSION_PROMPT.format(
        chapter_num=chapter_num,
        chapter_title=title,
        sections=sections
    ) + part_note

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk_text}
        ],
        max_tokens=16000,
        temperature=0.1
    )

    return response.choices[0].message.content.strip()


def convert_chapter(chapter_num: int, chapter_text: str) -> str:
    """Convert a chapter, splitting into smaller chunks if needed."""
    chunks = split_large_text(chapter_text)
    total = len(chunks)

    if total == 1:
        log.info(f"  single chunk ({len(chapter_text):,} chars)")
        return convert_chunk(chapter_num, chapter_text, 1, 1)

    log.info(f"  splitting into {total} parts")
    parts = []
    for i, chunk in enumerate(chunks, 1):
        log.info(f"  [part {i}/{total}] ({len(chunk):,} chars)...")
        converted = convert_chunk(chapter_num, chunk, i, total)
        parts.append(converted)

    return "\n\n".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    CHAPTER_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Reading {INPUT_FILE}...")
    raw_text = INPUT_FILE.read_text(encoding="utf-8")
    log.info(f"  {len(raw_text):,} characters")

    log.info("Splitting into chapters...")
    chapters = split_into_chapters(raw_text)
    log.info(f"  Found {len(chapters)} sections (including front matter)")

    for ch_num in sorted(chapters.keys()):
        output_file = CHAPTER_DIR / f"chapter_{ch_num:02d}.md"

        if output_file.exists():
            log.info(f"[skip] Chapter {ch_num} — already converted")
            continue

        ch_text = chapters[ch_num]
        ch_label = "Front Matter" if ch_num == 0 else f"Chapter {ch_num}"
        log.info(f"[convert] {ch_label} ({len(ch_text):,} chars)...")

        try:
            converted = convert_chapter(ch_num, ch_text)
            output_file.write_text(converted, encoding="utf-8")
            log.info(f"  done ({len(converted):,} chars)")
        except Exception as e:
            log.error(f"  FAILED: {e}")
            output_file.write_text(f"[ERROR: {e}]", encoding="utf-8")

    # ── Combine all chapters ──────────────────────────────────────────────
    log.info("Combining chapters into full_book.md...")
    parts = []
    for f in sorted(CHAPTER_DIR.glob("chapter_*.md")):
        text = f.read_text(encoding="utf-8").strip()
        if text and not text.startswith("[ERROR"):
            parts.append(text)

    OUTPUT_FILE.write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    total_chars = len(OUTPUT_FILE.read_text())
    log.info(f"Done. {len(parts)} chapters → {OUTPUT_FILE} ({total_chars:,} chars)")


if __name__ == "__main__":
    run()
