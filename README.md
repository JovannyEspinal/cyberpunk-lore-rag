# Cyberpunk Lore RAG

A side project built during week 2 of a 17-week agentic AI program. Takes The World of Cyberpunk 2077 — a 153-page scanned art/lore book with no selectable text — and turns it into a queryable RAG knowledge base.

The frontend puts GPT-4o and the RAG system side by side so you can see the difference. GPT makes stuff up. RAG stays grounded.

## The demo

Ask both systems: "Why is Santo Domingo called Santo Domingo?"

GPT confidently says it's named after the Dominican Republic capital, reflecting cultural immigration patterns. Whole paragraph. Sounds great. None of it is in the lore book.

The RAG system says it doesn't have info on the name, then tells you what it actually knows — factories, corporate suburbs, 6th Street gang territory. Less impressive sounding. Actually correct.

## How it works

### Extraction

The PDF is scanned images — no selectable text. GPT-4o Vision OCRs every page, one at a time.

- 153 pages, $0.69 total
- Per-page output files (safe to re-run if interrupted)
- Cost tracking and run stats logged to JSON

### Markdown conversion

Raw OCR text has inconsistent formatting — code blocks, missing headers, page numbers mixed in. A conversion script sends each chapter through GPT-4o-mini to produce clean markdown with proper `#`, `##`, `###` hierarchy.

This structure is what makes the chunking strategy work.

### Chunking + indexing

Markdown header splitting. The book already has chapters and sections, so the headers become free metadata on every chunk.

A query about the Voodoo Boys returns chunks tagged:
```
chapter: Chapter 5: Law and Disorder
section: Gangs: The Bad and The Ugly
subsection: Voodoo Boys
```

Each chunk is also enriched with its full header path in the text content, so the embedding captures hierarchical context.

Embedded with `text-embedding-3-small`, stored in ChromaDB.

### Retrieval + generation

MMR search (relevance + diversity) pulls the top 8 chunks. Those chunks are injected into a system prompt that constrains GPT to only use the provided context.

Two endpoints stream responses in parallel — one raw (training data only), one grounded (RAG).

### Frontend

Cyberpunk-themed terminal interface. Dark background, hot pink borders, cyan text, scanlines, glitch effects. Type a question or click a preset, enter your API key, and watch both responses stream in side by side.

## Architecture

```
src/
  config.py              Shared config + logging
  prompts.py             System prompts (single source of truth)
  extraction/
    extract_text.py      PDF → per-page text via GPT-4o Vision
    convert_to_markdown.py  Raw text → structured markdown
  indexing/
    index.py             Chunk → embed → store in ChromaDB
  retrieval/
    retrieve.py          Query ChromaDB with MMR
    generate.py          RAG vs raw generation
  api/
    app.py               FastAPI with streaming endpoints

frontend/
  index.html             Single-file cyberpunk terminal UI

data/                    (gitignored)
  extracted/             OCR output + markdown chapters
  chroma_db/             Vector store
```

Endpoints:
```
GET  /questions       Predefined lore questions
POST /generate/raw    Stream GPT raw response
POST /generate/rag    Stream RAG-grounded response
```

## Stack

| Component | Tool | Why |
|---|---|---|
| PDF extraction | GPT-4o Vision | Scanned pages, no selectable text |
| Markdown conversion | GPT-4o-mini | Structural formatting at scale |
| Chunking | LangChain MarkdownHeaderTextSplitter | Headers become free metadata |
| Embeddings | text-embedding-3-small | Cheap, good quality |
| Vector store | ChromaDB | Persistent, metadata filtering, MMR |
| Backend | FastAPI + StreamingResponse | Real-time token streaming |
| Frontend | Vanilla HTML/CSS/JS | Single file, no build step |

## Setup

**Requirements:** Python 3.12+, OpenAI API key

```bash
git clone https://github.com/JovannyEspinal/cyberpunk-lore-rag.git
cd cyberpunk-lore-rag

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI key.

### Run the pipeline

```bash
# 1. Extract text from PDF (needs PDF in configured path)
python -m src.extraction.extract_text

# 2. Convert to structured markdown
python -m src.extraction.convert_to_markdown

# 3. Chunk, embed, store
python -m src.indexing.index
```

### Run the app

```bash
# Start API
uvicorn src.api.app:app --reload --port 8000

# Open frontend
open frontend/index.html
```

Your API key is entered in the UI and sent directly to OpenAI per request. Never stored.

## Cost

Total pipeline cost: under $1.

| Step | Cost |
|---|---|
| PDF extraction (153 pages) | $0.69 |
| Markdown conversion (6 chapters) | ~$0.10 |
| Embedding (~200 chunks) | ~$0.02 |
| Per query (retrieval + generation) | ~$0.01 |

## Limitations

- OCR quality depends on page design — heavily stylized text may extract imperfectly
- The lore book doesn't cover everything in the game — post-launch DLC content isn't included
- Chunk boundaries occasionally split mid-section despite overlap
- Metadata labels have minor artifacts from the conversion step
