"""
index.py
--------
Chunks the extracted markdown chapters, embeds them, and stores in ChromaDB.

Usage:
    python -m src.indexing.index

Input:
    data/extracted/chapters/chapter_01.md ... chapter_06.md

Output:
    data/chroma_db/ (persistent ChromaDB store)
"""

import os
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import config, get_logger

log = get_logger(__name__)

cfg             = config["indexing"]
CHAPTER_DIR     = Path("data/extracted/chapters")
CHROMA_DIR      = cfg["chroma_dir"]
COLLECTION_NAME = cfg["collection_name"]
CHUNK_SIZE      = cfg["chunk_size"]
CHUNK_OVERLAP   = cfg["chunk_overlap"]
EMBEDDING_MODEL = cfg["embedding_model"]

# - Headers to split on
HEADERS_TO_SPLIT = [
    ("#", "chapter"),
    ("##", "section"),
    ("###", "subsection")
]

def chunk_chapters() -> list[Document]:
    """Read all chapter markdown files, chunk by headers."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False
    )

    all_chunks = []

    for md_file in sorted(CHAPTER_DIR.glob("chapter_*.md")):
        if md_file.name == "chapter_00.md":
            continue

        log.info(f"[chunk] {md_file.name}...")
        text = md_file.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)

        #Add source file to each chunk's metadata
        for chunk in chunks:
            chunk.metadata["source"] = md_file.name

        log.info(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    log.info(f"Total: {len(all_chunks)} chunks from {len(list(CHAPTER_DIR.glob('chapter_*.md'))) - 1} chapters")
    return all_chunks

def embed_and_store(chunks: list[Document]) -> Chroma:
    """Embed chunks and store in ChromaDB"""
    log.info(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Delete existing collection to avoid duplicates on re-run
    existing = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    existing.delete_collection()

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    log.info(f"✓ Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR} / ")
    return store

def run():
    log.info("=" * 60)
    log.info("Cyberpunk Lore — Indexing Pipeline")
    log.info("=" * 60)

    chunks = chunk_chapters()
    store = embed_and_store(chunks)

    # Quick sanity check — run a test query
    log.info("\nSanity check query: 'Who are the Voodoo Boys?'")
    results = store.similarity_search("Who are the Voodoo Boys?", k=3)

    for i, doc in enumerate(results, 1):
        log.info(f"\n  Result {i}:")
        log.info(f"    Chapter: {doc.metadata.get('chapter', 'N/A')}")
        log.info(f"    Section: {doc.metadata.get('section', 'N/A')}")
        log.info(f"    Subsection: {doc.metadata.get('subsection', 'N/A')}")
        log.info(f"    Preview: {doc.page_content[:150]}...")

    log.info("\n" + "=" * 60)
    log.info("Indexing complete.")
    log.info("=" * 60)

if __name__ == "__main__":
      run()