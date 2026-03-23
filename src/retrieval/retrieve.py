"""
 retrieve.py
 -----------
 Queries the ChromaDB lore store and returns grounded context.

 Usage:
     python -m src.retrieval.retrieve

 Input:
     data/chroma_db/ (built by indexing pipeline)

 Output:
     Retrieved chunks with metadata and relevance scores
 """

import os

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.config import config, get_logger

log = get_logger(__name__)

idx_cfg = config["indexing"]
ret_cfg = config["retrieval"]

CHROMA_DIR = idx_cfg["chroma_dir"]
COLLECTION_NAME = idx_cfg["collection_name"]
EMBEDDING_MODEL = idx_cfg["embedding_model"]
TOP_K = ret_cfg["top_k"]
USE_MRR = ret_cfg["use_mmr"]

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("OPENAI_API_KEY")
)

store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

def query(question: str) -> list:
    """Query the lore store and return relevant chunks."""
    log.info(f"Query: '{question}'")

    if USE_MRR:
        results = store.max_marginal_relevance_search(question, k=TOP_K)
    else:
        results = store.similarity_search(question, k=TOP_K)

    for i, doc in enumerate(results, 1):
        log.info(f"\n  Result {i}:")
        log.info(f"    Chapter: {doc.metadata.get('chapter', 'N/A')}")
        log.info(f"    Section: {doc.metadata.get('section', 'N/A')}")
        log.info(f"    Subsection: {doc.metadata.get('subsection',
                                                     'N/A')}")
        log.info(f"    Preview: {doc.page_content[:150]}...")

    return results

def run():
    log.info("=" * 60)
    log.info("Cyberpunk Lore — Retrieval Test")
    log.info("=" * 60)

    test_queries = [
        "Who are the Voodoo Boys?",
        "What happened to the old Net?",
        "Tell me about Pacifica",
        "What is braindance?",
        "Why is Santo Domingo called Santo Domingo?"
    ]

    for q in test_queries:
        query(q)
        log.info("")

    log.info("=" * 60)
    log.info("Retrieval test complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    run()