"""
  generate.py
  -----------
  Takes retrieved chunks and a question, sends to GPT for a grounded
  answer.

  Usage:
      Called by the retrieval pipeline or API layer.
  """

import os

from openai import OpenAI
from src.config import config, get_logger
from src.prompts import RAW_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT

log = get_logger(__name__)

MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_raw(question: str) -> str:
    """Ask GPT directly - no context, just training data."""
    log.info(f"[raw] Generating response for: '{question}")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RAW_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content.strip()
    log.info(f"[raw] Done ({len(answer)} chars)")
    return answer

def generate_rag(question: str, chunks: list) -> str:
    """Ask GPT with retrieved chunks as context."""
    log.info(f"[rag] Generating response: '{question}")

    # Build context from retrieved chunks
    context_parts = []

    for i, doc in enumerate(chunks, 1):
        chapter = doc.metadata.get("chapter", "Unknown")
        section = doc.metadata.get("section", "Unknown")
        context_parts.append(f"[Source {i}: {chapter} > {section}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    # Fill the context into the system prompt
    system = RAG_SYSTEM_PROMPT.replace("{context}", context)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
              {"role": "system", "content": system},
              {"role": "user", "content": question}
          ]
    )

    answer = response.choices[0].message.content.strip()
    log.info(f"[rag] Done ({len(answer)} chars)")
    return answer


def run():
    from src.retrieval.retrieve import query

    test_questions = [
        "Who are the Voodoo Boys?",
        "Why is Santo Domingo called Santo Domingo?"
    ]

    for q in test_questions:
        log.info("=" * 60)
        chunks = query(q)

        raw_answer = generate_raw(q)
        rag_answer = generate_rag(q, chunks)

        print(f"\n{'=' * 60}")
        print(f"QUESTION: {q}")
        print(f"\n--- GPT RAW ---")
        print(raw_answer)
        print(f"\n--- RAG ---")
        print(rag_answer)
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run()