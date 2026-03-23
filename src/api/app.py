"""
app.py
------
FastAPI backend for the Lore vs LLM comparison app.

Endpoints:
  GET  /questions      → predefined lore questions
  POST /generate/raw   → stream GPT raw response
  POST /generate/rag   → stream RAG response

Usage:
  uvicorn src.api.app:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from src.config import config, get_logger
from src.retrieval.retrieve import query
from src.prompts import RAW_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT

log = get_logger(__name__)

# - App
app = FastAPI(title="Cyberpunk Lore vs LLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL = config["api"]["chat_model"]

# - Request model
class GenerateRequest(BaseModel):
    question: str
    api_key: str

# - Predefined Questions
QUESTIONS = [
    "Who are the Voodoo Boys?",
    "What happened to the old Net?",
    "Tell me about Pacifica",
    "What is braindance?",
    "Why is Santo Domingo called Santo Domingo?",
    "Who is Rogue?",
    "What was the Fourth Corporate War?",
    "What are the Tyger Claws known for?",
]

def stream_raw(question: str, api_key: str):
    """Generator that streams the raw GPT response"""
    client = OpenAI(api_key=api_key)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RAW_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content

        if content:
            yield content

def stream_rag(question: str, api_key: str):
    """Generator that streams the RAG response"""
    client = OpenAI(api_key=api_key)

    # Retrieve relevant chunks
    chunks = query(question)

    # Build context
    context_parts = []

    for i, doc in enumerate(chunks, 1):
        chapter = doc.metadata.get("chapter", "Unknown")
        section = doc.metadata.get("section", "Unknown")
        context_parts.append(f"[Source {i}: {chapter} > {section}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    system = RAG_SYSTEM_PROMPT.replace("{context}", context)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ],
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content

        if content:
            yield content

@app.get("/questions")
def get_questions():
    return {"questions": QUESTIONS}


@app.post("/generate/raw")
def generate_raw(req: GenerateRequest):
    return StreamingResponse(
        stream_raw(req.question, req.api_key),
        media_type="text/plain"
    )

@app.post("/generate/rag")
def generate_rag(req: GenerateRequest):
    return StreamingResponse(
        stream_rag(req.question, req.api_key),
        media_type="text/plain"
    )