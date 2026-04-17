"""
PART D: Full RAG Pipeline Implementation
User Query → Retrieval → Context Selection → Prompt → LLM → Response
Logging at every stage.

Student: Maame Yaa Adumaba Appiah | Index: 10022200171
Course : CS4241 - Introduction to Artificial Intelligence | ACity 2026
"""
from __future__ import annotations
import json
import os
import time
import datetime
from rag.retriever import VectorStore
from rag.prompt import build_prompt, build_no_rag_prompt

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "pipeline_log.jsonl")


def _call_llm(prompt: str, api_key: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """Call Anthropic Claude API."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def run_rag(
    query: str,
    store: VectorStore,
    api_key: str,
    top_k: int = 5,
    use_hybrid: bool = True,
    conversation_history: list[dict] | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """
    Full RAG pipeline. Returns a dict with all stage outputs for display + logging.
    """
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "mode": "hybrid" if use_hybrid else "vector",
    }

    # Stage 1: Retrieval
    t0 = time.time()
    if use_hybrid:
        retrieved = store.hybrid_retrieve(query, top_k=top_k)
    else:
        retrieved = store.retrieve_with_expansion(query, top_k=top_k)
    retrieval_ms = round((time.time() - t0) * 1000, 1)
    log_entry["retrieval_ms"] = retrieval_ms
    log_entry["retrieved_chunks"] = [
        {"source": c["source"], "score": round(c["score"], 4), "snippet": c["text"][:120]}
        for c in retrieved
    ]

    # Stage 2: Prompt construction
    prompt = build_prompt(query, retrieved, conversation_history)
    log_entry["prompt_length_chars"] = len(prompt)

    # Stage 3: LLM generation
    t1 = time.time()
    response = _call_llm(prompt, api_key, model)
    generation_ms = round((time.time() - t1) * 1000, 1)
    log_entry["generation_ms"] = generation_ms
    log_entry["response_length_chars"] = len(response)

    # Write log
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "query": query,
        "retrieved": retrieved,
        "prompt": prompt,
        "response": response,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
    }


def run_pure_llm(query: str, api_key: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """Part E: Pure LLM without retrieval for comparison."""
    prompt = build_no_rag_prompt(query)
    return _call_llm(prompt, api_key, model)
