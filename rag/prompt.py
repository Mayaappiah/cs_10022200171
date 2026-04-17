"""
PART C: Prompt Engineering & Generation
- Structured prompt template with context injection
- Hallucination control via explicit grounding instructions
- Context window management: truncate to fit token budget

Student: Maame Yaa Adumaba Appiah | Index: 10022200171
Course : CS4241 - Introduction to Artificial Intelligence | ACity 2026
"""

MAX_CONTEXT_CHARS = 6000  # ~1500 tokens — leaves room for system + response


SYSTEM_PROMPT = """You are an AI assistant for Academic City University, specialised in:
1. Ghana Presidential Election results (1992–2020) by region and party.
2. Ghana's 2025 Budget Statement and Economic Policy.

Rules you MUST follow:
- Answer ONLY using the provided context. Do not invent figures, names, or events.
- If the context does not contain enough information, say: "I don't have enough information in the provided documents to answer that."
- Quote specific numbers and sources when available (e.g., "According to the 2025 Budget...").
- Be concise and factual."""


def build_prompt(query: str, chunks: list[dict], conversation_history: list[dict] | None = None) -> str:
    """
    Construct the full prompt:
      [System] → [Memory] → [Context] → [User Query]
    """
    # --- context window management ---
    context_parts = []
    total = 0
    for chunk in chunks:
        text = chunk["text"]
        src = chunk.get("source", "unknown")
        score = chunk.get("score", 0.0)
        entry = f"[Source: {src} | Relevance: {score:.3f}]\n{text}"
        if total + len(entry) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(entry)
        total += len(entry)

    context_block = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

    # --- conversation memory (last 3 turns) ---
    history_block = ""
    if conversation_history:
        recent = conversation_history[-3:]
        turns = []
        for turn in recent:
            turns.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
        history_block = "\n\n".join(turns)

    prompt_parts = [SYSTEM_PROMPT]
    if history_block:
        prompt_parts.append(f"\n\n### Conversation History:\n{history_block}")
    prompt_parts.append(f"\n\n### Retrieved Context:\n{context_block}")
    prompt_parts.append(f"\n\n### User Question:\n{query}\n\n### Answer:")

    return "".join(prompt_parts)


def build_no_rag_prompt(query: str) -> str:
    """Pure LLM prompt (no retrieval) — used in Part E comparison."""
    return (
        "You are a helpful assistant. Answer the following question to the best of your knowledge.\n\n"
        f"Question: {query}\n\nAnswer:"
    )
