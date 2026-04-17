"""
PART A: Chunking Strategy
Two strategies compared:
  1. Fixed-size sliding window (chunk_size=500 chars, overlap=100)
  2. Paragraph-based (split on double newlines)

Student: Maame Yaa Adumaba Appiah | Index: 10022200171
Course : CS4241 - Introduction to Artificial Intelligence | ACity 2026

Design rationale:
  - chunk_size=500 chars (~80-100 tokens) keeps chunks small enough for precise
    retrieval yet large enough to hold meaningful context.
  - overlap=100 chars prevents boundary cutoffs from losing key sentences.
  - Paragraph chunking respects natural document structure and avoids splitting
    mid-sentence; better for the budget PDF which uses clear section headings.
"""


def chunk_sliding_window(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Fixed-size sliding window chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_by_paragraph(text: str, max_size: int = 800) -> list[str]:
    """Paragraph-aware chunking — splits on double newlines, merges short paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= max_size:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para
    if buffer:
        chunks.append(buffer)
    return chunks


def chunk_csv_records(records: list[str]) -> list[str]:
    """Each election-region-year block is its own chunk (natural boundary)."""
    return [r for r in records if r.strip()]


def add_metadata(chunks: list[str], source: str) -> list[dict]:
    """Wrap each chunk with source metadata."""
    return [{"id": i, "source": source, "text": c} for i, c in enumerate(chunks)]
