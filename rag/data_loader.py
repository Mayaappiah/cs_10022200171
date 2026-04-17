"""
PART A: Data Engineering & Preparation
Loads and cleans Ghana Election CSV and 2025 Budget PDF.

Student: Maame Yaa Adumaba Appiah | Index: 10022200171
Course : CS4241 - Introduction to Artificial Intelligence | ACity 2026
"""
import os
import re
import csv
import fitz  # pymupdf


def load_election_csv(csv_path: str) -> list[dict]:
    """Load and clean Ghana election results CSV."""
    records = []
    with open(csv_path, encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f)
        for row in reader:
            # Clean: strip whitespace, non-breaking spaces, and BOM artefacts
            cleaned = {k.strip(): v.strip().replace("\xa0", " ") for k, v in row.items()}
            # Remove rows with missing core fields
            if not cleaned.get("Year") or not cleaned.get("New Region") or not cleaned.get("Candidate"):
                continue
            # Normalize vote percentage — remove trailing %
            votes_pct = cleaned.get("Votes(%)", "0").replace("%", "").strip()
            cleaned["Votes(%)"] = votes_pct
            # Normalize votes to integer
            votes_raw = cleaned.get("Votes", "0").replace(",", "").strip()
            cleaned["Votes"] = votes_raw
            records.append(cleaned)
    return records


def election_records_to_text(records: list[dict]) -> list[str]:
    """Convert election records to natural-language text chunks (one per region/year group)."""
    # Group by Year + New Region
    groups: dict[tuple, list] = {}
    for r in records:
        key = (r["Year"], r["New Region"])
        groups.setdefault(key, []).append(r)

    chunks = []
    for (year, region), rows in sorted(groups.items()):
        lines = [f"Ghana Presidential Election {year} — {region}:"]
        for r in rows:
            party = r.get("Party", r.get("Code", "?"))
            candidate = r["Candidate"]
            votes = r["Votes"]
            pct = r["Votes(%)"]
            lines.append(f"  {candidate} ({party}): {votes} votes ({pct}%)")
        chunks.append("\n".join(lines))
    return chunks


def load_budget_pdf(pdf_path: str) -> str:
    """Extract raw text from Ghana 2025 Budget Statement PDF."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text()
        # Basic cleaning: collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if text:
            pages_text.append(text)
    doc.close()
    return "\n\n".join(pages_text)


def clean_text(text: str) -> str:
    """Remove noise from extracted PDF text."""
    # Remove page numbers standalone lines like "  12  "
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove repeated dashes/underscores
    text = re.sub(r"[-_]{4,}", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
