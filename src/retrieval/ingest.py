import os
import json
import re
from typing import List, Dict


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def chunk_document(
    text: str,
    chunk_size: int = 350,
    chunk_overlap: int = 50,
    source_name: str = "nvda_10k",
) -> List[Dict]:
    """
    Splits raw filings text into fixed-word sliding window with unique chunk IDs
    """

    words = text.split()
    chunks = []
    chunk_idx = 0
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunk_record = {
            "chunk_id": f"{source_name}_chunk_{chunk_idx:04d}",
            "text": chunk_text,
            "metadata": {
                "source": source_name,
                "start_word": start,
                "end_word": min(end, len(words)),
                "word_count": len(chunk_words),
            },
        }
        chunks.append(chunk_record)
        chunk_idx += 1

        step = chunk_size - chunk_overlap
        if step <= 0:
            raise ValueError("chunk_size must be strictly greater than chunk_overlap")
        start += step
    return chunks


def save_chunks(chunks: List[Dict], output_path: str):
    """Saves chunks into JSONL format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"Indexed and saved {len(chunks)} chunks to: {output_path}")


if __name__ == "__main__":
    raw_filing_path = os.path.join("data", "raw", "sec_10k_filing.txt")
    output_chunk_path = os.path.join("data", "processed", "chunks.jsonl")

    if not os.path.exists(raw_filing_path):
        print(f"Error: Could not find {raw_filing_path}. Run fetch_sec.py first")
    else:
        print(f"Reading raw filing from {raw_filing_path}...")
        with open(raw_filing_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        cleaned = clean_text(raw_content)
        chunks = chunk_document(
            cleaned, chunk_size=350, chunk_overlap=50, source_name="nvda_10k"
        )
        save_chunks(chunks=chunks, output_path=output_chunk_path)
