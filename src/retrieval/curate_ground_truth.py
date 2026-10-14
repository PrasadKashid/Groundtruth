import os
import json

CHUNKS_PATH = os.path.join("data", "processed", "chunks.jsonl")
BENCHMARK_PATH = os.path.join("data", "benchmark", "ground_truth.jsonl")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]


def search_text(phrase):
    results = []
    for c in chunks:
        if phrase.lower() in c["text"].lower():
            results.append((c["chunk_id"], c["text"][:160]))
    return results


# Let's inspect where key facts actually live:
print("--- Delaware / Incorporation ---")
for cid, snippet in search_text("incorporated in delaware")[:3]:
    print(cid, "->", snippet)

print("\n--- Revenue Table / Net Income ---")
for cid, snippet in search_text("consolidated statements of income")[:3]:
    print(cid, "->", snippet)

print("\n--- Cybersecurity Risk ---")
for cid, snippet in search_text("cybersecurity")[:3]:
    print(cid, "->", snippet)

print("\n--- TSMC / Taiwan Semiconductor ---")
for cid, snippet in search_text("taiwan semiconductor manufacturing")[:3]:
    print(cid, "->", snippet)
