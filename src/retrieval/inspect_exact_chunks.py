import os
import json

CHUNKS_PATH = os.path.join("data", "processed", "chunks.jsonl")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]


def find_best_candidates(terms, title):
    print(f"\n{'='*70}\nTarget: {title}")
    matches = []
    for c in chunks:
        text_lower = c["text"].lower()
        score = sum(1 for t in terms if t.lower() in text_lower)
        if score > 0:
            matches.append((score, c["chunk_id"], c["text"]))

    # Sort by number of matched terms descending
    matches.sort(key=lambda x: x[0], reverse=True)
    for score, cid, text in matches[:3]:
        clean_text = " ".join(text.split())
        print(f"\n[{cid}] (Matched {score}/{len(terms)} terms):")
        print(f"   {clean_text[:220]}...")


# 1. State of Incorporation
find_best_candidates(["delaware", "incorporated", "state of"], "State of Incorporation")

# 2. Total Net Revenue
find_best_candidates(
    ["total revenue", "net revenue", "fiscal year 2026", "data center"],
    "Total Revenue & Data Center",
)

# 3. Gross Margin
find_best_candidates(
    ["gross margin", "cost of revenue", "percent"], "Gross Margin Performance"
)

# 4. R&D Expense
find_best_candidates(
    ["research and development", "r&d expense", "fiscal year"], "R&D Expense Increase"
)

# 5. Cybersecurity Risk
find_best_candidates(
    ["cybersecurity", "breach", "incident", "unauthorized access"],
    "Cybersecurity & Security Risks",
)

# 6. Supply Chain / Assembly
find_best_candidates(
    ["manufacturing", "packaging", "tsmc", "foundry"],
    "Supply Chain & Foundry Dependence",
)
