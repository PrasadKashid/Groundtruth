import os
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "EvaluationProject research_student@example.com",
    "Accept-Encoding": "gzip, deflate",
}

CIK = "0001045810"


def get_latest_10k_url(cik: str) -> str:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()

    recent = data["filings"]["recent"]
    for idx, form in enumerate(recent["form"]):
        if form == "10-K":
            acc_num = recent["accessionNumber"][idx].replace("-", "")
            primary_doc = recent["primaryDocument"][idx]
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_num}/{primary_doc}"
            print(f"Target 10-K URL: {doc_url}")
            return doc_url

    raise RuntimeError("No 10-K found.")


def download_and_clean_filing(doc_url: str, output_path: str):
    print("Downloading filing from SEC EDGAR...")
    response = requests.get(doc_url, headers=HEADERS)
    response.raise_for_status()

    print("Cleaning XBRL metadata and parsing narrative text...")
    soup = BeautifulSoup(response.content, "html.parser")

    # Strip scripts, styles, and hidden XBRL tagging metadata
    for tag in soup(["script", "style", "noscript", "ix:header", "xbrli:xbrl"]):
        tag.decompose()

    # Decompose any elements explicitly hidden by SEC reporting stylesheets
    for hidden in soup.find_all(
        attrs={"style": lambda s: s and "display:none" in s.replace(" ", "").lower()}
    ):
        hidden.decompose()

    # Extract text with newlines separating block elements
    text = soup.get_text(separator="\n")

    # Clean up empty lines and trailing whitespace
    cleaned_lines = [
        line.strip() for line in text.splitlines() if len(line.strip()) > 1
    ]
    cleaned_text = "\n".join(cleaned_lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(
        f"Success! Clean filing saved to {output_path} ({len(cleaned_text)} characters)."
    )


if __name__ == "__main__":
    output_file = os.path.join("data", "raw", "sec_10k_filing.txt")
    url = get_latest_10k_url(CIK)
    download_and_clean_filing(url, output_file)
