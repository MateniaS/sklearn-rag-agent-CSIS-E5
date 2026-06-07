import json
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parents[2]

SOURCES_FILE = BASE_DIR / "data" / "corpus_sources.csv"
OUTPUT_DIR = BASE_DIR / "data" / "raw"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body

    if main is None:
        return ""

    text = main.get_text(separator="\n")
    return clean_text(text)


def main():
    df = pd.read_csv(SOURCES_FILE)

    summary = []

    for _, row in df.iterrows():
        doc_id = str(row["doc_id"]).strip()
        title = str(row["title"]).strip()
        topic = str(row["topic"]).strip()
        url = str(row["url"]).strip()
        why_selected = str(row["why_selected"]).strip()

        print(f"Downloading {doc_id}: {title}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        text = extract_main_text(response.text)

        txt_path = OUTPUT_DIR / f"{doc_id}.txt"
        metadata_path = OUTPUT_DIR / f"{doc_id}_metadata.json"

        txt_path.write_text(text, encoding="utf-8")

        metadata = {
            "doc_id": doc_id,
            "title": title,
            "topic": topic,
            "url": url,
            "why_selected": why_selected,
            "text_file": txt_path.name,
            "num_characters": len(text),
            "num_words": len(text.split())
        }

        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        summary.append(metadata)

    summary_path = OUTPUT_DIR / "corpus_download_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nDone.")
    print(f"Saved raw corpus files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()