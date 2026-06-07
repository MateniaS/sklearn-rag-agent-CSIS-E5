import csv
import json
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = BASE_DIR / "configs" / "v2_structured_chunking.json"
SOURCES_FILE = BASE_DIR / "data" / "corpus_sources.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "v2_structured_chunks.jsonl"
STATS_FILE = BASE_DIR / "outputs" / "chunk_stats_v2.csv"
SAMPLES_FILE = BASE_DIR / "outputs" / "sample_chunks_v2.json"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
STATS_FILE.parent.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_sections_from_html(html: str):
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body

    if main is None:
        return []

    sections = []
    current_section = "General"
    current_level = 0
    current_parts = []

    useful_tags = ["h1", "h2", "h3", "h4", "p", "li", "pre", "dt", "dd"]

    for element in main.find_all(useful_tags):
        text = clean_text(element.get_text(" ", strip=True))

        if not text:
            continue

        if element.name in ["h1", "h2", "h3", "h4"]:
            if current_parts:
                sections.append({
                    "section": current_section,
                    "heading_level": current_level,
                    "text": "\n".join(current_parts)
                })

            current_section = text
            current_level = int(element.name[1])
            current_parts = []
        else:
            if len(text) > 20:
                current_parts.append(text)

    if current_parts:
        sections.append({
            "section": current_section,
            "heading_level": current_level,
            "text": "\n".join(current_parts)
        })

    return sections


def split_long_section(section_text: str, max_chunk_size: int):
    paragraphs = [p.strip() for p in section_text.split("\n") if p.strip()]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""

            for i in range(0, len(paragraph), max_chunk_size):
                chunks.append(paragraph[i:i + max_chunk_size].strip())

        elif len(current) + len(paragraph) + 2 <= max_chunk_size:
            current += "\n" + paragraph if current else paragraph
        else:
            chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def main():
    config = load_json(CONFIG_PATH)
    max_chunk_size = int(config["max_chunk_size"])

    df = pd.read_csv(SOURCES_FILE)

    all_chunks = []
    stats_rows = []

    for _, row in df.iterrows():
        doc_id = str(row["doc_id"]).strip()
        title = str(row["title"]).strip()
        topic = str(row["topic"]).strip()
        url = str(row["url"]).strip()

        print(f"Processing {doc_id}: {title}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        sections = extract_sections_from_html(response.text)

        doc_chunks = []
        chunk_counter = 1

        for section in sections:
            section_name = section["section"]
            heading_level = section["heading_level"]
            section_text = section["text"]

            section_chunks = split_long_section(section_text, max_chunk_size)

            for chunk_text in section_chunks:
                chunk_id = f"{doc_id}_v2_{chunk_counter:03d}"

                record = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "topic": topic,
                    "url": url,
                    "section": section_name,
                    "heading_level": heading_level,
                    "chunking_strategy": "v2_structured",
                    "chunk_index": chunk_counter,
                    "text": chunk_text
                }

                all_chunks.append(record)
                doc_chunks.append(record)
                chunk_counter += 1

        chunk_lengths = [len(chunk["text"]) for chunk in doc_chunks]
        unique_sections = sorted(set(chunk["section"] for chunk in doc_chunks))

        stats_rows.append({
            "doc_id": doc_id,
            "title": title,
            "topic": topic,
            "num_sections": len(unique_sections),
            "num_chunks": len(doc_chunks),
            "avg_chunk_length": round(sum(chunk_lengths) / len(chunk_lengths), 2) if chunk_lengths else 0,
            "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
            "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0
        })

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in all_chunks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with STATS_FILE.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "doc_id",
            "title",
            "topic",
            "num_sections",
            "num_chunks",
            "avg_chunk_length",
            "min_chunk_length",
            "max_chunk_length"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)

    samples = all_chunks[:5]
    SAMPLES_FILE.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Done.")
    print(f"Total structured chunks created: {len(all_chunks)}")
    print(f"Chunks saved to: {OUTPUT_FILE}")
    print(f"Stats saved to: {STATS_FILE}")
    print(f"Samples saved to: {SAMPLES_FILE}")


if __name__ == "__main__":
    main()
