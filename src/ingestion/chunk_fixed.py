import json
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = BASE_DIR / "configs" / "v1_fixed_chunking.json"
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "v1_fixed_chunks.jsonl"
STATS_FILE = BASE_DIR / "outputs" / "chunk_stats_v1.csv"
SAMPLES_FILE = BASE_DIR / "outputs" / "sample_chunks_v1.json"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
STATS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_size_chunks(text: str, chunk_size: int, overlap: int):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if start >= text_length:
            break

    return chunks


def main():
    config = load_json(CONFIG_PATH)

    chunk_size = int(config["chunk_size"])
    chunk_overlap = int(config["chunk_overlap"])

    all_chunks = []
    stats_rows = []

    txt_files = sorted(RAW_DIR.glob("D*.txt"))

    for txt_file in txt_files:
        doc_id = txt_file.stem
        metadata_file = RAW_DIR / f"{doc_id}_metadata.json"

        if not metadata_file.exists():
            print(f"Skipping {doc_id}: metadata file not found.")
            continue

        metadata = load_json(metadata_file)
        text = txt_file.read_text(encoding="utf-8")

        chunks = fixed_size_chunks(text, chunk_size, chunk_overlap)

        chunk_lengths = [len(chunk) for chunk in chunks]

        stats_rows.append({
            "doc_id": doc_id,
            "title": metadata["title"],
            "topic": metadata["topic"],
            "num_chunks": len(chunks),
            "avg_chunk_length": round(sum(chunk_lengths) / len(chunk_lengths), 2) if chunk_lengths else 0,
            "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
            "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0
        })

        for i, chunk_text in enumerate(chunks, start=1):
            chunk_id = f"{doc_id}_v1_{i:03d}"

            record = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "title": metadata["title"],
                "topic": metadata["topic"],
                "url": metadata["url"],
                "chunking_strategy": "v1_fixed",
                "chunk_index": i,
                "text": chunk_text
            }

            all_chunks.append(record)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in all_chunks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with STATS_FILE.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "doc_id",
            "title",
            "topic",
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
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Chunks saved to: {OUTPUT_FILE}")
    print(f"Stats saved to: {STATS_FILE}")
    print(f"Samples saved to: {SAMPLES_FILE}")


if __name__ == "__main__":
    main()
