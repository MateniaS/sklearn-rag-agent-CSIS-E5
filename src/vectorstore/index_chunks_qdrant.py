import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.models import Distance, VectorParams, PointStruct


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))

from qdrant_config import create_qdrant_client

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 50


def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def create_embeddings(openai_client, texts):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def recreate_collection(qdrant_client, collection_name: str):
    existing_collections = [
        collection.name
        for collection in qdrant_client.get_collections().collections
    ]

    if collection_name in existing_collections:
        qdrant_client.delete_collection(collection_name=collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE
        )
    )


def index_chunks(input_file: Path, collection_name: str):
    load_dotenv(dotenv_path=BASE_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

    openai_client = OpenAI()
    qdrant_client = create_qdrant_client()

    chunks = load_chunks(input_file)

    print(f"Loaded {len(chunks)} chunks from {input_file}")
    print(f"Creating Qdrant collection: {collection_name}")

    recreate_collection(qdrant_client, collection_name)

    point_id = 1

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [item["text"] for item in batch]

        embeddings = create_embeddings(openai_client, texts)

        points = []

        for item, embedding in zip(batch, embeddings):
            payload = {
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "topic": item.get("topic"),
                "url": item.get("url"),
                "section": item.get("section"),
                "heading_level": item.get("heading_level"),
                "chunking_strategy": item.get("chunking_strategy"),
                "chunk_index": item.get("chunk_index"),
                "text": item.get("text")
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )

            point_id += 1

        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

        print(f"Indexed chunks {start + 1} to {start + len(batch)}")

    print("Done.")
    print(f"Indexed {len(chunks)} chunks into collection: {collection_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--collection-name", required=True)
    args = parser.parse_args()

    input_file = BASE_DIR / args.input_file
    index_chunks(input_file, args.collection_name)


if __name__ == "__main__":
    main()
