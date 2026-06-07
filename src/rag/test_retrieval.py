import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient


BASE_DIR = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL = "text-embedding-3-small"


def create_query_embedding(openai_client, question: str):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    return response.data[0].embedding


def search_qdrant(qdrant_client, collection_name: str, query_vector, top_k: int):
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )
    return response.points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(dotenv_path=BASE_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found.")

    openai_client = OpenAI()
    qdrant_client = QdrantClient(host="localhost", port=6333)

    query_vector = create_query_embedding(openai_client, args.question)

    results = search_qdrant(
        qdrant_client=qdrant_client,
        collection_name=args.collection_name,
        query_vector=query_vector,
        top_k=args.top_k
    )

    print(f"\nQuestion: {args.question}")
    print(f"Collection: {args.collection_name}")
    print(f"Top {args.top_k} retrieved chunks:\n")

    markdown_lines = [
        f"# Retrieval test — {args.collection_name}",
        "",
        f"Question: {args.question}",
        "",
        f"Top {args.top_k} retrieved chunks:",
        ""
    ]

    for i, point in enumerate(results, start=1):
        payload = point.payload
        text_preview = payload.get("text", "")[:600].replace("\n", " ")

        print(f"{i}. Score: {point.score:.4f}")
        print(f"   Chunk ID: {payload.get('chunk_id')}")
        print(f"   Title: {payload.get('title')}")
        print(f"   Topic: {payload.get('topic')}")
        print(f"   Section: {payload.get('section')}")
        print(f"   URL: {payload.get('url')}")
        print(f"   Preview: {text_preview}")
        print()

        markdown_lines.extend([
            f"## Result {i}",
            "",
            f"- Score: {point.score:.4f}",
            f"- Chunk ID: {payload.get('chunk_id')}",
            f"- Title: {payload.get('title')}",
            f"- Topic: {payload.get('topic')}",
            f"- Section: {payload.get('section')}",
            f"- URL: {payload.get('url')}",
            "",
            "Preview:",
            "",
            text_preview,
            ""
        ])

    safe_name = args.collection_name.replace("/", "_")
    output_path = BASE_DIR / "outputs" / f"retrieval_test_{safe_name}.md"
    output_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    print(f"Saved retrieval test to: {output_path}")


if __name__ == "__main__":
    main()
