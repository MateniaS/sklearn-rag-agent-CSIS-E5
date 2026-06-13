import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))

from qdrant_config import create_qdrant_client

EMBEDDING_MODEL = "text-embedding-3-small"


def create_query_embedding(openai_client, question: str):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    return response.data[0].embedding


def search_qdrant_with_filter(
    qdrant_client,
    collection_name: str,
    query_vector,
    top_k: int,
    topic_filter: str
):
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="topic",
                match=MatchValue(value=topic_filter)
            )
        ]
    )

    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True
    )

    return response.points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--topic-filter", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(dotenv_path=BASE_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found.")

    openai_client = OpenAI()
    qdrant_client = create_qdrant_client()

    query_vector = create_query_embedding(openai_client, args.question)

    results = search_qdrant_with_filter(
        qdrant_client=qdrant_client,
        collection_name=args.collection_name,
        query_vector=query_vector,
        top_k=args.top_k,
        topic_filter=args.topic_filter
    )

    print(f"\nQuestion: {args.question}")
    print(f"Collection: {args.collection_name}")
    print(f"Metadata filter: topic = {args.topic_filter}")
    print(f"Top {args.top_k} filtered retrieved chunks:\n")

    markdown_lines = [
        f"# Filtered retrieval test — {args.collection_name}",
        "",
        f"Question: {args.question}",
        "",
        f"Metadata filter: topic = `{args.topic_filter}`",
        "",
        f"Top {args.top_k} retrieved chunks:",
        ""
    ]

    for i, point in enumerate(results, start=1):
        payload = point.payload
        text_preview = payload.get("text", "")[:700].replace("\n", " ")

        print(f"{i}. Score: {point.score:.4f}")
        print(f"   Chunk ID: {payload.get('chunk_id')}")
        print(f"   Doc ID: {payload.get('doc_id')}")
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
            f"- Doc ID: {payload.get('doc_id')}",
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

    safe_collection = args.collection_name.replace("/", "_")
    safe_topic = args.topic_filter.replace("/", "_")
    output_path = BASE_DIR / "outputs" / f"filtered_retrieval_{safe_collection}_{safe_topic}.md"

    output_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    print(f"Saved filtered retrieval test to: {output_path}")


if __name__ == "__main__":
    main()
