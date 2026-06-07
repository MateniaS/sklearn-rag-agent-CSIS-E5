import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient


BASE_DIR = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

PROMPT_FILE = BASE_DIR / "prompts" / "rag_system_prompt.txt"


def create_query_embedding(openai_client, question: str):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    return response.data[0].embedding


def retrieve_chunks(qdrant_client, collection_name: str, query_vector, top_k: int):
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )
    return response.points


def build_context(points):
    context_blocks = []

    for i, point in enumerate(points, start=1):
        payload = point.payload

        title = payload.get("title")
        url = payload.get("url")
        section = payload.get("section")
        chunk_id = payload.get("chunk_id")
        text = payload.get("text")

        block = f"""
[Context {i}]
Chunk ID: {chunk_id}
Title: {title}
Section: {section}
URL: {url}
Text:
{text}
"""
        context_blocks.append(block)

    return "\n".join(context_blocks)


def generate_answer(openai_client, question: str, context: str):
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

    user_message = f"""
Question:
{question}

Context:
{context}
"""

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content


def save_result(collection_name, question, points, answer):
    output_path = BASE_DIR / "outputs" / f"rag_answer_test_{collection_name}.md"

    lines = [
        f"# RAG answer test — {collection_name}",
        "",
        f"## Question",
        "",
        question,
        "",
        "## Answer",
        "",
        answer,
        "",
        "## Retrieved chunks",
        ""
    ]

    for i, point in enumerate(points, start=1):
        payload = point.payload
        preview = payload.get("text", "")[:700].replace("\n", " ")

        lines.extend([
            f"### Retrieved chunk {i}",
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
            preview,
            ""
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(dotenv_path=BASE_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

    openai_client = OpenAI()
    qdrant_client = QdrantClient(host="localhost", port=6333)

    query_vector = create_query_embedding(openai_client, args.question)

    points = retrieve_chunks(
        qdrant_client=qdrant_client,
        collection_name=args.collection_name,
        query_vector=query_vector,
        top_k=args.top_k
    )

    context = build_context(points)

    answer = generate_answer(
        openai_client=openai_client,
        question=args.question,
        context=context
    )

    print("\nQuestion:")
    print(args.question)

    print("\nGenerated answer:")
    print(answer)

    print("\nRetrieved sources:")
    for i, point in enumerate(points, start=1):
        payload = point.payload
        print(f"{i}. {payload.get('title')} | {payload.get('section')} | score={point.score:.4f}")
        print(f"   {payload.get('url')}")

    output_path = save_result(
        collection_name=args.collection_name,
        question=args.question,
        points=points,
        answer=answer
    )

    print(f"\nSaved RAG answer test to: {output_path}")


if __name__ == "__main__":
    main()
