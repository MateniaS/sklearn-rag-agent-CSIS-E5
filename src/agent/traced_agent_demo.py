import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


BASE_DIR = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
COLLECTION_NAME = "sklearn_rag_v2_structured"


def create_embedding(question):
    response = openai.embeddings.create(
        name="agent-query-embedding",
        model=EMBEDDING_MODEL,
        input=question,
        metadata={
            "component": "embedding",
            "project": "sklearn-rag-agent"
        }
    )
    return response.data[0].embedding


def retrieve_with_metadata_filter(question, topic_filter, top_k=5):
    qdrant_client = QdrantClient(host="localhost", port=6333)

    query_vector = create_embedding(question)

    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="topic",
                match=MatchValue(value=topic_filter)
            )
        ]
    )

    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True
    )

    return response.points


def build_context(points):
    blocks = []

    for i, point in enumerate(points, start=1):
        payload = point.payload

        blocks.append(f"""
[Context {i}]
Score: {point.score}
Doc ID: {payload.get("doc_id")}
Chunk ID: {payload.get("chunk_id")}
Title: {payload.get("title")}
Topic: {payload.get("topic")}
Section: {payload.get("section")}
URL: {payload.get("url")}

Text:
{payload.get("text")}
""")

    return "\n".join(blocks)


def generate_grounded_answer(question, context):
    system_prompt = """
You are a grounded assistant answering questions about scikit-learn.

Use only the provided context.

If the answer is not contained in the context, say:
"The available context does not contain enough information to answer this question."

Do not use outside knowledge.

At the end, include a Sources section with document titles and URLs.
"""

    user_prompt = f"""
Question:
{question}

Context:
{context}
"""

    response = openai.chat.completions.create(
        name="agent-grounded-answer",
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        metadata={
            "component": "grounded_generation",
            "project": "sklearn-rag-agent",
            "demo_question": "Q28"
        }
    )

    return response.choices[0].message.content


def main():
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found.")

    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        raise ValueError("Langfuse keys were not found.")

    langfuse = get_client()

    question = "Which RandomForestClassifier parameters can control model complexity?"

    agent_decision = {
        "thought": "The question asks about RandomForestClassifier parameters, so the API-specific metadata filter should be used.",
        "tool": "metadata_filtered_retriever",
        "arguments": {
            "question": question,
            "topic_filter": "random_forest_classifier",
            "top_k": 5
        }
    }

    print("\nQuestion:")
    print(question)

    print("\nAgent decision:")
    print(json.dumps(agent_decision, indent=2))

    points = retrieve_with_metadata_filter(
        question=question,
        topic_filter="random_forest_classifier",
        top_k=5
    )

    context = build_context(points)

    answer = generate_grounded_answer(
        question=question,
        context=context
    )

    print("\nFinal answer:")
    print(answer)

    print("\nRetrieved chunks:")
    for i, point in enumerate(points, start=1):
        payload = point.payload
        print(
            f"{i}. {payload.get('doc_id')} | "
            f"{payload.get('title')} | "
            f"{payload.get('topic')} | "
            f"score={point.score:.4f}"
        )

    output_path = BASE_DIR / "outputs" / "traced_agent_demo_q28.md"

    lines = [
        "# Traced agent demo — Q28",
        "",
        "## Question",
        question,
        "",
        "## Agent decision",
        "```json",
        json.dumps(agent_decision, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Final answer",
        answer,
        "",
        "## Retrieved chunks"
    ]

    for i, point in enumerate(points, start=1):
        payload = point.payload
        preview = payload.get("text", "")[:500].replace("\n", " ")

        lines.extend([
            "",
            f"### Chunk {i}",
            f"- Score: {point.score:.4f}",
            f"- Doc ID: {payload.get('doc_id')}",
            f"- Title: {payload.get('title')}",
            f"- Topic: {payload.get('topic')}",
            f"- Section: {payload.get('section')}",
            f"- URL: {payload.get('url')}",
            "",
            "Preview:",
            preview
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")

    langfuse.flush()

    print(f"\nSaved traced demo output to: {output_path}")
    print("Flushed Langfuse events. Check Langfuse → Tracing.")


if __name__ == "__main__":
    main()
