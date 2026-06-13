import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))

from observability.langfuse_tracing import (
    chunk_metadata,
    flush_langfuse,
    get_langfuse_client,
    observation,
    trace_context,
    update_observation,
)
from qdrant_config import create_qdrant_client

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
COLLECTION_NAME = "sklearn_rag_v2_structured"


def create_embedding(openai_client, question):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    return response.data[0].embedding


def retrieve_with_metadata_filter(openai_client, question, topic_filter, top_k=5):
    qdrant_client = create_qdrant_client()

    query_vector = create_embedding(openai_client, question)

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


def generate_grounded_answer(openai_client, question, context):
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

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content


def main():
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found.")

    openai_client = OpenAI()
    langfuse_client = get_langfuse_client()

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

    with trace_context(
        langfuse_client,
        name="traced-agent-demo",
        input_data={"question": question},
        metadata={
            "component": "agent",
            "collection_name": COLLECTION_NAME,
            "demo_question": "Q28",
        },
        tags=["agent", "rag"],
    ):
        with observation(
            langfuse_client,
            name="demo-tool-selection",
            input_data={"question": question},
            metadata={"component": "tool_selection"},
            as_type="agent",
        ) as tool_span:
            update_observation(
                tool_span,
                output={
                    "tool": agent_decision["tool"],
                    "arguments": agent_decision["arguments"],
                },
            )

        with observation(
            langfuse_client,
            name="demo-metadata-filtered-retrieval",
            input_data=agent_decision["arguments"],
            metadata={"component": "retrieval"},
            as_type="retriever",
        ) as retrieval_span:
            points = retrieve_with_metadata_filter(
                openai_client=openai_client,
                question=question,
                topic_filter="random_forest_classifier",
                top_k=5
            )
            update_observation(
                retrieval_span,
                output=chunk_metadata(points),
                metadata={"retrieved_chunk_count": len(points)},
            )

        context = build_context(points)

        with observation(
            langfuse_client,
            name="demo-grounded-generation",
            input_data={
                "question": question,
                "retrieved_chunk_count": len(points),
            },
            metadata={"component": "grounded_generation"},
            as_type="generation",
        ) as generation_span:
            answer = generate_grounded_answer(
                openai_client=openai_client,
                question=question,
                context=context
            )
            update_observation(generation_span, output=answer)

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

    flush_langfuse(langfuse_client)

    print(f"\nSaved traced demo output to: {output_path}")
    if langfuse_client is not None:
        print("Flushed Langfuse events. Check Langfuse Tracing.")


if __name__ == "__main__":
    main()
