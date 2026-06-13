import argparse
import hashlib
import json
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
CHAT_MODEL = "gpt-4.1-mini"
COLLECTION_NAME = "sklearn_rag_v2_structured"
MAX_ITERATIONS = 2


def create_embedding(client, text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def choose_tool(question):
    q = question.lower()

    # Broad multi-step workflow questions use the full corpus.
    if "complete classification workflow" in q or "combine preprocessing" in q:
        return {
            "thought": "The question combines multiple workflow stages, so full-corpus retrieval is more appropriate.",
            "tool": "rag_retriever",
            "arguments": {
                "question": question,
                "top_k": 5
            }
        }

    # Estimator/API-specific questions use metadata filtering.
    if "randomforestclassifier" in q and ("parameter" in q or "complexity" in q):
        return {
            "thought": "The question asks about RandomForestClassifier parameters, so the API-specific topic should be used.",
            "tool": "metadata_filtered_retriever",
            "arguments": {
                "question": question,
                "topic_filter": "random_forest_classifier",
                "top_k": 5
            }
        }

    if "logisticregression" in q and "parameter" in q:
        return {
            "thought": "The question asks about LogisticRegression parameters, so the API-specific topic should be used.",
            "tool": "metadata_filtered_retriever",
            "arguments": {
                "question": question,
                "topic_filter": "logistic_regression",
                "top_k": 5
            }
        }

    if "standardscaler" in q or "scaling" in q or "preprocessing" in q:
        return {
            "thought": "The question is about preprocessing or scaling, so the preprocessing topic should be used.",
            "tool": "metadata_filtered_retriever",
            "arguments": {
                "question": question,
                "topic_filter": "preprocessing",
                "top_k": 5
            }
        }

    if "pipeline" in q:
        return {
            "thought": "The question is about pipelines, so the pipelines topic should be used.",
            "tool": "metadata_filtered_retriever",
            "arguments": {
                "question": question,
                "topic_filter": "pipelines",
                "top_k": 5
            }
        }

    if "gridsearchcv" in q or "randomizedsearchcv" in q or "hyperparameter" in q:
        return {
            "thought": "The question is about hyperparameter tuning, so the hyperparameter_tuning topic should be used.",
            "tool": "metadata_filtered_retriever",
            "arguments": {
                "question": question,
                "topic_filter": "hyperparameter_tuning",
                "top_k": 5
            }
        }

    if "cross-validation" in q or "cross validation" in q or "k-fold" in q:
        return {
            "thought": "The question is about cross-validation, so the cross_validation topic should be used.",
            "tool": "metadata_filtered_retriever",
            "arguments": {
                "question": question,
                "topic_filter": "cross_validation",
                "top_k": 5
            }
        }

    if "accuracy" in q or "precision" in q or "recall" in q or "f1" in q or "metric" in q:
        return {
            "thought": "The question is about evaluation metrics, so the metrics topic should be used.",
            "tool": "metadata_filtered_retriever",
            "arguments": {
                "question": question,
                "topic_filter": "metrics",
                "top_k": 5
            }
        }

    # Default fallback.
    return {
        "thought": "No specific topic filter was clearly required, so full-corpus retrieval is used.",
        "tool": "rag_retriever",
        "arguments": {
            "question": question,
            "top_k": 5
        }
    }


def retrieve_chunks(qdrant_client, openai_client, question, top_k=5, topic_filter=None):
    query_vector = create_embedding(openai_client, question)

    query_filter = None
    if topic_filter:
        query_filter = Filter(
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
        query_filter=query_filter,
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
Chunk ID: {payload.get("chunk_id")}
Doc ID: {payload.get("doc_id")}
Title: {payload.get("title")}
Topic: {payload.get("topic")}
Section: {payload.get("section")}
URL: {payload.get("url")}

Text:
{payload.get("text")}
""")

    return "\n".join(blocks)


def generate_answer(openai_client, question, context):
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


def save_agent_run(question, decision, points, answer):
    safe_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    output_path = BASE_DIR / "outputs" / f"agent_run_{safe_hash}.md"

    lines = [
        "# Agent run",
        "",
        "## Question",
        question,
        "",
        "## Agent decision",
        f"Thought: {decision['thought']}",
        f"Tool: {decision['tool']}",
        "",
        "Arguments:",
        "```json",
        json.dumps(decision["arguments"], ensure_ascii=False, indent=2),
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
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    load_dotenv(dotenv_path=BASE_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

    openai_client = OpenAI()
    qdrant_client = create_qdrant_client()

    print("\nQuestion:")
    print(args.question)

    decision = choose_tool(args.question)

    print("\nAgent thought:")
    print(decision["thought"])

    print("\nSelected tool:")
    print(decision["tool"])

    print("\nTool arguments:")
    print(json.dumps(decision["arguments"], ensure_ascii=False, indent=2))

    tool = decision["tool"]
    tool_args = decision["arguments"]

    points = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {iteration}/{MAX_ITERATIONS}")

        if tool == "rag_retriever":
            points = retrieve_chunks(
                qdrant_client=qdrant_client,
                openai_client=openai_client,
                question=tool_args["question"],
                top_k=tool_args.get("top_k", 5)
            )
            break

        if tool == "metadata_filtered_retriever":
            points = retrieve_chunks(
                qdrant_client=qdrant_client,
                openai_client=openai_client,
                question=tool_args["question"],
                top_k=tool_args.get("top_k", 5),
                topic_filter=tool_args["topic_filter"]
            )
            break

        raise ValueError(f"Unknown tool: {tool}")

    context = build_context(points)
    answer = generate_answer(openai_client, args.question, context)

    print("\nFinal answer:")
    print(answer)

    print("\nRetrieved chunks:")
    for i, point in enumerate(points, start=1):
        payload = point.payload
        print(f"{i}. {payload.get('doc_id')} | {payload.get('title')} | {payload.get('topic')} | score={point.score:.4f}")

    output_path = save_agent_run(args.question, decision, points, answer)

    print(f"\nSaved agent run to: {output_path}")


if __name__ == "__main__":
    main()
