import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR / "src" / "agent"))
sys.path.append(str(BASE_DIR / "src" / "vectorstore"))

from qdrant_config import create_qdrant_client, get_qdrant_host, get_qdrant_port
from react_agent import build_context, choose_tool, generate_answer, retrieve_chunks
from index_chunks_qdrant import index_chunks


COLLECTION_NAME = "sklearn_rag_v2_structured"
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "v2_structured_chunks.jsonl"
OUTPUT_FILE = BASE_DIR / "outputs" / "docker_demo_run.md"
DEMO_QUESTIONS = [
    "How can a complete classification workflow combine preprocessing, model training, cross-validation and evaluation?",
    "Which RandomForestClassifier parameters can control model complexity?",
    "Who won the FIFA World Cup in 2022?",
]


def wait_for_qdrant(max_attempts=30, delay_s=2):
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            client = create_qdrant_client()
            client.get_collections()
            print(f"Qdrant is reachable at {get_qdrant_host()}:{get_qdrant_port()}.")
            return client
        except Exception as exc:
            last_error = exc
            print(f"Waiting for Qdrant ({attempt}/{max_attempts}): {exc}")
            time.sleep(delay_s)

    raise RuntimeError("Qdrant did not become reachable.") from last_error


def collection_has_points(qdrant_client, collection_name):
    collections = {
        collection.name
        for collection in qdrant_client.get_collections().collections
    }

    if collection_name not in collections:
        return False

    count_result = qdrant_client.count(
        collection_name=collection_name,
        exact=True,
    )
    return count_result.count > 0


def ensure_v2_collection(qdrant_client):
    if collection_has_points(qdrant_client, COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} already exists and has points.")
        return

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Required chunks file not found: {CHUNKS_FILE}")

    print(f"Collection {COLLECTION_NAME} missing or empty. Indexing {CHUNKS_FILE}.")
    index_chunks(CHUNKS_FILE, COLLECTION_NAME)


def run_question(qdrant_client, openai_client, question):
    decision = choose_tool(question)
    tool_args = decision["arguments"]

    if decision["tool"] == "metadata_filtered_retriever":
        points = retrieve_chunks(
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            question=tool_args["question"],
            top_k=tool_args.get("top_k", 5),
            topic_filter=tool_args["topic_filter"],
        )
    else:
        points = retrieve_chunks(
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            question=tool_args["question"],
            top_k=tool_args.get("top_k", 5),
        )

    context = build_context(points)
    answer = generate_answer(openai_client, question, context)

    return decision, points, answer


def write_markdown(results):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Docker demo run",
        "",
        f"Qdrant endpoint: `{get_qdrant_host()}:{get_qdrant_port()}`",
        f"Collection: `{COLLECTION_NAME}`",
        "",
    ]

    for index, result in enumerate(results, start=1):
        decision = result["decision"]
        points = result["points"]

        lines.extend(
            [
                f"## Demo question {index}",
                "",
                result["question"],
                "",
                "### Agent decision",
                "",
                f"- Tool: `{decision['tool']}`",
                f"- Thought: {decision['thought']}",
                "",
                "Arguments:",
                "",
                "```json",
                json.dumps(decision["arguments"], ensure_ascii=False, indent=2),
                "```",
                "",
                "### Answer",
                "",
                result["answer"],
                "",
                "### Retrieved chunks",
                "",
            ]
        )

        for chunk_index, point in enumerate(points, start=1):
            payload = point.payload
            lines.extend(
                [
                    f"{chunk_index}. `{payload.get('doc_id')}` | "
                    f"{payload.get('title')} | {payload.get('topic')} | "
                    f"score={point.score:.4f}",
                    f"   URL: {payload.get('url')}",
                ]
            )

        lines.append("")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved Docker demo output to: {OUTPUT_FILE}")


def main():
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required. Create .env from .env.example and fill it in.")

    qdrant_client = wait_for_qdrant()
    ensure_v2_collection(qdrant_client)

    openai_client = OpenAI()
    results = []

    for question in DEMO_QUESTIONS:
        print(f"\nRunning demo question: {question}")
        decision, points, answer = run_question(qdrant_client, openai_client, question)
        results.append(
            {
                "question": question,
                "decision": decision,
                "points": points,
                "answer": answer,
            }
        )

    write_markdown(results)


if __name__ == "__main__":
    main()
