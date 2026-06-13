import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))

from qdrant_config import create_qdrant_client

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
PROMPT_FILE = BASE_DIR / "prompts" / "rag_system_prompt.txt"


def find_golden_file():
    candidates = sorted((BASE_DIR / "evaluation").glob("golden_test_set*.xlsx"))

    if not candidates:
        raise FileNotFoundError(
            "No golden_test_set*.xlsx file found in the evaluation folder."
        )

    return candidates[0]


def normalize_columns(df):
    df.columns = [str(col).strip() for col in df.columns]
    return df


def get_column(df, possible_names, required=True):
    for name in possible_names:
        if name in df.columns:
            return name

    if required:
        raise ValueError(
            f"Could not find any of these columns: {possible_names}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def create_query_embedding(openai_client, question):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    return response.data[0].embedding


def retrieve_chunks(qdrant_client, collection_name, query_vector, top_k):
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

        block = f"""
[Context {i}]
Chunk ID: {payload.get("chunk_id")}
Doc ID: {payload.get("doc_id")}
Title: {payload.get("title")}
Topic: {payload.get("topic")}
Section: {payload.get("section")}
URL: {payload.get("url")}
Score: {point.score}

Text:
{payload.get("text")}
"""
        context_blocks.append(block)

    return "\n".join(context_blocks)


def generate_answer(openai_client, question, context):
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


def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--golden-file", default=None)
    args = parser.parse_args()

    load_dotenv(dotenv_path=BASE_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

    golden_file = Path(args.golden_file) if args.golden_file else find_golden_file()

    if not golden_file.is_absolute():
        golden_file = BASE_DIR / golden_file

    df = pd.read_excel(golden_file)
    df = normalize_columns(df)

    qid_col = get_column(df, ["question_id", "qid", "id", "ID"], required=False)
    question_col = get_column(df, ["question", "Question", "QUESTION"])
    question_type_col = get_column(df, ["question_type", "type", "Question Type"], required=False)
    expected_source_col = get_column(df, ["expected_source", "expected_doc_id", "source"], required=False)
    expected_url_col = get_column(df, ["expected_source_url", "expected_url", "url"], required=False)
    expected_answer_col = get_column(df, ["expected_answer_draft", "expected_answer", "answer"], required=False)

    if args.max_questions:
        df = df.head(args.max_questions)

    openai_client = OpenAI()
    qdrant_client = create_qdrant_client()

    rows = []
    jsonl_records = []

    print(f"Golden file: {golden_file}")
    print(f"Collection: {args.collection_name}")
    print(f"Questions to run: {len(df)}")
    print()

    for index, row in df.iterrows():
        question_id = safe_str(row[qid_col]) if qid_col else f"Q{index + 1:03d}"
        question = safe_str(row[question_col])
        question_type = safe_str(row[question_type_col]) if question_type_col else ""
        expected_source = safe_str(row[expected_source_col]) if expected_source_col else ""
        expected_url = safe_str(row[expected_url_col]) if expected_url_col else ""
        expected_answer = safe_str(row[expected_answer_col]) if expected_answer_col else ""

        print(f"Running {question_id}: {question}")

        query_vector = create_query_embedding(openai_client, question)

        points = retrieve_chunks(
            qdrant_client=qdrant_client,
            collection_name=args.collection_name,
            query_vector=query_vector,
            top_k=args.top_k
        )

        context = build_context(points)
        answer = generate_answer(openai_client, question, context)

        retrieved_chunk_ids = []
        retrieved_doc_ids = []
        retrieved_titles = []
        retrieved_sections = []
        retrieved_urls = []
        retrieved_scores = []

        for point in points:
            payload = point.payload
            retrieved_chunk_ids.append(str(payload.get("chunk_id")))
            retrieved_doc_ids.append(str(payload.get("doc_id")))
            retrieved_titles.append(str(payload.get("title")))
            retrieved_sections.append(str(payload.get("section")))
            retrieved_urls.append(str(payload.get("url")))
            retrieved_scores.append(round(point.score, 4))

        top1_doc_id = retrieved_doc_ids[0] if retrieved_doc_ids else ""
        top1_url = retrieved_urls[0] if retrieved_urls else ""

        expected_source_found = expected_source in retrieved_doc_ids if expected_source else ""
        expected_url_found = expected_url in retrieved_urls if expected_url else ""
        top1_expected_source = top1_doc_id == expected_source if expected_source else ""
        top1_expected_url = top1_url == expected_url if expected_url else ""

        output_row = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "expected_source": expected_source,
            "expected_source_url": expected_url,
            "expected_answer_draft": expected_answer,
            "generated_answer": answer,
            "retrieved_chunk_ids": " | ".join(retrieved_chunk_ids),
            "retrieved_doc_ids": " | ".join(retrieved_doc_ids),
            "retrieved_titles": " | ".join(retrieved_titles),
            "retrieved_sections": " | ".join(retrieved_sections),
            "retrieved_urls": " | ".join(retrieved_urls),
            "retrieved_scores": " | ".join(map(str, retrieved_scores)),
            "top1_doc_id": top1_doc_id,
            "top1_url": top1_url,
            "expected_source_found_top_k": expected_source_found,
            "expected_url_found_top_k": expected_url_found,
            "top1_expected_source": top1_expected_source,
            "top1_expected_url": top1_expected_url
        }

        rows.append(output_row)
        jsonl_records.append(output_row)

    output_csv = BASE_DIR / "evaluation" / f"rag_outputs_{args.run_name}.csv"
    output_jsonl = BASE_DIR / "evaluation" / f"rag_outputs_{args.run_name}.jsonl"

    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8")

    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in jsonl_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    result_df = pd.DataFrame(rows)

    total = len(result_df)

    if "expected_source_found_top_k" in result_df.columns:
        valid_source_rows = result_df[result_df["expected_source_found_top_k"] != ""]
        if len(valid_source_rows) > 0:
            source_topk_rate = valid_source_rows["expected_source_found_top_k"].mean()
        else:
            source_topk_rate = None
    else:
        source_topk_rate = None

    if "top1_expected_source" in result_df.columns:
        valid_top1_rows = result_df[result_df["top1_expected_source"] != ""]
        if len(valid_top1_rows) > 0:
            top1_rate = valid_top1_rows["top1_expected_source"].mean()
        else:
            top1_rate = None
    else:
        top1_rate = None

    print()
    print("Done.")
    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSONL: {output_jsonl}")
    print(f"Total questions: {total}")

    if source_topk_rate is not None:
        print(f"Expected source found in top-k: {source_topk_rate:.2%}")

    if top1_rate is not None:
        print(f"Expected source was top-1: {top1_rate:.2%}")


if __name__ == "__main__":
    main()
