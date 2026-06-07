import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient


BASE_DIR = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "text-embedding-3-small"
JUDGE_MODEL = "gpt-4.1-mini"


def create_embedding(client, text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def retrieve_context(qdrant_client, openai_client, collection_name, question, top_k=5):
    query_vector = create_embedding(openai_client, question)

    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    return response.points


def build_context(points, max_chars=12000):
    blocks = []

    for i, point in enumerate(points, start=1):
        payload = point.payload

        block = f"""
[Context {i}]
Score: {point.score}
Doc ID: {payload.get("doc_id")}
Title: {payload.get("title")}
Topic: {payload.get("topic")}
Section: {payload.get("section")}
URL: {payload.get("url")}

Text:
{payload.get("text")}
"""
        blocks.append(block)

    context = "\n".join(blocks)

    if len(context) > max_chars:
        context = context[:max_chars]

    return context


def judge_answer(openai_client, question, generated_answer, expected_answer, context):
    system_prompt = """
You are an evaluator for a Retrieval-Augmented Generation system.

You will evaluate one generated answer using the question, the retrieved context, and the expected answer draft.

Return only valid JSON.

Use scores from 1 to 5:
1 = very poor
2 = weak
3 = acceptable
4 = good
5 = excellent

Evaluate these metrics:

answer_relevancy:
Does the generated answer directly answer the user's question?

faithfulness:
Is the generated answer supported by the retrieved context? Penalize unsupported claims.

context_precision:
Are the retrieved context chunks relevant to the question?

context_recall:
Does the retrieved context contain enough information to answer the question?

expected_answer_alignment:
Does the generated answer align with the expected answer draft?

Return this JSON format:
{
  "answer_relevancy": 1-5,
  "faithfulness": 1-5,
  "context_precision": 1-5,
  "context_recall": 1-5,
  "expected_answer_alignment": 1-5,
  "short_explanation": "brief explanation"
}
"""

    user_prompt = f"""
Question:
{question}

Expected answer draft:
{expected_answer}

Generated answer:
{generated_answer}

Retrieved context:
{context}
"""

    response = openai_client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)


def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    load_dotenv(dotenv_path=BASE_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

    input_csv = BASE_DIR / args.input_csv
    df = pd.read_csv(input_csv)

    if args.max_questions:
        df = df.head(args.max_questions)

    openai_client = OpenAI()
    qdrant_client = QdrantClient(host="localhost", port=6333)

    rows = []

    print(f"Input CSV: {input_csv}")
    print(f"Collection: {args.collection_name}")
    print(f"Questions to evaluate: {len(df)}")
    print()

    for _, row in df.iterrows():
        question_id = safe_str(row["question_id"])
        question = safe_str(row["question"])
        generated_answer = safe_str(row["generated_answer"])
        expected_answer = safe_str(row.get("expected_answer_draft", ""))

        print(f"Judging {question_id}: {question}")

        points = retrieve_context(
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            collection_name=args.collection_name,
            question=question,
            top_k=args.top_k
        )

        context = build_context(points)

        judge = judge_answer(
            openai_client=openai_client,
            question=question,
            generated_answer=generated_answer,
            expected_answer=expected_answer,
            context=context
        )

        output_row = {
            "question_id": question_id,
            "question": question,
            "answer_relevancy": judge["answer_relevancy"],
            "faithfulness": judge["faithfulness"],
            "context_precision": judge["context_precision"],
            "context_recall": judge["context_recall"],
            "expected_answer_alignment": judge["expected_answer_alignment"],
            "short_explanation": judge["short_explanation"]
        }

        rows.append(output_row)

    results = pd.DataFrame(rows)

    output_csv = BASE_DIR / "evaluation" / f"llm_judge_results_{args.run_name}.csv"
    results.to_csv(output_csv, index=False, encoding="utf-8")

    summary = {
        "run_name": args.run_name,
        "collection_name": args.collection_name,
        "total_questions": len(results),
        "mean_answer_relevancy": results["answer_relevancy"].mean(),
        "mean_faithfulness": results["faithfulness"].mean(),
        "mean_context_precision": results["context_precision"].mean(),
        "mean_context_recall": results["context_recall"].mean(),
        "mean_expected_answer_alignment": results["expected_answer_alignment"].mean()
    }

    output_summary_json = BASE_DIR / "evaluation" / f"llm_judge_summary_{args.run_name}.json"
    output_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    md = f"""# LLM-as-judge evaluation — {args.run_name}

| Metric | Mean score |
|---|---:|
| Answer Relevancy | {summary["mean_answer_relevancy"]:.2f} / 5 |
| Faithfulness | {summary["mean_faithfulness"]:.2f} / 5 |
| Context Precision | {summary["mean_context_precision"]:.2f} / 5 |
| Context Recall | {summary["mean_context_recall"]:.2f} / 5 |
| Expected Answer Alignment | {summary["mean_expected_answer_alignment"]:.2f} / 5 |

Total questions evaluated: {summary["total_questions"]}

Results saved in `{output_csv}`.
"""

    output_summary_md = BASE_DIR / "evaluation" / f"llm_judge_summary_{args.run_name}.md"
    output_summary_md.write_text(md, encoding="utf-8")

    print()
    print(md)
    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON summary: {output_summary_json}")
    print(f"Saved MD summary: {output_summary_md}")


if __name__ == "__main__":
    main()
