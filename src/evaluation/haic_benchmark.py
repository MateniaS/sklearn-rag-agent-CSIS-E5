import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parents[2]
JUDGE_MODEL = "gpt-4.1-mini"


def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def judge_haic(openai_client, question, question_type, generated_answer, expected_answer):
    system_prompt = """
You are evaluating a Human-AI Collaboration (HAIC) benchmark for a scikit-learn RAG assistant.

The assistant is meant to help beginner/intermediate users understand scikit-learn documentation and complete supervised classification workflow tasks.

Return only valid JSON.

Use scores from 1 to 5:
1 = very poor
2 = weak
3 = acceptable
4 = good
5 = excellent

Evaluate:

task_completion:
Does the answer help the user complete the intended task or understand the concept?

user_effort_reduction:
Does the answer reduce the effort needed to search/read the documentation?

actionability:
Does the answer provide practical, usable guidance?

trust_grounding:
Does the answer appear grounded, cautious, and source-aware rather than overconfident?

learning_support:
Does the answer help the user learn the concept or workflow?

Return this JSON:
{
  "task_completion": 1-5,
  "user_effort_reduction": 1-5,
  "actionability": 1-5,
  "trust_grounding": 1-5,
  "learning_support": 1-5,
  "short_explanation": "brief explanation"
}
"""

    user_prompt = f"""
Question type:
{question_type}

User question:
{question}

Expected answer draft:
{expected_answer}

Generated answer:
{generated_answer}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found.")

    input_csv = BASE_DIR / args.input_csv
    df = pd.read_csv(input_csv)

    if args.max_questions:
        df = df.head(args.max_questions)

    openai_client = OpenAI()

    rows = []

    print(f"Input CSV: {input_csv}")
    print(f"Questions to evaluate: {len(df)}")
    print()

    for _, row in df.iterrows():
        question_id = safe_str(row["question_id"])
        question_type = safe_str(row.get("question_type", ""))
        question = safe_str(row["question"])
        generated_answer = safe_str(row["generated_answer"])
        expected_answer = safe_str(row.get("expected_answer_draft", ""))

        print(f"HAIC judging {question_id}: {question}")

        scores = judge_haic(
            openai_client=openai_client,
            question=question,
            question_type=question_type,
            generated_answer=generated_answer,
            expected_answer=expected_answer
        )

        output_row = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "task_completion": scores["task_completion"],
            "user_effort_reduction": scores["user_effort_reduction"],
            "actionability": scores["actionability"],
            "trust_grounding": scores["trust_grounding"],
            "learning_support": scores["learning_support"],
            "short_explanation": scores["short_explanation"]
        }

        output_row["overall_haic_score"] = round(
            (
                output_row["task_completion"]
                + output_row["user_effort_reduction"]
                + output_row["actionability"]
                + output_row["trust_grounding"]
                + output_row["learning_support"]
            ) / 5,
            2
        )

        rows.append(output_row)

    results = pd.DataFrame(rows)

    output_csv = BASE_DIR / "evaluation" / f"haic_results_{args.run_name}.csv"
    results.to_csv(output_csv, index=False, encoding="utf-8")

    summary = {
        "run_name": args.run_name,
        "total_questions": len(results),
        "mean_task_completion": results["task_completion"].mean(),
        "mean_user_effort_reduction": results["user_effort_reduction"].mean(),
        "mean_actionability": results["actionability"].mean(),
        "mean_trust_grounding": results["trust_grounding"].mean(),
        "mean_learning_support": results["learning_support"].mean(),
        "mean_overall_haic_score": results["overall_haic_score"].mean()
    }

    output_json = BASE_DIR / "evaluation" / f"haic_summary_{args.run_name}.json"
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"""# HAIC Benchmarking — {args.run_name}

| Metric | Mean score |
|---|---:|
| Task Completion | {summary["mean_task_completion"]:.2f} / 5 |
| User Effort Reduction | {summary["mean_user_effort_reduction"]:.2f} / 5 |
| Actionability | {summary["mean_actionability"]:.2f} / 5 |
| Trust / Grounding | {summary["mean_trust_grounding"]:.2f} / 5 |
| Learning Support | {summary["mean_learning_support"]:.2f} / 5 |
| Overall HAIC Score | {summary["mean_overall_haic_score"]:.2f} / 5 |

Total questions evaluated: {summary["total_questions"]}

Results saved in `{output_csv}`.
"""

    output_md = BASE_DIR / "evaluation" / f"haic_summary_{args.run_name}.md"
    output_md.write_text(md, encoding="utf-8")

    print()
    print(md)
    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")
    print(f"Saved MD: {output_md}")


if __name__ == "__main__":
    main()
