import re
from pathlib import Path

import pandas as pd

from react_agent import choose_tool


BASE_DIR = Path(__file__).resolve().parents[2]
EVALUATION_DIR = BASE_DIR / "evaluation"


def find_golden_file():
    candidates = sorted(EVALUATION_DIR.glob("golden_test_set*.xlsx"))

    if not candidates:
        raise FileNotFoundError("No golden_test_set*.xlsx file found in evaluation folder.")

    return candidates[0]


def expected_routing(question):
    q = question.lower()

    if "complete classification workflow" in q or "combine preprocessing" in q:
        return "rag_retriever", ""

    if "randomforestclassifier" in q and ("parameter" in q or "complexity" in q):
        return "metadata_filtered_retriever", "random_forest_classifier"

    if "logisticregression" in q and "parameter" in q:
        return "metadata_filtered_retriever", "logistic_regression"

    if "standardscaler" in q or "scaling" in q or "preprocessing" in q:
        return "metadata_filtered_retriever", "preprocessing"

    if "pipeline" in q:
        return "metadata_filtered_retriever", "pipelines"

    if "gridsearchcv" in q or "randomizedsearchcv" in q or "hyperparameter" in q:
        return "metadata_filtered_retriever", "hyperparameter_tuning"

    if "cross-validation" in q or "cross validation" in q or "k-fold" in q:
        return "metadata_filtered_retriever", "cross_validation"

    if "accuracy" in q or "precision" in q or "recall" in q or "f1" in q or "metric" in q:
        return "metadata_filtered_retriever", "metrics"

    return "rag_retriever", ""


def get_column(df, possible_names):
    for col in possible_names:
        if col in df.columns:
            return col

    raise ValueError(f"None of these columns found: {possible_names}. Available: {list(df.columns)}")


def main():
    golden_file = find_golden_file()

    df = pd.read_excel(golden_file)
    df.columns = [str(col).strip() for col in df.columns]

    qid_col = get_column(df, ["question_id", "qid", "id", "ID"])
    question_col = get_column(df, ["question", "Question", "QUESTION"])

    rows = []

    for _, row in df.iterrows():
        question_id = str(row[qid_col]).strip()
        question = str(row[question_col]).strip()

        expected_tool, expected_topic = expected_routing(question)
        decision = choose_tool(question)

        predicted_tool = decision.get("tool", "")
        predicted_topic = decision.get("arguments", {}).get("topic_filter", "")

        tool_correct = predicted_tool == expected_tool

        if expected_tool == "metadata_filtered_retriever":
            topic_correct = predicted_topic == expected_topic
        else:
            topic_correct = True

        full_correct = tool_correct and topic_correct

        rows.append({
            "question_id": question_id,
            "question": question,
            "expected_tool": expected_tool,
            "expected_topic_filter": expected_topic,
            "predicted_tool": predicted_tool,
            "predicted_topic_filter": predicted_topic,
            "tool_correct": tool_correct,
            "topic_correct": topic_correct,
            "full_tool_call_correct": full_correct,
            "agent_thought": decision.get("thought", "")
        })

    results = pd.DataFrame(rows)

    output_csv = EVALUATION_DIR / "tool_call_accuracy_results.csv"
    results.to_csv(output_csv, index=False, encoding="utf-8")

    total = len(results)
    tool_accuracy = results["tool_correct"].mean()
    topic_accuracy = results["topic_correct"].mean()
    full_accuracy = results["full_tool_call_correct"].mean()

    failed = results[results["full_tool_call_correct"] == False]
    failed_output = EVALUATION_DIR / "tool_call_accuracy_failures.csv"
    failed.to_csv(failed_output, index=False, encoding="utf-8")

    md = f"""# Tool Call Accuracy

Tool Call Accuracy was used as the agentic metric.

The metric evaluates whether the agent selected the correct retrieval tool for each golden test question.

Two aspects were evaluated:

1. Whether the correct tool was selected.
2. Whether the correct metadata topic filter was selected when the metadata-filtered retriever was used.

| Metric | Score |
|---|---:|
| Tool selection accuracy | {tool_accuracy:.2%} |
| Topic filter accuracy | {topic_accuracy:.2%} |
| Full tool call accuracy | {full_accuracy:.2%} |
| Total questions | {total} |
| Failed tool calls | {len(failed)} |

The results were saved in `evaluation/tool_call_accuracy_results.csv`.
"""

    output_md = EVALUATION_DIR / "tool_call_accuracy_summary.md"
    output_md.write_text(md, encoding="utf-8")

    print(md)

    if len(failed) > 0:
        print("\nFailed tool calls:")
        for _, row in failed.iterrows():
            print(f"\n{row['question_id']}: {row['question']}")
            print(f"Expected: {row['expected_tool']} / {row['expected_topic_filter']}")
            print(f"Predicted: {row['predicted_tool']} / {row['predicted_topic_filter']}")

    print(f"\nSaved CSV: {output_csv}")
    print(f"Saved failures: {failed_output}")
    print(f"Saved summary: {output_md}")


if __name__ == "__main__":
    main()
