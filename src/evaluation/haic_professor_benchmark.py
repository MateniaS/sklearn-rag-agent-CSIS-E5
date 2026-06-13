import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR / "src" / "agent"))

from qdrant_config import create_qdrant_client
from react_agent import choose_tool, retrieve_chunks, build_context, generate_answer


BASELINE_S = 30.0
RT_MAX_S = 30.0


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def extract_doc_ids(text):
    if pd.isna(text):
        return []
    return re.findall(r"D\d+", str(text))


def log_event(events, session_id, actor_type, action, object_id, duration_s, correct, metadata):
    events.append({
        "schema_version": "haic.decisions_artifact.v1",
        "session_id": session_id,
        "timestamp": now_iso(),
        "actor_type": actor_type,
        "action": action,
        "object_id": object_id,
        "duration_s": round(duration_s, 4),
        "latency_ms": int(duration_s * 1000),
        "correct": correct,
        "metadata": metadata
    })


def clamp(value, low, high):
    return max(low, min(high, value))


def fmt(value, decimals=3):
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def compute_metrics(events, baseline_s=BASELINE_S, rt_max_s=RT_MAX_S):
    included = [e for e in events if e["actor_type"] in ["human", "ai", "surrogate"]]

    durations = [e["duration_s"] for e in included]
    d_mean = sum(durations) / len(durations) if durations else 0.0

    sessions = {}
    for e in included:
        sessions.setdefault(e["session_id"], []).append(e)

    session_durations = [sum(e["duration_s"] for e in session_events) for session_events in sessions.values()]
    mean_session_duration = sum(session_durations) / len(session_durations) if session_durations else 0.0

    effort_loss = max(0.0, (mean_session_duration - baseline_s) / baseline_s)

    total_duration_min = sum(durations) / 60.0
    frequency = len(included) / total_duration_min if total_duration_min > 0 else 0.0

    response_events = [e for e in included if e["actor_type"] == "ai" and e["action"] == "respond"]

    hcl_values = [
        clamp(1 - (e["duration_s"] / rt_max_s), 0, 1)
        for e in response_events
    ]
    hcl = sum(hcl_values) / len(hcl_values) if hcl_values else None

    human_events = [
        e for e in included
        if e["actor_type"] == "human" and e["action"] in ["accept", "reject"]
    ]
    accepted = sum(1 for e in human_events if e["action"] == "accept")
    trust = accepted / len(human_events) if human_events else None

    correctness_sequence = [
        1 if e["correct"] is True else 0
        for e in response_events
        if e["correct"] is not None
    ]

    if len(correctness_sequence) >= 10:
        window = max(1, int(len(correctness_sequence) * 0.2))
        early = correctness_sequence[:window]
        late = correctness_sequence[-window:]
        acc_early = sum(early) / len(early)
        acc_late = sum(late) / len(late)
        adaptability = 0.0 if acc_early == 0 else math.tanh((acc_late - acc_early) / acc_early)
    else:
        adaptability = None

    efficiency_score = 1 / (1 + effort_loss)

    return {
        "baseline_s": baseline_s,
        "rt_max_s": rt_max_s,
        "num_events": len(included),
        "num_sessions": len(sessions),
        "EL": effort_loss,
        "Tr": trust,
        "HCL": hcl,
        "F": frequency,
        "A": adaptability,
        "D": d_mean,
        "EfficiencyScore": efficiency_score,
        "S": None
    }


def quadrant_interpretation(metrics):
    el = metrics["EL"]
    tr = metrics["Tr"]
    hcl = metrics["HCL"]
    f = metrics["F"]

    if tr is None:
        el_tr = "Trust not available"
    elif el <= 0.2 and tr >= 0.7:
        el_tr = "Efficient & Trusted"
    elif el > 0.2 and tr >= 0.7:
        el_tr = "Trusted but Slow"
    elif el <= 0.2 and tr < 0.7:
        el_tr = "Fast but Distrusted"
    else:
        el_tr = "Inefficient & Distrusted"

    if hcl is None:
        hcl_f = "HCL not available"
    elif hcl >= 0.7 and f >= 2:
        hcl_f = "Smooth collaboration"
    elif hcl >= 0.7 and f < 2:
        hcl_f = "Underutilised AI"
    elif hcl < 0.7 and f >= 2:
        hcl_f = "Overloaded human"
    else:
        hcl_f = "Struggling"

    return el_tr, hcl_f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="evaluation/rag_outputs_v2_structured_full.csv")
    parser.add_argument("--collection-name", default="sklearn_rag_v2_structured")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

    input_csv = BASE_DIR / args.input_csv
    df = pd.read_csv(input_csv)

    if args.max_questions:
        df = df.head(args.max_questions)

    openai_client = OpenAI()
    qdrant_client = create_qdrant_client()

    events = []
    summary_rows = []

    print(f"Running professor-style HAIC benchmark for {len(df)} questions")
    print(f"baseline_s={BASELINE_S}, rt_max_s={RT_MAX_S}")
    print()

    for _, row in df.iterrows():
        question_id = str(row["question_id"]).strip()
        question = str(row["question"]).strip()
        expected_ids = extract_doc_ids(row.get("expected_source", ""))

        session_id = f"haic_{question_id}"

        print(f"HAIC session {question_id}: {question}")

        log_event(
            events, session_id, "human", "query", question_id,
            duration_s=1.0,
            correct=None,
            metadata={"question": question}
        )

        t0 = time.perf_counter()
        decision = choose_tool(question)
        route_s = time.perf_counter() - t0

        log_event(
            events, session_id, "ai", "route", decision["tool"],
            duration_s=route_s,
            correct=True,
            metadata={
                "tool": decision["tool"],
                "arguments": decision["arguments"],
                "thought": decision["thought"]
            }
        )

        tool_args = decision["arguments"]

        t0 = time.perf_counter()
        if decision["tool"] == "metadata_filtered_retriever":
            points = retrieve_chunks(
                qdrant_client=qdrant_client,
                openai_client=openai_client,
                question=tool_args["question"],
                top_k=tool_args.get("top_k", 5),
                topic_filter=tool_args["topic_filter"]
            )
        else:
            points = retrieve_chunks(
                qdrant_client=qdrant_client,
                openai_client=openai_client,
                question=tool_args["question"],
                top_k=tool_args.get("top_k", 5)
            )
        retrieve_s = time.perf_counter() - t0

        retrieved_ids = [p.payload.get("doc_id") for p in points]
        expected_found = any(doc_id in retrieved_ids for doc_id in expected_ids) if expected_ids else True

        log_event(
            events, session_id, "ai", "tool_call", decision["tool"],
            duration_s=retrieve_s,
            correct=expected_found,
            metadata={
                "retrieved_doc_ids": retrieved_ids,
                "expected_doc_ids": expected_ids,
                "topic_filter": tool_args.get("topic_filter", "")
            }
        )

        context = build_context(points)

        t0 = time.perf_counter()
        answer = generate_answer(openai_client, question, context)
        respond_s = time.perf_counter() - t0

        response_correct = expected_found

        log_event(
            events, session_id, "ai", "respond", question_id,
            duration_s=respond_s,
            correct=response_correct,
            metadata={
                "answer_preview": answer[:300],
                "source_found": expected_found
            }
        )

        human_action = "accept" if response_correct else "reject"

        log_event(
            events, session_id, "human", human_action, question_id,
            duration_s=1.0,
            correct=response_correct,
            metadata={
                "acceptance_proxy": "accepted if expected source was retrieved"
            }
        )

        summary_rows.append({
            "question_id": question_id,
            "question": question,
            "tool": decision["tool"],
            "topic_filter": tool_args.get("topic_filter", ""),
            "retrieved_doc_ids": " | ".join(retrieved_ids),
            "expected_doc_ids": " | ".join(expected_ids),
            "expected_found": expected_found,
            "human_action": human_action,
            "route_s": round(route_s, 4),
            "retrieve_s": round(retrieve_s, 4),
            "respond_s": round(respond_s, 4)
        })

    events_path = BASE_DIR / "evaluation" / "haic_events_v2_structured.jsonl"
    with events_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = BASE_DIR / "evaluation" / "haic_event_summary_v2_structured.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    metrics = compute_metrics(events)

    metrics_path = BASE_DIR / "evaluation" / "haic_metrics_v2_structured.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    el_tr, hcl_f = quadrant_interpretation(metrics)

    md = f"""# HAIC Benchmarking — v2_structured

Professor-style HAIC evaluation based on `haic.decisions_artifact.v1`.

## Configuration

| Parameter | Value |
|---|---:|
| baseline_s | {metrics["baseline_s"]:.2f} |
| rt_max_s | {metrics["rt_max_s"]:.2f} |
| sessions | {metrics["num_sessions"]} |
| logged events | {metrics["num_events"]} |

## Metrics

| Metric | Value | Interpretation |
|---|---:|---|
| EL | {fmt(metrics["EL"])} | Effort loss compared to baseline. |
| Tr | {fmt(metrics["Tr"])} | Fraction of accepted AI responses. |
| HCL | {fmt(metrics["HCL"])} | Higher means lower cognitive load. |
| F | {fmt(metrics["F"])} | Interaction events per minute. |
| A | {fmt(metrics["A"])} | Adaptability across early vs late decisions. |
| D | {fmt(metrics["D"])} s | Mean duration per decision event. |
| EfficiencyScore | {fmt(metrics["EfficiencyScore"])} | Composite efficiency score. |
| S | N/A | Excluded because no surrogate simulation was used. |

## Quadrant diagnostics

- EL × Tr: {el_tr}
- HCL × F: {hcl_f}

## Limitation

The human accept/reject event is approximated offline using an acceptance proxy: a response is accepted when the expected source was retrieved. This is a limitation because no real user study was conducted.
"""

    md_path = BASE_DIR / "evaluation" / "haic_metrics_v2_structured.md"
    md_path.write_text(md, encoding="utf-8")

    print()
    print(md)
    print(f"Saved events: {events_path}")
    print(f"Saved event summary: {summary_csv}")
    print(f"Saved metrics JSON: {metrics_path}")
    print(f"Saved metrics MD: {md_path}")


if __name__ == "__main__":
    main()
