import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from openai import OpenAI
from qdrant_client import QdrantClient

from react_agent import CHAT_MODEL, build_context, choose_tool, generate_answer, retrieve_chunks
from tool_definitions import (
    VALID_TOPIC_FILTERS,
    create_retrieval_tools,
    get_last_retrieval,
    reset_last_retrieval,
)


BASE_DIR = Path(__file__).resolve().parents[2]

AGENT_SYSTEM_PROMPT = """You are a scikit-learn documentation assistant agent.

You must answer user questions using ONLY information retrieved from the documentation tools.

Available tools:
1. rag_retriever - search the full scikit-learn corpus. Use for broad or multi-topic workflow questions.
2. metadata_filtered_retriever - search within a specific topic. Use when the question targets a specific scikit-learn topic or API page.

Valid topic_filter values for metadata_filtered_retriever:
general_intro, preprocessing, pipelines, train_test_split, cross_validation, hyperparameter_tuning, metrics, logistic_regression, random_forest, random_forest_classifier

Process:
1. Choose the most appropriate tool and call it once.
2. Read the retrieved context carefully.
3. Answer using only that context.
4. If the context is insufficient, say:
"The available context does not contain enough information to answer this question."
5. End with a Sources section listing document titles and URLs used.

Do not use outside knowledge.
"""


def extract_tool_usage(messages):
    tool_calls = []

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(
                    {
                        "name": tool_call.get("name"),
                        "args": tool_call.get("args", {}),
                    }
                )

    return tool_calls


def validate_langgraph_run(messages):
    tool_calls = extract_tool_usage(messages)

    if not tool_calls:
        return False, "No tool call was made by the LangGraph agent."

    for tool_call in tool_calls:
        tool_name = tool_call["name"]

        if tool_name == "metadata_filtered_retriever":
            topic = tool_call["args"].get("topic_filter", "")
            if topic not in VALID_TOPIC_FILTERS:
                return False, f"Invalid topic_filter: {topic}"
        elif tool_name != "rag_retriever":
            return False, f"Unknown tool: {tool_name}"

    return True, ""


def get_final_answer(messages):
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            return message.content

    return None


def build_decision_from_tool_calls(tool_calls, router):
    if not tool_calls:
        return {
            "thought": "LangGraph ReAct agent did not select a tool.",
            "tool": "unknown",
            "arguments": {},
            "router": router,
            "tool_calls": [],
        }

    last_tool_call = tool_calls[-1]

    return {
        "thought": "LangGraph ReAct agent selected a retrieval tool based on the question.",
        "tool": last_tool_call["name"],
        "arguments": last_tool_call["args"],
        "router": router,
        "tool_calls": tool_calls,
    }


def run_langgraph(question, qdrant_client, openai_client):
    reset_last_retrieval()

    tools = create_retrieval_tools(qdrant_client, openai_client)
    model = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    agent = create_react_agent(model, tools, prompt=AGENT_SYSTEM_PROMPT)

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 8},
    )

    messages = result["messages"]
    is_valid, reason = validate_langgraph_run(messages)

    if not is_valid:
        raise ValueError(reason)

    answer = get_final_answer(messages)

    if not answer:
        raise ValueError("LangGraph agent did not produce a final answer.")

    tool_calls = extract_tool_usage(messages)
    decision = build_decision_from_tool_calls(tool_calls, router="langgraph")
    points = get_last_retrieval()

    return decision, points, answer


def run_rule_based(question, qdrant_client, openai_client, router="rules"):
    decision = choose_tool(question)
    decision["router"] = router

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


def run_hybrid(question, qdrant_client, openai_client):
    try:
        return run_langgraph(question, qdrant_client, openai_client)
    except Exception as exc:
        print(f"\nLangGraph routing failed ({exc}). Falling back to rule-based router.")

        decision, points, answer = run_rule_based(
            question,
            qdrant_client,
            openai_client,
            router="hybrid_fallback_rules",
        )
        decision["fallback_reason"] = str(exc)

        return decision, points, answer


def save_agent_run(question, decision, points, answer):
    safe_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    output_path = BASE_DIR / "outputs" / f"langgraph_agent_run_{safe_hash}.md"

    lines = [
        "# LangGraph agent run",
        "",
        "## Question",
        question,
        "",
        "## Agent decision",
        f"Router: {decision.get('router', 'unknown')}",
        f"Thought: {decision.get('thought', '')}",
        f"Tool: {decision.get('tool', '')}",
        "",
        "Arguments:",
        "```json",
        json.dumps(decision.get("arguments", {}), ensure_ascii=False, indent=2),
        "```",
        "",
    ]

    if decision.get("fallback_reason"):
        lines.extend(
            [
                "## Fallback",
                decision["fallback_reason"],
                "",
            ]
        )

    if decision.get("tool_calls"):
        lines.extend(
            [
                "## Tool calls",
                "```json",
                json.dumps(decision["tool_calls"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Final answer",
            answer,
            "",
            "## Retrieved chunks",
        ]
    )

    for index, point in enumerate(points, start=1):
        payload = point.payload
        preview = payload.get("text", "")[:500].replace("\n", " ")

        lines.extend(
            [
                "",
                f"### Chunk {index}",
                f"- Score: {point.score:.4f}",
                f"- Doc ID: {payload.get('doc_id')}",
                f"- Title: {payload.get('title')}",
                f"- Topic: {payload.get('topic')}",
                f"- Section: {payload.get('section')}",
                f"- URL: {payload.get('url')}",
                "",
                "Preview:",
                preview,
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph ReAct agent for scikit-learn RAG"
    )
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--router",
        choices=["langgraph", "hybrid"],
        default="langgraph",
        help="langgraph = LLM ReAct routing; hybrid = fallback to rule-based router on failure",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=BASE_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

    openai_client = OpenAI()
    qdrant_client = QdrantClient(host="localhost", port=6333)

    print("\nQuestion:")
    print(args.question)
    print(f"\nRouter mode: {args.router}")

    if args.router == "hybrid":
        decision, points, answer = run_hybrid(
            args.question,
            qdrant_client,
            openai_client,
        )
    else:
        decision, points, answer = run_langgraph(
            args.question,
            qdrant_client,
            openai_client,
        )

    print("\nAgent thought:")
    print(decision.get("thought", ""))

    print("\nSelected tool:")
    print(decision.get("tool", ""))

    print("\nTool arguments:")
    print(json.dumps(decision.get("arguments", {}), ensure_ascii=False, indent=2))

    if decision.get("fallback_reason"):
        print("\nFallback reason:")
        print(decision["fallback_reason"])

    print("\nFinal answer:")
    print(answer)

    print("\nRetrieved chunks:")
    for index, point in enumerate(points, start=1):
        payload = point.payload
        print(
            f"{index}. {payload.get('doc_id')} | "
            f"{payload.get('title')} | "
            f"{payload.get('topic')} | "
            f"score={point.score:.4f}"
        )

    output_path = save_agent_run(args.question, decision, points, answer)

    print(f"\nSaved LangGraph agent run to: {output_path}")


if __name__ == "__main__":
    main()
