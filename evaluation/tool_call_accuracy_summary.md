# Tool Call Accuracy

Tool Call Accuracy was used as the agentic metric.

The metric evaluates whether the agent selected the correct retrieval tool for each golden test question.

Two aspects were evaluated:

1. Whether the correct tool was selected.
2. Whether the correct metadata topic filter was selected when the metadata-filtered retriever was used.

| Metric | Score |
|---|---:|
| Tool selection accuracy | 100.00% |
| Topic filter accuracy | 100.00% |
| Full tool call accuracy | 100.00% |
| Total questions | 30 |
| Failed tool calls | 0 |

The results were saved in `evaluation/tool_call_accuracy_results.csv`.
