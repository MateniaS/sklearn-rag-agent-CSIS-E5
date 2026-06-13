# Scikit-learn Agentic RAG Agent

## 1. Project title

**Scikit-learn Agentic RAG Agent**

## 2. Short description

Το project υλοποιεί ένα Agentic Retrieval-Augmented Generation (RAG) σύστημα για το επίσημο documentation του scikit-learn. Το σύστημα ανακτά σχετικά αποσπάσματα από curated corpus, παράγει grounded απαντήσεις μέσω LLM και υποστηρίζει agentic routing μεταξύ δύο retrieval tools: πλήρους corpus search και metadata-filtered search.

Το corpus καλύπτει supervised classification workflows (preprocessing, pipelines, classifiers, metrics, cross-validation, hyperparameter tuning).

## 3. Assignment context

Η εργασία ανήκει στο πλαίσιο του μαθήματος **CSIS130** και αφορά την ανάπτυξη Agentic RAG συστήματος πάνω σε τεκμηριωμένο corpus. Το project συγκρίνει δύο ingestion strategies (`v1_fixed`, `v2_structured`), αξιολογεί retrieval και generation quality, υλοποιεί agent routing, HAIC benchmarking και Langfuse tracing.

## 4. Architecture overview

```text
Corpus (10 scikit-learn pages)
        │
        ▼
Ingestion (v1_fixed / v2_structured chunking)
        │
        ▼
Embeddings (text-embedding-3-small) + Qdrant vector store
        │
        ├──► RAG pipeline (retrieve → context → generate)
        │
        └──► Agent layer
                 ├── rule-based agent (react_agent.py)
                 └── LangGraph ReAct agent (langgraph_agent.py)
                          ├── rag_retriever
                          └── metadata_filtered_retriever
        │
        ▼
Evaluation (golden test set, LLM-as-judge, tool call accuracy, HAIC)
        │
        └──► Observability (Langfuse demo)
```

**Βασικά components:**

- **Ingestion:** fixed-size και structured/section-based chunking
- **Vector store:** Qdrant (`sklearn_rag_v1_fixed`, `sklearn_rag_v2_structured`)
- **RAG:** semantic retrieval + grounded generation (`gpt-4.1-mini`)
- **Agents:** rule-based routing και LangGraph ReAct agent με hybrid fallback
- **Evaluation:** golden test set (30 ερωτήσεις), LLM judge, HAIC metrics

## 5. Folder structure

```text
sklearn-rag-agent/
├── configs/                    # Chunking configuration (v1, v2)
├── data/
│   ├── corpus_sources.csv      # Corpus catalog (D01–D10)
│   ├── raw/                    # Downloaded documentation text + metadata
│   └── processed/              # Chunk files (JSONL)
├── Dockerfile                  # Python app/demo image
├── docker-compose.yml          # Qdrant + Python demo service
├── docs/                       # Technical report and project notes
├── evaluation/                 # Golden test set, metrics, HAIC artifacts
├── outputs/                    # Demo runs, retrieval tests, chunk stats
├── prompts/
│   └── rag_system_prompt.txt
├── requirements.txt
├── src/
│   ├── agent/                  # react_agent, langgraph_agent, tool_definitions
│   ├── demo/                   # Docker demo runner
│   ├── evaluation/             # HAIC benchmarking scripts
│   ├── ingestion/              # download_corpus, chunk_fixed, chunk_structured
│   ├── rag/                    # RAG pipeline and evaluation scripts
│   └── vectorstore/            # Qdrant indexing
├── .dockerignore
├── .env.example
└── README.md
```

**Σημείωση:** Τα `.env`, `.venv/`, `qdrant_storage/` δεν περιλαμβάνονται στο repository.

## 6. Prerequisites

- **Python 3.9+** (δοκιμασμένο με Python 3.9.6)
- **Docker** και **Docker Compose** (για αναπαραγώγιμη εκτέλεση Qdrant + app demo)
- **OpenAI API key** (υποχρεωτικό για embeddings, generation, evaluation)
- **Langfuse keys** (προαιρετικά — μόνο για το Langfuse demo script)

## 7. Docker / Reproducible execution

Το προτεινόμενο reproducible path είναι το Docker Compose setup. Εκκινεί:

- `qdrant` — vector store service
- `app` — Python demo service που περιμένει το Qdrant, ελέγχει/δημιουργεί το `sklearn_rag_v2_structured` collection και τρέχει 3 representative demo questions

Από τη ρίζα του project:

```bash
cp .env.example .env
```

Συμπληρώστε τουλάχιστον:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

Έπειτα:

```bash
docker compose up --build
```

Στο πρώτο run, αν το collection `sklearn_rag_v2_structured` λείπει ή είναι άδειο, το app service κάνει indexing του `data/processed/v2_structured_chunks.jsonl` στο Qdrant. Το demo αποθηκεύει:

`outputs/docker_demo_run.md`

Το `.env` χρησιμοποιείται μέσω `env_file` στο `docker-compose.yml`, παραμένει gitignored και δεν αντιγράφεται στο Docker image.

## 8. Local environment setup

Από τη ρίζα του project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Δημιουργία `.env` από το template:

```bash
cp .env.example .env
```

Συμπλήρωση των τιμών στο `.env`:

```text
OPENAI_API_KEY=your_openai_api_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_BASE_URL=https://cloud.langfuse.com
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

Το `OPENAI_API_KEY` απαιτείται για όλα τα βασικά scripts. Τα Langfuse keys απαιτούνται μόνο για `src/agent/traced_agent_demo.py`.

## 9. Qdrant configuration

Όλα τα scripts που συνδέονται στο Qdrant διαβάζουν:

- `QDRANT_HOST` με default `localhost`
- `QDRANT_PORT` με default `6333`

Στο Docker Compose, το app service χρησιμοποιεί `QDRANT_HOST=qdrant` και `QDRANT_PORT=6333`, ώστε να συνδέεται στο Qdrant container μέσω του compose network.

Για local-only εκτέλεση του Qdrant:

```bash
docker compose up -d qdrant
```

Έλεγχος ότι το Qdrant τρέχει:

```bash
curl http://localhost:6333/collections
```

Το Qdrant εκτίθεται στα ports `6333` (HTTP) και `6334` (gRPC). Τα δεδομένα αποθηκεύονται τοπικά στον φάκελο `qdrant_storage/` (gitignored).

## 9. Corpus ingestion

Το corpus ορίζεται στο `data/corpus_sources.csv` (10 σελίδες: D01–D10).

Για λήψη raw documentation:

```bash
python src/ingestion/download_corpus.py
```

**Output:**

- `data/raw/D01.txt` … `data/raw/D10.txt`
- `data/raw/D01_metadata.json` … `data/raw/D10_metadata.json`
- `data/raw/corpus_download_summary.json`

**Σημείωση:** Απαιτείται network access. Αν τα raw files υπάρχουν ήδη στο repo, το βήμα μπορεί να παραλειφθεί.

## 10. Chunking v1 and v2

### v1 — fixed-size chunking

```bash
python src/ingestion/chunk_fixed.py
```

- Config: `configs/v1_fixed_chunking.json`
- Output: `data/processed/v1_fixed_chunks.jsonl`
- Stats: `outputs/chunk_stats_v1.csv`

### v2 — structured/section-based chunking

```bash
python src/ingestion/chunk_structured.py
```

- Config: `configs/v2_structured_chunking.json`
- Output: `data/processed/v2_structured_chunks.jsonl`
- Stats: `outputs/chunk_stats_v2.csv`

**Σημείωση:** Το `chunk_structured.py` κατεβάζει HTML από URLs (όχι από `data/raw/`). Απαιτείται network access.

## 11. Index chunks in Qdrant

Μετά την εκκίνηση του Qdrant:

```bash
python src/vectorstore/index_chunks_qdrant.py \
  --input-file data/processed/v1_fixed_chunks.jsonl \
  --collection-name sklearn_rag_v1_fixed

python src/vectorstore/index_chunks_qdrant.py \
  --input-file data/processed/v2_structured_chunks.jsonl \
  --collection-name sklearn_rag_v2_structured
```

**Embeddings model:** `text-embedding-3-small` (1536 dimensions)

**Collections:**

- `sklearn_rag_v1_fixed` — 402 chunks
- `sklearn_rag_v2_structured` — 375 chunks

## 12. Run the RAG pipeline

### Single-question RAG answer

```bash
python src/rag/rag_answer.py \
  --collection-name sklearn_rag_v2_structured \
  --question "When should I use StandardScaler in scikit-learn?"
```

**Output:** `outputs/rag_answer_test_<collection_name>.md`

### Retrieval-only test

```bash
python src/rag/test_retrieval.py \
  --collection-name sklearn_rag_v2_structured \
  --question "When should I use StandardScaler in scikit-learn?"
```

### Metadata-filtered retrieval test

```bash
python src/rag/test_filtered_retrieval.py \
  --collection-name sklearn_rag_v2_structured \
  --question "Which RandomForestClassifier parameters can control model complexity?" \
  --topic-filter random_forest_classifier
```

## 13. Run the rule-based agent

Ο rule-based agent (`react_agent.py`) χρησιμοποιεί keyword routing και δύο tools:

- `rag_retriever`
- `metadata_filtered_retriever`

### Agent tool schemas

`rag_retriever`

```json
{
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "User question to search against the full scikit-learn documentation corpus."
    },
    "top_k": {
      "type": "integer",
      "default": 5,
      "description": "Number of chunks to retrieve."
    }
  },
  "required": ["question"]
}
```

`metadata_filtered_retriever`

```json
{
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "User question to search against the scikit-learn documentation corpus."
    },
    "topic_filter": {
      "type": "string",
      "description": "Metadata topic to filter on before retrieval.",
      "enum": [
        "general_intro",
        "preprocessing",
        "pipelines",
        "train_test_split",
        "cross_validation",
        "hyperparameter_tuning",
        "metrics",
        "logistic_regression",
        "random_forest",
        "random_forest_classifier"
      ]
    },
    "top_k": {
      "type": "integer",
      "default": 5,
      "description": "Number of chunks to retrieve."
    }
  },
  "required": ["question", "topic_filter"]
}
```

Το `src/agent/tool_definitions.py` εκθέτει τα ίδια εργαλεία στον LangGraph ReAct agent με `@tool`, ώστε οι περιγραφές και τα type hints να μετατρέπονται σε structured tool schemas.

### Loop protection

- `src/agent/react_agent.py` ορίζει `MAX_ITERATIONS = 2`.
- `src/agent/langgraph_agent.py` καλεί τον LangGraph agent με `recursion_limit = 8`.

```bash
python src/agent/react_agent.py \
  --question "Which RandomForestClassifier parameters can control model complexity?"
```

**Default collection:** `sklearn_rag_v2_structured`

**Output:** `outputs/agent_run_<hash>.md`

## 14. Run the LangGraph ReAct agent

Ο LangGraph agent (`langgraph_agent.py`) χρησιμοποιεί LLM ReAct routing με τα ίδια retrieval tools.

### LangGraph mode (default)

```bash
python src/agent/langgraph_agent.py \
  --question "Which RandomForestClassifier parameters can control model complexity?"
```

### Hybrid mode (LangGraph + rule-based fallback)

```bash
python src/agent/langgraph_agent.py \
  --question "Which RandomForestClassifier parameters can control model complexity?" \
  --router hybrid
```

**Output:** `outputs/langgraph_agent_run_<hash>.md`

## 15. Run evaluation

Το golden test set βρίσκεται στο:

`evaluation/golden_test_set_v0-3.xlsx`

### Golden test set RAG run

```bash
python src/rag/run_golden_rag.py \
  --collection-name sklearn_rag_v2_structured \
  --run-name v2_structured_full
```

**Output:**

- `evaluation/rag_outputs_v2_structured_full.csv`
- `evaluation/rag_outputs_v2_structured_full.jsonl`

Για `v1_fixed`:

```bash
python src/rag/run_golden_rag.py \
  --collection-name sklearn_rag_v1_fixed \
  --run-name v1_fixed_full
```

### Tool Call Accuracy (rule-based agent)

```bash
python src/agent/evaluate_tool_calls.py
```

**Output:**

- `evaluation/tool_call_accuracy_results.csv`
- `evaluation/tool_call_accuracy_summary.md`

### LLM-as-judge evaluation

```bash
python src/rag/llm_judge_evaluation.py \
  --input-csv evaluation/rag_outputs_v2_structured_full.csv \
  --collection-name sklearn_rag_v2_structured \
  --run-name v2_structured_full
```

**Output:**

- `evaluation/llm_judge_results_v2_structured_full.csv`
- `evaluation/llm_judge_summary_v2_structured_full.md`

## 16. Run HAIC benchmarking

### LLM-as-judge HAIC scores

```bash
python src/evaluation/haic_benchmark.py \
  --input-csv evaluation/rag_outputs_v2_structured_full.csv \
  --run-name v2_structured_full
```

**Output:**

- `evaluation/haic_results_v2_structured_test3.csv`
- `evaluation/haic_summary_v2_structured_test3.md`

### Professor-style HAIC artifact (`haic.decisions_artifact.v1`)

```bash
python src/evaluation/haic_professor_benchmark.py
```

**Default input:** `evaluation/rag_outputs_v2_structured_full.csv`

**Output:**

- `evaluation/haic_events_v2_structured.jsonl`
- `evaluation/haic_event_summary_v2_structured.csv`
- `evaluation/haic_metrics_v2_structured.json`
- `evaluation/haic_metrics_v2_structured.md`

## 17. Run Langfuse demo

Απαιτούνται Langfuse credentials στο `.env`.

```bash
python src/agent/traced_agent_demo.py
```

Το script εκτελεί demo ερώτηση Q28 (RandomForestClassifier parameters), καταγράφει traces στο Langfuse dashboard και αποθηκεύει:

`outputs/traced_agent_demo_q28.md`

## 18. Known limitations

1. **Rule-based vs LangGraph agent:** Ο `react_agent.py` χρησιμοποιεί deterministic keyword routing, όχι πλήρες LLM ReAct loop. Ο LangGraph agent προσθέτει genuine agentic routing, με optional hybrid fallback.

2. **Tool Call Accuracy metric:** Το `evaluate_tool_calls.py` αξιολογεί τον rule-based router. Το LangGraph routing αξιολογείται ξεχωριστά.

3. **HAIC accept/reject proxy:** Στο `haic_professor_benchmark.py`, το human accept/reject υπολογίζεται offline (accepted όταν βρέθηκε expected source στα retrieved chunks). Δεν πραγματοποιήθηκε πραγματικό user study.

4. **Qdrant data not in repo:** Το `qdrant_storage/` είναι gitignored. Μετά clone, το Docker demo μπορεί να δημιουργήσει ξανά το `sklearn_rag_v2_structured` collection από τα processed chunks.

5. **Network dependencies:** Τα `download_corpus.py` και `chunk_structured.py` απαιτούν πρόσβαση στο scikit-learn.org.

6. **OpenAI package version:** Το `langchain-openai` απαιτεί `openai<2.0.0` (pinned στο `requirements.txt`).

7. **Retrieval ambiguity:** Ερωτήσεις που συνδυάζουν γενικές και API-specific πηγές (π.χ. Q28) μπορεί να αποτυγχάνουν χωρίς metadata filtering.

8. **Langfuse integration scope:** Το Langfuse tracing υλοποιείται στο demo script, όχι στον κύριο agent flow.

## Quick start (minimal path)

```bash
# 1. Setup
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Qdrant only
docker compose up -d qdrant

# 3. Index (if not already indexed)
python src/vectorstore/index_chunks_qdrant.py \
  --input-file data/processed/v2_structured_chunks.jsonl \
  --collection-name sklearn_rag_v2_structured

# 4. Run LangGraph agent
python src/agent/langgraph_agent.py \
  --question "Which RandomForestClassifier parameters can control model complexity?"
```

## Related documentation

- `docs/notes_for_final_report.md` — αναλυτικά project notes
- `docs/Τεχνική Αναφορά — Scikit-learn RAG Agent-4.txt` — τεχνική αναφορά
