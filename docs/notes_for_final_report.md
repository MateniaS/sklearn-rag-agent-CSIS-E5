
## Corpus collection

Οι 10 επιλεγμένες σελίδες του επίσημου scikit-learn documentation κατέβηκαν και αποθηκεύτηκαν ως καθαρά text files στον φάκελο `data/raw`.

Για κάθε σελίδα αποθηκεύτηκε ξεχωριστό metadata αρχείο με `doc_id`, `title`, `topic`, `url`, `why_selected`, αριθμό χαρακτήρων και αριθμό λέξεων.

Τα raw documents θα χρησιμοποιηθούν ως είσοδος για τις δύο ingestion strategies του συστήματος.

## Ingestion strategy v1 — Fixed-size chunking

Η πρώτη ingestion strategy ονομάστηκε `v1_fixed` και βασίστηκε σε fixed-size chunking.

Κάθε raw document χωρίστηκε σε chunks μεγέθους 1200 χαρακτήρων με overlap 200 χαρακτήρων. Η στρατηγική αυτή χρησιμοποιήθηκε ως baseline, επειδή είναι απλή, εύκολα αναπαραγώγιμη και δεν βασίζεται στη δομή των documentation pages.

Για κάθε chunk αποθηκεύτηκαν metadata όπως `chunk_id`, `doc_id`, `title`, `topic`, `url`, `chunking_strategy` και `chunk_index`.

Τα παραγόμενα chunks αποθηκεύτηκαν στο αρχείο `data/processed/v1_fixed_chunks.jsonl`, ενώ τα στατιστικά του chunking αποθηκεύτηκαν στο `outputs/chunk_stats_v1.csv`.

## Ingestion strategy v2 — Structured chunking

Η δεύτερη ingestion strategy ονομάστηκε `v2_structured` και βασίστηκε σε structured / section-based chunking.

Σε αυτή την εκδοχή, το κείμενο κάθε documentation page χωρίστηκε με βάση τη λογική δομή της σελίδας, όπως headings, sections, paragraphs και list items. Στόχος ήταν τα chunks να διατηρούν καλύτερα τη σημασιολογική συνοχή τους σε σχέση με το fixed-size chunking.

Το v2 δημιούργησε συνολικά 375 chunks.

Για κάθε chunk αποθηκεύτηκαν metadata όπως `chunk_id`, `doc_id`, `title`, `topic`, `url`, `section`, `heading_level`, `chunking_strategy` και `chunk_index`.

Τα παραγόμενα chunks αποθηκεύτηκαν στο αρχείο `data/processed/v2_structured_chunks.jsonl`, ενώ τα στατιστικά του chunking αποθηκεύτηκαν στο `outputs/chunk_stats_v2.csv`.

## Vector store setup — Qdrant

Για το vector store χρησιμοποιήθηκε το Qdrant. Το Qdrant εκτελέστηκε τοπικά μέσω Docker Compose, με exposed ports `6333` και `6334`.

Το αρχείο `docker-compose.yml` δημιουργήθηκε στη ρίζα του project και ορίζει το Qdrant service με persistent storage στον φάκελο `qdrant_storage`.

Η λειτουργία του Qdrant ελέγχθηκε με την εντολή `curl http://localhost:6333`, η οποία επέστρεψε επιτυχώς πληροφορίες για το Qdrant service.

Το Qdrant θα χρησιμοποιηθεί για την αποθήκευση και αναζήτηση των embeddings που θα παραχθούν από τα chunks του corpus.

## Embeddings and Qdrant indexing

Για τη δημιουργία embeddings χρησιμοποιήθηκε το μοντέλο `text-embedding-3-small`, το οποίο παράγει vectors διάστασης 1536.

Τα chunks των δύο ingestion strategies αποθηκεύτηκαν σε δύο ξεχωριστές Qdrant collections:

- `sklearn_rag_v1_fixed`: περιέχει τα 402 chunks του fixed-size chunking.
- `sklearn_rag_v2_structured`: περιέχει τα 375 chunks του structured / section-based chunking.

Η επιτυχής δημιουργία των collections ελέγχθηκε με το endpoint `http://localhost:6333/collections`, το οποίο επέστρεψε και τις δύο collections.

## Embeddings and Qdrant indexing

Για τη δημιουργία embeddings χρησιμοποιήθηκε το μοντέλο `text-embedding-3-small`, το οποίο παράγει vectors διάστασης 1536.

Τα chunks των δύο ingestion strategies αποθηκεύτηκαν σε δύο ξεχωριστές Qdrant collections:

- `sklearn_rag_v1_fixed`: περιέχει τα 402 chunks του fixed-size chunking.
- `sklearn_rag_v2_structured`: περιέχει τα 375 chunks του structured / section-based chunking.

Η επιτυχής δημιουργία των collections ελέγχθηκε με το endpoint `http://localhost:6333/collections`, το οποίο επέστρεψε και τις δύο collections.

## Retrieval test — StandardScaler question

Πραγματοποιήθηκε αρχικό retrieval test με την ερώτηση:

"When should I use StandardScaler in scikit-learn?"

Στο `v1_fixed`, το πρώτο αποτέλεσμα είχε score 0.5636 και προήλθε από το document D02, δηλαδή τη σελίδα "Preprocessing data". Ωστόσο, επειδή το fixed-size chunking δεν διατηρεί section information, το πεδίο `section` ήταν κενό.

Στο `v2_structured`, το πρώτο αποτέλεσμα είχε score 0.6393 και προήλθε επίσης από το D02, αλλά από το συγκεκριμένο section "7.3.1. Standardization, or mean removal and variance scaling". Αυτό δείχνει ότι το structured chunking μπορεί να επιστρέψει πιο στοχευμένα και σημασιολογικά καθαρά chunks.

Το συγκεκριμένο retrieval test δείχνει αρχικά ότι το `v2_structured` είναι πιθανό να προσφέρει καλύτερη ποιότητα ανάκτησης σε σχέση με το `v1_fixed`, επειδή διατηρεί τη δομή των documentation pages.

## RAG answer generation test

Υλοποιήθηκε RAG answer generation pipeline, το οποίο εκτελεί τα εξής βήματα:

1. Δημιουργεί embedding για την ερώτηση του χρήστη.
2. Αναζητά τα πιο σχετικά chunks στο Qdrant.
3. Δημιουργεί context από τα retrieved chunks.
4. Δίνει το context και την ερώτηση σε LLM με οδηγία να απαντήσει μόνο με βάση το διαθέσιμο context.
5. Επιστρέφει τελική απάντηση μαζί με τις πηγές που χρησιμοποιήθηκαν.

Πραγματοποιήθηκε δοκιμή με την ερώτηση:

"When should I use StandardScaler in scikit-learn?"

Στο `v1_fixed`, το πρώτο retrieved chunk είχε score 0.5636 και προήλθε από το document "Preprocessing data", αλλά χωρίς section information.

Στο `v2_structured`, το πρώτο retrieved chunk είχε score 0.6394 και προήλθε από το section "7.3.1. Standardization, or mean removal and variance scaling". Η απάντηση του συστήματος ήταν πιο στοχευμένη, καθώς το structured chunking διατήρησε πληροφορία για το section της αρχικής σελίδας.

Τα αποτελέσματα αποθηκεύτηκαν στα αρχεία:
- `outputs/rag_answer_test_sklearn_rag_v1_fixed.md`
- `outputs/rag_answer_test_sklearn_rag_v2_structured.md`

## Grounding / out-of-context test

Πραγματοποιήθηκε δοκιμή με ερώτηση εκτός του corpus:

"Who won the FIFA World Cup in 2022?"

Η ερώτηση δεν σχετίζεται με το scikit-learn documentation. Το σύστημα ανέκτησε chunks με πολύ χαμηλά similarity scores, όμως η τελική απάντηση ήταν:

"The available context does not contain enough information to answer this question."

Αυτό δείχνει ότι το RAG prompt χειρίζεται σωστά περιπτώσεις όπου η απάντηση δεν υπάρχει στο διαθέσιμο context και αποφεύγει να απαντήσει με εξωτερική γνώση ή hallucination.

## Golden test set retrieval comparison

Το golden test set των 30 ερωτήσεων εκτελέστηκε και στις δύο Qdrant collections:

- `sklearn_rag_v1_fixed`
- `sklearn_rag_v2_structured`

Τα αρχικά retrieval αποτελέσματα ήταν:

| Metric | v1_fixed | v2_structured |
|---|---:|---:|
| Expected source found in top-k | 60.00% | 60.00% |
| Expected source was top-1 | 50.00% | 53.33% |

Το `v2_structured` παρουσίασε μικρή βελτίωση στο top-1 retrieval accuracy, από 50.00% σε 53.33%. Ωστόσο, επειδή και οι δύο εκδοχές είχαν 60.00% expected source found in top-k, χρειάζεται failure analysis για να εξεταστεί αν οι αστοχίες οφείλονται στο retrieval ή σε αυστηρά expected_source labels.

## Corrected retrieval evaluation

Η αρχική αξιολόγηση retrieval υποτίμησε την απόδοση του συστήματος, επειδή ορισμένες ερωτήσεις του golden test set είχαν περισσότερες από μία αποδεκτές πηγές, όπως `D02/D03` ή `D02/D03/D05/D06/D07`.

Για τον λόγο αυτό πραγματοποιήθηκε διορθωμένος υπολογισμός των retrieval metrics, όπου τα πολλαπλά expected sources διαχωρίστηκαν σε ξεχωριστά αποδεκτά document IDs.

Τα διορθωμένα αποτελέσματα ήταν:

| Metric | v1_fixed | v2_structured |
|---|---:|---:|
| Expected source found in top-k | 93.33% | 96.67% |
| Expected source was top-1 | 73.33% | 80.00% |
| Failed top-k questions | 2 | 1 |

Τα αποτελέσματα δείχνουν ότι το `v2_structured` παρουσίασε καλύτερη απόδοση από το `v1_fixed`, ειδικά στο top-1 retrieval. Αυτό υποστηρίζει την επιλογή του structured / section-based chunking ως βελτιωμένη ingestion strategy.

## Failure analysis after corrected retrieval evaluation

Μετά τον διορθωμένο υπολογισμό των retrieval metrics, το `v1_fixed` είχε 2 failed top-k questions, ενώ το `v2_structured` είχε μόνο 1 failed top-k question.

Στο `v1_fixed`, οι αποτυχημένες ερωτήσεις ήταν οι Q28 και Q29. Στο `v2_structured`, απέτυχε μόνο η Q28, γεγονός που δείχνει ότι το structured chunking βελτίωσε την ανάκτηση για την Q29.

Η Q28 ήταν:

"Which RandomForestClassifier parameters can control model complexity?"

Η αναμενόμενη πηγή ήταν το D10, δηλαδή η API page του `RandomForestClassifier`. Ωστόσο, το σύστημα ανέκτησε κυρίως chunks από το D09, δηλαδή τη γενική σελίδα για Random Forests. Αυτό δείχνει ότι το retriever μπέρδεψε τη γενική θεωρητική σελίδα Random Forests με την ειδική API page του RandomForestClassifier.

Το failure αυτό είναι λογικό, επειδή οι δύο πηγές είναι θεματικά πολύ κοντινές. Πιθανή βελτίωση θα ήταν η χρήση metadata filtering ή query rewriting, ώστε ερωτήσεις που περιέχουν συγκεκριμένο estimator name και ζητούν parameters να δίνουν προτεραιότητα σε API documentation pages.

## Additional failure analysis for Q28

Για την Q28:

"Which RandomForestClassifier parameters can control model complexity?"

το `v2_structured` δεν ανέκτησε το expected source D10 μέσα στο top-5. Ωστόσο, όταν αυξήθηκε το `top_k` σε 10, chunks από το D10 εμφανίστηκαν στις θέσεις 6, 7 και 10.

Αυτό δείχνει ότι το σύστημα δεν αποτυγχάνει πλήρως να εντοπίσει τη σωστή πηγή. Αντίθετα, η σωστή API page βρίσκεται κοντά στα κορυφαία αποτελέσματα, αλλά κατατάσσεται χαμηλότερα από γενικότερα chunks του D09 και D01.

Η πιθανή αιτία είναι ότι η ερώτηση περιέχει όρους όπως `RandomForestClassifier`, `parameters` και `model complexity`, οι οποίοι ταιριάζουν τόσο στη γενική σελίδα Random Forests όσο και στην API page του RandomForestClassifier.

Πιθανή βελτίωση είναι η χρήση metadata filtering ή query routing, ώστε ερωτήσεις που αναφέρονται σε συγκεκριμένο estimator και ζητούν parameters να δίνουν προτεραιότητα σε API documentation pages.

## Metadata filtering test

Υλοποιήθηκε metadata filtering στο Qdrant με βάση το πεδίο `topic`.

Το filtering δοκιμάστηκε στην Q28:

"Which RandomForestClassifier parameters can control model complexity?"

Χωρίς metadata filtering, το `v2_structured` ανέκτησε κυρίως chunks από το D09, δηλαδή τη γενική σελίδα "Random forests", ενώ το expected source ήταν το D10, δηλαδή η API page του `RandomForestClassifier`.

Με metadata filter `topic = random_forest_classifier`, τα top-5 retrieved chunks προήλθαν όλα από το D10 / `RandomForestClassifier`. Αυτό δείχνει ότι το metadata filtering μπορεί να βελτιώσει την ανάκτηση όταν η ερώτηση αναφέρεται σε συγκεκριμένο estimator ή API page.

Το αποτέλεσμα αποθηκεύτηκε στο αρχείο:
`outputs/filtered_retrieval_sklearn_rag_v2_structured_random_forest_classifier.md`

## ReAct-style agent with two tools

Υλοποιήθηκε ReAct-style routing agent με δύο διαθέσιμα εργαλεία:

1. `rag_retriever`: εκτελεί semantic retrieval σε ολόκληρο το scikit-learn corpus.
2. `metadata_filtered_retriever`: εκτελεί semantic retrieval με metadata filtering στο πεδίο `topic`.

Ο agent χρησιμοποιεί απλή λογική routing για να επιλέξει το κατάλληλο tool ανάλογα με την ερώτηση. Για γενικές ή σύνθετες ερωτήσεις workflow χρησιμοποιεί το `rag_retriever`, ενώ για ερωτήσεις που αφορούν συγκεκριμένο estimator, API page ή parameters χρησιμοποιεί το `metadata_filtered_retriever`.

Υπάρχει επίσης `MAX_ITERATIONS = 2`, ώστε να υπάρχει προστασία από infinite loops.

Πραγματοποιήθηκαν δύο demo runs:

1. Για την ερώτηση:
"How can a complete classification workflow combine preprocessing, model training, cross-validation and evaluation?"

ο agent επέλεξε το `rag_retriever`, επειδή η ερώτηση συνδυάζει πολλά στάδια του machine learning workflow.

2. Για την ερώτηση:
"Which RandomForestClassifier parameters can control model complexity?"

ο agent επέλεξε το `metadata_filtered_retriever` με `topic_filter = random_forest_classifier`. Τα retrieved chunks προήλθαν όλα από το D10 / RandomForestClassifier API page, διορθώνοντας το προηγούμενο failure case της Q28.

Τα demo agent runs αποθηκεύτηκαν στον φάκελο `outputs/`.

## Agentic metric — Tool Call Accuracy

Ως agentic metric χρησιμοποιήθηκε το Tool Call Accuracy.

Το metric αξιολογεί αν ο agent επέλεξε το σωστό retrieval tool για κάθε ερώτηση του golden test set. Αξιολογήθηκαν δύο στοιχεία:

1. Αν επιλέχθηκε το σωστό tool (`rag_retriever` ή `metadata_filtered_retriever`).
2. Αν επιλέχθηκε το σωστό `topic_filter` όταν χρησιμοποιήθηκε το `metadata_filtered_retriever`.

Τα αποτελέσματα στο golden test set των 30 ερωτήσεων ήταν:

| Metric | Score |
|---|---:|
| Tool selection accuracy | 100.00% |
| Topic filter accuracy | 100.00% |
| Full tool call accuracy | 100.00% |
| Total questions | 30 |
| Failed tool calls | 0 |

Το αποτέλεσμα δείχνει ότι ο routing mechanism του agent εφάρμοσε σωστά τους προκαθορισμένους κανόνες επιλογής εργαλείου στο golden test set.

Τα αποτελέσματα αποθηκεύτηκαν στα αρχεία:
- `evaluation/tool_call_accuracy_results.csv`
- `evaluation/tool_call_accuracy_summary.md`

## LLM-as-judge evaluation — v2_structured

Πραγματοποιήθηκε LLM-as-judge evaluation για τις 30 απαντήσεις του `v2_structured` pipeline.

Οι μετρικές αξιολόγησης ήταν:

- Answer Relevancy
- Faithfulness
- Context Precision
- Context Recall
- Expected Answer Alignment

Τα αποτελέσματα ήταν:

| Metric | Mean score |
|---|---:|
| Answer Relevancy | 4.97 / 5 |
| Faithfulness | 4.97 / 5 |
| Context Precision | 4.80 / 5 |
| Context Recall | 4.73 / 5 |
| Expected Answer Alignment | 4.97 / 5 |

Τα αποτελέσματα δείχνουν ότι το `v2_structured` pipeline παράγει απαντήσεις υψηλής συνάφειας και πιστότητας ως προς το retrieved context. Οι ελαφρώς χαμηλότερες τιμές σε Context Precision και Context Recall δείχνουν ότι σε ορισμένες ερωτήσεις το retrieved context περιέχει σχετικές αλλά όχι πάντα πλήρως ιδανικές πηγές, κάτι που φάνηκε και στο failure analysis της Q28.

Τα αποτελέσματα αποθηκεύτηκαν στα αρχεία:
- `evaluation/llm_judge_results_v2_structured_full.csv`
- `evaluation/llm_judge_summary_v2_structured_full.md`

## LLM-as-judge comparison v1 vs v2

Πραγματοποιήθηκε LLM-as-judge evaluation και για τα δύο pipelines, `v1_fixed` και `v2_structured`, πάνω στο golden test set των 30 ερωτήσεων.

Τα αποτελέσματα ήταν:

| Metric | v1_fixed | v2_structured |
|---|---:|---:|
| Answer Relevancy | 4.97 / 5 | 4.97 / 5 |
| Faithfulness | 4.97 / 5 | 4.97 / 5 |
| Context Precision | 4.60 / 5 | 4.80 / 5 |
| Context Recall | 4.57 / 5 | 4.73 / 5 |
| Expected Answer Alignment | 4.97 / 5 | 4.97 / 5 |

Τα αποτελέσματα δείχνουν ότι και τα δύο pipelines παράγουν υψηλής ποιότητας απαντήσεις. Ωστόσο, το `v2_structured` βελτίωσε το Context Precision από 4.60 σε 4.80 και το Context Recall από 4.57 σε 4.73.

Αυτό υποστηρίζει ότι το structured / section-based chunking βοηθά το retrieval να επιστρέφει πιο σχετικά και πληρέστερα context chunks σε σχέση με το fixed-size chunking.

## Langfuse tracing setup

Ρυθμίστηκε Langfuse observability για την παρακολούθηση των OpenAI calls του συστήματος.

Τα Langfuse credentials αποθηκεύτηκαν στο `.env` αρχείο και δεν συμπεριλαμβάνονται στο GitHub repository. Πραγματοποιήθηκε αρχικό test trace με OpenAI chat completion, το οποίο εκτελέστηκε επιτυχώς και εμφανίστηκε στο Langfuse dashboard.

Το Langfuse θα χρησιμοποιηθεί για την παρακολούθηση RAG και agent calls κατά το demo και την παρουσίαση.

## Traced agent demo with Langfuse

Πραγματοποιήθηκε traced agent demo με Langfuse για την ερώτηση:

"Which RandomForestClassifier parameters can control model complexity?"

Ο agent επέλεξε το εργαλείο `metadata_filtered_retriever` με `topic_filter = random_forest_classifier`, επειδή η ερώτηση αφορά συγκεκριμένο estimator και ζητά parameters.

Τα retrieved chunks προήλθαν όλα από το D10 / RandomForestClassifier API page. Η τελική απάντηση δημιουργήθηκε με βάση το retrieved context και περιλάμβανε source citation προς το scikit-learn documentation.

Το demo αποθηκεύτηκε στο αρχείο:
`outputs/traced_agent_demo_q28.md`

Το run καταγράφηκε στο Langfuse, ώστε να μπορεί να παρουσιαστεί ως trace κατά το live demo.

## HAIC Benchmarking — Professor-style evaluation

Πραγματοποιήθηκε HAIC benchmarking σύμφωνα με το schema `haic.decisions_artifact.v1`.

Για κάθε ερώτηση του golden test set δημιουργήθηκε ένα session με events τύπου:

- human query
- ai route
- ai tool_call
- ai respond
- human accept/reject

Συνολικά καταγράφηκαν 30 sessions και 150 HAIC events.

Οι παράμετροι που χρησιμοποιήθηκαν ήταν:

| Parameter | Value |
|---|---:|
| baseline_s | 30.00 |
| rt_max_s | 30.00 |
| sessions | 30 |
| logged events | 150 |

Τα αποτελέσματα ήταν:

| Metric | Value | Interpretation |
|---|---:|---|
| EL | 0.000 | No effort loss compared to baseline. |
| Tr | 0.967 | High acceptance of AI responses. |
| HCL | 0.865 | High human-centeredness / low cognitive load. |
| F | 46.983 | High interaction frequency. |
| A | 0.197 | Positive adaptability across the session. |
| D | 1.277 s | Mean duration per decision event. |
| EfficiencyScore | 1.000 | High composite efficiency. |
| S | N/A | Excluded because no surrogate simulation was used. |

Quadrant diagnostics:

- EL × Tr: Efficient & Trusted
- HCL × F: Smooth collaboration

Limitation: το human accept/reject event υπολογίστηκε offline με proxy. Συγκεκριμένα, μία απάντηση θεωρήθηκε accepted όταν το expected source βρέθηκε στα retrieved chunks. Αυτό αποτελεί περιορισμό, επειδή δεν πραγματοποιήθηκε πραγματικό user study.
