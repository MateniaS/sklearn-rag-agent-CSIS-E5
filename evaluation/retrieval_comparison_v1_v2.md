# Retrieval comparison v1 vs v2

| Metric | v1_fixed | v2_structured |
|---|---:|---:|
| Expected source found in top-k | 60.00% | 60.00% |
| Expected source was top-1 | 50.00% | 53.33% |

## Initial interpretation

Both ingestion strategies retrieved the expected source within the top-k results for 60.00% of the golden test set questions.

The structured chunking strategy showed a small improvement in top-1 retrieval accuracy, increasing from 50.00% in v1_fixed to 53.33% in v2_structured.

Further failure analysis is required to determine whether the remaining failures are caused by retrieval limitations or by strict expected_source labels in the golden test set.