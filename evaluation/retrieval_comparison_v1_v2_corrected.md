# Corrected retrieval comparison v1 vs v2

The first retrieval comparison underestimated performance because some golden test questions had multiple acceptable expected sources, such as `D02/D03` or `D02/D03/D05/D06/D07`.

The corrected evaluation parses these labels as separate acceptable document IDs.

| Metric | v1_fixed | v2_structured |
|---|---:|---:|
| Expected source found in top-k | 93.33% | 96.67% |
| Expected source was top-1 | 73.33% | 80.00% |
| Failed top-k questions | 2 | 1 |

## Interpretation

After correcting the handling of multiple expected sources, both retrieval pipelines perform substantially better than the initial strict-label evaluation suggested.

The structured chunking strategy performs better overall, especially in top-1 retrieval. This indicates that preserving section-level structure helps the retriever return more directly relevant chunks.
