# Chunking comparison: v1 vs v2

## Summary

| Strategy | Description | Total chunks | Mean avg chunk length |
|---|---|---:|---:|
| v1_fixed | Fixed-size chunking with 1200 characters and 200 overlap | 402 | 1166.03 |
| v2_structured | Section-based chunking using documentation structure | 375 | 1279.55 |

## Initial interpretation

The v1_fixed strategy produced 402 chunks, while the v2_structured strategy produced 375 chunks.

The first strategy is useful as a simple baseline because it splits text by character length. However, it may cut meaningful sections in the middle.

The second strategy attempts to preserve the structure of the original documentation pages by using headings and sections. This may improve retrieval quality because each chunk is more likely to contain semantically coherent information.

Final conclusions will be based on retrieval and evaluation results in the next stages.
