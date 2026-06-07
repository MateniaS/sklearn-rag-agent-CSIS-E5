# HAIC Benchmarking — v2_structured

Professor-style HAIC evaluation based on `haic.decisions_artifact.v1`.

## Configuration

| Parameter | Value |
|---|---:|
| baseline_s | 30.00 |
| rt_max_s | 30.00 |
| sessions | 30 |
| logged events | 150 |

## Metrics

| Metric | Value | Interpretation |
|---|---:|---|
| EL | 0.000 | Effort loss compared to baseline. |
| Tr | 0.967 | Fraction of accepted AI responses. |
| HCL | 0.883 | Higher means lower cognitive load. |
| F | 51.404 | Interaction events per minute. |
| A | 0.197 | Adaptability across early vs late decisions. |
| D | 1.167 s | Mean duration per decision event. |
| EfficiencyScore | 1.000 | Composite efficiency score. |
| S | N/A | Excluded because no surrogate simulation was used. |

## Quadrant diagnostics

- EL × Tr: Efficient & Trusted
- HCL × F: Smooth collaboration

## Limitation

The human accept/reject event is approximated offline using an acceptance proxy: a response is accepted when the expected source was retrieved. This is a limitation because no real user study was conducted.
