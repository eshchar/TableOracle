# Eval results

Generated 2026-09-02T10:16:24+00:00 · 56 questions · embed `bge-small-en-v1.5` · answer `claude-sonnet-5`

Regenerate with `make eval`. `make eval-retrieval` reruns the retrieval half only, which needs no API key and costs nothing.

## Retrieval

| metric | value |
|---|---|
| retrieval@1 | 72.9% |
| retrieval@3 | 91.7% |
| **retrieval@5** | **93.8%** |
| retrieval@10 | 97.9% |
| MRR | 0.820 |
| latency p50 / p95 | 16.82 ms / 20.0 ms |

Scored over 48 answerable questions (8 abstention questions are scored separately).

**Missed at k=5** (3): `avoid-opportunity-attack`, `death-save-damage`, `two-weapon-fighting`

### Can retrieval score predict answerability?

Furthest answerable question: **0.3303**. Nearest unanswerable question: **0.2856**.

These ranges **overlap**, so no distance threshold separates answerable from unanswerable questions. Questions that are in-domain but absent from the corpus retrieve plenty of adjacent text that does not answer them. The threshold (`0.4`) is therefore set above the answerable range and only catches genuinely off-domain questions; everything inside the overlap is caught after the fact by the no-citation check.

