"""Render eval results as committed markdown.

The failure lists are not an afterthought. A results file that reports only
aggregate percentages is unauditable: naming the questions that missed is what
lets a reader disagree with the numbers.
"""

from __future__ import annotations

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "evals" / "results"


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def write_markdown(payload: dict, stem: str = "latest") -> Path:
    r = payload["retrieval"]
    a = payload.get("answers")
    lines: list[str] = []

    lines.append("# Eval results")
    lines.append("")
    lines.append(f"Generated {payload['generated_utc']} · "
                 f"{payload['questions']} questions · "
                 f"embed `{payload['embed_model']}` · answer `{payload['answer_model']}`")
    lines.append("")
    lines.append("Regenerate with `make eval`. `make eval-retrieval` reruns the "
                 "retrieval half only, which needs no API key and costs nothing.")
    lines.append("")

    lines.append("## Retrieval")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| retrieval@1 | {_pct(r['retrieval_at_1'])} |")
    lines.append(f"| retrieval@3 | {_pct(r['retrieval_at_3'])} |")
    lines.append(f"| **retrieval@5** | **{_pct(r['retrieval_at_5'])}** |")
    lines.append(f"| retrieval@10 | {_pct(r['retrieval_at_10'])} |")
    lines.append(f"| MRR | {r['mrr']:.3f} |")
    lines.append(f"| latency p50 / p95 | {r['latency_ms']['p50']} ms / {r['latency_ms']['p95']} ms |")
    lines.append("")
    lines.append(f"Scored over {r['answerable']} answerable questions "
                 f"({r['abstention']} abstention questions are scored separately).")
    lines.append("")

    if r["misses_at_5"]:
        lines.append(f"**Missed at k=5** ({len(r['misses_at_5'])}): "
                     + ", ".join(f"`{m}`" for m in r["misses_at_5"]))
        lines.append("")

    d = r["distance"]
    lines.append("### Can retrieval score predict answerability?")
    lines.append("")
    lines.append(f"Furthest answerable question: **{d['answerable_max']}**. "
                 f"Nearest unanswerable question: **{d['abstention_min']}**.")
    lines.append("")
    if not d["separable"]:
        lines.append("These ranges **overlap**, so no distance threshold separates "
                     "answerable from unanswerable questions. Questions that are "
                     "in-domain but absent from the corpus retrieve plenty of "
                     "adjacent text that does not answer them. The threshold "
                     f"(`{d['threshold']}`) is therefore set above the answerable "
                     "range and only catches genuinely off-domain questions; "
                     "everything inside the overlap is caught after the fact by "
                     "the no-citation check.")
    else:
        lines.append(f"These ranges are separable at the current threshold "
                     f"(`{d['threshold']}`).")
    lines.append("")

    if a:
        lines.append("## Answers")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| **citation accuracy** | **{_pct(a['citation_accuracy'])}** |")
        lines.append(f"| citation precision | {_pct(a['citation_precision'])} |")
        lines.append(f"| answered with at least one citation | {_pct(a['answered_with_a_citation'])} |")
        lines.append(f"| expected phrase recall | {_pct(a['expected_phrase_recall'])} |")
        if a.get("judge"):
            lines.append(f"| judge: correct | {_pct(a['judge']['correct_rate'])} |")
        lines.append(f"| abstention correct | {_pct(a['abstention']['correct'])} |")
        lines.append(f"| false abstention rate | {_pct(a['abstention']['false_abstain_rate'])} |")
        lines.append(f"| mean $/query | ${a['cost_usd']['mean']:.5f} |")
        lines.append(f"| p50 / p95 latency | {a['latency_ms']['p50']} ms / {a['latency_ms']['p95']} ms |")
        lines.append(f"| tool calls | {a['tool_calls_total']} |")
        lines.append(f"| total run cost | ${a['cost_usd']['total'] + a.get('judge_cost_usd', 0):.4f} |")
        lines.append("")

        lines.append("**Definitions.** *Citation accuracy* is the share of "
                     "answerable questions where at least one emitted citation "
                     "resolves to a chunk the corpus says holds the rule; citing "
                     "something else earns no partial credit. *Citation precision* "
                     "is the share of all emitted citations that land on an "
                     "expected chunk. *Abstention correct* is the share of "
                     "unanswerable questions where the system declined or cited "
                     "nothing. *False abstention* is refusing a question the "
                     "corpus can answer, which is the costly mistake.")
        lines.append("")

        if a["citation_failures"]:
            lines.append(f"**Cited no expected chunk** ({len(a['citation_failures'])}): "
                         + ", ".join(f"`{x}`" for x in a["citation_failures"]))
            lines.append("")
        if a["abstention"]["missed"]:
            lines.append("**Should have abstained but did not**: "
                         + ", ".join(f"`{x}`" for x in a["abstention"]["missed"]))
            lines.append("")
        if a["abstention"]["false_abstained"]:
            lines.append("**Wrongly abstained**: "
                         + ", ".join(f"`{x}`" for x in a["abstention"]["false_abstained"]))
            lines.append("")
        if a.get("judge") and a["judge"]["failed"]:
            lines.append("**Judged not correct**: "
                         + ", ".join(f"`{x}`" for x in a["judge"]["failed"]))
            lines.append("")
        if a["errors"]:
            lines.append("**Errors**: " + ", ".join(f"`{e['id']}`" for e in a["errors"]))
            lines.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{stem}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
