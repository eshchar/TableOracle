"""`make eval` entry point."""

from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime, timezone

from tableoracle.config import get_settings
from tableoracle.evals import harness
from tableoracle.evals.report import write_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tableoracle-eval")
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="score retrieval only: no model calls, no cost, no API key needed",
    )
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    parser.add_argument("--tag", default=None, help="only questions carrying this tag")
    parser.add_argument("--rejudge", default=None, metavar="RESULTS_JSON",
                        help="re-grade the answers in a saved results file "
                             "without paying to answer them again")
    parser.add_argument("--name", default=None,
                        help="results filename stem (defaults to the answer model)")
    args = parser.parse_args(argv)

    settings = get_settings()

    if args.rejudge:
        return rejudge(args, settings)

    questions = harness.load_questions()

    problems = harness.validate(questions, settings)
    if problems:
        print("Eval set does not match the index:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if args.tag:
        questions = [q for q in questions if args.tag in q.tags]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("No questions selected.", file=sys.stderr)
        return 1

    print(f"Table Oracle eval -- {len(questions)} questions")
    print(f"  embed model  {settings.embed_model}")
    print(f"  answer model {settings.answer_model}")
    print()

    print("Retrieval (no model calls, no cost)")
    retrieval_rows = harness.run_retrieval(questions, settings)
    retrieval = harness.score_retrieval(retrieval_rows, settings)
    print(f"  retrieval@1 {retrieval['retrieval_at_1']:.3f}"
          f"  @3 {retrieval['retrieval_at_3']:.3f}"
          f"  @5 {retrieval['retrieval_at_5']:.3f}"
          f"  MRR {retrieval['mrr']:.3f}")
    print(f"  p95 {retrieval['latency_ms']['p95']} ms")
    print()

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embed_model": settings.embed_model,
        "answer_model": settings.answer_model,
        "questions": len(questions),
        "retrieval": retrieval,
        "retrieval_rows": harness.rows_to_dicts(retrieval_rows),
    }

    if args.retrieval_only:
        path = harness.save(payload, args.name or "retrieval")
        write_markdown(payload, args.name or "retrieval")
        print(f"wrote {path}")
        return 0

    print("Answers (this costs money)")
    answer_rows = harness.run_answers(questions, settings)

    judge_cost = 0.0
    if not args.no_judge:
        print("\nJudge")
        from tableoracle.evals.judge import judge_answers
        judge_cost = judge_answers(answer_rows, settings)

    answers = harness.score_answers(answer_rows)
    answers["judge_cost_usd"] = round(judge_cost, 4)
    payload["answers"] = answers
    payload["answer_rows"] = harness.rows_to_dicts(answer_rows)

    print()
    print(f"  citation accuracy   {fmt(answers['citation_accuracy'])}")
    print(f"  citation precision  {fmt(answers['citation_precision'])}")
    print(f"  abstention correct  {fmt(answers['abstention']['correct'])}")
    print(f"  false abstain rate  {fmt(answers['abstention']['false_abstain_rate'])}")
    if answers["judge"]:
        print(f"  judge correct       {fmt(answers['judge']['correct_rate'])}")
    print(f"  mean $/query        ${answers['cost_usd']['mean']:.5f}")
    print(f"  p95 latency         {answers['latency_ms']['p95']} ms")
    print(f"  total spend         ${answers['cost_usd']['total'] + judge_cost:.4f}")

    stem = args.name or f"eval-{settings.answer_model}"
    path = harness.save(payload, stem)
    write_markdown(payload, stem)
    print(f"\nwrote {path}")
    return 0


def rejudge(args, settings) -> int:
    """Re-grade saved answers. Fixing the grader must not cost another run."""
    import json

    from tableoracle.evals.judge import judge_answers

    payload = json.loads(pathlib.Path(args.rejudge).read_text(encoding="utf-8"))
    rows = [harness.AnswerRow(**row) for row in payload["answer_rows"]]
    for row in rows:
        row.judge_verdict = None
        row.judge_reason = None

    print(f"Re-judging {len(rows)} saved answers from {args.rejudge}")
    cost = judge_answers(rows, settings)

    answers = harness.score_answers(rows)
    answers["judge_cost_usd"] = round(
        payload.get("answers", {}).get("judge_cost_usd", 0) + cost, 4
    )
    payload["answers"] = answers
    payload["answer_rows"] = harness.rows_to_dicts(rows)

    stem = args.name or f"eval-{payload['answer_model']}"
    path = harness.save(payload, stem)
    write_markdown(payload, stem)
    print()
    print(f"  judge correct  {fmt(answers['judge']['correct_rate'])}")
    print(f"  re-judge cost  ${cost:.4f}")
    print(f"wrote {path}")
    return 0


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
