"""Command line entry points: ingest, search, ask, status."""

from __future__ import annotations

import argparse
import sys

from tableoracle.config import get_settings


def _force_utf8_stdout() -> None:
    """Print UTF-8 regardless of the console's code page.

    Rules text is full of em dashes and typographic quotes. On a Windows
    console defaulting to cp1252 those stream out as mojibake, which looks
    like the model produced garbage when in fact the answer was fine.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def cmd_ingest(args: argparse.Namespace) -> int:
    from tableoracle.ingest.pipeline import ingest

    report = ingest(get_settings())
    print()
    print(f"  documents        {report.documents}")
    print(f"  chunks           {report.chunks}")
    print(f"  chunk tokens     {report.tokens:,}")
    print(f"  offsets verified {report.offsets_verified}")
    print(f"  embed tokens     {report.embed_tokens:,}")
    print(f"  embed cost       ${report.embed_cost_usd:.4f}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from tableoracle.store import db

    settings = get_settings()
    conn = db.connect(settings)
    try:
        status = db.index_status(conn)
        print(f"database     {settings.db_path}")
        print(f"chunks       {status['chunks']}")
        print(f"vectors      {status['vectors']}")
        print(f"rulebook     {status['rulebook']}")
        print(f"last ingest  {status['last_ingest']}")
        try:
            db.assert_index_usable(conn, settings)
            print("index        ready")
        except db.StaleIndexError as exc:
            print(f"index        NOT USABLE -- {exc}")
            return 1
    finally:
        conn.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from tableoracle.ingest.embed import get_provider
    from tableoracle.store import db, search

    settings = get_settings()
    conn = db.connect(settings)
    try:
        db.assert_index_usable(conn, settings)
        provider = get_provider(settings)
        vector = provider.embed_query(args.query)
        retrieval = search.search(
            conn,
            args.query,
            vector,
            k=args.k or settings.top_k,
            candidate_k=settings.candidate_k,
            rrf_k=settings.rrf_k,
        )
    except db.StaleIndexError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    best = retrieval.best_vector_distance
    print(f'query: {args.query!r}')
    print(
        f"vector hits {retrieval.vector_hits} | keyword hits {retrieval.keyword_hits} | "
        f"best distance {'n/a' if best is None else f'{best:.4f}'}"
    )
    print()
    for i, result in enumerate(retrieval.results, start=1):
        print(f"[{i}] rrf={result.rrf_score:.5f}  vec_rank={result.vec_rank}  bm25_rank={result.bm25_rank}")
        print(f"    {result.section_path}")
        print(f"    {result.anchor}")
        print(f"    {result.source_file} [{result.source_start}:{result.source_end}]  {result.token_count} tokens")
        if args.verbose:
            body = result.text if args.full else result.text[:400]
            print("    " + body.replace("\n", "\n    "))
        print()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from tableoracle.answer.service import AnswerService

    service = AnswerService(get_settings())
    citations: list[dict] = []
    exit_code = 0

    for event in service.answer_stream(args.question, args.k):
        if event.type == "token":
            sys.stdout.write(event.data["text"])
            sys.stdout.flush()
        elif event.type == "citation":
            citations.append(event.data)
        elif event.type == "retrieval":
            best = event.data["best_vector_distance"]
            print(
                f"[retrieved {len(event.data['chunks'])} chunks in "
                f"{event.data['retrieval_ms']}ms, best distance "
                f"{'n/a' if best is None else f'{best:.4f}'}]\n"
            )
        elif event.type == "warning":
            print(f"\n\n[warning] {event.data['message']}", file=sys.stderr)
        elif event.type == "error":
            print(f"\n\nerror: {event.data['message']}", file=sys.stderr)
            exit_code = 1
        elif event.type == "usage":
            data = event.data
            print("\n")
            print(
                f"[{data['input_tokens']} in / {data['output_tokens']} out"
                f" | ${data['cost_usd']:.5f} | ttft {data['ttft_ms']}ms"
                f" | total {data['total_ms']}ms"
                f" | cache read {data['cache_read_tokens']}]"
            )

    if citations:
        print("\nSources")
        seen = set()
        for citation in citations:
            key = (citation["anchor"], citation["source_start"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {citation['section_path']}")
            print(
                f"    {citation['source_file']}"
                f" [{citation['source_start']}:{citation['source_end']}]"
            )
            quoted = citation["cited_text"].strip().replace("\n", " ")
            print(f"    \"{quoted[:160]}\"")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tableoracle", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="build the index from the corpus").set_defaults(func=cmd_ingest)
    sub.add_parser("status", help="show index status").set_defaults(func=cmd_status)

    search_parser = sub.add_parser("search", help="retrieve chunks, no model call")
    search_parser.add_argument("query")
    search_parser.add_argument("-k", type=int, default=None)
    search_parser.add_argument("-v", "--verbose", action="store_true", help="print chunk text")
    search_parser.add_argument("--full", action="store_true", help="print full chunk text")
    search_parser.set_defaults(func=cmd_search)

    ask_parser = sub.add_parser("ask", help="stream a grounded, cited answer")
    ask_parser.add_argument("question")
    ask_parser.add_argument("-k", type=int, default=None)
    ask_parser.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
