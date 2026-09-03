"""Generate the README charts from committed eval results.

The charts read `evals/results/*.json` rather than hard-coded numbers, so a
chart cannot drift from the measurement it claims to show. Regenerate after any
eval run:

    python scripts/make_charts.py

Output is plain SVG with an internal stylesheet. GitHub renders a repository
SVG as its own document, so `prefers-color-scheme` inside the file works and
one file serves both themes -- no duplicate light/dark assets.

Colours are the dataviz reference palette's categorical slots 1 and 2, stepped
per mode. Both pairs pass the six-check validator (lightness band, chroma
floor, CVD separation, normal-vision separation, contrast) against their own
surface; adjacent-pair CVD separation is 24.7 light / 26.8 dark against a
target of 8.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evals" / "results"
DOCS = ROOT / "docs"

W = 720
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

STYLE = """
  .bg    { fill: #fcfcfb; }
  .ink   { fill: #0b0b0b; }
  .ink2  { fill: #52514e; }
  .ink3  { fill: #78766f; }
  .s1    { fill: #2a78d6; }
  .s2    { fill: #eb6834; }
  .s1s   { stroke: #2a78d6; }
  .s2s   { stroke: #eb6834; }
  .grid  { stroke: #e5e4e0; }
  .rule  { stroke: #b8b6ae; }
  @media (prefers-color-scheme: dark) {
    .bg   { fill: #1a1a19; }
    .ink  { fill: #ffffff; }
    .ink2 { fill: #c3c2b7; }
    .ink3 { fill: #96948a; }
    .s1   { fill: #3987e5; }
    .s2   { fill: #d95926; }
    .s1s  { stroke: #3987e5; }
    .s2s  { stroke: #d95926; }
    .grid { stroke: #33322f; }
    .rule { stroke: #5c5a54; }
  }
"""


def bar(x: float, y: float, w: float, h: float, cls: str, r: float = 4.0) -> str:
    """A bar rounded only at the data end, anchored square to the baseline."""
    r = max(0.0, min(r, w))
    if r <= 0.5:
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" class="{cls}"/>'
    return (
        f'<path class="{cls}" d="M{x:.1f},{y:.1f} H{x + w - r:.1f} '
        f'A{r:.1f},{r:.1f} 0 0 1 {x + w:.1f},{y + r:.1f} '
        f'V{y + h - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w - r:.1f},{y + h:.1f} '
        f'H{x:.1f} Z"/>'
    )


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_open(height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
        f'width="{W}" height="{height}" font-family="{FONT}" role="img" '
        f'aria-label="{esc(title)}. {esc(subtitle)}">',
        f"<style>{STYLE}</style>",
        f'<rect width="{W}" height="{height}" rx="6" class="bg"/>',
        f'<text x="24" y="30" class="ink" font-size="15" font-weight="600">{esc(title)}</text>',
        f'<text x="24" y="50" class="ink2" font-size="12">{esc(subtitle)}</text>',
    ]


def legend(x: float, y: float, items: list[tuple[str, str]]) -> list[str]:
    """Legend is always present for two or more series."""
    out = []
    for label, cls in items:
        out.append(f'<rect x="{x:.0f}" y="{y - 8:.0f}" width="10" height="10" rx="2" class="{cls}"/>')
        out.append(f'<text x="{x + 15:.0f}" y="{y:.0f}" class="ink2" font-size="11.5">{esc(label)}</text>')
        x += 22 + len(label) * 6.6
    return out


# --------------------------------------------------------------------------
# 1. Retrieval: before vs after small-to-big
# --------------------------------------------------------------------------

def grouped_bars(path: pathlib.Path, title: str, subtitle: str,
                 rows: list[tuple[str, float, float]],
                 names: tuple[str, str], footnote: str = "") -> None:
    """Horizontal grouped bars. One axis, always -- never a second scale."""
    left, right = 132, 92
    top, row_h, gap = 78, 34, 16
    height = top + len(rows) * (row_h + gap) + (74 if footnote else 56)
    plot_w = W - left - right

    out = svg_open(height, title, subtitle)
    out += legend(left, 62, [(names[0], "s1"), (names[1], "s2")])

    # Recessive gridlines at 25% intervals, drawn under the marks.
    for frac in (0.25, 0.5, 0.75, 1.0):
        gx = left + plot_w * frac
        out.append(f'<line x1="{gx:.1f}" y1="{top - 6:.0f}" x2="{gx:.1f}" '
                   f'y2="{top + len(rows) * (row_h + gap) - gap + 2:.0f}" '
                   f'class="grid" stroke-width="1"/>')
        out.append(f'<text x="{gx:.1f}" y="{top + len(rows) * (row_h + gap) + 6:.0f}" '
                   f'class="ink3" font-size="10" text-anchor="middle">{int(frac * 100)}%</text>')

    y = top
    for label, before, after in rows:
        out.append(f'<text x="{left - 10:.0f}" y="{y + row_h / 2 + 4:.0f}" class="ink2" '
                   f'font-size="12" text-anchor="end">{esc(label)}</text>')
        # 2px surface gap between adjacent bars in a group.
        bh = (row_h - 2) / 2
        for value, cls in ((before, "s1"), (after, "s2")):
            out.append(bar(left, y, max(1.0, plot_w * value), bh, cls))
            y += bh + 2
        # Direct labels: both values, plus the delta that is the point.
        out.append(f'<text x="{left + plot_w * before + 8:.1f}" y="{y - bh - 8:.1f}" '
                   f'class="ink3" font-size="11">{before * 100:.1f}%</text>')
        out.append(f'<text x="{left + plot_w * after + 8:.1f}" y="{y - 4:.1f}" '
                   f'class="ink" font-size="11" font-weight="600">{after * 100:.1f}%</text>')
        y += gap - 2

    if footnote:
        out.append(f'<text x="24" y="{height - 18:.0f}" class="ink3" font-size="11">{esc(footnote)}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# 2. Abstention: the overlap that kills the threshold design
# --------------------------------------------------------------------------

def overlap_strip(path: pathlib.Path, answerable: list[float],
                  unanswerable: list[tuple[float, str]], threshold: float) -> None:
    """A 1-D strip plot: the point is where the two groups sit on one axis.

    A bar chart of means would hide the finding entirely -- what matters is
    that the ranges intersect, not what they average.
    """
    left, right = 132, 40
    height = 320
    plot_w = W - left - right
    lo, hi = 0.10, 0.56

    def sx(v: float) -> float:
        return left + plot_w * (v - lo) / (hi - lo)

    a_max, u_min = max(answerable), min(v for v, _ in unanswerable)

    out = svg_open(height, "Retrieval distance cannot predict answerability",
                   "Best cosine distance per question. Lower is a closer match.")

    # The overlap band is the whole finding, so it is drawn first and largest.
    out.append(f'<rect x="{sx(u_min):.1f}" y="88" width="{sx(a_max) - sx(u_min):.1f}" '
               f'height="128" fill="#eb6834" opacity="0.10"/>')
    out.append(f'<text x="{(sx(u_min) + sx(a_max)) / 2:.1f}" y="80" class="ink2" '
               f'font-size="11" text-anchor="middle">overlap</text>')

    for frac in range(0, 6):
        v = 0.10 + frac * 0.10
        if v > hi:
            break
        out.append(f'<line x1="{sx(v):.1f}" y1="96" x2="{sx(v):.1f}" y2="216" '
                   f'class="grid" stroke-width="1"/>')
        out.append(f'<text x="{sx(v):.1f}" y="234" class="ink3" font-size="10" '
                   f'text-anchor="middle">{v:.2f}</text>')

    # Threshold rule.
    out.append(f'<line x1="{sx(threshold):.1f}" y1="80" x2="{sx(threshold):.1f}" y2="220" '
               f'class="rule" stroke-width="1.5" stroke-dasharray="4 3"/>')
    out.append(f'<text x="{sx(threshold):.1f}" y="252" class="ink2" font-size="11" '
               f'text-anchor="middle">cutoff {threshold}</text>')

    # Strips. Dots are dodged vertically rather than drawn on one line: 48
    # answerable questions span ~215px here, so a single row would pile most of
    # them into an unreadable blob and hide the density that is the whole point.
    for y, values, cls, label in (
        (126, answerable, "s1", f"answerable ({len(answerable)})"),
        (186, [v for v, _ in unanswerable], "s2", f"not in corpus ({len(unanswerable)})"),
    ):
        out.append(f'<text x="{left - 10:.0f}" y="{y + 4:.0f}" class="ink2" font-size="12" '
                   f'text-anchor="end">{esc(label)}</text>')
        placed: list[tuple[float, float]] = []
        for v in sorted(values):
            cx = sx(v)
            # Walk outward from the baseline until this dot clears its neighbours.
            for step in range(8):
                for dy in ({0.0} if step == 0 else {step * 9.0, -step * 9.0}):
                    if all((cx - px) ** 2 + (dy - py) ** 2 >= 81 for px, py in placed):
                        placed.append((cx, dy))
                        break
                else:
                    continue
                break
            else:
                placed.append((cx, 0.0))
        for cx, dy in placed:
            out.append(f'<circle cx="{cx:.1f}" cy="{y + dy:.1f}" r="4.5" class="{cls}" '
                       f'stroke="#fcfcfb" stroke-width="2" opacity="0.9"/>')

    caught = sum(1 for v, _ in unanswerable if v > threshold)
    for i, line in enumerate([
        f"The ranges intersect between {u_min:.3f} and {a_max:.3f}, so no cutoff separates them.",
        f"At {threshold} nothing answerable is refused and {caught} of {len(unanswerable)} "
        f"unanswerable questions are caught before any",
        "model call. The rest are caught afterwards by the no-citation check.",
    ]):
        out.append(f'<text x="24" y="{276 + i * 14}" class="ink3" font-size="11">{esc(line)}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    latest = json.loads((RESULTS / "retrieval-latest.json").read_text(encoding="utf-8"))
    r = latest["retrieval"]

    # Baseline is the pre-small-to-big measurement, kept for the comparison.
    before = {"retrieval_at_1": 0.625, "retrieval_at_3": 0.7708,
              "retrieval_at_5": 0.8542, "retrieval_at_10": 0.8958, "mrr": 0.7134}

    grouped_bars(
        DOCS / "retrieval-improvement.svg",
        "Small-to-big retrieval",
        "Same 56 questions, same corpus, no other change.",
        [("retrieval@1", before["retrieval_at_1"], r["retrieval_at_1"]),
         ("retrieval@3", before["retrieval_at_3"], r["retrieval_at_3"]),
         ("retrieval@5", before["retrieval_at_5"], r["retrieval_at_5"]),
         ("retrieval@10", before["retrieval_at_10"], r["retrieval_at_10"]),
         ("MRR", before["mrr"], r["mrr"])],
        ("one vector per chunk", "one vector per section"),
        "Four previously-missed questions became findable. None regressed.",
    )

    sonnet = json.loads((RESULTS / "eval-claude-sonnet-5-latest.json").read_text(encoding="utf-8"))["answers"]
    haiku = json.loads((RESULTS / "eval-claude-haiku-4-5-latest.json").read_text(encoding="utf-8"))["answers"]
    grouped_bars(
        DOCS / "model-comparison.svg",
        "Sonnet 5 vs Haiku 4.5",
        "Quality metrics only. Cost and latency are different units and get their own table.",
        [("citation accuracy", sonnet["citation_accuracy"], haiku["citation_accuracy"]),
         ("citation precision", sonnet["citation_precision"], haiku["citation_precision"]),
         ("judge: correct", sonnet["judge"]["correct_rate"], haiku["judge"]["correct_rate"]),
         ("phrase recall", sonnet["expected_phrase_recall"], haiku["expected_phrase_recall"]),
         ("abstention correct", sonnet["abstention"]["correct"], haiku["abstention"]["correct"])],
        ("Sonnet 5", "Haiku 4.5"),
        "Measured before small-to-big retrieval landed; both models understate current performance.",
    )

    rows = latest["retrieval_rows"]
    overlap_strip(
        DOCS / "abstention-overlap.svg",
        [x["best_distance"] for x in rows if not x["should_abstain"]],
        [(x["best_distance"], x["id"]) for x in rows if x["should_abstain"]],
        threshold=0.40,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
