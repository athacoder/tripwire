"""Render results as Markdown: readable in a terminal and as a pull-request comment.

The same Markdown can be turned into one self-contained HTML page, for a CI artifact.
"""

from __future__ import annotations

import html
import re
from typing import Any

MEANING = {
    "IMPROVED": "the candidate is confidently better.",
    "PASS": "any drop is confidently smaller than the margin.",
    "REGRESSED": "the candidate is confidently worse, and may be worse than the margin.",
    "INCONCLUSIVE": "cannot tell 'fine' from 'too much worse' on this data. More cases would help.",
    "INVALID": "too many cases have no usable result, so nothing can be concluded.",
    "UNCHANGED": "base and head produce the same target, so there is nothing to compare.",
}


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _trace(trace_id: str | None, base_url: str | None) -> str:
    if not trace_id:
        return ""
    trace_id = str(trace_id)  # a target may report a number
    return f"[{trace_id[:8]}]({base_url.rstrip('/')}/traces/{trace_id})" if base_url else trace_id


def comparison(
    suite: str,
    r: dict[str, Any],
    is_blocked: bool = False,
    trace_url: str | None = None,
    shown: tuple[int, int] = (10, 5),
) -> str:
    """`shown` caps how many broken and fixed cases are listed; the result holds more."""
    verdict = r["verdict"]
    parts = [
        f"## Tripwire: {verdict} · `{suite}`",
        f"**{verdict}**: {MEANING[verdict]}"
        + (" **This blocks the merge.**" if is_blocked else ""),
    ]
    if "base_fingerprint" in r:
        parts.append(f"base `{r['base_fingerprint']}` → head `{r['head_fingerprint']}`")
    if "reason" in r:
        parts.append(r["reason"])
    if "delta" not in r:
        return "\n\n".join(parts) + "\n"

    level = round((1 - 2 * r["alpha"]) * 100)
    parts.append(
        _table(
            ["metric", "base", "head", "Δ", f"{level}% interval", "p"],
            [
                [
                    r["metric"],
                    f"{r['base']:.3f}",
                    f"{r['head']:.3f}",
                    f"{r['delta']:+.3f}",
                    f"[{r['lo']:+.3f}, {r['hi']:+.3f}]",
                    f"{r['p']:.4f}",
                ]
            ],
        )
    )
    mcnemar = f" · McNemar p {r['mcnemar_p']:.4f}" if r.get("mcnemar_p") is not None else ""
    parts.append(
        f"margin {r['margin']:.3f} · α {r['alpha']} · paired on {r['paired']} of {r['cases']} cases"
        + mcnemar
    )
    if r.get("stages", 1) > 1:
        note = (
            "decided on the first-stage subset; the remaining cases were not run"
            if r["stage"] == 1
            else "the first-stage subset could not decide, so every case was run"
        )
        parts.append(f"Stage {r['stage']} of {r['stages']}: {note}. Each stage uses half of α.")

    if r["slices"]:
        rows = [
            [
                s["tag"],
                s["n"],
                f"{s['delta']:+.3f}",
                f"[{s['lo']:+.3f}, {s['hi']:+.3f}]",
                f"{s['p_adjusted']:.3f}",
                "**regressed**" if s["regressed"] else "",
            ]
            for s in r["slices"]
        ]
        note = f"{r['small_slices']} smaller slices have too few cases to judge."
        parts.append(
            "### Slices\n\n"
            + _table(["slice", "n", "Δ", "interval", "p (FDR)", ""], rows)
            + f"\n\n{note}"
        )

    if r["guardrails"]:
        rows = [
            [
                g["name"],
                f"{g['value']:.3f}",
                f"[{g['lo']:.3f}, {g['hi']:.3f}]" if "lo" in g else "",
                g["limit"],
                "ok" if g["ok"] else "**over limit**",
            ]
            for g in r["guardrails"]
        ]
        parts.append(
            "### Guardrails\n\n" + _table(["guardrail", "value", "interval", "limit", ""], rows)
        )

    f = r["flips"]
    parts.append(
        f"### Flips\n\n{f['broke']} broke · {f['fixed']} fixed · {f['flaky']} flaky on base · "
        f"{f['stable_pass']} stable pass · {f['stable_fail']} stable fail"
    )
    for (title, key), limit in zip(
        (("Broke", "broke_examples"), ("Fixed", "fixed_examples")), shown, strict=True
    ):
        examples = f[key][:limit]
        if examples:
            # When the target reported a trace id, point at the stage-level diagnosis.
            traced = any(e.get("trace_id") for e in examples)
            listed = [
                [
                    _cell(e["text"]),
                    _cell(e["expected"]),
                    _cell(e["base_output"]),
                    _cell(e["head_output"]),
                    *([_trace(e.get("trace_id"), trace_url)] if traced else []),
                ]
                for e in examples
            ]
            header = ["input", "expected", "base said", "head said", *(["trace"] if traced else [])]
            parts.append(
                f"**{title}** (top {len(listed)} of {f[key.split('_')[0]]})\n\n"
                + _table(header, listed)
            )
    return "\n\n".join(parts) + "\n"


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; --ink: #1c1c1a; --soft: #5b5a55; --line: #dddcd5;
  --wash: #f1f0eb; --page: #fcfcfb; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink: #f2f1ec; --soft: #b5b4ab; --line: #3a3a37; --wash: #2a2a28; --page: #1a1a19; }}
}}
body {{ font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink);
  background: var(--page); max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
h2 {{ font-size: 1.35rem; margin: 0 0 .5rem; }}
h3 {{ font-size: 1.05rem; margin: 1.75rem 0 .5rem; }}
p {{ margin: .5rem 0; }}
table {{ display: block; overflow-x: auto; border-collapse: collapse; margin: .5rem 0 1rem;
  font-size: 14px; }}
th, td {{ text-align: left; padding: .35rem .6rem; border-bottom: 1px solid var(--line);
  vertical-align: top; }}
th {{ color: var(--soft); font-weight: 600; white-space: nowrap; }}
code {{ background: var(--wash); padding: .1rem .3rem; border-radius: 3px; font-size: .92em; }}
a {{ color: inherit; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _inline(text: str) -> str:
    """Bold, code and links. Escaped first: cells quote model output, which is not trusted."""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', text)


def to_html(markdown: str, title: str) -> str:
    """One self-contained page from a report written by this module.

    It understands only what those reports use (headings, paragraphs, tables, bold, code,
    links), and carries its styles with it, so the file works on its own as a CI artifact.
    """
    body = []
    for block in markdown.strip().split("\n\n"):
        lines = block.split("\n")
        if lines[0].startswith("|"):
            cells = [
                [_inline(c.strip().replace("\\|", "|")) for c in re.split(r"(?<!\\)\|", line)[1:-1]]
                for line in lines
                if not set(line) <= set("|-")  # the rule under the header
            ]
            head = "".join(f"<th>{c}</th>" for c in cells[0])
            rest = "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells[1:]
            )
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rest}</tbody></table>")
        elif lines[0].startswith("#"):
            level = len(lines[0]) - len(lines[0].lstrip("#"))
            body.append(f"<h{level}>{_inline(lines[0].lstrip('# '))}</h{level}>")
            body += [f"<p>{_inline(line)}</p>" for line in lines[1:]]
        else:
            body.append("<p>" + "<br>".join(_inline(line) for line in lines) + "</p>")
    return PAGE.format(title=html.escape(title), body="\n".join(body))


def single(suite: str, fingerprint: str, s: dict[str, Any]) -> str:
    parts = [
        f"## `{suite}` · `{fingerprint}`",
        f"**{s['metric']} = {s['mean']:.3f}** [{s['lo']:.3f}, {s['hi']:.3f}] "
        f"on {s['scored']} of {s['cases']} cases",
        f"statuses {s['statuses']} · {s['output_tokens']} output tokens · "
        f"latency p50 {s['latency_p50_ms']:.0f} ms, p95 {s['latency_p95_ms']:.0f} ms",
    ]
    if s["saturated"]:
        parts.append(
            "Warning: the score is above 0.95, so this eval can barely tell changes apart."
        )
    if "variance" in s:
        v = s["variance"]
        advice = (
            "more repetitions help" if v["within_share"] > 0.5 else "add cases, not repetitions"
        )
        parts.append(
            f"variance: {v['between']:.4f} between cases, {v['within']:.4f} within "
            f"({v['within_share']:.0%} within; {advice})"
        )
    if s.get("robustness"):
        rows = [
            [
                x["kind"],
                x["n"],
                f"{x['delta']:+.3f}",
                f"[{x['lo']:+.3f}, {x['hi']:+.3f}]",
                f"{x['changed']:.1%}",
            ]
            for x in s["robustness"]
        ]
        parts.append(
            "### Robustness\n\nEach perturbed case against the original it was made from.\n\n"
            + _table(["perturbation", "pairs", "Δ score", "interval", "outcome changed"], rows)
        )
    if s["slices"]:
        rows = [[x["tag"], x["n"], f"{x['mean']:.3f}"] for x in s["slices"]]
        parts.append("### Slices\n\n" + _table(["slice", "n", "mean"], rows))
    if s["lowest"]:
        rows = [
            [_cell(x["text"]), _cell(x["expected"]), _cell(x["output"]), f"{x['score']:.2f}"]
            for x in s["lowest"]
        ]
        parts.append(
            "### Lowest-scoring cases\n\n" + _table(["input", "expected", "output", "score"], rows)
        )
    return "\n\n".join(parts) + "\n"
