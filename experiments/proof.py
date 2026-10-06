"""Measure the gate itself, from stored scores. Calls no model.

    python experiments/proof.py              # report and figure from reports/scores.json
    python experiments/proof.py --extract    # first refresh scores.json from tripwire.db

The zoo (zoo.py) is a set of deliberate edits to a prompt, each run once on every case of
a large reference set. A variant's effect on all of those cases is taken as the truth.
The gate is then replayed a thousand times per variant on random draws of cases, to count
how often it reaches each verdict when the truth is known.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import zoo
from scipy.stats import norm

from tripwire import datasets, stats, store
from tripwire.compare import MIN_SLICE
from tripwire.config import load_config
from tripwire.models import Case
from tripwire.scorers import primary

HERE = Path(__file__).parent
REPORTS = HERE / "reports"
ALPHA, MARGIN, ERROR_RATE_MAX = 0.05, 0.03, 0.02  # the gated suite's settings
DRAWS, SLICE_DRAWS, N_BOOT = 1_000, 300, 2_000
SIZES = {"banking": (250, 700), "policy_qa": (120, 240)}  # first stage, whole gate
VERDICTS = ("REGRESSED", "INCONCLUSIVE", "PASS", "IMPROVED", "INVALID")
# What a variant really did to the score, by its exact effect on every case.
BEYOND, INSIDE, NO_DROP = "beyond the margin", "inside the margin", "no drop"
IDENTICAL, INVALID = "identical scores", "invalid"
RERUN = zoo.Variant("rerun", zoo.HARMLESS, "The baseline again, with new sampling seeds")

Key = tuple[str, str, str]  # task, metric, variant


# --- from the database to a small file that can be committed ---------------------------


def marks(
    db: sqlite3.Connection, fingerprint: str, key: tuple[str, str, str], rep: int, cases: list[Case]
) -> str:
    """One character per case, in dataset order: 1 pass, 0 fail, - no score."""
    found = {
        r[0]: r[1]
        for r in db.execute(
            "SELECT case_hash, value FROM scores WHERE fingerprint=? AND scorer=? AND version=? "
            "AND metric=? AND rep=?",
            (fingerprint, *key, rep),
        )
    }
    return "".join(str(int(found[c.hash])) if c.hash in found else "-" for c in cases)


def extract() -> None:
    cfg = load_config(HERE / "zoo.toml")
    db = store.connect(cfg.db_path)
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True
    )
    out: dict[str, Any] = {
        "commit": head.stdout.strip(),
        "date": date.today().isoformat(),
        "tasks": {},
    }
    for task in zoo.tasks():
        cases = datasets.load(HERE.parent / task.dataset)
        version = datasets.version(cases)
        variants: dict[str, Any] = {}
        reference: dict[str, str] = {}
        for v in task.variants:
            suite = cfg.get_suite(v.name)
            run = db.execute(
                "SELECT fingerprint FROM runs WHERE suite=? AND dataset_version=? "
                "ORDER BY started_at DESC",
                (v.name, version),
            ).fetchone()
            fp = run["fingerprint"] if run else ""
            accounted = {
                (r["case_hash"], r["rep"])
                for table in ("samples", "errors")
                for r in db.execute(
                    f"SELECT case_hash, rep FROM {table} WHERE fingerprint=?", (fp,)
                )
            }
            if any((c.hash, rep) not in accounted for c in cases for rep in range(v.reps)):
                print(f"skipped {v.name}: not fully run yet")
                if v.name == task.baseline:
                    break  # nothing to compare the others with
                continue
            rows = {
                r["case_hash"]: r
                for r in db.execute(
                    "SELECT case_hash, status, output, prompt_tokens, output_tokens, latency_ms "
                    "FROM samples WHERE fingerprint=? AND rep=0",
                    (fp,),
                )
            }
            got = [rows[c.hash] for c in cases if c.hash in rows]
            outputs = {c.hash: rows[c.hash]["output"].strip() for c in cases if c.hash in rows}
            reference = reference or outputs
            spec = json.loads(
                db.execute("SELECT spec FROM targets WHERE fingerprint=?", (fp,)).fetchone()["spec"]
            )
            # (scorer, version, metric) under which each metric's scores are stored.
            keys = {
                m: primary(suite.model_copy(update={"primary_metric": m})) for m in task.metrics
            }
            variants[v.name] = {
                "fingerprint": fp,
                "model": spec["model"],
                "provider": spec["provider_kind"],
                "digest": spec["digest"],
                "seconds": round(float(np.mean([r["latency_ms"] for r in got])) / 1000, 3),
                "prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in got]))),
                "output_tokens": round(float(np.mean([r["output_tokens"] for r in got])), 2),
                "cut_off": sum(r["status"] == "truncated" for r in got) / len(cases),
                "same_output": sum(outputs.get(h) == o for h, o in reference.items()) / len(cases),
                "scores": {
                    m: [marks(db, fp, keys[m], rep, cases) for rep in range(v.reps)]
                    for m in task.metrics
                },
            }
            # Unscored answers are data (cut off, or a judge that gave no verdict), but a
            # judging pass that has not run yet looks the same from here: make it visible.
            for metric, reps in variants[v.name]["scores"].items():
                missing = sum(marked.count("-") for marked in reps)
                if missing:
                    print(f"note: {v.name} has {missing} unscored answers for {metric}")
        out["tasks"][task.key] = {
            "dataset": task.dataset,
            "version": version,
            "cases": len(cases),
            "variants": variants,
        }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "scores.json").write_text(json.dumps(out, indent=1) + "\n", "utf-8", newline="\n")


# --- the analysis: pure functions over score arrays ------------------------------------


def array(marks: str) -> np.ndarray:
    """One score per case; NaN where there is none (a cut-off answer, a failed request)."""
    return np.array([math.nan if m == "-" else float(m) for m in marks])


def truth(base: np.ndarray, head: np.ndarray) -> dict[str, Any]:
    """A variant's effect on every case, and what that makes it.

    The replay below draws cases from exactly these, so within it the effect is known
    without error and a variant can be sorted by it: a drop beyond the margin, a drop
    inside it, or no drop. The interval is what the same cases say about the task at large.
    """
    ok = ~np.isnan(base) & ~np.isnan(head)
    unscored = float(1 - ok.mean())
    if ok.sum() < 2:
        return {"kind": INVALID, "unscored": unscored, "delta": math.nan, "changed": math.nan}
    b, h = base[ok], head[ok]
    r = stats.paired(b, h, ALPHA)
    if unscored > ERROR_RATE_MAX:
        kind = INVALID
    elif (b == h).all():
        kind = IDENTICAL
    elif r.delta <= -MARGIN:
        kind = BEYOND
    else:
        kind = INSIDE if r.delta < 0 else NO_DROP
    return {
        "kind": kind,
        "unscored": unscored,
        "base": r.base,
        "head": r.head,
        "delta": r.delta,
        "lo": r.lo,
        "hi": r.hi,
        "changed": float((b != h).mean()),
    }


def decide(base: np.ndarray, head: np.ndarray, alpha: float, seed: int) -> str:
    """The gate's verdict on one set of cases: what compare() does, without the database."""
    ok = ~np.isnan(base) & ~np.isnan(head)
    if ok.sum() < 2 or 1 - ok.mean() > ERROR_RATE_MAX:
        return "INVALID"
    _, lo, hi = stats.mean_ci(head[ok] - base[ok], 1 - 2 * alpha, N_BOOT, seed)
    return stats.verdict(lo, hi, MARGIN)


def replay(
    base: np.ndarray, head: np.ndarray, first: int, full: int, draws: int = DRAWS
) -> dict[str, Any]:
    """Run the gate on `draws` random sets of cases: at two sizes, staged, and naively.

    Cases are drawn with replacement, so each draw is an independent sample from a
    population in which the variant's effect is exactly the one measured on all cases.
    Subsets drawn without replacement would each resemble the whole set too closely,
    which shrinks the spread between draws and flatters the gate.
    """
    rng = np.random.default_rng(0)  # every variant sees the same draws
    tally: dict[str, Counter[str]] = {d: Counter() for d in ("first", "full", "staged")}
    naive: Counter[str] = Counter()
    used = early = settled = 0
    for i in range(draws):
        pick = rng.integers(0, len(base), full)
        b, h = base[pick], head[pick]
        tally["first"][decide(b[:first], h[:first], ALPHA, i)] += 1
        tally["full"][decide(b, h, ALPHA, i)] += 1
        # Staged: half the error budget on the first cases, the other half on all of them.
        look = decide(b[:first], h[:first], ALPHA / 2, i)
        if look == "INCONCLUSIVE":
            look = decide(b, h, ALPHA / 2, i)
            used += full
            settled += look != "INCONCLUSIVE"
        else:
            used += first
            early += 1
        tally["staged"][look] += 1
        # Naive: compare two accuracies, with no interval. An unscored answer counts as wrong.
        for n in (first, full):
            drop = np.nan_to_num(b[:n]).mean() - np.nan_to_num(h[:n]).mean()
            naive[f"{n}:any"] += drop > 0
            naive[f"{n}:margin"] += drop > MARGIN
    out: dict[str, Any] = {
        design: {v: count[v] / draws for v in VERDICTS} for design, count in tally.items()
    }
    out["early"], out["cases"] = early / draws, used / draws
    out["settled"] = settled / (draws - early) if draws > early else math.nan
    out["naive"] = {key: count / draws for key, count in sorted(naive.items())}
    return out


def slice_alarms(
    base: np.ndarray,
    head: np.ndarray,
    tags: dict[str, np.ndarray],
    full: int,
    draws: int = SLICE_DRAWS,
) -> dict[str, Any]:
    """How often a slice flag blocks a change, by the gate's own slice test.

    "alone" counts the draws where the overall verdict would have let the change through.
    "worst" is the slice that really fell furthest on all cases: a flag is not a false
    alarm when a slice did get worse while the total held.
    """
    rng = np.random.default_rng(0)
    flagged = alone = 0
    for i in range(draws):
        pick = rng.integers(0, len(base), full)
        b, h = base[pick], head[pick]
        ok = ~np.isnan(b) & ~np.isnan(h)
        found = []
        for member in tags.values():
            idx = np.flatnonzero(member[pick] & ok)
            if len(idx) >= MIN_SLICE:
                found.append(stats.paired(b[idx], h[idx], ALPHA, n_boot=N_BOOT, seed=i))
        adjusted = stats.adjust([s.p for s in found])
        if any(p < ALPHA and s.delta < 0 for p, s in zip(adjusted, found, strict=True)):
            flagged += 1
            alone += decide(b, h, ALPHA, i) in ("PASS", "IMPROVED")
    falls = {tag: float(np.nanmean((head - base)[member])) for tag, member in tags.items()}
    worst = min(falls, key=lambda tag: falls[tag])
    return {"any": flagged / draws, "alone": alone / draws, "worst": [worst, falls[worst]]}


def expected(delta: float, changed: float, n: int) -> float:
    """Normal-approximation chance of REGRESSED for a true `delta` on n pass/fail cases."""
    se = math.sqrt(max(changed - delta**2, 1e-12) / n)
    z = norm.ppf(1 - ALPHA)
    return float(norm.cdf((min(-z * se, -MARGIN + z * se) - delta) / se))


def work(job: tuple[Key, np.ndarray, np.ndarray, int, int, dict[str, np.ndarray] | None]) -> Any:
    key, base, head, first, full, tags = job
    result = {"truth": truth(base, head), **replay(base, head, first, full)}
    if tags and result["truth"]["kind"] in (INSIDE, NO_DROP):
        result["slices"] = slice_alarms(base, head, tags, full)
    return key, result


def analyse(data: dict[str, Any]) -> dict[Key, Any]:
    jobs = []
    for task in zoo.tasks():
        stored = data["tasks"][task.key]["variants"]
        if task.baseline not in stored:
            continue
        first, full = SIZES[task.key]
        cases = datasets.load(HERE.parent / task.dataset)
        counts = Counter(tag for c in cases for tag in c.tags)
        tags = {
            tag: np.array([tag in c.tags for c in cases])
            for tag, n in sorted(counts.items())
            if n >= MIN_SLICE
        }
        for metric in task.metrics:
            base = [array(m) for m in stored[task.baseline]["scores"][metric]]
            pairs = {RERUN.name: (base[0], base[1])} if len(base) > 1 else {}
            for v in task.variants[1:]:
                if v.name in stored:
                    head = [array(m) for m in stored[v.name]["scores"][metric]]
                    pairs[v.name] = (base[0], head[0])
                    if len(head) > 1 and len(base) > 1:
                        pairs[v.name + " (repeat)"] = (base[1], head[1])
            sliced = tags if metric == task.metrics[0] else None
            jobs += [
                ((task.key, metric, n), b, h, first, full, sliced) for n, (b, h) in pairs.items()
            ]
    with ProcessPoolExecutor() as pool:
        return dict(pool.map(work, jobs))


# --- the report --------------------------------------------------------------------------


def signed(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:+.3f}".replace("-", "−")


def pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{100 * x:.0f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines) + "\n"


def passed(result: dict[str, Any], design: str) -> float:
    return float(result[design]["PASS"] + result[design]["IMPROVED"])


def shares(result: dict[str, Any], design: str) -> str:
    s = result[design]
    if s["INVALID"] == 1:
        return "`INVALID` 100%"
    text = f"{pct(s['REGRESSED'])} / {pct(s['INCONCLUSIVE'])} / {pct(passed(result, design))}"
    # A variant near the unscored limit is invalid in some draws only: say so, or the
    # three shares would quietly fail to add up.
    return text + (f", `INVALID` {pct(s['INVALID'])}" if s["INVALID"] else "")


def mean(values: Any) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def task_report(task: zoo.Task, data: dict[str, Any], results: dict[Key, Any]) -> str:
    stored = data["tasks"][task.key]
    first, full = SIZES[task.key]
    designs = (("first", f"{first} cases"), ("full", f"{full} cases"), ("staged", "staged"))
    meta = {v.name: v for v in [*task.variants, RERUN]}
    text = ""
    for metric in task.metrics:
        found = {n: r for (key, m, n), r in results.items() if key == task.key and m == metric}
        main = {n: r for n, r in found.items() if not n.endswith("(repeat)")}
        if not main:
            continue
        order = sorted(
            main, key=lambda n: (main[n]["truth"]["kind"] == INVALID, main[n]["truth"]["delta"])
        )

        def group(kind: str, main: dict[str, Any] = main) -> list[dict[str, Any]]:
            return [r for r in main.values() if r["truth"]["kind"] == kind]

        base_score = next(r["truth"]["base"] for r in main.values() if "base" in r["truth"])
        text += f"\n### `{metric}`\n\n"
        text += f"Baseline {base_score:.3f} on {stored['cases']:,} cases. "
        text += (
            "Δ is the variant's score minus the baseline's on all of them; the interval is "
            "what those cases say about the task at large.\n\n"
        )
        rows = []
        for name in order:
            t, v = main[name]["truth"], meta[name]
            scored = "base" in t
            rows.append(
                [
                    f"`{name}`",
                    v.what,
                    v.intent,
                    f"{t['head']:.3f}" if scored else "n/a",
                    signed(t["delta"]),
                    f"[{signed(t['lo'])}, {signed(t['hi'])}]" if scored else "n/a",
                    f"{100 * t['changed']:.1f}%" if scored else "n/a",
                    f"{100 * t['unscored']:.1f}%",
                    t["kind"],
                ]
            )
        text += table(
            [
                "Variant",
                "Edit",
                "Meant to be",
                "Score",
                "Δ",
                "90% interval",
                "Scores changed",
                "Unscored",
                "What it did",
            ],
            rows,
        )

        text += (
            f"\nVerdicts over {DRAWS:,} simulated gates per variant, as "
            "`REGRESSED` / `INCONCLUSIVE` / `PASS` or `IMPROVED`:\n\n"
        )
        rows = [
            [
                f"`{name}`",
                signed(main[name]["truth"]["delta"]),
                main[name]["truth"]["kind"],
                *(shares(main[name], design) for design, _ in designs),
                f"{main[name]['cases']:.0f}",
            ]
            for name in order
        ]
        text += table(
            [
                "Variant",
                "True Δ",
                "What it did",
                f"{first} cases",
                f"{full} cases",
                f"Staged {first} → {full}",
                "Staged: cases run",
            ],
            rows,
        )

        rows = []
        for kind, label, verdicts in (
            (NO_DROP, "A change with no drop is called `REGRESSED`", ("REGRESSED",)),
            (BEYOND, "A drop beyond the margin is let through", ("PASS", "IMPROVED")),
        ):
            if group(kind):
                rows.append(
                    [
                        label,
                        str(len(group(kind))),
                        *(
                            pct(max(sum(r[d][v] for v in verdicts) for r in group(kind)))
                            for d, _ in designs
                        ),
                    ]
                )
        if rows:
            text += (
                "\nThe two promises the gate makes, each checked on the variant that tests it "
                f"hardest (the limit for both is alpha, {ALPHA:.0%}):\n\n"
            )
            text += table(["Error", "Variants", *(f"Worst, {label}" for _, label in designs)], rows)

        text += "\nOn average, by what the change really did (each variant counts once):\n\n"
        rows = []
        for kind in (BEYOND, INSIDE, NO_DROP, IDENTICAL, INVALID):
            for design, label in designs:
                if group(kind):
                    rows.append(
                        [
                            f"{kind} ({len(group(kind))})",
                            label,
                            pct(mean(r[design]["REGRESSED"] for r in group(kind))),
                            pct(mean(r[design]["INCONCLUSIVE"] for r in group(kind))),
                            pct(mean(passed(r, design) for r in group(kind))),
                            pct(mean(r[design]["INVALID"] for r in group(kind))),
                        ]
                    )
        text += table(
            [
                "What it did (variants)",
                "Gate",
                "`REGRESSED`",
                "`INCONCLUSIVE`",
                "`PASS` or `IMPROVED`",
                "`INVALID`",
            ],
            rows,
        )

        better = [n for n in order if main[n]["truth"].get("lo", -1) > 0]
        if better:
            text += "\nVariants whose interval is wholly above zero, and how often the gate said so:\n\n"
            text += table(
                ["Variant", "True Δ", f"`IMPROVED` at {first}", f"`IMPROVED` at {full}"],
                [
                    [
                        f"`{n}`",
                        signed(main[n]["truth"]["delta"]),
                        pct(main[n]["first"]["IMPROVED"]),
                        pct(main[n]["full"]["IMPROVED"]),
                    ]
                    for n in better
                ],
            )

        if metric != task.metrics[0]:
            continue

        text += "\n#### Cost of a decision\n\n"
        text += (
            f"The staged gate looks at {first} cases with half the error budget and runs all "
            f"{full} only when those cannot decide. Model time is the candidate's cases at its "
            "measured seconds per case; the baseline side is normally already cached.\n\n"
        )
        rows = []
        for kind in (BEYOND, INSIDE, NO_DROP):
            names = [n for n in order if main[n]["truth"]["kind"] == kind]
            if names:
                fallback = stored["variants"][task.baseline]
                seconds = mean(stored["variants"].get(n, fallback)["seconds"] for n in names)
                cases = mean(main[n]["cases"] for n in names)
                rows.append(
                    [
                        f"{kind} ({len(names)})",
                        pct(mean(main[n]["early"] for n in names)),
                        f"{cases:.0f} of {full} ({pct(1 - cases / full)} fewer)",
                        f"{cases * seconds / 60:.1f} min, against {full * seconds / 60:.1f} min",
                        pct(float(np.nanmean([main[n]["settled"] for n in names]))),
                        f"{pct(mean(main[n]['staged']['REGRESSED'] for n in names))}, against "
                        f"{pct(mean(main[n]['full']['REGRESSED'] for n in names))}",
                    ]
                )
        text += table(
            [
                "What it did (variants)",
                "Decided at the first look",
                "Cases run on average",
                "Model time per decision",
                "Undecided first looks settled by the second",
                f"`REGRESSED`, staged against one look at {full}",
            ],
            rows,
        )

        text += "\n#### Against a plain threshold\n\n"
        text += (
            "The same draws, judged by comparing the two accuracies with no interval: block on "
            f"any drop at all, or block on a drop of more than the margin ({MARGIN}).\n\n"
        )
        rows = []
        for kind in (BEYOND, INSIDE, NO_DROP):
            for n, design in ((first, "first"), (full, "full")):
                if group(kind):
                    rows.append(
                        [
                            f"{kind} ({len(group(kind))})",
                            str(n),
                            pct(mean(r["naive"][f"{n}:any"] for r in group(kind))),
                            pct(mean(r["naive"][f"{n}:margin"] for r in group(kind))),
                            pct(mean(r[design]["REGRESSED"] for r in group(kind))),
                            pct(mean(1 - passed(r, design) for r in group(kind))),
                        ]
                    )
        text += table(
            [
                "What it did (variants)",
                "Cases",
                "Threshold: any drop",
                "Threshold: drop over the margin",
                "Tripwire: `REGRESSED`",
                "Tripwire: not passed",
            ],
            rows,
        )

        sliced = {n: main[n]["slices"] for n in order if "slices" in main[n]}
        if sliced:
            text += "\n#### Slice flags\n\n"
            text += (
                "The gate also blocks when one slice regresses significantly. For the variants "
                f"that did not fall beyond the margin, over {SLICE_DRAWS} draws of {full} cases "
                "with the gate's own slice test:\n\n"
            )
            rows = [
                [
                    f"`{n}`",
                    signed(main[n]["truth"]["delta"]),
                    pct(s["any"]),
                    pct(s["alone"]),
                    f"`{s['worst'][0]}` {signed(s['worst'][1])}",
                ]
                for n, s in sliced.items()
            ]
            rows.append(
                [
                    "**mean**",
                    "",
                    pct(mean(s["any"] for s in sliced.values())),
                    pct(mean(s["alone"] for s in sliced.values())),
                    "",
                ]
            )
            text += table(
                [
                    "Variant",
                    "True Δ",
                    "A slice was flagged",
                    "Flagged while the overall verdict passed",
                    "Slice that really fell furthest, on all cases",
                ],
                rows,
            )

        repeats = [n for n in found if n.endswith("(repeat)")]
        if repeats:
            text += "\n#### A second repetition\n\n"
            text += (
                "Everything above uses one stored answer per case. These variants and the "
                "baseline were generated a second time with different seeds:\n\n"
            )
            rows = []
            for name in repeats:
                once, twice = main[name.removesuffix(" (repeat)")], found[name]
                rows.append(
                    [
                        f"`{name.removesuffix(' (repeat)')}`",
                        f"{signed(once['truth']['delta'])} ({once['truth']['kind']})",
                        f"{signed(twice['truth']['delta'])} ({twice['truth']['kind']})",
                        shares(once, "full"),
                        shares(twice, "full"),
                    ]
                )
            text += table(
                [
                    "Variant",
                    "Δ, first repetition",
                    "Δ, second repetition",
                    f"Verdicts at {full}, first",
                    f"Verdicts at {full}, second",
                ],
                rows,
            )
    return text


def figure(results: dict[Key, Any], data: dict[str, Any]) -> str:
    """The dose-response curve as plain SVG, so regenerating it needs no plotting library."""
    task = zoo.tasks()[0]
    first, full = SIZES[task.key]
    points = [
        (
            name,
            r["truth"]["delta"],
            r["truth"]["changed"],
            r["first"]["REGRESSED"],
            r["full"]["REGRESSED"],
        )
        for (key, metric, name), r in results.items()
        if key == task.key
        and metric == task.metrics[0]
        and r["truth"]["kind"] in (BEYOND, INSIDE, NO_DROP)
        and not name.endswith("(repeat)")
    ]
    x0, x1 = -0.12, 0.06
    shown = [p for p in points if x0 <= p[1] <= x1]
    beyond = [p for p in points if p[1] < x0]
    typical = float(np.median([p[2] for p in shown])) if shown else 0.1
    left, right, top, bottom = 64, 736, 104, 380
    ink, soft, muted, grid, axis, surface = (
        "#0b0b0b",
        "#52514e",
        "#898781",
        "#e1e0d9",
        "#c3c2b7",
        "#fcfcfb",
    )
    series = ((first, "#2a78d6"), (full, "#eb6834"))

    def x(delta: float) -> float:
        return left + (delta - x0) / (x1 - x0) * (right - left)

    def y(share: float) -> float:
        return bottom - share * (bottom - top)

    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 460" role="img" '
        'font-family="system-ui, -apple-system, \'Segoe UI\', sans-serif" font-size="11">',
        "<title>How often the gate returns REGRESSED, by the true size of the change</title>",
        f'<rect width="760" height="460" fill="{surface}"/>',
        f'<text x="{left}" y="28" font-size="15" font-weight="600" fill="{ink}">How often the gate says REGRESSED, by the true size of the change</text>',
        f'<text x="{left}" y="46" font-size="12" fill="{soft}">Each dot is one deliberate change to the Banking77 prompt: {DRAWS:,} simulated gates on random draws of its scored cases.</text>',
    ]
    for share in (0, 0.25, 0.5, 0.75, 1):
        stroke = axis if share == 0 else grid
        out.append(
            f'<line x1="{left}" x2="{right}" y1="{y(share):.1f}" y2="{y(share):.1f}" stroke="{stroke}"/>'
        )
        out.append(
            f'<text x="{left - 8}" y="{y(share) + 4:.1f}" text-anchor="end" fill="{muted}">{share:.0%}</text>'
        )
    for tick in np.arange(x0, x1 + 1e-9, 0.02):
        out.append(
            f'<text x="{x(tick):.1f}" y="{bottom + 18}" text-anchor="middle" fill="{muted}">{signed(tick)[:-1]}</text>'
        )
    for at, label in ((-MARGIN, "margin"), (0.0, "no change")):
        out.append(
            f'<line x1="{x(at):.1f}" x2="{x(at):.1f}" y1="{top}" y2="{bottom}" stroke="{axis}"/>'
        )
        out.append(
            f'<text x="{x(at):.1f}" y="{top - 8}" text-anchor="middle" fill="{soft}">{label}</text>'
        )
    for n, colour in series:
        curve = " ".join(
            f"{x(d):.1f},{y(expected(d, typical, n)):.1f}" for d in np.linspace(x0, x1, 91)
        )
        out.append(
            f'<polyline points="{curve}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    for name, delta, _, *hit in shown:
        for (n, colour), share in zip(series, hit, strict=True):
            out.append(
                f'<circle cx="{x(delta):.1f}" cy="{y(share):.1f}" r="5" fill="{colour}" stroke="{surface}" stroke-width="2">'
                f"<title>{name}: true change {signed(delta)}, REGRESSED in {share:.0%} of gates on {n} cases</title></circle>"
            )
    for row, (n, colour) in enumerate(series):
        cx = left + 6 + 96 * row
        out.append(
            f'<circle cx="{cx}" cy="68" r="5" fill="{colour}" stroke="{surface}" stroke-width="2"/>'
        )
        out.append(f'<text x="{cx + 12}" y="72" fill="{ink}">{n} cases</text>')
    out.append(
        f'<line x1="{left + 200}" x2="{left + 224}" y1="68" y2="68" stroke="{soft}" stroke-width="2"/>'
    )
    out.append(
        f'<text x="{left + 232}" y="72" fill="{soft}">expected by the normal approximation when {typical:.0%} of scores change (the median here)</text>'
    )
    cases = data["tasks"][task.key]["cases"]
    out.append(
        f'<text x="{(left + right) / 2}" y="{bottom + 40}" text-anchor="middle" font-size="12" fill="{soft}">True change in accuracy, measured on all {cases:,} reference cases</text>'
    )
    if beyond:
        caught = min(min(p[3], p[4]) for p in beyond)
        often = "every gate" if caught == 1 else f"at least {caught:.0%} of gates"
        out.append(
            f'<text x="{left}" y="446" fill="{muted}">Not shown: {len(beyond)} larger drops, down to {signed(min(p[1] for p in beyond))}, '
            f"each called REGRESSED in {often} at both sizes.</text>"
        )
    return "\n".join(out) + "\n</svg>\n"


def report(data: dict[str, Any], results: dict[Key, Any]) -> str:
    # Ollama identifies a model by digest; the OpenAI-compatible API only by where it is.
    models = sorted(
        {
            (
                v["model"],
                f"digest {v['digest'][:12]}"
                if v["provider"] == "ollama"
                else f"no digest; served at {v['provider'].split(' ', 1)[-1]}",
            )
            for t in data["tasks"].values()
            for v in t["variants"].values()
        }
    )
    text = (
        "# The gate, measured\n\n"
        "Generated by `python experiments/proof.py` from `experiments/reports/scores.json`. "
        "Do not edit by hand. The method and its limits are in "
        "[docs/benchmark.md](../../docs/benchmark.md).\n\n"
        f"- Scores extracted on {data['date']} at commit `{data['commit']}`.\n"
    )
    for t in data["tasks"].values():
        text += f"- `{t['dataset']}`: {t['cases']:,} cases, version `{t['version']}`, {len(t['variants']) - 1} variants.\n"
    text += "- Models: " + "; ".join(f"`{model}` ({digest})" for model, digest in models) + ".\n"
    text += (
        f"- Gate settings: margin {MARGIN}, alpha {ALPHA}, at most {ERROR_RATE_MAX:.0%} of cases "
        f"unscored. {DRAWS:,} simulated gates per variant, {N_BOOT:,} bootstrap resamples each.\n\n"
        "![How often the gate says REGRESSED, by the true size of the change](dose_response.svg)\n"
    )
    titles = {
        "banking": "Banking77 intent classification",
        "policy_qa": "Policy question answering, scored by an LLM judge",
    }
    for task in zoo.tasks():
        if task.key in data["tasks"] and task.baseline in data["tasks"][task.key]["variants"]:
            text += f"\n## {titles[task.key]}\n" + task_report(task, data, results)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--extract", action="store_true", help="refresh scores.json first")
    if parser.parse_args().extract:
        extract()
    stored = json.loads((REPORTS / "scores.json").read_text(encoding="utf-8"))
    measured = analyse(stored)
    (REPORTS / "proof.md").write_text(report(stored, measured), encoding="utf-8", newline="\n")
    (REPORTS / "dose_response.svg").write_text(
        figure(measured, stored), encoding="utf-8", newline="\n"
    )
    print(f"wrote {REPORTS / 'proof.md'} and {REPORTS / 'dose_response.svg'}")
