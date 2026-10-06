"""A dashboard over a Tripwire database. Read-only, except for the labelling page.

    tripwire dashboard                                  # the project in this directory
    tripwire dashboard -c experiments/zoo.toml --read-only

It needs the optional dependency: pip install "tripwire-eval[dashboard]". Every number
shown comes from the same functions the command line uses; nothing is computed twice in
two different ways.
"""

from __future__ import annotations

import difflib
import json
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import streamlit as st

from tripwire import datasets, report, stats, store
from tripwire.compare import _brief, blocked, case_scores, compare, summarise
from tripwire.config import Config, load_config
from tripwire.judge import Judge, against_rule, calibration
from tripwire.models import Case
from tripwire.runner import now
from tripwire.scorers import JUDGE

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"


def settings() -> tuple[Path, bool]:
    """Where the project is, and whether labelling is switched off (a public demo)."""
    args = sys.argv[1:]
    given = next((a for a in args if a.endswith(".toml")), "tripwire.toml")
    read_only = "--read-only" in args or os.environ.get("TRIPWIRE_READ_ONLY") == "1"
    return Path(os.environ.get("TRIPWIRE_CONFIG") or given), read_only


@st.cache_resource
def _open(config: str) -> tuple[Config, sqlite3.Connection]:
    cfg = load_config(Path(config))
    uri = cfg.db_path.resolve().as_uri() + "?mode=ro"  # the dashboard never writes through this
    db = sqlite3.connect(uri, uri=True, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return cfg, db


def project() -> tuple[Config, sqlite3.Connection]:
    return _open(str(settings()[0].resolve()))


@st.cache_resource
def cases_of(suite: str) -> list[Case]:
    cfg, _ = project()
    return datasets.load(cfg.root / cfg.get_suite(suite).dataset)


@st.cache_data(ttl=60)
def configurations() -> list[dict[str, Any]]:
    """Every target that has run under a suite of this project, oldest first."""
    cfg, db = project()
    found = db.execute(
        "SELECT suite, fingerprint, min(started_at) AS first, max(git_sha) AS git_sha "
        "FROM runs GROUP BY suite, fingerprint ORDER BY first"
    )
    rows = [dict(r) for r in found if r["suite"] in cfg.suite]
    # When a target first ran at all, under any name: the oldest one is the natural baseline.
    born = dict(db.execute("SELECT fingerprint, min(started_at) FROM runs GROUP BY fingerprint"))
    for r in rows:
        r["label"] = f"{r['suite']} · {r['fingerprint'][:8]}"
        r["dataset"] = cfg.get_suite(r["suite"]).dataset
        r["born"] = born[r["fingerprint"]]
    return rows


@st.cache_data(show_spinner="Scoring")
def summary(suite: str, fingerprint: str) -> dict[str, Any] | None:
    cfg, db = project()
    try:
        return summarise(db, cfg.get_suite(suite), cases_of(suite), fingerprint)
    except ValueError:  # nothing scored under the suite's current scorers
        return None


@st.cache_data(show_spinner="Comparing")
def comparison(base: str, suite: str, head: str) -> dict[str, Any]:
    cfg, db = project()
    return compare(db, cfg.get_suite(suite), cases_of(suite), base, head)


def table(rows: list[dict[str, Any]], **options: Any) -> Any:
    return st.dataframe(rows, hide_index=True, width="stretch", **options)


def pick(label: str, options: list[dict[str, Any]], key: str) -> dict[str, Any]:
    chosen = st.sidebar.selectbox(label, [o["label"] for o in options], key=key)
    return next(o for o in options if o["label"] == chosen)


def pick_pair() -> tuple[dict[str, Any], dict[str, Any]] | None:
    """A candidate and a baseline that ran on the same dataset."""
    found = configurations()
    head = pick("Candidate", found[::-1], "head")
    others = [c for c in found if c["dataset"] == head["dataset"] and c is not head]
    if not others:
        st.info("Nothing else has run on this dataset, so there is nothing to compare with.")
        return None
    return pick("Baseline", sorted(others, key=lambda c: c["born"]), "base"), head


# --- pages -------------------------------------------------------------------------------


def history() -> None:
    """The primary metric of every target a suite has had, in the order they first ran."""
    found = configurations()
    suite = st.sidebar.selectbox("Suite", sorted({c["suite"] for c in found}))
    rows = []
    for c in (c for c in found if c["suite"] == suite):
        s = summary(suite, c["fingerprint"])
        if s:
            rows.append(
                {
                    "target": c["fingerprint"][:8],
                    "first run": c["first"][:16].replace("T", " "),
                    "commit": (c["git_sha"] or "")[:10],
                    "score": round(s["mean"], 4),
                    "low": round(s["lo"], 4),
                    "high": round(s["hi"], 4),
                    "cases": s["scored"],
                }
            )
    if not rows:
        st.info("This suite has no scored samples yet.")
        return
    cfg, _ = project()
    metric = cfg.get_suite(suite).primary_metric
    st.subheader(f"{suite}: {metric}")
    st.caption("One point per target the suite has had, with its 90% interval.")
    base = alt.Chart(alt.InlineData(values=rows)).encode(
        x=alt.X(
            "target:N",
            sort=None,
            title="target, in order of first run",
            axis=alt.Axis(labelAngle=0),
        )
    )
    interval = base.mark_rule(color=BLUE, strokeWidth=2).encode(
        y=alt.Y("low:Q", title=metric, scale=alt.Scale(zero=False)), y2="high:Q"
    )
    point = base.mark_point(color=BLUE, filled=True, size=90, opacity=1).encode(
        y="score:Q", tooltip=["target:N", "first run:N", "commit:N", "score:Q", "low:Q", "high:Q"]
    )
    st.altair_chart(interval + point, width="stretch")
    table(rows)


def compare_page() -> None:
    pair = pick_pair()
    if pair is None:
        return
    base, head = pair
    cfg, _ = project()
    suite = cfg.get_suite(head["suite"])
    result = comparison(base["fingerprint"], head["suite"], head["fingerprint"])
    is_blocked = blocked(result, suite)
    if "delta" in result:
        tiles = st.columns(4)
        tiles[0].metric("Verdict", result["verdict"])
        tiles[1].metric("Baseline", f"{result['base']:.3f}")
        tiles[2].metric("Candidate", f"{result['head']:.3f}", f"{result['delta']:+.3f}")
        tiles[3].metric("Interval", f"[{result['lo']:+.3f}, {result['hi']:+.3f}]")
    label = f"{base['label']} → {head['label']}"
    st.markdown(report.comparison(label, result, is_blocked, cfg.tracelens_url))


@st.cache_data(show_spinner="Reading answers")
def flip_rows(base: str, suite: str, head: str) -> list[dict[str, Any]]:
    """Every case scored on both sides, with what each side said."""
    cfg, db = project()
    conf = cfg.get_suite(suite)
    scores = [case_scores(db, conf, fp) for fp in (base, head)]
    said = [
        {
            r["case_hash"]: r
            for r in db.execute(
                "SELECT case_hash, output, response FROM samples WHERE fingerprint=? AND rep=0",
                (fp,),
            )
        }
        for fp in (base, head)
    ]
    rows = []
    for case in cases_of(suite):
        if all(case.hash in found for found in (*scores, *said)):
            before, after = (float(np.mean(s[case.hash])) for s in scores)
            text, expected = _brief(case)
            rows.append(
                {
                    "change": "broke" if after < before else "fixed" if after > before else "same",
                    "input": text,
                    "expected": str(expected),
                    "baseline said": said[0][case.hash]["output"].strip(),
                    "candidate said": said[1][case.hash]["output"].strip(),
                    "baseline": before,
                    "candidate": after,
                    "tags": ", ".join(case.tags),
                    "trace": json.loads(said[1][case.hash]["response"] or "{}").get("trace_id"),
                }
            )
    return rows


def flips() -> None:
    pair = pick_pair()
    if pair is None:
        return
    base, head = pair
    rows = flip_rows(base["fingerprint"], head["suite"], head["fingerprint"])
    counts = {kind: sum(r["change"] == kind for r in rows) for kind in ("broke", "fixed", "same")}
    st.subheader(f"{base['label']} → {head['label']}")
    left, right = st.columns([1, 2])
    kind = left.radio(
        "Show", list(counts), horizontal=True, format_func=lambda k: f"{k} ({counts[k]})"
    )
    wanted = right.text_input("Filter by text or tag").lower()
    shown = [
        r
        for r in rows
        if r["change"] == kind and wanted in f"{r['input']} {r['tags']} {r['expected']}".lower()
    ]
    columns = ["input", "expected", "baseline said", "candidate said", "tags"]
    chosen = table(
        [{k: r[k] for k in columns} for r in shown],
        on_select="rerun",
        selection_mode="single-row",
        key="flip-table",
    )
    picked = chosen.selection.rows if chosen else []
    if not picked:
        st.caption("Select a row to see the two answers side by side.")
        return
    row = shown[picked[0]]
    st.markdown(f"**Input:** {row['input']}  \n**Expected:** `{row['expected']}`")
    one, two = st.columns(2)
    one.caption(f"baseline, score {row['baseline']:.2f}")
    one.code(row["baseline said"] or "(empty)", language=None)
    two.caption(f"candidate, score {row['candidate']:.2f}")
    two.code(row["candidate said"] or "(empty)", language=None)
    diff = difflib.unified_diff(
        row["baseline said"].splitlines(),
        row["candidate said"].splitlines(),
        "baseline",
        "candidate",
        lineterm="",
    )
    st.code("\n".join(diff) or "(identical text)", language="diff")
    cfg, _ = project()
    if row["trace"] and cfg.tracelens_url:
        st.markdown(f"[Open this trace in TraceLens]({cfg.tracelens_url}/traces/{row['trace']})")


def speed_and_quality() -> None:
    """Which target is the fastest that still holds quality?"""
    found = configurations()
    dataset = st.sidebar.selectbox("Dataset", sorted({c["dataset"] for c in found}))
    rows = []
    for c in (c for c in found if c["dataset"] == dataset):
        s = summary(c["suite"], c["fingerprint"])
        # A target that has only been tried on a few cases is not a point on this chart.
        if s and s["scored"] >= 0.95 * s["cases"]:
            rows.append(
                {
                    "target": c["label"],
                    "score": round(s["mean"], 4),
                    "seconds per case": round(s["latency_p50_ms"] / 1000, 3),
                    "output tokens per case": round(s["output_tokens"] / max(1, s["scored"]), 1),
                }
            )
    if not rows:
        st.info("No target has been scored on the whole of this dataset yet.")
        return
    for r in rows:  # on the frontier: nothing else is both faster and better
        dominated = any(
            o["seconds per case"] <= r["seconds per case"]
            and o["score"] >= r["score"]
            and (o["seconds per case"], o["score"]) != (r["seconds per case"], r["score"])
            for o in rows
        )
        r["frontier"] = "no" if dominated else "yes"
    st.subheader("Speed and quality")
    st.caption(
        "Median seconds per case against the primary metric. Blue targets are on the "
        "frontier: no other target is both faster and better. Latency recorded on "
        "different days is only roughly comparable."
    )
    chart = (
        alt.Chart(alt.InlineData(values=rows))
        .mark_point(filled=True, size=110, opacity=1, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("seconds per case:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("score:Q", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "frontier:N",
                scale=alt.Scale(domain=["yes", "no"], range=[BLUE, MUTED]),
                legend=alt.Legend(title="on the frontier"),
            ),
            tooltip=["target:N", "score:Q", "seconds per case:Q", "output tokens per case:Q"],
        )
    )
    st.altair_chart(chart, width="stretch")
    table(sorted(rows, key=lambda r: (r["frontier"] == "no", r["seconds per case"])))


def judge_page() -> None:
    cfg, db = project()
    judged = [c for c in configurations() if JUDGE in cfg.get_suite(c["suite"]).scorers]
    if not judged or cfg.judge is None:
        st.info("No suite in this project is scored by a judge.")
        return
    chosen = pick("Judged target", judged[::-1], "judged")
    suite, fingerprint = chosen["suite"], chosen["fingerprint"]
    st.subheader(f"Judge `{cfg.judge.model}` on {chosen['label']}")
    rows = []
    for criterion, found in calibration(cfg, suite, fingerprint, db).items():
        row = {"criterion": criterion, "judge mean": found["judge_mean"], "labels": found["n"]}
        if found["n"] >= 2:
            row["agreement with labels"] = round(found["agreement"], 3)
            row["kappa"] = (
                f"{found['kappa']:.2f} [{found['kappa_lo']:.2f}, {found['kappa_hi']:.2f}]"
            )
        if "human_estimate" in found:
            row["human-equivalent score"] = (
                f"{found['human_estimate']:.3f} [{found['human_lo']:.3f}, {found['human_hi']:.3f}]"
            )
        rows.append(row)
    st.caption("Against blind human labels (`tripwire label`, or the Label page).")
    table(rows)
    if "fact" not in cfg.get_suite(suite).scorers:
        return
    criterion = st.selectbox("Criterion the rule also measures", list(Judge(cfg).criteria))
    versus = against_rule(db, cfg, suite, fingerprint, criterion, "fact.pass")
    if versus.get("n", 0) < 2:
        st.info("No verdicts to compare with the rule yet.")
        return
    st.caption("Against the rule-based check of the same thing (`fact.pass`).")
    tiles = st.columns(3)
    tiles[0].metric("Answers", versus["n"])
    tiles[1].metric("Agreement", f"{versus['agreement']:.1%}")
    tiles[2].metric("Kappa", f"{versus['kappa']:.2f}")
    disagreed = versus["disagreements"]
    if not disagreed:
        return
    cases = {c.hash: c for c in cases_of(suite)}
    case_hash = st.selectbox(
        f"Disagreements ({len(disagreed)})",
        disagreed,
        format_func=lambda h: _brief(cases[h])[0][:90],
    )
    answer = db.execute(
        "SELECT output FROM samples WHERE fingerprint=? AND case_hash=? AND rep=0",
        (fingerprint, case_hash),
    ).fetchone()
    verdict = db.execute(
        "SELECT value, detail FROM scores WHERE fingerprint=? AND case_hash=? AND rep=0 "
        "AND scorer='judge' AND metric=? AND version=?",
        (fingerprint, case_hash, criterion, Judge(cfg).versions[criterion]),
    ).fetchone()
    if answer is None or verdict is None:
        st.info("The disagreement is on a later repetition of this case.")
        return
    detail = json.loads(verdict["detail"] or "{}")
    for key, value in cases[case_hash].input.items():
        st.markdown(f"**{key}**")
        st.text(str(value))
    st.markdown(f"**reference:** {_brief(cases[case_hash])[1]}")
    st.markdown("**answer**")
    st.code(answer["output"].strip(), language=None)
    said = "passes" if verdict["value"] else "fails"
    st.markdown(f"The judge says it **{said}**; the rule says the opposite.")
    st.markdown(
        f"*Evidence:* {detail.get('evidence', '')}  \n*Reasoning:* {detail.get('reasoning', '')}"
    )


def label_page() -> None:
    """Blind labelling: no model name and no judge verdict are shown."""
    cfg, db = project()
    judged = [c for c in configurations() if JUDGE in cfg.get_suite(c["suite"]).scorers]
    if settings()[1] or not judged or cfg.judge is None:
        st.info("Labelling is switched off here. It works on a local copy of the project.")
        return
    chosen = pick("Target to label", judged[::-1], "labelled")
    suite, fingerprint = chosen["suite"], chosen["fingerprint"]
    labeller = st.sidebar.text_input("Your name", "me")
    judge = Judge(cfg)
    cases = {c.hash: c for c in cases_of(suite)}
    rows = db.execute(
        "SELECT case_hash, rep, output FROM samples WHERE fingerprint=? AND status='ok' "
        "AND rep<? ORDER BY case_hash, rep",
        (fingerprint, cfg.get_suite(suite).reps),
    ).fetchall()
    rows = [r for r in rows if r["case_hash"] in cases]
    random.Random(0).shuffle(rows)  # the same random order as `tripwire label`
    done = {
        tuple(r)
        for r in db.execute(
            "SELECT case_hash, rep, criterion FROM human_labels WHERE fingerprint=? AND labeller=?",
            (fingerprint, labeller),
        )
    }
    skipped = st.session_state.setdefault("skipped", set())
    taken = done | skipped
    todo = [
        (r, criterion)
        for r in rows
        for criterion in judge.criteria
        if (r["case_hash"], r["rep"], criterion) not in taken
    ]
    st.subheader(f"Labelling {chosen['label']}")
    st.caption(f"{len(done)} labels stored for {labeller}; {len(todo)} to go.")
    if not todo:
        st.success("Nothing left to label. `tripwire judge calibrate` compares the judge with you.")
        return
    row, criterion = todo[0]
    case = cases[row["case_hash"]]
    for name, value in case.input.items():
        st.markdown(f"**{name}**")
        st.text(str(value))
    st.markdown(f"**reference:** {_brief(case)[1]}")
    st.markdown("**answer**")
    st.code(row["output"].strip(), language=None)
    st.markdown(f"**{criterion}:** {judge.criteria[criterion]}")
    yes, no, skip = st.columns(3)
    answer = 1 if yes.button("Yes", type="primary") else 0 if no.button("No") else None
    if answer is not None:
        entry = {
            "fingerprint": fingerprint,
            "case_hash": row["case_hash"],
            "rep": row["rep"],
            "criterion": criterion,
            "value": answer,
            "labeller": labeller,
            "created_at": now(),
        }
        writer = store.connect(cfg.db_path)
        store.insert(writer, "human_labels", [entry], "OR REPLACE")
        writer.close()
        st.rerun()
    if skip.button("Skip"):
        skipped.add((row["case_hash"], row["rep"], criterion))
        st.rerun()


def power() -> None:
    """How many cases a gate needs, from a suite's real pass and fail outcomes."""
    cfg, db = project()
    oldest_first = sorted(configurations(), key=lambda c: c["born"])
    chosen = pick("Baseline outcomes from", oldest_first, "power")
    conf = cfg.get_suite(chosen["suite"])
    scores = case_scores(db, conf, chosen["fingerprint"])
    outcomes = np.array([float(np.mean(v) >= 0.5) for v in scores.values()])
    if not len(outcomes) or not 0 < outcomes.mean() < 1:
        st.info("This target has no mix of passes and failures to resample from.")
        return
    left, right = st.columns(2)
    drop = left.slider("True drop to catch", 0.01, 0.15, 0.03, 0.01)
    discordant = left.slider("Share of cases on which two runs disagree", 0.02, 0.40, 0.12, 0.01)
    margin = right.slider("Margin: the drop you tolerate", 0.0, 0.10, conf.margin, 0.005)
    alpha = right.slider("Alpha: the error rate of each bound", 0.01, 0.10, conf.alpha, 0.01)
    needed = stats.sample_size(drop, discordant + drop, alpha)
    st.metric(f"Cases for 80% power against a drop of {drop:.2f}", f"{needed:,}")
    rows = []
    for n in (100, 250, 500, 700, 1000, 1500, 2500):
        hit = stats.simulate(outcomes, drop, n, discordant, alpha, margin, sims=400)
        null = stats.simulate(outcomes, 0.0, n, discordant, alpha, margin, sims=400)
        rows.append({"cases": n, "share": hit["REGRESSED"], "outcome": "a real drop is caught"})
        passed = null["PASS"] + null["IMPROVED"]
        rows.append({"cases": n, "share": passed, "outcome": "a harmless change passes"})
        rows.append(
            {"cases": n, "share": null["REGRESSED"], "outcome": "a harmless change is blocked"}
        )
    order = ["a real drop is caught", "a harmless change passes", "a harmless change is blocked"]
    chart = (
        alt.Chart(alt.InlineData(values=rows))
        .mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2)
        .encode(
            x=alt.X("cases:Q", title="cases in the gate"),
            y=alt.Y("share:Q", title="share of simulated gates", axis=alt.Axis(format="%")),
            color=alt.Color(
                "outcome:N",
                sort=order,
                scale=alt.Scale(domain=order, range=[BLUE, ORANGE, MUTED]),
                legend=alt.Legend(title=None, orient="bottom", labelLimit=320),
            ),
            tooltip=["outcome:N", "cases:Q", alt.Tooltip("share:Q", format=".1%")],
        )
    )
    st.altair_chart(chart, width="stretch")
    st.caption(f"400 simulated gates per point, resampled from {chosen['label']}.")


def drift() -> None:
    """Canary runs: a fixed subset rerun afresh and compared with a pinned reference."""
    _, db = project()
    found = db.execute(
        "SELECT suite, verdict, result, created_at FROM comparisons WHERE suite LIKE 'canary:%' "
        "AND verdict != 'PINNED' ORDER BY id"
    ).fetchall()
    if not found:
        st.info("No canary has run yet: `tripwire canary SUITE`.")
        return
    suite = st.sidebar.selectbox("Suite", sorted({r["suite"][7:] for r in found}))
    rows = []
    for r in (r for r in found if r["suite"][7:] == suite):
        result = json.loads(r["result"])
        env = result.get("env", {})
        rows.append(
            {
                "run": r["created_at"][:16].replace("T", " "),
                "verdict": r["verdict"],
                "identical answers": f"{result.get('identical', 0)} of {result.get('subset', 0)}",
                "delta": result.get("delta"),
                "low": result.get("lo"),
                "high": result.get("hi"),
                "runtime then": env.get("reference", {}).get("ollama"),
                "runtime now": env.get("now", {}).get("ollama"),
            }
        )
    st.subheader(f"Drift canary: {suite}")
    st.caption(
        "Each run repeats the same cases with the same seeds. With nothing changed "
        "underneath, the answers come back identical and the difference is exactly zero."
    )
    scored = [r for r in rows if r["delta"] is not None]
    if scored:
        base = alt.Chart(alt.InlineData(values=scored)).encode(x=alt.X("run:N", title=None))
        interval = base.mark_rule(color=BLUE, strokeWidth=2).encode(
            y=alt.Y("low:Q", title="change against the pinned reference"), y2="high:Q"
        )
        point = base.mark_point(color=BLUE, filled=True, size=90, opacity=1).encode(
            y="delta:Q", tooltip=["run:N", "verdict:N", "identical answers:N", "delta:Q"]
        )
        st.altair_chart(interval + point, width="stretch")
    table(rows)


PAGES = {
    "History": history,
    "Compare": compare_page,
    "Flips": flips,
    "Speed and quality": speed_and_quality,
    "Drift": drift,
    "Judge": judge_page,
    "Label": label_page,
    "Power": power,
}


def main() -> None:
    st.set_page_config(page_title="Tripwire", layout="wide")
    try:
        cfg, _ = project()
    except (OSError, ValueError, sqlite3.Error) as e:
        st.error(f"Cannot open the project at {settings()[0]}: {e}")
        return
    st.sidebar.title("Tripwire")
    page = st.sidebar.radio("View", list(PAGES))
    st.sidebar.caption(f"{cfg.db_path.name} · {'read-only' if settings()[1] else 'local'}")
    if not configurations():
        st.info("Nothing has run in this project yet: `tripwire run SUITE`.")
        return
    PAGES[page]()


if __name__ == "__main__":  # which is how `streamlit run` executes this file
    main()
