import asyncio
from pathlib import Path

import pytest
from test_judge import qa  # noqa: F401  (a fixture: a judged suite on mock providers)

from tripwire import store
from tripwire.gate import canary
from tripwire.judge import judge_suite
from tripwire.runner import run_suite
from tripwire.scorers import score_suite

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
PAGE = str(Path(__file__).parents[1] / "src" / "tripwire" / "dashboard.py")


@pytest.fixture(autouse=True)
def fresh_caches():
    """One dashboard serves one project; each test brings its own, so nothing may carry over."""
    import streamlit

    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()


def open_page(cfg, monkeypatch, view=None, read_only=False):
    monkeypatch.setenv("TRIPWIRE_CONFIG", str(cfg.root / "tripwire.toml"))
    monkeypatch.setenv("TRIPWIRE_READ_ONLY", "1" if read_only else "0")
    app = AppTest.from_file(PAGE, default_timeout=120).run()
    if view:
        app.sidebar.radio[0].set_value(view).run()
    assert not app.exception, app.exception
    return app


def text(app):
    parts = [*app.markdown, *app.caption, *app.info, *app.subheader, *app.metric, *app.success]
    return " ".join(str(getattr(e, "value", "")) + str(getattr(e, "label", "")) for e in parts)


@pytest.fixture
def two_targets(project):
    """The mock project with a second, weaker target on the same cases, both scored."""
    (project.root / "weak.toml").write_text('provider = "mock"\nmodel = "mock:0.4"\n')
    path = project.root / "tripwire.toml"
    extra = '[suite.weak]\ndataset = "data.jsonl"\ntarget = "weak.toml"\nreps = 2\n'
    path.write_text(path.read_text() + extra)
    from tripwire.config import load_config

    cfg = load_config(path)
    db = store.connect(cfg.db_path)
    for name in ("demo", "weak"):
        s = asyncio.run(run_suite(cfg, name))
        score_suite(cfg, name, s.fingerprint, db)
    db.close()
    return cfg


def test_an_empty_project_says_what_to_do(project, monkeypatch):
    store.connect(project.db_path).close()
    assert "tripwire run" in text(open_page(project, monkeypatch))


def test_every_view_renders_on_a_scored_project(two_targets, monkeypatch):
    asyncio.run(canary(two_targets, "demo", size=12))
    history = open_page(two_targets, monkeypatch)
    assert len(history.dataframe) == 1 and "exact.pass" in text(history)

    compared = open_page(two_targets, monkeypatch, "Compare")
    assert any(m.value in ("REGRESSED", "IMPROVED") for m in compared.metric)
    assert "paired on 40 of 40 cases" in text(compared)

    flips = open_page(two_targets, monkeypatch, "Flips")
    assert "broke (" in str(flips.main.radio[0].options) and len(flips.dataframe) == 1

    for view in ("Speed and quality", "Power"):
        open_page(two_targets, monkeypatch, view)
    assert "12 of 12" in str(open_page(two_targets, monkeypatch, "Drift").dataframe[0].value)
    assert "No suite in this project" in text(open_page(two_targets, monkeypatch, "Judge"))


def test_a_judged_project_shows_the_judge_and_takes_labels(qa, monkeypatch):  # noqa: F811
    s = asyncio.run(run_suite(qa, "qa"))
    db = store.connect(qa.db_path)
    score_suite(qa, "qa", s.fingerprint, db)
    asyncio.run(judge_suite(qa, "qa", s.fingerprint, db))

    judged = open_page(qa, monkeypatch, "Judge")
    assert "Against the rule-based check" in text(judged)

    label = open_page(qa, monkeypatch, "Label")
    assert "0 labels stored" in text(label)
    label.button[0].click().run()  # "Yes" to the first answer shown
    assert not label.exception and "1 labels stored" in text(label)
    assert db.execute("SELECT value FROM human_labels").fetchall()[0]["value"] == 1

    closed = open_page(qa, monkeypatch, "Label", read_only=True)
    assert "switched off" in text(closed) and not closed.button
