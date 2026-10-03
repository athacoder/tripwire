import pytest

from tripwire import datasets
from tripwire.config import Config, load_config
from tripwire.models import Case


def make_cases(n: int = 40) -> list[Case]:
    return [
        Case(input={"text": f"question number {i}"}, expected=f"answer {i}", tags=[f"kind:{i % 4}"])
        for i in range(n)
    ]


@pytest.fixture
def project(tmp_path) -> Config:
    """A complete project on the mock provider: 40 cases, 2 repetitions, no retry delay."""
    datasets.save(tmp_path / "data.jsonl", make_cases())
    (tmp_path / "target.toml").write_text('provider = "mock"\nmodel = "mock:0.8"\n')
    (tmp_path / "tripwire.toml").write_text(
        'backoff = 0\n[suite.demo]\ndataset = "data.jsonl"\ntarget = "target.toml"\nreps = 2\n'
    )
    return load_config(tmp_path / "tripwire.toml")
