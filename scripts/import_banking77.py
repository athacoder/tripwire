"""One-off import of Banking77 (PolyAI, CC BY 4.0) into Tripwire's dataset format.

Writes the dev / gate / reference splits, a manifest, and a starter system prompt whose
few-shot examples come only from cases left out of every split.

    python scripts/import_banking77.py
"""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

from tripwire import datasets
from tripwire.models import Case

SOURCE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "banking77"
PROMPT = ROOT / "prompts" / "banking_intent" / "system.md"
SIZES = {"dev": 200, "gate": 700, "reference": 1500}
SEED = 77
CANARY = "tripwire-eval-data 6f1d2c0e-banking77 (do not train on this file)"

# First matching rule wins, so the order matters: "top_up_by_bank_transfer_charge" is a top-up.
GROUPS = [
    ("top_ups", ("top_up", "topping_up")),
    ("transfers", ("transfer", "beneficiary", "receiving_money")),
    ("exchange", ("exchange", "currency", "currencies")),
    ("cash", ("cash", "atm")),
    ("identity", ("identity", "verify", "pin", "passcode", "lost_or_stolen_phone")),
    ("payments", ("payment", "refund", "charge")),
    ("cards", ("card", "contactless", "apple_pay")),
]

INSTRUCTIONS = """You are an intent classifier for a retail banking app.
Read the customer message and reply with exactly one intent label from the list below.
Reply with the label only: no explanation, no punctuation, no quotes.
"""


def group_of(intent: str) -> str:
    name = intent.lower()
    return next((g for g, keys in GROUPS if any(k in name for k in keys)), "account")


def length_of(text: str) -> str:
    return "short" if len(text) < 40 else "medium" if len(text) < 60 else "long"


def fetch() -> list[Case]:
    cases = []
    for part in ("train", "test"):
        with urllib.request.urlopen(f"{SOURCE}/{part}.csv") as reply:
            rows = csv.DictReader(io.StringIO(reply.read().decode("utf-8")))
            for row in rows:
                text, intent = row["text"].strip(), row["category"]
                tags = [f"intent:{intent}", f"group:{group_of(intent)}", f"len:{length_of(text)}"]
                cases.append(
                    Case(
                        input={"text": text},
                        expected=intent,
                        tags=tags,
                        source="banking77",
                        provenance={"part": part},
                    )
                )
    return cases


def clean(cases: list[Case]) -> list[Case]:
    """Drop exact repeats, and any message that appears under more than one intent."""
    labels = defaultdict(set)
    for c in cases:
        labels[" ".join(c.text.lower().split())].add(c.expected)
    unique = {c.hash: c for c in cases}.values()
    return [c for c in unique if len(labels[" ".join(c.text.lower().split())]) == 1]


def main() -> None:
    raw = fetch()
    cases = clean(raw)
    splits = datasets.split(cases, SIZES, SEED)
    for name, part in splits.items():
        datasets.save(OUT / f"{name}.jsonl", part)
    used = {c.hash for part in splits.values() for c in part}
    spare = sorted((c for c in cases if c.hash not in used), key=lambda c: c.hash)

    manifest = datasets.manifest("banking77", splits, CANARY)
    manifest["imported"] = {"raw": len(raw), "after_cleaning": len(cases), "seed": SEED}
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    if not PROMPT.exists():  # never overwrite a prompt that has been edited by hand
        rng = random.Random(SEED)
        by_group = defaultdict(list)
        for c in spare:
            by_group[c.tags[1]].append(c)
        shots = [c for group in sorted(by_group) for c in rng.sample(by_group[group], 2)]
        intents = "\n".join(f"- {i}" for i in sorted({c.expected for c in cases}))
        examples = "\n\n".join(f"Message: {c.text}\nIntent: {c.expected}" for c in shots)
        PROMPT.parent.mkdir(parents=True, exist_ok=True)
        PROMPT.write_text(
            f"{INSTRUCTIONS}\nIntents:\n{intents}\n\nExamples:\n\n{examples}\n",
            encoding="utf-8",
            newline="\n",
        )

    print(f"{len(raw)} rows, {len(cases)} after cleaning, {len(spare)} left out of every split")
    for name, info in manifest["splits"].items():
        print(f"{name}: {info['cases']} cases, version {info['version']}, groups {info['groups']}")


if __name__ == "__main__":
    main()
