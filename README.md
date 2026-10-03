# Tripwire

A statistical regression gate for LLM systems: run the old and the new version of a
prompt or pipeline on the same test cases, and block the change when the new one is
measurably worse.

LLM outputs are samples, not return values. "Accuracy went from 84% to 82%" might be a
real regression or might be noise, and a fixed threshold cannot tell the difference.
Tripwire treats an eval as a paired experiment and reports the difference with a
confidence interval.

Everything runs locally against open models through [Ollama](https://ollama.com). No API
key is needed.

## Status

Early. The data layer and the runner work end to end; the statistics and the CI gate are
next.

| Piece | State |
|---|---|
| Datasets: content-hashed cases, versioning, lint, stratified splits | done |
| Runner: resumable, seeded, cached samples; Ollama and OpenAI-compatible backends | done |
| Scorers and single-run reports | next |
| Paired comparison, verdicts, power analysis | planned |
| CI gate with pull-request comments | planned |
| LLM judge with calibration against human labels | planned |

## Quick start

Requires Python 3.12+ and Ollama with `gemma3:4b` pulled.

```bash
git clone https://github.com/athacoder/tripwire.git
cd tripwire
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

tripwire doctor                          # is the model reachable and on the GPU?
tripwire bench banking-intent            # measure throughput, project run times
tripwire run banking-intent              # generate every missing sample
tripwire run banking-intent              # a second run makes zero model calls
```

The tests need neither Ollama nor a GPU:

```bash
pytest -q
```

## How it works

A **case** is one test item, identified by a hash of its input and expected answer. A
**target** is the system under test: a prompt and a model, a Python function, or an HTTP
endpoint. Its **fingerprint** is a hash of everything that can change its output: the
model's digest, the exact prompt text, sampling parameters, context size, and any source
files it declares.

A **sample** is one output, stored under `(fingerprint, case, repetition)`. That one key
gives three things for free:

- **Caching.** An unchanged target never calls the model twice for the same case.
- **Resume.** An interrupted run picks up only the missing samples.
- **Baselines.** "The baseline" is simply the fingerprint of the target on the main
  branch; there is no separate notion of a baseline run.

Each sample's seed is derived from the case and the repetition index, so repetitions are
distinct draws and a whole run can be regenerated exactly. On the reference machine,
deleting 50 stored samples and generating them again reproduced all 50 outputs.

Failed requests are never scored. A timeout or an HTTP error goes to an `errors` table,
a cut-off answer is stored as `truncated`, and neither is confused with a wrong answer.

Before a run starts, Tripwire checks that the longest prompt fits the context window.
Ollama silently drops whatever does not fit and answers anyway, which would otherwise
look like the model getting worse.

## Commands

| Command | What it does |
|---|---|
| `tripwire doctor` | Checks each suite's model is pulled, loads, and how much of it is on the GPU |
| `tripwire bench SUITE` | Times a few cases and projects the duration of every split |
| `tripwire run SUITE` | Generates missing samples. `--limit`, `--max-minutes`, `--dry-run` |
| `tripwire dataset lint FILE` | Duplicates, conflicting labels, empty fields, leakage into prompts |
| `tripwire dataset split SRC OUT --size dev=200 --size gate=700` | Disjoint stratified splits |

Suites live in `tripwire.toml`; each points at a dataset and a target file.

## Other backends

Ollama on `localhost` is the default. Anything that speaks the OpenAI chat API also
works: LM Studio, llama.cpp's server, vLLM, or a hosted API. Add a provider block to
`tripwire.toml` and name it in the target file:

```toml
[provider.lmstudio]
kind     = "openai_compat"
base_url = "http://localhost:1234/v1"

[provider.groq]
kind        = "openai_compat"
base_url    = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"   # the variable's name; the key itself is never stored
```

## Example dataset

`datasets/banking77` holds three splits of [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets)
(customer-service intent classification, 77 intents, CC BY 4.0): `dev` for iterating on
prompts, `gate` for the regression gate, and `reference` for measuring the gate itself.
See the [dataset card](datasets/banking77/README.md).

On a laptop RTX 3050 (6 GB) with `gemma3:4b`, a case takes about 0.36 s, so the 700-case
gate split runs in roughly four minutes.

## Layout

```text
src/tripwire/   models, datasets, providers, targets, runner, store, cli
datasets/       JSONL cases and dataset cards
prompts/        system and user prompts, versioned in git
targets/        one file per system under test
scripts/        one-off dataset importers
tests/          runs on a deterministic mock provider
docs/           design notes
```

## License

MIT. The Banking77 data is redistributed under CC BY 4.0; see its dataset card.
