# Pointing Tripwire at your own system

Tripwire needs three things from you: cases, a target, and a way to score an answer. A
runnable version of everything on this page is in [examples/](../examples/README.md).

## 1. Cases

A dataset is a JSONL file, one case per line:

```json
{"input": {"text": "My card still has not arrived"}, "expected": "card_arrival", "tags": ["group:cards"]}
```

| Field | Meaning |
|---|---|
| `input` | An object. Its fields fill the prompt template, or are passed to your function |
| `expected` | The reference answer: a string, a number, or an object, depending on the scorer |
| `tags` | Optional. Each tag with at least 30 cases becomes a slice in every comparison |
| `source`, `provenance` | Optional. Where the case came from |

A case is identified by a hash of `input` and `expected`, so editing either makes it a new
case and retagging does not. Check a file before trusting it, and split it at random:

```bash
tripwire dataset lint cases.jsonl --prompts prompts/
tripwire dataset split cases.jsonl out/ --size dev=200 --size gate=700
```

How many cases a gate needs depends on how often two runs disagree; `tripwire power`
answers it from your own results. As a first guess, about 700 cases settle a 3-point
margin when two runs disagree on a tenth of the cases.

## 2. A target

A target file says what is under test. There are three kinds.

### A prompt and a model

```toml
provider    = "local"                 # a [provider.*] block; "local" is Ollama on localhost
model       = "gemma3:4b"
system      = "prompts/system.md"
template    = "prompts/user.md"       # filled from the case input, e.g. "Message: {text}"
temperature = 0.7
num_ctx     = 4096
max_tokens  = 32
```

Put what every case shares first and what varies last: the server reuses the shared part,
which made runs 2.7 times faster here. Literal braces in a template must be doubled.

### A Python function

When the thing you ship is code, not one prompt, test the code:

```toml
kind  = "python"
entry = "my_app:answer"       # my_app.py beside tripwire.toml, function `answer`
watch = ["my_app.py", "pipeline/**/*.py", "prompts/**"]
```

```python
def answer(case: dict) -> str:  # may also be `async def`
    return run_my_pipeline(case["text"])
```

- The function receives the case's `input` and returns the answer, or a dict with an
  `"output"` key. Anything else in the dict is stored with the sample; a `"trace_id"`
  is shown next to each flipped case, linked to TraceLens when `tracelens_url` is set.
- **`watch` is what makes caching safe.** Tripwire cannot see inside a function, so the
  target's identity is the hash of the watched files. List everything that can change the
  answer. A file left out means a change to it reuses old samples and passes unnoticed.
- An exception is recorded as an error for that case, never as a wrong answer. If more
  than `error_rate_max` of cases fail, the verdict is `INVALID`.
- At a git ref (`tripwire gate --base REF`), the function is loaded from that ref's
  files, together with the modules it imports from the project.

### An HTTP endpoint

```toml
kind  = "http"
url   = "http://localhost:8080/predict"
watch = ["service/**"]
```

Tripwire sends `POST {"input": {...}, "seed": 123}` and expects JSON back: the answer, or
an object with an `"output"` key.

One thing differs from the other kinds. A prompt or a function can be run as it was at an
older commit; a running service cannot. `tripwire gate --base REF` computes the base's
identity from the watched files at that ref, but if no samples are stored for it, the
calls go to whatever is running now. So store the baseline while it is deployed: run the
suite against each release, and gate the next one against those samples.

## 3. A scorer

```toml
[suite.my-suite]
dataset        = "cases.jsonl"
target         = "targets/my_app.toml"
scorers        = ["exact"]
primary_metric = "exact.pass"       # "<scorer>.<metric>": the one that decides the verdict
margin         = 0.03               # the largest drop you accept
```

| Scorer | `expected` is | Passes when |
|---|---|---|
| `exact` | a label | the answer is that label, or a sentence naming only that label |
| `contains` | a string or a list | every string appears in the answer |
| `regex` | a pattern | the pattern matches |
| `numeric` | a number | the last number in the answer equals it |
| `json_valid` | anything | the answer parses as JSON |
| `field_f1` | an object | per-field precision, recall and F1; `pass` when every field matches |
| `token_f1` | a reference text | word overlap, as a score from 0 to 1 |
| `fact` | `{"answerable": true, "value": 30}` | the number appears, or the answer abstains when it should |
| `judge` | `{"reference": "..."}` | an LLM judge says yes to a criterion; needs a `[judge]` section |

Before believing a score, run `tripwire selfcheck my-suite`: the reference answers must
pass and junk must fail. A judged metric is checked with `tripwire judge probe` instead
([judge.md](judge.md)). Changing a scorer never rewrites history: scores carry the
scorer's version, and `tripwire score my-suite` re-scores stored answers for free.

## 4. Gate it

```bash
tripwire run my-suite                        # the baseline's samples
tripwire gate my-suite --base origin/main    # after a change: runs what is missing, decides
```

Exit codes: 0 for `PASS`, `IMPROVED` and `UNCHANGED`; 1 for `REGRESSED`, a guardrail over
its limit or a slice that fell; 2 for `INCONCLUSIVE`; 3 for `INVALID`.

For CI without a GPU, add `--bundle`, commit the `bundles/` directory, and run
`tripwire gate my-suite --verify-only` in the workflow;
[.github/workflows/gate.yml](../.github/workflows/gate.yml) is a working one.

## Another backend

Anything that speaks the OpenAI chat API works as a provider:

```toml
[provider.lmstudio]
kind     = "openai_compat"
base_url = "http://localhost:1234/v1"

[provider.groq]
kind        = "openai_compat"
base_url    = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"      # the variable's name; the key is never stored
```

Two things Tripwire can do with Ollama and not through this API: check that a prompt fits
the context window, and start each run from a freshly loaded model. Set `concurrency` and
`retries` in `tripwire.toml` for a hosted API with rate limits.
