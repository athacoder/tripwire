# Examples: every command, no model

The suites in [tripwire.toml](tripwire.toml) run on the built-in mock provider, so they
need no Ollama, no GPU and no key. The datasets and prompts are the real ones from this
repository; only the "model" is fake. It answers a fixed share of cases correctly
(`mock:0.8` is about 80%), deterministically, and the same cases are hard for every mock
model, so two of them give paired results the way two real prompts do.

From the repository root, after `pip install -e .`:

```bash
tripwire run banking -c examples/tripwire.toml                      # 200 samples, scored
tripwire report banking -c examples/tripwire.toml                   # score, interval, slices
tripwire compare banking banking-worse -c examples/tripwire.toml    # a blocked change
```

The last one prints a verdict and exits with code 1, as a gate would:

```text
Tripwire: REGRESSED · banking → banking-worse        This blocks the merge.

| metric     | base  | head  | Δ      | 90% interval     | p      |
| exact.pass | 0.780 | 0.675 | -0.105 | [-0.145, -0.070] | 0.0002 |

margin 0.030 · α 0.05 · paired on 200 of 200 cases
```

Run any of them twice: the second time nothing is generated, because every sample is
already stored (in `examples/tripwire.db`, which git ignores).

## The other suites

| Suite | Shows | Try |
|---|---|---|
| `invoices` | A continuous metric: per-field F1 on structured output | `tripwire report invoices -c examples/tripwire.toml` |
| `policy-qa` | A judged metric, with a rule-based check beside it. The judge is the mock too | `tripwire report policy-qa -c examples/tripwire.toml` |
| `my-app` | Your own code as the system under test: [my_app.py](my_app.py), a Python function | `tripwire report my-app -c examples/tripwire.toml` |

More to try on any of them: `tripwire selfcheck banking`, `tripwire power banking`,
`tripwire report banking --out report.html`, each with `-c examples/tripwire.toml`.

## From here to a real model

Change two lines in a target file, for example [targets/banking.toml](targets/banking.toml):

```toml
provider = "local"        # Ollama on localhost
model    = "gemma3:4b"
```

The suites in the repository root's `tripwire.toml` are exactly that. Pointing Tripwire at
your own application is covered in [docs/integrating.md](../docs/integrating.md).
