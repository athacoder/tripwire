# Running at scale on one GPU

On a laptop the budget is model time. This page covers what was measured, and the three
tools that spend less of it: the sample cache, the queue and the two-stage gate.

## What was measured

`gemma3:4b` on an RTX 3050 laptop GPU (6 GB), the Banking77 prompt (about 1,200 tokens of
instructions and examples, then one short message), 100 fresh cases per row.

| Setting | Seconds per case |
|---|---|
| Baseline: shared prefix, `num_ctx` 4096, one request at a time | 0.29 |
| Two requests at a time | 0.27 |
| Four requests at a time | 0.27 |
| `num_ctx` 2048 | 0.29 |
| Per-case text first, so no two prompts share a prefix | 0.78 |

- **Prompt layout matters most.** With the instructions first and the per-case text last,
  the server reuses 98% of each prompt from the previous request. Putting the varying text
  first makes every case 2.7 times slower. Keep what changes at the end.
- **Parallel requests buy nothing here.** The server works through them one after another;
  each request simply waits longer. `concurrency` stays at 1 for local models.
- **A smaller context window did not help** once the prompt fits either way.

## The cache

A sample is stored under what produced it, so nothing is generated twice.
`tripwire usage` shows the effect: over this project's runs so far, 70% of requested
samples were already stored.

## The queue

```bash
tripwire queue                         # every suite
tripwire queue --match "banking-*"     # suites whose name fits a pattern
tripwire queue suite-a suite-b --max-minutes 240
```

runs suites back to back. It sorts them by model so each model loads once (two suites
alternating between models reload on every switch and both crawl), prints how much each
suite has left and roughly how long that will take, reports progress every 100 samples,
and judges the judged suites after all generation is done. `--max-minutes` is one budget
for the whole queue: when it runs out the queue stops cleanly, exits with code 1, and the
same command later picks up where it stopped.

The 1,500-case reference split ran through it in 7.2 minutes.

To use another machine's GPU, point a provider's `base_url` at an Ollama server there.
Nothing else changes: samples are keyed by the model's digest, not by where it ran.

## The two-stage gate

```toml
on_inconclusive = "escalate"
first_stage     = 250
```

The gate first looks at a fixed, stratified subset of 250 cases. If that gives a decisive
verdict it stops, and the other cases are never run. Otherwise it runs all of them.

Looking twice at the same error budget would raise the false-alarm rate, which is the
usual mistake with early stopping. Here each look uses half of `alpha`, so by the union
bound the two together stay within it. A simulation test checks this: under no true
difference the two-stage gate's false `REGRESSED` rate stays at or below `alpha`.

Two real runs on the Banking77 suite:

| Change | Decided at | Cases run | Result |
|---|---|---|---|
| Whitespace-only edit to the prompt | stage 1 | 250 of 700 | `PASS`, Δ +0.008 [−0.008, +0.028] |
| Few-shot examples removed | stage 2 | 700 of 700 | `REGRESSED`, Δ −0.046 [−0.071, −0.021] |

### It is not free

Halving `alpha` widens each interval, so a staged gate has less power than a single look
at everything. `tripwire power SUITE --first-stage 250` simulates both from the suite's
real scores. With a discordant rate of 12% (what a real prompt change produced here):

| True situation | Stops at stage 1 | Cases run on average | `REGRESSED` | `INCONCLUSIVE` |
|---|---|---|---|---|
| A 3-point drop | 25% | 586 of 700 (16% fewer) | 0.55 (single look: 0.64) | 0.41 (0.30) |
| No change | 32% | 557 of 700 (20% fewer) | 0.04 (0.05) | 0.29 (0.21) |

So for this suite the staged gate saves about a fifth of the model time on average and
gives up roughly nine points of power on a drop that sits exactly at the margin. It pays
off most when changes are usually harmless and clearly so, as the whitespace edit was. A
group-sequential boundary would lose less power than the even split used here.

In a bundle, a gate that stopped at stage 1 carries only the first-stage samples. CI
verifies it the same way, and reaches the same verdict at the same stage.
