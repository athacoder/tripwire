# Statistical method

How Tripwire decides whether a candidate is worse than a baseline. The code is in
`src/tripwire/stats.py`; every function there is pure and tested by simulation.

## Setup

Both configurations run on the same cases. For each case, average the primary metric over
its repetitions, then take the difference:

```text
d_i = candidate_i - baseline_i
Δ   = mean(d_i)
```

The case is the unit of analysis. Repetitions of one case are correlated, so 700 cases
with 3 repetitions are 700 observations, not 2,100.

## Interval and test

| Quantity | Method |
|---|---|
| Interval on Δ | BCa bootstrap over cases, 10,000 resamples. Falls back to the percentile bootstrap if BCa degenerates, and to `[Δ, Δ]` when every difference is identical |
| p-value | Sign-flip permutation test on `d_i` |
| Cross-check | Exact McNemar test on discordant pairs, reported only when every score is 0 or 1 |

A two-sided 90% interval is used, which makes each end a one-sided 95% bound. `alpha` in
the configuration is that one-sided error rate (0.05 by default).

## Verdict

With interval `[L, U]` and margin `δ` (the largest drop you are willing to accept):

| Condition | Verdict |
|---|---|
| `L > 0` | `IMPROVED` |
| `-δ < L ≤ 0` | `PASS`: any drop is confidently smaller than the margin |
| `L ≤ -δ` and `U < 0` | `REGRESSED` |
| `L ≤ -δ` and `U ≥ 0` | `INCONCLUSIVE` |

`INVALID` is returned before any statistics when more than `error_rate_max` of the cases
lack a score on either side (failed requests, cut-off answers, missing samples).

This is a non-inferiority test. An ordinary significance test asks "is there evidence of
a difference?", and a lack of evidence then gets read as "safe". Here the candidate has to
demonstrate that it is not worse by more than `δ`; with too little data the answer is
`INCONCLUSIVE`, never a silent pass.

A gate is also blocked when a guardrail's whole interval is over its limit, or when a
slice regresses significantly.

## Looking twice

With `on_inconclusive = "escalate"` the gate judges a fixed, stratified subset first
and the whole set only if the subset is `INCONCLUSIVE`. Each look uses `alpha / 2`, so
the chance of a false alarm at either look is at most `alpha` (a union bound). The
even split is conservative; it costs some power, measured in
[running-at-scale.md](running-at-scale.md).

## Slices

Each tag with at least 30 paired cases gets its own Δ, interval and permutation p-value.
The p-values are adjusted with Benjamini–Hochberg, so the share of false alarms among the
flagged slices stays below `alpha`. Smaller slices are counted and reported as too small
to judge.

## Guardrails

Output-token ratio and p95-latency ratio (candidate over baseline) get a paired percentile
bootstrap interval. The interval is widened by a Bonferroni correction for the number of
ratio guardrails. A guardrail fails only when its entire interval is above the limit.
Truncation rate is a plain threshold.

## How many cases

For a pass/fail metric the standard error of Δ depends on the **discordant rate**: the
share of cases on which the two runs disagree.

```text
n ≈ ( z_(1-α) · sqrt(p_d) + z_(1-β) · sqrt(p_d − Δ²) )² / Δ²
```

`p_d` here is the total discordance: rerun noise plus the cases the drop itself flips.

`tripwire power` also simulates the gate: it resamples the suite's real pass/fail outcomes,
injects symmetric noise and a drop of a chosen size, and counts the verdicts. The
simulation uses the normal approximation to the interval so that thousands of gates run in
seconds; a test checks that approximation against the bootstrap.

Two measurements from the Banking77 suite on `gemma3:4b` show why the discordant rate has
to be measured rather than assumed:

| | Discordant rate |
|---|---|
| Same prompt, two repetitions (rerun noise) | 1.9% |
| Few-shot examples removed (a real change) | 12.3% (59 broke, 27 fixed, of 700) |

A real change flips far more cases than noise does, so planning from rerun noise alone is
optimistic. Pass `--discordant` with the flip rate of a past comparison.

## Cases or repetitions

```text
Var(mean) = between-case variance / n  +  within-case variance / (n · R)
```

`tripwire report` prints the split for suites with more than one repetition. On the
Banking77 suite 96% of the variance is between cases, so repetitions buy almost nothing
there and more cases are the only way to a tighter estimate.

## A/A calibration

`tripwire aa` compares a configuration with itself. Each split deals every case's
repetitions at random into two halves; the true difference is zero, so any `REGRESSED` or
`IMPROVED` verdict is a false alarm. On the Banking77 suite, 500 splits gave a false-alarm
rate of 4.4% against a nominal ceiling of 10% (5% in each direction).

## Limits

- The power simulation uses a simple noise model (independent symmetric flips).
- With two repetitions the A/A test has few discordant cases to work with; it checks the
  interval's calibration, not the whole pipeline.
- Slice p-values come from 2,000 permutations, so they are coarse below about 0.001.
