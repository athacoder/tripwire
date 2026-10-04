# Design decisions

Short notes on choices that are not obvious from the code.

## Samples are keyed by fingerprint, not by run

A sample's key is `(fingerprint, case_hash, rep)`. A run owns no data; it is a manifest
that points at samples. Caching, resume and baseline lookup all fall out of that one key,
and a change that does not touch the target costs nothing to evaluate.

## The fingerprint uses the model digest and the exact prompt bytes

A tag such as `gemma3:4b` can be re-pointed by a later pull, so the tag alone would let
old samples be reused for a different model. Prompt whitespace is hashed as written,
because reformatting a prompt can change the output. Only newlines are normalised, so a
Windows and a Linux checkout agree.

Case hashes are the opposite: whitespace is collapsed and tags are excluded, so tidying
or retagging a dataset keeps its cached samples.

## Seeded sampling at a realistic temperature

Targets run at temperature 0.7 by default, with the seed derived from the case hash and
the repetition index. Temperature 0 would also be repeatable, but it hides the variance
a real user sees, and measuring under that variance is the point.

## SQLite, written from the event loop

The runner is async, but asyncio runs on one thread and each write happens between
awaits, so there is never a concurrent writer. No queue or lock is needed. Revisit this
if generation ever moves to worker threads or processes.

## Requests are not stored per sample

The system prompt lives once in `targets.spec` and the case input once in `cases`; the
request can be rebuilt from the two. Storing it on every sample would repeat a
kilobyte-scale prompt tens of thousands of times.

## Near-duplicate detection is token-set overlap

Jaccard similarity over lowercase word sets, with an O(n²) pair scan. It catches
re-punctuated and reordered copies and is fast enough for a few thousand cases.
Embedding similarity is available behind `--embed` for paraphrases.

## Sample bundles live in the repository

CI has no GPU, so the gate verifies committed samples instead of generating them. The
bundles sit under `bundles/` on the same branch as the change, not on a separate storage
branch: a pull request then carries its own evidence, the base bundles arrive with the
base branch's history, and there is nothing extra to fetch. A bundle is a gzip file of
about 26 KB for 700 cases, written with a fixed timestamp so identical samples give
identical bytes.

Bundles are looked up by a *static key*: the fingerprint without the model digest. CI can
compute that from the files alone; the digest is known only to the machine that holds the
model, and travels inside the bundle.

## The base target runs on the head dataset

`tripwire gate` takes the prompt and target files from the base ref but the cases and
scorers from the head. Adding or fixing cases therefore never breaks the comparison: the
question is always "old prompt against new prompt, on today's cases, by today's rules".

## Python targets are loaded by file path

The gate loads the same module from two checkouts. A plain `import` would return the
first one for both, so the entry module is loaded from its file under a name unique to
its checkout. Modules that the entry module imports are still shared; a target with
several local modules is better served by the `http` kind.

## A failed request is never a wrong answer

Anything raised while producing one sample (a timeout, an HTTP error, a malformed reply,
an exception in a python target) is recorded in `errors` and the run continues. If more
than `error_rate_max` of the cases end up without a score on either side, the comparison
is `INVALID` rather than `REGRESSED`.

## Slices and guardrails can block on their own

A change that holds the overall score but breaks one slice, or doubles the output length,
is blocked. Slice p-values are FDR-adjusted, and a ratio guardrail fails only when its
whole interval is over the limit, so neither fires on noise alone.

## Not built yet


- A dollar-cost column and price table: every backend in use is local and free.
- A rate limiter for hosted APIs: retries with backoff cover the free tiers so far.
- Two-stage gating (a cheap first look, escalating only when undecided).
- A composite GitHub Action; the workflow file is the integration for now.
- `import_bundle` picks one bundle per static key. Two digests of the same tag would need
  a rule for which to prefer.
