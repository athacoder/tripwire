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

## Not built yet

- A dollar-cost column and price table: every backend in use is local and free.
- A rate limiter for hosted APIs: retries with backoff cover the free tiers so far.
- A killed process leaves its run row marked `running`. Samples are unaffected and the
  next run resumes normally.
