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

## The judge is a scorer with a prompt-derived version

Judge verdicts are stored in the same `scores` table as rule-based scores, under scorer
`judge`, metric = criterion. The version is a hash of the judge model's name and every
word it reads (system prompt, template, criterion). Rewording one criterion re-judges
that criterion only and keeps the old verdicts. The model digest is recorded with each
verdict but is not part of the version, so CI can compute versions without a model.

Verdicts travel in sample bundles: CI cannot run a judge, and re-judging on every machine
would also make a comparison depend on who ran it.

## Per-sentence criteria and the claim pattern

A 7B judge asked whether a whole answer is grounded checks the first sentence and stops.
Asked about one sentence at a time it is accurate, except that it calls greetings and
"the documents do not cover this" ungrounded. Rather than keep rewording the question,
the rubric can give a criterion a `claim_pattern`: sentences that do not match it are
taken to make no claim and pass without a call. For the policy dataset the pattern is
"contains a digit", because every policy fact there is a number. This is a deliberate
domain rule, and it means a non-numeric invented claim would go unnoticed there.

## Which model judges

`qwen2.5:7b`, not `llama3.1:8b` as first planned. On the tuning probes llama misread the
documents (calling a number absent when it was present) under every wording tried; qwen
did not. The judge must still differ from the system under test, so `qwen2.5:7b` is no
longer available as a comparison system for judged suites.

## Labelling is a random sample, not a stratified one

`tripwire label` draws answers uniformly at random. Sampling by judge verdict would give
more failures to label, but the prediction-powered estimate is only valid when the
labelled answers are a random subset of all answers.

## Candidates are files, and review is the only way into a dataset

Drafted and imported cases live in a candidates JSONL file with a status per case, not in
the database: they are small, they belong next to the data in git while under review, and
a file can be read without the tool. `dataset review` writes both files after every
decision. There is no flag to skip review.

## A run refuses a dataset drafted by its own model

If any case's provenance says it was drafted by the model under test, the run stops with
an error instead of a warning. The failure mode it prevents (a model graded against its
own guesses) produces numbers that look fine, so a warning would be ignored.

## Robustness sets keep their originals

`dataset perturb` writes the unmodified cases too, tagged `perturb:none`, with unchanged
hashes. They reuse samples already generated for the source dataset, and the report can
pair each variant with its parent without a second file.

## The first stage is a fixed subset, and each look gets half of alpha

The two-stage gate's first 250 cases are the same every time: a stratified draw with a
fixed seed, a function of the dataset alone. A subset re-drawn per run would let a
borderline change pass by luck on a retry. Splitting alpha evenly is the simplest rule
that keeps the overall false-alarm rate honest; it is conservative, and the docs say what
it costs.

## The queue orders by model, and concurrency stays at 1

Measured, not assumed: two or four simultaneous requests to a local model finish no
sooner than one at a time, and alternating between two models reloads each on every
switch. So the queue sorts suites by model and runs them in sequence, and judging happens
after all generation.

## A subset run keeps the dataset's identity

`--limit`, a first-stage subset and a full run all record the version of the whole
dataset file. The version says which dataset a sample belongs to, not how much of it has
been run so far; bundles are filed under it.

## A server's address is part of the fingerprint only when there is no digest

An Ollama model is identified by its digest, so the address of the server is left out:
the same model on another machine is the same target, and its samples are reused. The
OpenAI-compatible API reports no digest. There the address is the only thing that tells
two servers apart, so it goes into the fingerprint; otherwise two servers offering one
model name would share samples, and moving a target from one to the other would look to
the gate like no change at all.

## The context check assumes 2.5 characters per token

The check has to run before any call, without a tokenizer. English prose here measured
3.5 to 4 characters per token, but the invoice prompts, which are mostly numbers,
measured 2.55, so the estimate uses 2.5 and errs towards asking for a larger window.
The judge's prompts get the same check as a target's. It cannot protect a target served
through the OpenAI-compatible API, which has no way to set the window.

## `allow_overflow` is left out of the fingerprint

A target can set `allow_overflow = true` to skip the check that its prompt fits the
context window. It exists to measure what an overflow does, and it lifts a check without
changing any output, so it is not hashed. Every other target field is, which means adding
a hashed field later would orphan all stored samples; a new field that cannot change the
output should be excluded the same way.

## The benchmark has its own configuration file

The variants used to measure the gate live in `experiments/zoo.toml`, generated by
`experiments/zoo.py`, and not in `tripwire.toml`. Thirty-odd suites that exist only to be
compared with each other would bury the few that describe the product, and a bare
`tripwire queue` would run all of them. Both files point at the same database, so the
baseline's samples are shared rather than generated twice.

## The benchmark draws cases with replacement

The gate is replayed on random draws from the 1,500 reference cases. Drawing subsets
without replacement looks natural and is wrong here: 700 of 1,500 is nearly half the pool,
so every subset resembles the whole, the spread between subsets shrinks by a factor of
`sqrt(1 - 700/1500) = 0.73`, and the gate looks both more powerful and more cautious than
it is. Drawing with replacement makes each draw an independent sample from a population
in which the variant's effect is exactly the one measured on all 1,500 cases, so the
truth the gate is checked against is known without error. See
[benchmark.md](benchmark.md).

## The benchmark figure is hand-written SVG

`experiments/proof.py` writes the dose-response figure as SVG text. Regenerating every
table and the figure then needs nothing beyond the packages Tripwire already depends on.

## Not built yet


- A dollar-cost column and price table: every backend in use is local and free.
- A rate limiter for hosted APIs: retries with backoff cover the free tiers so far.
- A pairwise judge (which of two answers is better). The three suites all have a
  reference answer, so nothing needs it yet.
- Hosted batch APIs and provider-side prompt caching: they only matter with a paid
  provider.
- A group-sequential boundary for the two-stage gate, in place of the even alpha split.
- A composite GitHub Action; the workflow file is the integration for now.
- `import_bundle` picks one bundle per static key. Two digests of the same tag would need
  a rule for which to prefer.
