# Growing a dataset

An eval set is never finished. There are four ways to add to one here, and they differ in
how far the expected answers can be trusted.

| Route | Where the expected answer comes from | Needs review? |
|---|---|---|
| Templates | the same parameters that render the case | no: correct by construction |
| Perturbation | the parent case; the answer must not change | no, except paraphrases |
| Drafted by a model | the model's guess | **yes, every case** |
| Imported from TraceLens | nobody yet; the trace shows what went wrong, not what is right | **yes, every case** |

Each case records its route in `source` and the details in `provenance`.

## Templates

`scripts/make_invoices.py` and `scripts/make_policy_qa.py` render a case from random
parameters and build the expected answer from the same parameters. No model and no
annotator is involved, and the result cannot have been seen in training.

## Perturbation: robustness without new labels

```bash
tripwire dataset perturb datasets/banking77/dev.jsonl datasets/banking77/robust.jsonl
```

writes the original cases followed by variants of each: swapped letters, lower case, an
irrelevant extra sentence (`--kind shuffle_lines --field context` reorders the lines of a
document instead). A variant keeps its parent's expected answer and remembers the parent's
hash. Perturbations are seeded per case, so the file is reproducible.

`tripwire report` then pairs every variant with its own original:

| perturbation | pairs | Δ score | 90% interval | outcome changed |
|---|---|---|---|---|
| distractor | 200 | +0.010 | [−0.025, +0.045] | 10.0% |
| lowercase | 187 | −0.027 | [−0.064, +0.011] | 9.1% |
| typo | 175 | −0.017 | [−0.051, +0.017] | 7.4% |

(`gemma3:4b`, Banking77 dev split.) No perturbation moves the score by a detectable
amount. But 7–10% of individual answers change, against 1.9% when the identical prompt is
simply run again: the model's answer to a given customer is far less stable under a typo
than its average accuracy suggests. The "outcome changed" column needs no expected answer
at all, only the knowledge that the answer was not supposed to change.

`--paraphrase-model` adds model-written rewordings. A paraphrase can shift the meaning
("I forgot my code" came back as "I failed to remember my password"), so read a sample
before trusting a paraphrase slice. None are committed here.

## Drafted by a model

```bash
tripwire dataset gen datasets/banking77/gate.jsonl seed.md candidates.jsonl \
    --model qwen2.5:7b --verifier llama3.1:8b --n 10
```

The drafting model is shown a few existing cases as the format and asked for new ones
about the seed material. Its reply is constrained to that format; for a classification
task the expected answer is restricted to labels that already exist. Each draft is then
dropped if it repeats or nearly repeats an existing case, or if a second model flags it as
ambiguous or wrongly answered.

What survives is still only a candidate. In one trial of ten drafts about card problems,
six passed both checks, and most of those six carried a wrong label: "I didn't receive the
card I ordered" was labelled `card_not_working`, not `card_arrival`. The screening model
did not notice. That is the reason review is not optional.

A run refuses to start if any case in its dataset was drafted by the model under test.
Expected answers written by a model reward agreeing with that model.

## Imported from TraceLens

```bash
tripwire dataset import-tracelens candidates.jsonl --url http://localhost:8000
```

reads the traces that [TraceLens](https://github.com/athacoder/tracelens) diagnosed as
failures. Each becomes a candidate whose input is what the pipeline was given, tagged with
the stage TraceLens blamed and the kind of failure, and carrying the diagnosis and the
pipeline's actual output as notes for the reviewer:

```json
{"input": {"user_input": "How long does a refund take to reach the original payment method?"},
 "tags": ["stage:retrieval", "failure:retrieval_failure", "pipeline:rag"],
 "source": "production-failure",
 "provenance": {"trace_id": "5c64…", "status": "pending",
   "root_cause": "retriever did not return the expected document(s) refund-2026; it returned refund-2019",
   "observed_output": "{\"answer\": \"Refunds are issued … within 68 business days.\"}"}}
```

Against TraceLens's demo data this produced 24 candidates from 28 traces. Importing again
adds nothing new. The expected answer is empty on purpose.

In the other direction, a target may return `{"output": ..., "trace_id": ...}` instead of
a bare answer. The trace id is stored with the sample, and when `tracelens_url` is set in
`tripwire.toml` every flipped case in a comparison links to its trace, so a regression
arrives with the stage that caused it.

## Review

```bash
tripwire dataset review candidates.jsonl --into datasets/banking77/gate.jsonl
```

shows each pending candidate with its tags and notes and asks to approve, edit the
expected answer, reject, skip or quit. A candidate without an expected answer cannot be
approved until one is typed in. Approved cases are appended to the dataset with the
reviewer's name; a case already in the dataset is not added twice. Every decision is
written back to the candidates file immediately, so quitting halfway loses nothing and a
rejected draft is not offered again.

## Not done here

- No dataset in this repository yet mixes all four routes. The TraceLens candidates are
  questions for a retrieval pipeline, which is a separate project, and the drafted
  Banking77 cases are waiting for a human reviewer.
- There is no shared template engine. Two generator scripts did not justify one.
