# Banking77 splits

Customer-service messages from a retail banking app, each labelled with one of 77
intents.

## Source and licence

- Casanueva, Temčinas, Gerz, Henderson and Vulić, *Efficient Intent Detection with Dual
  Sentence Encoders* (2020).
- Data: <https://github.com/PolyAI-LDN/task-specific-datasets>, `banking_data/`.
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

## What was changed

`scripts/import_banking77.py` produces these files. It:

1. pools the published train and test files (13,083 rows);
2. removes exact repeats and any message that appears under more than one intent
   (13,071 rows remain);
3. tags each case with its intent, one of eight intent groups, and a length bucket;
4. draws three disjoint random splits, stratified by intent, with seed 77.

| Split | Cases | Used for |
|---|---|---|
| `dev.jsonl` | 200 | iterating on prompts |
| `gate.jsonl` | 700 | the regression gate |
| `reference.jsonl` | 1,500 | measuring the gate's own error rates |
| `robust.jsonl` | 762 | robustness: the 200 dev cases plus typo, lower-case and distractor variants |

The 10,671 remaining rows are not stored here. The few-shot examples in
`prompts/banking_intent/system.md` are drawn from them, so no evaluated case appears in
the prompt; `tripwire dataset lint --prompts prompts/banking_intent` checks this.

Splits are drawn at random, never by how a model scores on them. Picking the cases a
model currently fails would guarantee they "improve" on a re-run.

## Known limits

- **Label noise.** Banking77 is known to contain mislabelled and overlapping intents.
  A hand audit of a sample of these splits has not been done yet, so the ceiling on
  measurable accuracy is unknown.
- **Public data.** Models may have seen it during training. That matters little for
  comparing two configurations of the same model, which is what the gate does.
- `manifest.json` records the version hash and group counts of each split.
