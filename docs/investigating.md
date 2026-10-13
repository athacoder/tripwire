# After the verdict

The gate says a change is worse. Three tools help with what comes next: finding the
commit, checking whether the ground moved, and looking at the answers themselves.

## Which commit did it: `tripwire bisect`

```bash
tripwire bisect banking-intent --good v0.3.0            # up to HEAD
tripwire bisect banking-intent --good 4c1f9aa --bad 9e2b771
```

Given a ref where the suite was fine and one where the gate blocks it, bisect finds the
first commit in between that is blocked. Each probe is an ordinary gate run of one commit
against the good ref, so "bad" means exactly what it means in CI: `REGRESSED`, a guardrail
over its limit, or a slice that fell.

- **The working tree is never touched.** Each commit's prompts and config are read
  straight from git, the same way `gate --base` reads its base. There is no bisect state
  to reset and nothing to stash first.
- **It costs little.** Samples are keyed by what produced them, so a commit that did not
  change the target makes no model call at all, and two commits with the same prompt
  share their samples. Only commits that changed the target are ever generated.
- **Undecided commits are stepped around.** `INCONCLUSIVE` and `INVALID` cannot say which
  side of the break a commit is on. Bisect tries the nearest decided neighbours instead,
  and if a run of undecided commits sits right at the break it says so and lists them
  rather than guessing. Exit code 0 means one commit was found, 2 that it was narrowed to
  a few.

It follows the first-parent history from good to bad, and assumes what any bisect
assumes: once the suite breaks, it stays broken.

## Did the ground move: `tripwire canary`

A local model can change underneath a project that has not changed at all: the runtime is
upgraded, a tag is pulled again and now points at different weights, a driver is updated.
Nothing in git records it, so no gate run is triggered.

```bash
tripwire canary banking-intent              # 150 cases; --cases N for another size
tripwire canary banking-intent --pin        # accept today's behaviour as the reference
```

The canary reruns a fixed, stratified subset of the suite under a fresh salt, so every
answer is generated again, and compares it with a pinned earlier run of the same subset
using the ordinary comparator.

- **With nothing changed, every answer comes back the same.** Each case's seed is derived
  from the case, every run starts from a freshly loaded model, and the reference is itself
  a run of this subset in this order. The canary reports how many answers are identical
  before it reports any statistics.
- **The reference is generated when it is pinned, not read from the cache.** The first
  canary run pins it; `--pin` pins it again. The suite's stored samples are not used:
  they may be older than whatever moved, and they were generated among other cases.
- **The reference is pinned by fingerprint.** A model pulled again has a new digest, hence
  a new fingerprint; looking the reference up afresh would compare the new model with
  itself. The pinned run is the old model's.
- **It compares with a fixed point, not with yesterday.** Day-to-day comparison lets slow
  drift through one small step at a time.

On the reference machine the reference was pinned under Ollama 0.35.1. Two days later the
runtime had been upgraded to 0.40.3, and nothing else in the project had changed:

```text
150 of 150 answers are identical to the reference pinned on 2026-10-11 (runtime then 0.35.1, now 0.40.3)
Tripwire: PASS · canary · banking-intent
exact.pass   base 0.580   head 0.580   Δ +0.000   [+0.000, +0.000]
```

That upgrade did not move this model on this prompt. It is one upgrade; the point of the
canary is that the next one is a one-minute check and not an assumption.

### What it takes for a rerun to be identical

Less obvious than "the same seed", and measured on `gemma3:4b` under Ollama 0.35.1:

| Rerun of stored answers | Identical |
|---|---|
| Same cases, same order, freshly loaded model, three days later | 400 of 400 |
| The 150 canary cases alone, freshly loaded, seven runs over five days | 150 of 150 each time |
| The same 150 cases generated inside a run of all 700 | 148 of 150 |
| 200 cases straight after five requests with another prompt, no fresh load | 186 of 200 |
| The same, with the fresh load every run now makes | 200 of 200 |

Ollama keeps the evaluated prefix of the previous request and reuses what the next prompt
shares with it, so an answer depends a little on what the server did just before. Most of
the changed answers differ only in a trailing newline; fewer than half change the label.
Every run therefore unloads the model before its first request (`keep_alive: 0`). That is
only possible through Ollama's own API: a target on an OpenAI-compatible server starts
from whatever that server last did.

An earlier version of the canary read its reference from the suite's stored samples, and
reported 140 of 150 identical answers against ones generated under Ollama 0.35.0. That
cannot be credited to the upgrade: those samples were generated before runs started from
a fresh load, and among other cases, and either alone moves that many answers.

Run it after anything that could have moved the model. Each run is stored, and the
dashboard's Drift page plots them.

## Looking at the answers

### A report you can attach

`--out report.html` on `report`, `compare`, `gate` and `canary` writes one self-contained
page: the same content as the terminal report, plus every changed case that was kept (up
to 50 broken and 50 fixed, against the first ten in the terminal). Styles are inline and
there are no external assets, so it works as a CI artifact. Model output is escaped before
it reaches the page.

### The dashboard

```bash
pip install "tripwire-eval[dashboard]"
tripwire dashboard                                   # this directory's tripwire.toml
tripwire dashboard -c experiments/zoo.toml --read-only
```

| Page | What it answers |
|---|---|
| History | How has this suite's score moved as its target changed? One point per target, with its interval |
| Compare | Is this target worse than that one? The gate's own report for any two targets that ran on the same cases |
| Flips | Which cases changed, and what did each side say? Filter by direction, text or tag; select a row for both answers and a diff |
| Speed and quality | Which target is the fastest that still holds quality? Seconds per case against the score, with the frontier marked |
| Drift | Have the canary runs stayed at zero? |
| Judge | How far does the judge agree with human labels, and with the rule that measures the same thing? Browse the disagreements |
| Label | Blind labelling of stored answers, as `tripwire label` does in a terminal |
| Power | How many cases would a gate need? Resamples a target's real outcomes as the sliders move |

The dashboard opens the database read-only and calls the same functions as the command
line, so a number on a page and the number in a report cannot disagree. Only the Label
page writes, and `--read-only` switches it off.

In a container, with the project mounted rather than copied:

```bash
docker build -t tripwire-dashboard .
docker run --rm -p 8501:8501 -v "$PWD:/project" tripwire-dashboard
```
