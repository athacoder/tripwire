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
answer is generated again, and compares it with a pinned earlier run of the same target
using the ordinary comparator.

- **Seeds make the null exact.** Each case's seed is derived from the case, so an
  unchanged runtime returns the same text. The canary reports how many answers are
  identical before it reports any statistics, and with nothing changed that number is all
  of them and the difference is exactly zero.
- **The reference is pinned by fingerprint.** The first canary run fixes it. A model
  pulled again has a new digest, hence a new fingerprint; looking the reference up afresh
  would compare the new model with itself. The pinned run is the old model's.
- **It compares with a fixed point, not with yesterday.** Day-to-day comparison lets slow
  drift through one small step at a time.

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
