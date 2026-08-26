# Contributing

Open threads are filed as [issues](../../issues). Several need hardware or data
this repository does not have, and those are labelled `needs-hardware` or
`needs-data` so you can tell before you start.

## What this project is now

It began as imitation learning on a frozen JEPA world model. The world model
works. What turned out to be harder, and more interesting, is **measuring
whether it works** — see [Findings](README.md#findings). Most contributions
that matter here are measurements, diagnostics, or refutations.

## Three conventions, and the reasons for them

### 1. Register a prediction before you measure

If you are adding an experiment whose outcome you care about, write what you
expect into `docs/prereg-*.md` and **commit it before running anything**.
Include what would falsify it and what would invalidate the instrument
entirely.

This is not ceremony. Two registered predictions in this repository came out
backwards, and both would otherwise have quietly become "what we expected all
along" — the domain-randomisation reversal in
[`docs/n1b-results.md`](docs/n1b-results.md) §6 and the camera-ruler failure in
[`docs/r1-results.md`](docs/r1-results.md). A third
([N1](docs/prereg-n1.md)) invalidated itself on a condition written down in
advance, which is the only reason the invalid result was never published.

Print the predictions that **failed**, not only those that passed. Every
measurement script here does.

### 2. Build the check before you trust the number

Every substantive defect in this project produced no error message. The loss
went down, the numbers looked plausible, and the system was wrong. The pattern
and its diagnostics are in [`docs/ledger.md`](docs/ledger.md).

Concretely, before comparing two runs:

```bash
python scripts/diff_checkpoints.py checkpoints/a.pt checkpoints/b.pt
```

It prints every non-weight field side by side and flags what differs. Three
false findings in this repository came from comparing conditions that differed
in more than the variable under test; that script exists so the fourth doesn't.

### 3. Record what didn't work

`docs/ledger.md` holds every failure, how it was diagnosed, and what fixed it —
including the ones that were my own reasoning errors rather than code. If you
spend a day on something that turned out to be wrong, that day is worth more
written down than deleted. Retractions go **next to** the original claim, not
in place of it.

## Running things

```bash
docker compose -f docker/compose.yaml --profile wsl2 run --rm dev-wsl bash
```

Profiles: `linux` (native, `/dev/kfd`), `wsl2` (Windows, `/dev/dxg`), `cpu`.
Setup and troubleshooting are in [README.md](README.md#setup).

Experiment settings live in
[`configs/experiments.json`](configs/experiments.json), committed next to the
numbers they produced. The reason is in ledger L7: `--pca-dim 128` lived only in
shell history, a later run omitted it, and two incomparable runs looked
comparable for a week.

## Scope

Work that fits: measurement validity, diagnostics, replication, refutation,
real-robot verification of anything currently only measured in simulation.

Work that fits less well: capability improvements to the policy or world model
without a measurement establishing that the improvement is real. The bar is not
"it scores higher" — it is "here is the check that would have caught it if it
didn't."
