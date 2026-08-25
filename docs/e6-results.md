# E6 — is a frozen video foundation model worth it?

Four encoders, three tasks, three seeds each, identical downstream pipeline.
Every arm writes the same cache layout, so nothing downstream can tell which
encoder produced it.

| arm | encoder | can it collapse? |
|---|---|---|
| `vjepa` | frozen V-JEPA 2, **326M** params | no — it cannot move |
| `rand` | frozen scratch CNN, **7.5M**, never trained | no |
| `joint` | same CNN, trained with the predictor | **yes** |
| `jointreg` | same, plus a VICReg variance hinge | **yes** |

---

## Results

| task | arm | cosine | ratio | probe R² | horizon |
|---|---|---|---|---|---|
| **push** | vjepa | 0.818 ± 0.016 | 0.937 ± 0.005 | 0.575 ± 0.000 | 96 |
| | **rand** | **0.921 ± 0.005** | 0.973 ± 0.007 | **0.663 ± 0.011** | 96 |
| | joint | 0.904 ± 0.022 | 0.995 ± 0.024 | 0.636 ± 0.066 | 96 |
| | jointreg | 0.784 ± 0.081 | 0.958 ± 0.037 | 0.713 ± 0.019 | 96 |
| **pickplace** | **vjepa** | **0.803 ± 0.002** | 0.848 ± 0.002 | **0.184 ± 0.000** | 96 |
| | rand | 0.774 ± 0.006 | 0.958 ± 0.017 | 0.179 ± 0.009 | 96 |
| | joint | 0.382 ± 0.241 | 1.327 ± 0.162 | 0.134 ± 0.021 | 63 |
| | jointreg | 0.195 ± 0.056 | 1.818 ± 0.471 | **0.033 ± 0.074** | 24 |
| **real_cubes** | vjepa | 0.824 ± 0.004 | 0.919 ± 0.001 | **0.743 ± 0.000** | 96 |
| | **rand** | **0.876 ± 0.009** | 1.017 ± 0.012 | 0.721 ± 0.010 | 96 |
| | joint | 0.603 ± 0.009 | 0.744 ± 0.022 | 0.707 ± 0.006 | 96 |
| | jointreg | 0.208 ± 0.011 | 0.450 ± 0.038 | 0.696 ± 0.006 | 64 |

All horizons censored at the tested maximum.

---

## 1. Pretraining buys nothing consistent

The decisive comparison is `vjepa` against `rand`. Both are frozen, so neither
can collapse, and the difference between them is exactly what 22M videos of
pretraining bought over random convolutional features.

| | push | pickplace | real_cubes | wins |
|---|---|---|---|---|
| cosine | **rand** +0.103 | vjepa +0.029 | **rand** +0.052 | rand 2–1 |
| probe R² | **rand** +0.088 | vjepa +0.005 | vjepa +0.022 | vjepa 2–1 |

**The two metrics disagree on real_cubes**: random takes cosine (0.876 vs
0.824), V-JEPA takes the inverse-dynamics probe (0.743 vs 0.721). Reported
rather than resolved — a metric disagreement is information about the metrics.

The supported claim is therefore narrower than the one push alone suggested:

> **A frozen 326M video foundation model buys nothing consistent over random
> convolutional features at this data scale.** Which encoder wins depends on the
> task and on which metric is asked.

### Retracted: "random CNN beats frozen V-JEPA"

Held on push with non-overlapping intervals across three seeds, and reported as
a general result. **It does not replicate on pickplace**, where V-JEPA wins
cosine outright.

Push has now turned out to be the atypical task twice — it is also the only task
where raw pixels beat latents on the E2 reward correlation. A single-task result
from push should be treated as provisional until replicated.

---

## 2. Trained encoders fail in two ways, and the controls separate them

Both `jointreg` arms look equally broken on cosine — 0.195 on pickplace, 0.208
on real_cubes. They are not the same failure:

| | cosine | ratio | **probe R²** | diagnosis |
|---|---|---|---|---|
| pickplace jointreg | 0.195 | 1.818 | **0.033** | **encoder collapsed** — latents carry no recoverable action |
| real_cubes jointreg | 0.208 | **0.450** | **0.696** | **encoder fine** — the *predictor* under-moves by half |

Without the inverse-dynamics probe both would have been filed as "the CNN
collapsed", and one of those filings would have been wrong. This is what the
control was built for.

### VICReg made things worse

The variance hinge was added so arm 3 would be a fair competitor rather than a
strawman that collapses trivially. It is **the worst arm on every task except
push**, and on pickplace it collapses harder than the unregularised arm it was
meant to rescue (probe R² 0.033 against 0.134).

Not the expected outcome, and it is recorded as measured.

---

## 3. Real teleoperation carries the most action signal

Inverse-dynamics R², all four arms:

| task | range |
|---|---|
| **real_cubes** | **0.696 – 0.743** |
| push | 0.575 – 0.713 |
| pickplace | 0.033 – 0.184 |

Human demonstrations are more action-informative than our scripted experts,
**regardless of encoder**. This is a property of the data, not the
representation.

Pickplace is the outlier in the other direction — actions are barely recoverable
from its latents at all, matching the standalone probe (0.176 pickplace against
0.741 reach). The likely reason is that its gripper open/close barely moves any
pixels, so a visual encoder cannot see the action that matters most.

---

## Limits

- One data scale, 20–30 episodes. Whether pretraining wins with more data is
  untested — [issue #7](../../issues/7).
- Simulation for two of three tasks; `real_cubes` is the only real-robot arm.
- All horizons censored, so that column separates nothing.
- Fréchet, cosine and probe R² are the three readouts; downstream *task success*
  is not measured, and a representation could rank differently on it.

**Reproduce:** `bash scripts/run_e6_seeds.sh push 30 "0 1 2" 96`
