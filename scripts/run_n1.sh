#!/usr/bin/env bash
# The N1 ladder, end to end, exactly as specified in docs/prereg-n1.md.
#
# Three stages, because each has a different bottleneck:
#   1. collect  simulation re-rendered with MESHES (CPU, MuJoCo)
#   2. encode   everything through the frozen encoder, COMB-FREE (GPU)
#   3. measure  centroid + MMD + Frechet on every rung (CPU, seconds)
#
# Both encoding settings are non-negotiable and both are audit items:
#   B2  our fast path renders collision primitives for an 11x speedup, and is
#       much further from real video than the mesh render. Measuring the
#       sim-to-real gap on blocky renders would partly measure our own
#       rendering shortcut.
#   comb  our default encode stride stamps a period-8 signature into simulated
#       latents (1.44-1.67x) and leaves real ones alone (1.014x). Every metric
#       below either projects or whitens, and PCA is exactly what concentrates
#       a low-rank periodic component -- see ledger L6 and L7.
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE=${1:-all}
EPS=${2:-30}

# ---------------------------------------------------------------- 1. collect
#
# "Directory exists" is NOT the same as "collection finished". An earlier run
# was killed partway through and left 18, 17 and 7 episodes of a requested 30.
# Skipping on existence would have accepted all three silently -- and because
# the measurement stage equalises n across rungs using the SMALLEST dataset, a
# 7-episode leftover would have quietly capped every rung in the ladder at a
# third of the data it should have had.
#
# So the check is on episode count, and a short directory is rebuilt rather
# than reused. A partial result that looks complete is worse than no result.
collect_one() {
    out="$1"; task="$2"; seed="$3"; shift 3
    have=$(ls "$out"/episode_*.npz 2>/dev/null | wc -l)
    if [ "$have" -ge "$EPS" ]; then
        echo "### $(basename "$out"): $have episodes, complete"
        return
    fi
    if [ "$have" -gt 0 ]; then
        echo "### $(basename "$out"): only $have/$EPS episodes from a killed run"
        echo "    rebuilding rather than reusing a partial set"
        rm -rf "$out"
    fi
    echo "### collecting $(basename "$out") with MESH rendering (11x slower, required)"
    python scripts/collect_demos.py --task "$task" --episodes "$EPS" \
        --pretty --out "$out" --seed "$seed" "$@" 2>&1 | tail -4
    echo
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "collect" ]; then
    collect_one data/episodes/n1_sim_push_pretty      push      0
    collect_one data/episodes/n1_sim_pickplace_pretty pickplace 0
    # The domain-randomisation arm of N1: same task, same renderer, DR on.
    collect_one data/episodes/n1_sim_push_pretty_dr   push      1 --randomize
fi

# ----------------------------------------------------------------- 2. encode
if [ "$STAGE" = "all" ] || [ "$STAGE" = "encode" ]; then
    for d in data/episodes/n1_*; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        out="cache/latents/${name}"
        eps=$(ls "$d"/episode_*.npz 2>/dev/null | wc -l)
        lat=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
        if [ -f "$out/info.json" ] && [ "$lat" -ge "$eps" ] && [ "$eps" -gt 0 ]; then
            echo "### $name latents complete ($lat/$eps), skipping"
            continue
        fi
        if [ "$lat" -gt 0 ]; then
            echo "### $name partially encoded ($lat/$eps), redoing"
            rm -rf "$out"
        fi
        echo "### encoding $name  (chunk 32, margin 15 = one-latent stride)"
        python scripts/cache_latents.py --task "$name" --data "$d" --out "$out" \
            --chunk 32 --margin 15 --pool-grid 4 \
            2>&1 | grep -vE "UserWarning|self.blocks|Loading weights" | tail -3
        echo
    done
fi

# ---------------------------------------------------------------- 3. measure
if [ "$STAGE" = "all" ] || [ "$STAGE" = "measure" ]; then
    L=cache/latents

    # EVERY RUNG AT THE SAME SAMPLE COUNT.
    #
    # measure_domain_gap.py equalises n *within* a pair, which is not enough.
    # The rungs hold very different amounts of data -- R2 contributes 745
    # latents, R1 over 2000 -- so S would be measured at n=745 while SIM used
    # n=2016. All three metrics move with n: Frechet through its covariance
    # estimate, centroid through the bias of a mean at small n, MMD through its
    # variance. A ladder built that way would partly rank rungs by how much
    # data each dataset happens to contain.
    #
    # That is the same defect as the PCA mismatch in ledger L7 and the
    # pixels-versus-latents mix-up in E2: two conditions differing in more than
    # the variable under test. Third occurrence, so it gets a constant rather
    # than a good intention.
    N=$(python3 scripts/min_latent_count.py)
    echo "sample count fixed at n=$N for every rung (smallest dataset governs)"
    echo

    gap() {
        rung="$1"; ref="$2"; oth="$3"
        if [ ! -d "$L/$ref" ] || [ ! -d "$L/$oth" ]; then
            echo "### $rung: missing latents ($ref or $oth)"; echo; return
        fi
        echo "############ rung $rung ############"
        python scripts/measure_domain_gap.py --reference "$L/$ref" --other "$L/$oth" \
            --label "$rung" --cap "$N" --out "cache/n1_${rung}.json" \
            2>&1 | grep -vE "UserWarning|self.blocks"
        echo
    }

    # V is the confound check: same dataset, same episodes, only the camera
    # differs. If it rivals L or T, viewpoint dominates and the ladder is
    # inconclusive -- that outcome is pre-registered as invalidating, not as an
    # answer.
    gap V   n1_R4_blocks_top   n1_V_blocks_wrist

    # S is the floor: same lab, same task, same hardware, different session.
    gap S   n1_R2_penmug_s9    n1_R3_penmug_s12

    # L: different labs, both rigid-object pick-and-place.
    gap L   n1_R1_cubes        n1_R4_blocks_top

    # T: different labs, different tasks. Upper end of "still real".
    gap T   n1_R1_cubes        n1_R2_penmug_s9

    # The measurement. Real is always the reference so simulation never
    # defines the coordinate system it is judged in.
    gap SIM_push       n1_R1_cubes  n1_sim_push_pretty
    gap SIM_pickplace  n1_R1_cubes  n1_sim_pickplace_pretty
    gap SIM_push_DR    n1_R1_cubes  n1_sim_push_pretty_dr

    echo "================== THE LADDER =================="
    python3 - <<'PY'
import glob
import json
order = ["V", "S", "L", "T", "SIM_push", "SIM_pickplace", "SIM_push_DR"]
rows = {}
for f in glob.glob("cache/n1_*.json"):
    d = json.load(open(f))
    rows[d["label"]] = d
if not rows:
    print("no results")
else:
    ns = {d["n_per_side"] for d in rows.values()}
    print(f"sample count per side: {sorted(ns)}"
          f"{'  <-- NOT EQUAL, rungs are not comparable' if len(ns) > 1 else ''}")
    print()
    print(f"{'rung':16s} {'n':>5} {'centroid':>10} {'MMD^2':>10} "
          f"{'Frechet':>10} {'p':>8}")
    print("-" * 64)
    for k in order:
        d = rows.get(k)
        if not d:
            continue
        print(f"{k:16s} {d['n_per_side']:>5} {d['centroid_pca']:>10.3f} "
              f"{d['mmd2_pca']:>10.5f} {d['frechet_pca']:>10.3f} "
              f"{d['mmd_pvalue']:>8.4f}")
    print()
    print("Reading rule, fixed in advance (docs/prereg-n1.md):")
    print("  SIM ~ S        frozen encoder aligns sim and real like two sessions")
    print("  S < SIM <= L   simulation looks like another lab")
    print("  L < SIM <= T   further than cross-lab, still inside the real spread")
    print("  SIM > T        simulation is outside real entirely -- assumption fails")
    print("  V >= L         viewpoint dominates; the ladder is INCONCLUSIVE")
PY
fi
