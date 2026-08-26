#!/usr/bin/env bash
# Isolate the comb with PCA dimensionality held constant.
#
# WHY THIS EXISTS. I reported that removing the window comb dropped push's
# direction cosine from 0.902 to 0.668, and concluded the comb had been
# inflating the model's apparent accuracy. That conclusion was wrong, and the
# checkpoint diff shows why:
#
#   predictor_push_seed0.pt          pca_basis (1024,)  hidden  128   cosine 0.902
#   predictor_push_decombed_seed0.pt pca_basis None     hidden 1024   cosine 0.668
#
# The two runs differed in PCA projection AND dimensionality, not only in the
# comb. Matching directions is far easier inside a 128-dimensional principal
# subspace -- which holds the dominant smooth trends -- than in the full 1024,
# where fine detail counts too. The drop was mine, not the artifact's.
#
# The controlled pair already exists and says the opposite:
#
#   push_s8n60  (combed,    no PCA)   cosine 0.692
#   push_s1n60  (stride-1,  no PCA)   cosine 0.695
#
# With everything but the encoding stride held fixed, the comb moves the cosine
# by 0.003. It is real in the latents and it wrecks the do-nothing baseline --
# which oscillates between 2.3 and 8.4 on encoder phase -- but it does not
# measurably change how well the model tracks.
#
# This script closes the loop by rebuilding the PCA-128 condition on top of
# comb-free latents. If push_s1n60 at PCA-128 lands near 0.90, the comb is
# confirmed irrelevant to cosine at both dimensionalities and the only honest
# comparison against real_cubes (0.847, also PCA-128) is the PCA-128 one.
set -uo pipefail
cd "$(dirname "$0")/.."

for s in s8n60 s1n60; do
    lat="cache/latents/push_${s}"
    [ -d "$lat" ] || { echo "### push_${s}: no cache"; continue; }
    name="push_${s}_pca128"
    echo "================ $name ================"
    python scripts/train_predictor.py --task "$name" --data data/episodes/push \
        --latents "$lat" --out checkpoints/pca128 --epochs 30 --seed 0 --pca-dim 128 \
        2>&1 | grep -vE "UserWarning|self.blocks" | tail -3
    echo
    python scripts/eval_horizon.py --task "$name" --data data/episodes/push \
        --latents "$lat" --checkpoint "checkpoints/pca128/predictor_${name}_seed0.pt" \
        --max-horizon 96 --out "cache/e3_pca128_${name}.json" \
        2>&1 | grep -vE "UserWarning|self.blocks" | tail -10
    echo
    python scripts/check_conservatism.py --task "$name" --data data/episodes/push \
        --latents "$lat" --checkpoint "checkpoints/pca128/predictor_${name}_seed0.pt" \
        --max-horizon 96 2>&1 | grep -vE "UserWarning|self.blocks" | tail -8
    echo
done

echo "=============================================================="
echo "  PCA-128 reference points:"
echo "    push  (combed, original)   cosine 0.902"
echo "    real_cubes                 cosine 0.847"
echo "  If the stride-1 PCA-128 run lands near 0.90, the comb does not"
echo "  affect cosine, and sim scores ABOVE real at matched settings --"
echo "  the reverse of what I reported from the mismatched pair."
echo "=============================================================="
