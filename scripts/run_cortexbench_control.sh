#!/usr/bin/env bash
# Run the E12d discriminability control on CortexBench's own demonstration data.
#
#   bash scripts/run_cortexbench_control.sh
#
# The point: E12 found five of eight nuisance axes in THIS project's benchmark
# cannot separate a trained encoder from an untrained CNN. The fair objection is
# that those axes are ours. This asks the same question of the benchmark the
# field actually uses to rank frozen encoders for embodied AI, on its published
# Adroit demonstrations.
#
# Needs the dataset from cortexbench/DATASETS.md unpacked at
# /home/lee-m/cortexbench-data/adroit-expert-v1.0 (1.7 GB zipped, 7.5 GB out).
set -uo pipefail
cd "$(dirname "$0")/.."
export DOCKER_UID=1000 DOCKER_GID=1000

DATA_HOST="${CBENCH_HOST_DATA:-/home/lee-m/cortexbench-data}"
DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T \
-e PYTHONPATH=/workspace/.pydeps \
-e CORTEXBENCH_DATA=/cbdata/adroit-expert-v1.0 \
-v ${DATA_HOST}:/cbdata dev-wsl"

# Every encoder that loads through the transformers path, plus the untrained
# control. V-JEPA 2 is absent on purpose: it is a video model reached through
# cache_latents.py, and mixing a different temporal rate into a single-frame
# comparison would confound representation quality with tubelet size.
MODELS="dinov2,dinov2-large,clip,clip-large,siglip2,aimv2,vit-in1k,vc1,\
vc1-large,convnext,resnet50,RANDOM"

for task in pen-v0 relocate-v0; do
    echo "########## ${task}  $(date +%T)"
    $DC python scripts/cortexbench_control.py \
        --task "${task}" --episodes 25 --models "${MODELS}" \
        --out "cache/cortexbench_${task}.json"
done
echo "########## done $(date +%T)"
