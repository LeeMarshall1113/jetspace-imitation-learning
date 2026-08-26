#!/usr/bin/env bash
# Install the extra packages the robotics encoders need, into the workspace.
#
#   bash scripts/setup_extra_deps.sh
#
# The container runs as a non-root user and /opt/venv is not writable, so pip
# cannot install into the image. Installing into /workspace/.pydeps instead
# keeps them on the mounted volume, which means they survive between
# `docker compose run` invocations without rebuilding the image.
#
#   timm     rebuilds the VC-1 architecture; the Hub ships only an MAE
#            ViT-B/16 state dict plus a hydra config, not a loadable model
#   einops   required by Theia's remote modelling code
set -uo pipefail
cd "$(dirname "$0")/.."

DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000

$DC bash -c '
set -e
mkdir -p /workspace/.pydeps
pip install -q --target /workspace/.pydeps timm einops 2>&1 | tail -3
export PYTHONPATH=/workspace/.pydeps
python -c "import timm, einops; print(\"timm\", timm.__version__, \"einops\", einops.__version__)"
'
echo
echo "add PYTHONPATH=/workspace/.pydeps to any run that needs these"
