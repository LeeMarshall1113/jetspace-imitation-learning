#!/usr/bin/env bash
# Can the robotics-specific encoders be installed at all?
#
#   bash scripts/install_robot_encoders.sh
#
# A manipulation benchmark that compares only general-vision encoders invites
# the obvious question: where are the models trained for manipulation? R3M
# (CoRL 2022), VC-1 (NeurIPS 2023) and Theia (CoRL 2024) are the ones a
# robotics reviewer expects to see, and their absence is a credibility problem
# rather than a coverage gap.
#
# They are awkward in different ways, which is why this probes rather than
# assumes:
#
#   VC-1    on the Hub, but as a raw MAE ViT-B/16 checkpoint with a hydra
#           config, not a transformers model. Needs timm to rebuild the
#           architecture before the weights mean anything.
#   R3M     not on the Hub under any name that resolves. Ships as a GitHub
#           package whose weights historically came from Google Drive, which
#           may or may not still work unattended.
#   Theia   on the Hub with custom modelling code, so it needs
#           trust_remote_code.
set -uo pipefail
cd "$(dirname "$0")/.."

DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000

echo "### timm (required to rebuild the VC-1 architecture)"
$DC pip install -q timm 2>&1 | tail -2
$DC python -c "import timm; print('  timm', timm.__version__)" 2>&1 | tail -1

echo
echo "### R3M from GitHub"
$DC bash -c 'timeout 240 pip install -q git+https://github.com/facebookresearch/r3m 2>&1 | tail -3; python -c "import r3m; print(\"  r3m import OK\")" 2>&1 | tail -1'

echo
echo "### Theia (trust_remote_code)"
$DC python -c "
from transformers import AutoModel
try:
    m = AutoModel.from_pretrained('theaiinstitute/theia-base-patch16-224-cddsv',
                                  trust_remote_code=True)
    print('  theia OK', sum(p.numel() for p in m.parameters()) / 1e6, 'M')
except Exception as e:
    print('  theia FAILED:', str(e).split(chr(10))[0][:110])
" 2>&1 | tail -2
