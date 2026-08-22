# JetSpace dev image.
#
# Base: ROCm 7.2.4 / Ubuntu 24.04 / Python 3.12 / PyTorch 2.10.
#   - ROCm >= 7.2.1 is the floor for RX 9000-series (gfx1201) support under WSL2.
#   - Ubuntu 24.04 is also the ROS 2 Jazzy target, so one image can host both
#     the learning stack and the eventual real-robot bridge.
# Newer rocm7.14_* tags exist (Ubuntu 26.04 / PyTorch 2.12) if you want to move up.
ARG BASE_IMAGE=rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=egl

# MuJoCo needs EGL/OSMesa to render headless; the rest is teleop + video I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglfw3 libosmesa6 \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg \
        git git-lfs curl \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Dependency layer first so source edits don't invalidate the pip cache.
# README.md is copied because pyproject declares it as the project readme.
COPY pyproject.toml README.md /workspace/
# An editable install resolves `where = ["src"]` at install time, so the package
# tree has to exist already. Stub it here to keep this layer cacheable; the real
# source is copied over the stub below.
RUN mkdir -p /workspace/src/jetspace && touch /workspace/src/jetspace/__init__.py
# torch is deliberately not a declared dependency: the base image already ships
# the ROCm build, and a plain `pip install torch` would replace it with the
# CPU/CUDA wheel.
RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]" \
    && python -c "import torch; print('torch', torch.__version__)"

COPY . /workspace

# Confirm the editable install still resolves subpackages now that the real
# source has replaced the stub. Fails the build rather than surfacing at runtime.
RUN python -c "import jetspace, jetspace.envs, jetspace.utils; print('jetspace', jetspace.__version__)"

CMD ["/bin/bash"]
