"""Device selection that does not assume CUDA."""

from __future__ import annotations


def get_device(prefer: str = "auto") -> str:
    """Return the torch device string.

    ROCm builds of torch report themselves through the `cuda` namespace, so
    `torch.cuda.is_available()` is the correct check on a Radeon too. Use
    `is_rocm()` when you need to branch on the actual backend.
    """
    import torch

    if prefer != "auto":
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_rocm() -> bool:
    import torch

    return getattr(torch.version, "hip", None) is not None


def describe() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    props = torch.cuda.get_device_properties(0)
    backend = "rocm" if is_rocm() else "cuda"
    return f"{props.name} ({backend}, {props.total_memory / 1024**3:.1f} GiB)"
