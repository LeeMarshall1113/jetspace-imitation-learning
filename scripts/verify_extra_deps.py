#!/usr/bin/env python3
"""Confirm the workspace-installed deps work and did not shadow the venv.

    PYTHONPATH=/workspace/.pydeps python scripts/verify_extra_deps.py

timm and einops have to be installed to /workspace/.pydeps because the
container runs as non-root and /opt/venv is read-only. That directory goes on
PYTHONPATH ahead of the venv, so anything pip drags in alongside them shadows
the real installation. The first attempt did exactly that -- it pulled fsspec
2026.7.0, which `datasets 5.0.1` refuses -- so this checks that the versions
actually imported are still the venv's.
"""

from __future__ import annotations

import sys


def main() -> int:
    print("PYTHONPATH entries:")
    for p in sys.path[:3]:
        print(f"  {p}")
    print()

    ok = True
    for name in ("timm", "einops"):
        try:
            m = __import__(name)
            print(f"{name:12s} {getattr(m, '__version__', '?'):12s} {m.__file__}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:12s} FAILED: {e}")
            ok = False

    print()
    print("venv packages that must NOT have been shadowed:")
    for name in ("datasets", "fsspec", "torch", "transformers", "numpy"):
        try:
            m = __import__(name)
            where = "pydeps" if ".pydeps" in (m.__file__ or "") else "venv"
            flag = "  <-- SHADOWED" if where == "pydeps" else ""
            print(f"  {name:14s} {getattr(m, '__version__', '?'):14s} {where}{flag}")
            if where == "pydeps":
                ok = False
        except Exception as e:  # noqa: BLE001
            print(f"  {name:14s} FAILED: {e}")
            ok = False

    print()
    print("all clear" if ok else "PROBLEM: a venv package was shadowed or failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
