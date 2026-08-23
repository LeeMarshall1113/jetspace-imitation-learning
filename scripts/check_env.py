#!/usr/bin/env python3
"""Verify the JetSpace stack end to end.

Run this first, in the container, before anything else:
    python scripts/check_env.py

It answers the only questions that matter on day one:
  1. Does torch see the Radeon through ROCm (and via which path, kfd or dxg)?
  2. Does a real matmul actually execute on the GPU?
  3. Does MuJoCo step and render headless?

Exit code is non-zero if any REQUIRED check fails, so CI can gate on it.
"""

from __future__ import annotations

import os
import platform
import sys

PASS, FAIL, WARN = "  [ok]  ", "  [FAIL]", "  [warn]"
_failures: list[str] = []


def _report(
    ok: bool, label: str, detail: str = "", hint: str = "", required: bool = True
) -> bool:
    """Report one check.

    `detail` is context worth seeing either way (a measurement, a version).
    `hint` is remediation and prints only on failure -- otherwise a passing run
    tells you to go fix things that are not broken.
    """
    if ok:
        mark = PASS
    elif required:
        mark = FAIL
        _failures.append(label)
    else:
        mark = WARN
    suffix = detail if ok else " ".join(x for x in (detail, hint) if x)
    print(f"{mark} {label}" + (f" - {suffix}" if suffix else ""))
    return ok


def check_host() -> None:
    print("\n== host ==")
    print(f"         python {platform.python_version()}  |  {platform.platform()}")
    # WSL2 reaches the GPU through /dev/dxg + ROCDXG, not the usual /dev/kfd.
    in_wsl = "microsoft" in platform.release().lower() or os.path.exists("/dev/dxg")
    if in_wsl:
        _report(
            os.path.exists("/dev/dxg"),
            "WSL2 GPU device /dev/dxg",
            hint="update Adrenalin to 26.2.2+ on the Windows host",
        )
        _report(
            os.path.exists("/usr/lib/libdxcore.so"),
            "libdxcore.so bind-mounted",
            hint="from /usr/lib/wsl/lib on the host (wsl2 compose profile)",
        )
        # The ROCDXG translation layer is the piece people miss: unlike NVIDIA,
        # AMD's Windows driver does not deliver a compute runtime into WSL. It
        # comes from https://github.com/ROCm/librocdxg, installed in the DISTRO.
        _report(
            os.path.exists("/usr/lib/librocdxg.so"),
            "librocdxg.so bind-mounted (ROCDXG bridge)",
            hint="run scripts/install_rocm_wsl.sh in the distro, not in this container",
        )
        # Upstream lists dids.conf among the required mounts, but librocdxg 1.2.0
        # does not ship it. Informational only - do not make this required.
        _report(
            os.path.exists("/usr/share/rocdxg/dids.conf"),
            "rocdxg dids.conf (absent in librocdxg 1.2.0, not required)",
            required=False,
        )
        # Mandatory below ROCm 7.13. Silently absent, it looks identical to a
        # driver problem: everything else passes and the GPU simply never appears.
        _report(
            os.environ.get("HSA_ENABLE_DXG_DETECTION") == "1",
            "HSA_ENABLE_DXG_DETECTION=1",
            hint="required for ROCm < 7.13; set in the wsl2 compose profile",
        )
    else:
        _report(os.path.exists("/dev/kfd"), "ROCm kernel driver /dev/kfd")
        _report(os.path.exists("/dev/dri"), "GPU render nodes /dev/dri")
    print(f"         GPU path: {'WSL2 / dxg / ROCDXG' if in_wsl else 'native / kfd'}")


def check_torch() -> None:
    print("\n== torch ==")
    try:
        import torch
    except ImportError as exc:
        _report(False, "import torch", str(exc))
        return

    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    print(f"         torch {torch.__version__}  |  hip={hip}  cuda={cuda}")
    _report(hip is not None, "torch is a ROCm build",
            hint="a CPU/CUDA wheel replaced the ROCm one")

    if not _report(torch.cuda.is_available(), "GPU visible to torch"):
        return

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gfx = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "?")
    print(f"         device 0: {name}  |  {vram:.1f} GiB  |  arch {gfx}")
    _report(vram >= 12, "VRAM >= 12 GiB", f"{vram:.1f} GiB",
            hint="see REQUIREMENTS.md", required=False)

    # Availability lies more often than a real kernel launch does.
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        _report(bool(torch.isfinite((a @ a).float().sum())), "bf16 matmul executes on GPU")
    except Exception as exc:  # noqa: BLE001 - surface whatever ROCm threw
        _report(False, "bf16 matmul executes on GPU", f"{type(exc).__name__}: {exc}")


def check_mujoco() -> None:
    print("\n== mujoco ==")
    try:
        import mujoco
    except ImportError as exc:
        _report(False, "import mujoco", str(exc))
        return
    print(f"         mujoco {mujoco.__version__}  |  MUJOCO_GL={os.environ.get('MUJOCO_GL', 'unset')}")

    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
             <light pos="0 0 3"/>
             <body pos="0 0 1"><joint type="free"/><geom type="sphere" size=".1"/></body>
           </worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    _report(data.time > 0, "physics steps", f"t={data.time:.3f}s after 100 steps")

    # Headless rendering is what breaks first on a fresh box; it is also what
    # every pixel-based policy depends on, so treat it as required.
    try:
        with mujoco.Renderer(model, height=64, width=64) as r:
            r.update_scene(data)
            frame = r.render()
        _report(frame.shape == (64, 64, 3), "headless render (EGL)", f"frame {frame.shape}")
    except Exception as exc:  # noqa: BLE001
        _report(False, "headless render (EGL)", f"{type(exc).__name__}: {exc}")


def main() -> int:
    print("=" * 62)
    print("JetSpace environment check")
    print("=" * 62)
    check_host()
    check_torch()
    check_mujoco()

    print("\n" + "=" * 62)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        print("See docs/setup.md for the fix for each check.")
        return 1
    print("All required checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
