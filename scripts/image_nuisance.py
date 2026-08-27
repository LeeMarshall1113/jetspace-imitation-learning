#!/usr/bin/env python3
"""Image-space nuisance axes, applied at encode time.

E12's axes each cost a rendering pass, and only two of the four survived their
own controls. Scaling that way is slow: every new axis means re-collecting
thirteen conditions before a single encoder sees them.

But a whole class of real deployment nuisances lives in image space, not in the
simulator. A camera that is slightly out of focus, a sensor with read noise, a
compressed video stream, an auto-exposure that drifted -- none of those need the
physics re-run. They are transforms of frames already on disk, so they can be
applied at encode time and cost nothing to store.

That is what makes a benchmark-shaped axis count reachable. These are not
substitutes for rendered nuisances -- a blurred render is not a real defocus,
and this is stated in the limitations rather than glossed -- but they are the
same *kind* of perturbation a deployed policy meets, and they discriminate
between encoders for the same reasons.

Every transform takes uint8 (H, W, 3) and returns uint8 (H, W, 3), so it drops
into the caching path ahead of the processor without touching any encoder.
"""

from __future__ import annotations

import numpy as np


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def sensor_noise(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian read noise. `level` is the sigma in 0-255 units.

    Deterministic per frame given the generator, so two encoders see the SAME
    corrupted pixels -- otherwise the arms would differ by their noise draw
    rather than by their features.
    """
    return _clip(img.astype(np.float32) + rng.normal(0, level, img.shape))


def defocus(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Separable box blur, `level` giving the radius in pixels.

    A box blur rather than a Gaussian because it is exact in integer arithmetic
    and needs no kernel library; the point is a controlled loss of high spatial
    frequency, not optical fidelity.
    """
    r = max(1, int(round(level)))
    k = 2 * r + 1
    x = img.astype(np.float32)

    def box(a, axis):
        # Sliding window sum via cumulative sums. The zero prepended before
        # differencing is what makes the output the same width as the padded
        # input minus 2r -- without it every pass loses a pixel.
        pad_w = [(0, 0)] * a.ndim
        pad_w[axis] = (r, r)
        b = np.pad(a, pad_w, mode="edge")
        cs = np.cumsum(b, axis=axis)
        zeros = np.zeros_like(np.take(cs, [0], axis=axis))
        cs = np.concatenate([zeros, cs], axis=axis)
        hi = np.take(cs, range(k, cs.shape[axis]), axis=axis)
        lo = np.take(cs, range(0, cs.shape[axis] - k), axis=axis)
        return (hi - lo) / k

    x = box(box(x, 1), 0)
    return _clip(x)


def compression(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Blockiness, as a stand-in for a low-bitrate video stream.

    `level` is the block edge in pixels. Averaging within blocks reproduces the
    dominant artefact of aggressive compression -- loss of detail at a fixed
    spatial scale -- without needing a codec in the container.
    """
    b = max(1, int(round(level)))
    h, w, _ = img.shape
    x = img.astype(np.float32)
    hh, ww = (h // b) * b, (w // b) * b
    blocks = x[:hh, :ww].reshape(hh // b, b, ww // b, b, 3).mean(axis=(1, 3))
    out = np.repeat(np.repeat(blocks, b, axis=0), b, axis=1)
    x[:hh, :ww] = out
    return _clip(x)


def exposure(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative gain, as an auto-exposure that settled wrong.

    Distinct from E12's lighting axis: that changes the SCENE illumination
    before rendering, so shadows and specular highlights move with it. This
    changes only what the sensor reports, leaving the physics of the light
    untouched. Whether encoders treat the two alike is itself a question the
    benchmark can answer.
    """
    return _clip(img.astype(np.float32) * level)


def downsample(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Resolution loss: decimate by `level` and scale back by pixel repeat."""
    f = max(1, int(round(level)))
    small = img[::f, ::f]
    up = np.repeat(np.repeat(small, f, axis=0), f, axis=1)
    return up[: img.shape[0], : img.shape[1]]


#: axis -> (function, levels). Levels are ordered from mild to severe and were
#: chosen so the mildest is visible and the harshest is not yet destructive;
#: an axis that saturates ranks nothing, exactly as clutter failed to.
AXES = {
    "noise": (sensor_noise, [4.0, 10.0, 20.0, 35.0]),
    "defocus": (defocus, [1, 2, 4, 7]),
    "compress": (compression, [4, 8, 14, 22]),
    "exposure": (exposure, [0.65, 0.80, 1.25, 1.55]),
    "lowres": (downsample, [2, 3, 5, 8]),
}


def apply_axis(frames: np.ndarray, axis: str, level, seed: int = 0) -> np.ndarray:
    """(T, H, W, 3) uint8 -> same, with the nuisance applied per frame.

    The generator is seeded from the axis, level and frame index so the
    corruption is reproducible and IDENTICAL across encoders. An unseeded draw
    would let two arms differ by their noise rather than by their features,
    which is the confound this whole comparison exists to avoid.
    """
    fn, _ = AXES[axis]
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        rng = np.random.default_rng((seed, hash(axis) % 2**31, int(level * 1000), i))
        out[i] = fn(f, level, rng)
    return out


def tag_for(axis: str, level) -> str:
    return f"{axis}_{str(level).replace('.', 'p')}"
