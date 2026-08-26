"""EpisodeBuffer -> EpisodeWriter -> EpisodeDataset, end to end.

There was no coverage of this path at all, which is how PR #9 could refactor
`EpisodeBuffer` into a separate `Episode` dataclass, break both callers
(`collect_demos.py` and `fetch_lerobot.py` with
`TypeError: object of type 'Episode' has no len()`), and still look mergeable.

The contract these tests pin down:

  * whatever `EpisodeBuffer` is internally, `EpisodeWriter.write` must accept
    what the collection scripts actually hand it;
  * `action` and `action_executed` are DIFFERENT fields and both survive the
    round trip -- ledger L5 and L6 are two separate bugs that came from
    conflating them, and a refactor that silently drops one would pass any test
    that only checks `action`;
  * actions stay float64. Quantising to float32 costs ~3e-08 rad, which the
    dynamics amplify ~6300x over a 17-step episode, which is enough to break
    exact replay verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import (  # noqa: E402
    EpisodeBuffer,
    EpisodeDataset,
    EpisodeWriter,
)

CAM = "front"


def fill(buffer: EpisodeBuffer, n: int = 5, *, distinct_executed: bool = True):
    """Populate a buffer with recognisable per-step values."""
    for i in range(n):
        action = np.full(6, float(i), dtype=np.float64)
        executed = action + 0.25 if distinct_executed else None
        buffer.add(
            pixels={CAM: np.full((4, 4, 3), i, dtype=np.uint8)},
            proprio=np.full(12, float(i), dtype=np.float32),
            action=action,
            action_executed=executed,
            reward=float(i),
            success=(i == n - 1),
        )
    return buffer


def make_writer(root: Path) -> EpisodeWriter:
    return EpisodeWriter(
        root, task="unit", fps=25, action_dim=6,
        cameras=(CAM,), image_size=4,
    )


def test_buffer_reports_its_length(tmp_path: Path):
    """`EpisodeWriter.write` calls len() on what it is given, so whatever the
    scripts pass must support it. This is the exact assertion PR #9 failed."""
    buf = fill(EpisodeBuffer(), 5)
    writer = make_writer(tmp_path)
    # Whatever the internal layout, this call is the one collect_demos.py makes.
    path = writer.write(buf, metadata={"policy": "test", "seed": 1})
    assert path.exists()


def test_roundtrip_preserves_values(tmp_path: Path):
    buf = fill(EpisodeBuffer(), 5)
    writer = make_writer(tmp_path)
    writer.write(buf, metadata={"policy": "test", "seed": 7})

    ds = EpisodeDataset(tmp_path)
    assert len(ds) == 1
    ep = ds[0]
    assert ep["action"].shape == (5, 6)
    np.testing.assert_allclose(ep["action"][:, 0], np.arange(5, dtype=np.float64))
    assert ep[f"pixels_{CAM}"].shape == (5, 4, 4, 3)


def test_action_and_executed_stay_distinct(tmp_path: Path):
    """Ledger L5/L6: label and executed action are different quantities.
    Learning needs `action`; replay verification needs `action_executed`.
    Collapsing them loses a property the project depends on."""
    buf = fill(EpisodeBuffer(), 4, distinct_executed=True)
    writer = make_writer(tmp_path)
    writer.write(buf)

    ep = EpisodeDataset(tmp_path)[0]
    assert not np.allclose(ep["action"], ep["action_executed"])
    np.testing.assert_allclose(ep["action_executed"] - ep["action"], 0.25)


def test_actions_are_float64(tmp_path: Path):
    """float32 would cost ~3e-08 rad, amplified ~6300x over an episode."""
    buf = fill(EpisodeBuffer(), 3)
    writer = make_writer(tmp_path)
    writer.write(buf)
    ep = EpisodeDataset(tmp_path)[0]
    assert ep["action"].dtype == np.float64
    assert ep["action_executed"].dtype == np.float64


def test_refuses_to_write_an_empty_episode(tmp_path: Path):
    writer = make_writer(tmp_path)
    with pytest.raises(ValueError):
        writer.write(EpisodeBuffer())


def test_index_records_length_and_success(tmp_path: Path):
    buf = fill(EpisodeBuffer(), 6)
    writer = make_writer(tmp_path)
    writer.write(buf, metadata={"policy": "scripted", "seed": 3})

    import json
    lines = (tmp_path / "episodes.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["length"] == 6
    assert rec["success"] is True
    assert rec["seed"] == 3


def test_reopening_continues_numbering(tmp_path: Path):
    """Demo collection happens across several sittings; reopening a dataset
    must append rather than overwrite episode_000000."""
    make_writer(tmp_path).write(fill(EpisodeBuffer(), 3))
    make_writer(tmp_path).write(fill(EpisodeBuffer(), 3))
    files = sorted(tmp_path.glob("episode_*.npz"))
    assert [f.name for f in files] == ["episode_000000.npz", "episode_000001.npz"]
