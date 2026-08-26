"""Episode recording and loading for teleoperated demonstrations.

Layout on disk, deliberately shaped like LeRobot's so conversion is mechanical:

    data/episodes/<task>/
        info.json           dataset-level metadata (fps, dims, cameras, ...)
        episodes.jsonl      one JSON object per episode (index, length, success)
        episode_000000.npz  arrays for one episode
        episode_000001.npz
        ...

Episodes are stored at their **natural length**, not padded or truncated. Demos
of the same task legitimately take different amounts of time, and Balaguer &
Carpin (see docs/papers/balaguer-carpin-2011.md) show that discarding that
variation loses real information about the action space. Fixing a horizon is a
*training-time* decision, so it belongs to the sampler, not the recorder --
`fps` and each episode's `length` are recorded here so that decision can be made
explicitly later rather than inherited by accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

INFO_FILE = "info.json"
INDEX_FILE = "episodes.jsonl"
EPISODE_GLOB = "episode_*.npz"


@dataclass
class Episode:
    pixels: dict[str, list[np.ndarray]] = field(default_factory=dict)
    proprio: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    action_executed: list[np.ndarray] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    success: list[bool] = field(default_factory=list)


class EpisodeBuffer:
    """Accumulates one episode in memory before it is flushed to disk."""

    def __init__(self) -> None:
        self.Episode: Episode = Episode()

    def buffer_size(self) -> int:
        return len(self.Episode.action)

    def add(
        self,
        *,
        pixels: dict[str, np.ndarray],
        proprio: np.ndarray,
        action: np.ndarray,
        reward: float,
        success: bool,
        action_executed: np.ndarray | None = None,
    ) -> None:
        
        for cam, frame in pixels.items():
            self.Episode.pixels.setdefault(cam, []).append(np.asarray(frame, dtype=np.uint8))

        self.Episode.proprio.append(np.asarray(proprio, dtype=np.float32))
        
        # Actions are stored float64, deliberately. Quantizing to float32 costs
        # ~3e-08 rad, which the dynamics amplify ~6300x over a 17-step episode
        # to ~2e-04 rad -- enough to break exact replay verification. Actions are
        # a rounding error in the byte budget next to 224x224x3 images, so there
        # is no reason to lose the precision.
        
        self.Episode.action.append(np.asarray(action, dtype=np.float64))
        self.Episode.action_executed.append(
            np.asarray(action if action_executed is None else action_executed, dtype=np.float64)
        )

        self.Episode.reward.append(float(reward))
        self.Episode.success.append(bool(success))


class EpisodeWriter:
    """Writes episodes into a dataset directory, appending to any existing set.

    Reopening an existing dataset continues numbering rather than overwriting,
    so demo collection can be done across several sittings.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        task: str,
        fps: int,
        action_dim: int,
        cameras: tuple[str, ...],
        image_size: int,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.info = {
            "task": task,
            "fps": fps,
            "action_dim": action_dim,
            "cameras": list(cameras),
            "image_size": image_size,
            "codebase": "jetspace",
            **(extra_info or {}),
        }
        info_path = self.root / INFO_FILE
        if info_path.exists():
            existing = json.loads(info_path.read_text())
            # Silently mixing 30 Hz and 60 Hz episodes into one dataset would
            # corrupt every downstream timing assumption. Fail loudly instead.
            for key in ("fps", "action_dim", "image_size", "task"):
                if existing.get(key) != self.info.get(key):
                    raise ValueError(
                        f"Dataset at {self.root} has {key}={existing.get(key)!r}, "
                        f"cannot append episodes with {key}={self.info.get(key)!r}"
                    )
        else:
            info_path.write_text(json.dumps(self.info, indent=2))
        self._next_index = len(list(self.root.glob(EPISODE_GLOB)))

    @property
    def num_episodes(self) -> int:
        return self._next_index

    def write(self, buffer: EpisodeBuffer, *, metadata: dict[str, Any] | None = None) -> Path:
        if (buffer.buffer_size()) == 0:
            raise ValueError("Refusing to write an empty episode")

        index = self._next_index
        path = self.root / f"episode_{index:06d}.npz"
        arrays: dict[str, np.ndarray] = {
            "proprio": np.stack(buffer.Episode.proprio),
            "action": np.stack(buffer.Episode.action),
            "action_executed": np.stack(buffer.Episode.action_executed),
            "reward": np.asarray(buffer.Episode.reward, dtype=np.float32),
            "success": np.asarray(buffer.Episode.success, dtype=bool),
        }
        for cam, frames in buffer.Episode.pixels.items():
            arrays[f"pixels_{cam}"] = np.stack(frames)
        np.savez_compressed(path, **arrays)

        record = {
            "index": index,
            "file": path.name,
            "length": len(buffer.buffer_size()),
            "duration_s": round(len(buffer.buffer_size()) / self.info["fps"], 3),
            "success": bool(buffer.Episode.success[-1]),
            "return": round(float(np.sum(buffer.Episode.reward)), 4),
            **(metadata or {}),
        }
        with (self.root / INDEX_FILE).open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        self._next_index += 1
        return path


class EpisodeDataset:
    """Reads a dataset written by `EpisodeWriter`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        info_path = self.root / INFO_FILE
        if not info_path.exists():
            raise FileNotFoundError(f"No {INFO_FILE} in {self.root}")
        self.info = json.loads(info_path.read_text())
        index_path = self.root / INDEX_FILE
        self.records = (
            [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
            if index_path.exists()
            else []
        )

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> dict[str, Any]:
        record = self.records[i]
        with np.load(self.root / record["file"]) as data:
            episode = {k: data[k] for k in data.files}
        episode["meta"] = record
        return episode

    # -- summary helpers ---------------------------------------------------
    @property
    def success_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(r["success"] for r in self.records) / len(self.records)

    def summary(self) -> str:
        if not self.records:
            return f"{self.root}: empty"
        lengths = [r["length"] for r in self.records]
        return (
            f"{self.root}\n"
            f"  task            {self.info['task']} @ {self.info['fps']} Hz\n"
            f"  episodes        {len(self.records)}\n"
            f"  success rate    {self.success_rate:.1%}\n"
            f"  length          min {min(lengths)}  max {max(lengths)}  "
            f"mean {sum(lengths) / len(lengths):.1f} frames\n"
            f"  duration        {min(lengths) / self.info['fps']:.2f}-"
            f"{max(lengths) / self.info['fps']:.2f} s"
        )
