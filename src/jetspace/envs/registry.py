"""Task registry.

One place that knows which environments exist and which scripted expert drives
each. Scripts take `--task <name>` and stay task-agnostic, which is what makes
the data-efficiency experiment in docs/task-hierarchy.md tractable: adding a
task should not mean editing the collector, the trainer and the evaluator.
"""

from __future__ import annotations

from typing import Any, Callable

TaskSpec = dict[str, Any]


def _reach() -> TaskSpec:
    from .so101_env import ReachExpert, SO101ReachEnv

    return {"env": SO101ReachEnv, "expert": ReachExpert, "default_steps": 150}


def _pickplace() -> TaskSpec:
    from .so101_pickplace import PickPlaceExpert, SO101PickPlaceEnv

    return {"env": SO101PickPlaceEnv, "expert": PickPlaceExpert, "default_steps": 400}


def _push() -> TaskSpec:
    from .so101_push import PushExpert, SO101PushEnv

    return {"env": SO101PushEnv, "expert": PushExpert, "default_steps": 300}


# Ordered by level in docs/task-hierarchy.md, not alphabetically: the order is
# the curriculum.
TASKS: dict[str, Callable[[], TaskSpec]] = {
    "reach": _reach,          # L0 - arm dynamics
    "push": _push,            # L1 - contact, non-prehensile
    "pickplace": _pickplace,  # L2 - grasp, payload, regime change
}


def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name]()
