"""Rolling-origin (expanding window) schedules, fixed in docs/protocol.md section 5.

Indices are 1-based, as in the protocol: origin ``t`` means the last observed index, the training window is
``y_1..y_t`` and the target of horizon ``h`` is ``y_{t+h}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Schedule:
    name: str
    T: int                       # series length in the confirmatory window
    m: int                       # seasonal period
    horizons: tuple[int, ...]
    spacing: int                 # distance between consecutive origins
    val_origins: tuple[int, ...]
    test_origins: tuple[int, ...]
    n_cal: int                   # split-conformal calibration length (protocol section 8)

    @property
    def H(self) -> int:
        return max(self.horizons)

    def validate(self) -> None:
        """Assert the structural rules of protocol section 5 (raises AssertionError on violation)."""
        assert list(self.test_origins) == sorted(self.test_origins)
        assert list(self.val_origins) == sorted(self.val_origins)
        assert len(self.test_origins) >= 5 and len(self.val_origins) >= 1
        assert self.test_origins[-1] + self.H == self.T, "last test origin + H must equal T"
        assert all(b - a == self.spacing for a, b in zip(self.test_origins, self.test_origins[1:]))
        assert all(b - a == self.spacing for a, b in zip(self.val_origins, self.val_origins[1:]))
        # rule (ii): validation targets never go beyond the first test origin
        assert self.val_origins[-1] + self.H <= self.test_origins[0]
        # a calibration window must fit inside the shortest training window used
        assert self.val_origins[0] - self.n_cal > self.m


MONTHLY = Schedule(
    name="monthly", T=405, m=12, horizons=(1, 3, 6, 12), spacing=6,
    val_origins=(300, 306, 312, 318, 324),
    test_origins=(351, 357, 363, 369, 375, 381, 387, 393),
    n_cal=36,
)

DAILY = Schedule(
    name="daily", T=1096, m=7, horizons=(1, 7, 14, 28), spacing=28,
    val_origins=(732, 760, 788, 816, 844),
    test_origins=(872, 900, 928, 956, 984, 1012, 1040, 1068),
    n_cal=112,
)

SCHEDULES = {"monthly": MONTHLY, "daily": DAILY}
for _s in SCHEDULES.values():
    _s.validate()


def origin_split(schedule: Schedule, origin: int, h: int) -> tuple[np.ndarray, int]:
    """Zero-based training index array (y_1..y_origin) and the 1-based target index ``origin + h``."""
    if h not in schedule.horizons:
        raise ValueError(f"horizon {h} not in schedule {schedule.horizons}")
    return np.arange(origin), origin + h


def iter_splits(schedule: Schedule, stage: str) -> Iterator[tuple[int, int, np.ndarray, int]]:
    """Yield (origin, h, train_index_zero_based, target_index_one_based) for stage 'validation' or 'test'."""
    if stage == "validation":
        origins = schedule.val_origins
    elif stage == "test":
        origins = schedule.test_origins
    else:
        raise ValueError("stage must be 'validation' or 'test'")
    for t in origins:
        for h in schedule.horizons:
            train, target = origin_split(schedule, t, h)
            yield t, h, train, target
