"""Thread-safe storage for the latest validated component state."""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional

import numpy as np

from tron2_env.errors import StateError


@dataclass(frozen=True)
class SnapshotValue:
    """One component sample with robot and local receive timestamps."""

    values: np.ndarray
    robot_timestamp: Optional[float]
    received_at: float
    metadata: dict


@dataclass(eq=False)
class _PredicateWaiter:
    predicate: Callable[[SnapshotValue], bool]
    value: Optional[SnapshotValue] = None
    error: Optional[BaseException] = None


class LatestSnapshot:
    """Keep the latest finite fixed-width sample and wake waiting readers."""

    def __init__(self, width: int, name: str) -> None:
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        self._width = width
        self._name = str(name)
        self._condition = threading.Condition()
        self._value: Optional[SnapshotValue] = None
        self._predicate_waiters: List[_PredicateWaiter] = []

    def update(
        self,
        values,
        *,
        robot_timestamp=None,
        received_at=None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Validate and atomically replace the current sample."""
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{self._name} must be finite shape ({self._width},)"
            ) from exc
        if array.shape != (self._width,) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{self._name} must be finite shape ({self._width},)"
            )

        received = time.monotonic() if received_at is None else float(received_at)
        if not math.isfinite(received):
            raise ValueError("received_at must be finite")

        stored_values = array.copy()
        stored_values.setflags(write=False)
        stored_metadata = copy.deepcopy({} if metadata is None else dict(metadata))
        item = SnapshotValue(
            values=stored_values,
            robot_timestamp=robot_timestamp,
            received_at=received,
            metadata=stored_metadata,
        )
        with self._condition:
            self._value = item
            for waiter in self._predicate_waiters:
                if waiter.value is not None or waiter.error is not None:
                    continue
                try:
                    matches = waiter.predicate(item)
                except BaseException as exc:
                    waiter.error = exc
                else:
                    if matches:
                        waiter.value = item
            self._condition.notify_all()

    def clear(self) -> None:
        """Atomically invalidate the current sample.

        Existing and future readers continue waiting for an update from the
        new producer session instead of observing a value from the old one.
        """
        with self._condition:
            self._value = None
            for waiter in self._predicate_waiters:
                waiter.value = None
            self._condition.notify_all()

    def get(
        self,
        timeout: float = 0.0,
        max_age: Optional[float] = None,
        now: Optional[float] = None,
        predicate: Optional[Callable[[SnapshotValue], bool]] = None,
    ) -> SnapshotValue:
        """Return a defensive copy, waiting for a value accepted by ``predicate``."""
        timeout_value = float(timeout)
        if not math.isfinite(timeout_value) or timeout_value < 0:
            raise ValueError("timeout must be finite and non-negative")
        if timeout_value > threading.TIMEOUT_MAX:
            raise ValueError("timeout must not exceed threading.TIMEOUT_MAX")

        max_age_value = None if max_age is None else float(max_age)
        if max_age_value is not None and (
            not math.isfinite(max_age_value) or max_age_value < 0
        ):
            raise ValueError("max_age must be finite and non-negative")

        requested_now = None if now is None else float(now)
        if requested_now is not None and not math.isfinite(requested_now):
            raise ValueError("now must be finite")
        if predicate is not None and not callable(predicate):
            raise TypeError("predicate must be callable")

        deadline = time.monotonic() + timeout_value
        with self._condition:
            if predicate is None:
                while self._value is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise StateError(f"{self._name} state timed out")
                    self._condition.wait(min(threading.TIMEOUT_MAX, remaining))
                item = self._value
            else:
                item = self._value
                if item is None or not predicate(item):
                    waiter = _PredicateWaiter(predicate=predicate)
                    self._predicate_waiters.append(waiter)
                    try:
                        while waiter.value is None and waiter.error is None:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise StateError(f"{self._name} state timed out")
                            self._condition.wait(
                                min(threading.TIMEOUT_MAX, remaining)
                            )
                        if waiter.error is not None:
                            raise waiter.error
                        item = waiter.value
                    finally:
                        self._predicate_waiters.remove(waiter)

        assert item is not None

        current = time.monotonic() if requested_now is None else requested_now
        if not math.isfinite(current):
            raise ValueError("now must be finite")
        if (
            max_age_value is not None
            and current - item.received_at > max_age_value
        ):
            raise StateError(f"{self._name} state is stale")
        return SnapshotValue(
            values=item.values.copy(),
            robot_timestamp=item.robot_timestamp,
            received_at=item.received_at,
            metadata=copy.deepcopy(item.metadata),
        )
