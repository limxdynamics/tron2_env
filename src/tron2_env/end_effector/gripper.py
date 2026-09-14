"""Two-finger gripper adapter for the robot-wide provider interface."""

from __future__ import annotations

import copy
import inspect
import threading
import time
from typing import Optional

import numpy as np

from tron2_env.errors import CommandError, StateError
from tron2_env.joints import JointIndex
from tron2_env.transport.base import GripperStateTransport


class GripperProvider:
    """Adapt normalized policy openings to the existing percentage command."""

    name = "gripper"
    width_per_side = 1

    def __init__(self, transport) -> None:
        has_independent_state = isinstance(transport, GripperStateTransport)
        has_legacy_state = callable(getattr(transport, "get_joint_state", None))
        if (
            not callable(getattr(transport, "set_gripper", None))
            or not (has_independent_state or has_legacy_state)
        ):
            raise TypeError("gripper provider requires gripper state and command transport")
        self._transport = transport
        self._has_independent_state = has_independent_state
        self._lifecycle_lock = threading.RLock()
        self._paused = False
        self._closed = False

    @staticmethod
    def _opening(value, side: str) -> float:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CommandError(
                f"{side} gripper must be one finite value in [0, 1]"
            ) from exc
        if array.shape == ():
            opening = float(array)
        elif array.shape == (1,):
            opening = float(array[0])
        else:
            raise CommandError(
                f"{side} gripper must be one finite value in [0, 1]"
            )
        if not np.isfinite(opening) or not 0.0 <= opening <= 1.0:
            raise CommandError(
                f"{side} gripper must be one finite value in [0, 1]"
            )
        return opening

    def read_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> tuple[np.ndarray, dict]:
        if self._has_independent_state:
            state = self._transport.get_gripper_state(
                timeout=timeout,
                max_age=max_age,
            )
            expected_width = 2
        else:
            getter = self._transport.get_joint_state
            try:
                parameters = inspect.signature(getter).parameters.values()
            except (TypeError, ValueError):
                supports_max_age = True
            else:
                supports_max_age = any(
                    parameter.name == "max_age"
                    or parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            if max_age is not None and supports_max_age:
                state = getter(timeout=timeout, max_age=max_age)
            else:
                state = getter(timeout=timeout)
            if max_age is not None:
                try:
                    received_at_value = float(state["received_at"])
                    max_age_value = float(max_age)
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise StateError("legacy joint state has invalid freshness") from exc
                if (
                    not np.isfinite(received_at_value)
                    or not np.isfinite(max_age_value)
                    or max_age_value < 0.0
                ):
                    raise StateError("legacy joint state has invalid freshness")
                if time.monotonic() - received_at_value > max_age_value:
                    raise StateError("legacy joint state is stale")
            expected_width = JointIndex.STATE_DIM
        try:
            values = np.asarray(state["states"], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise StateError("gripper provider received invalid state") from exc
        if values.shape != (expected_width,) or not np.all(np.isfinite(values)):
            raise StateError("gripper provider received invalid state")
        if not self._has_independent_state:
            values = values[[JointIndex.LEFT_GRIPPER, JointIndex.RIGHT_GRIPPER]]
        metadata = {
            key: copy.deepcopy(value)
            for key, value in state.items()
            if key != "states"
        }
        return values.copy(), metadata

    def command(self, left: np.ndarray, right: np.ndarray) -> None:
        with self._lifecycle_lock:
            self._raise_if_command_blocked()
            left_target, right_target = self.validate_command(left, right)
            self._transport.set_gripper(
                float(left_target[0]) * 100.0,
                float(right_target[0]) * 100.0,
            )

    def validate_command(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate paired normalized openings without touching provider state."""
        return (
            np.array([self._opening(left, "left")], dtype=np.float64),
            np.array([self._opening(right, "right")], dtype=np.float64),
        )

    def pause(self) -> None:
        with self._lifecycle_lock:
            if not self._closed:
                self._paused = True

    def resume(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("gripper provider is closed")
            self._paused = False

    def close(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
            self._paused = False

    def raise_if_faulted(self) -> None:
        return None

    def _raise_if_command_blocked(self) -> None:
        if self._closed:
            raise RuntimeError("gripper provider is closed")
        if self._paused:
            raise RuntimeError("gripper provider is paused")
