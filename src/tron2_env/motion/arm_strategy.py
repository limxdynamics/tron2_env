"""Fixed arm-mode strategies used by :class:`MotionController`."""

from __future__ import annotations

import inspect
import time
from typing import ClassVar, Literal, Optional, Protocol, runtime_checkable

import numpy as np

from tron2_env.errors import StateError
from tron2_env.interpolation import (
    JointInterpolator,
    LinearInterpolator,
    PoseInterpolator,
)
from tron2_env.joints import JointIndex

ArmMode = Literal["servoj", "servop"]


def validate_finite_vector(value, *, width: int, name: str) -> np.ndarray:
    """Return an owned finite vector with one exact, documented width."""
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite shape ({width},)") from exc
    if vector.shape != (width,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite shape ({width},)")
    return vector.copy()


def state_to_servoj(state18) -> np.ndarray:
    """Convert canonical 18D feedback into a 16D ServoJ setpoint."""
    state = validate_finite_vector(
        state18,
        width=JointIndex.STATE_DIM,
        name="measured joint state",
    )
    return np.concatenate(
        (
            state[JointIndex.LEFT_ARM],
            state[JointIndex.RIGHT_ARM],
            state[JointIndex.HEAD],
        )
    )


def _state_values(payload, *, width: int, name: str) -> np.ndarray:
    try:
        values = payload["states"]
    except (KeyError, TypeError) as exc:
        raise StateError(f"{name} response has no valid 'states' vector") from exc
    try:
        return validate_finite_vector(values, width=width, name=name)
    except ValueError as exc:
        raise StateError(str(exc)) from exc


def _positive_timeout(timeout: float) -> float:
    try:
        result = float(timeout)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timeout must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("timeout must be finite and positive")
    return result


def _supports_keyword(callable_, keyword: str) -> bool:
    """Return whether an introspectable legacy hook accepts ``keyword``."""
    try:
        parameters = inspect.signature(callable_).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        (
            parameter.name == keyword
            and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
        )
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _read_correlated_pose_payload(
    transport,
    *,
    timeout: float,
    max_age: Optional[float],
) -> dict:
    """Read one fresh pose correlated with a single request GUID."""
    timeout_value = _positive_timeout(timeout)
    request = getattr(transport, "request_pose_state", None)
    getter = getattr(transport, "get_pose_state", None)
    if not callable(request) or not callable(getter):
        raise StateError("servop transport does not provide pose state")
    requested_guid = (
        request(timeout=timeout_value)
        if _supports_keyword(request, "timeout")
        else request()
    )
    if not isinstance(requested_guid, str) or not requested_guid.strip():
        raise StateError("pose state request returned an invalid GUID")

    getter_supports_expected_guid = _supports_keyword(getter, "expected_guid")
    deadline = time.monotonic() + timeout_value
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise StateError(
                "pose state timed out waiting for the requested response"
            )
        getter_kwargs = {"timeout": remaining, "max_age": max_age}
        if getter_supports_expected_guid:
            getter_kwargs["expected_guid"] = requested_guid
        try:
            payload = getter(**getter_kwargs)
        except StateError as exc:
            if getter_supports_expected_guid:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise StateError(
                    "pose state timed out waiting for the requested response"
                ) from exc
            time.sleep(min(0.001, remaining))
            continue
        if isinstance(payload, dict) and payload.get("guid") == requested_guid:
            return payload
        time.sleep(min(0.001, max(0.0, remaining)))


@runtime_checkable
class ArmStrategy(Protocol):
    """Mode-specific measured-state, interpolation, and publication policy."""

    mode: ArmMode
    target_dim: int

    def read_measured_target(
        self,
        timeout: float,
        max_age: Optional[float] = None,
    ) -> np.ndarray:
        """Read one safe measured seed for this arm mode."""

    def reset(self, measured: np.ndarray) -> None:
        """Replace the current trajectory with ``measured``."""

    def set_destination(self, target: np.ndarray, eta: float) -> None:
        """Retarget the active trajectory."""

    def publish_current(self) -> None:
        """Publish one interpolated setpoint through the selected primitive."""


class ServoJStrategy:
    """Joint-space strategy with independent arm/head feedback when available."""

    __slots__ = ("_transport", "_interpolator")

    mode: ClassVar[Literal["servoj"]] = "servoj"
    target_dim: ClassVar[int] = JointIndex.SERVOJ_DIM

    def __init__(
        self,
        transport,
        interpolator: JointInterpolator | None = None,
    ) -> None:
        if isinstance(interpolator, PoseInterpolator):
            raise ValueError("servoj mode does not accept a PoseInterpolator")
        if interpolator is not None and not isinstance(
            interpolator,
            JointInterpolator,
        ):
            raise TypeError("servoj interpolator must implement JointInterpolator")
        self._transport = transport
        self._interpolator = (
            LinearInterpolator() if interpolator is None else interpolator
        )

    def read_measured_target(
        self,
        timeout: float,
        max_age: Optional[float] = None,
    ) -> np.ndarray:
        timeout_value = _positive_timeout(timeout)
        independent_getter = getattr(self._transport, "get_arm_head_state", None)
        if callable(independent_getter):
            payload = self._read_state(
                independent_getter,
                timeout=timeout_value,
                max_age=max_age,
            )
            return _state_values(
                payload,
                width=self.target_dim,
                name="measured arm/head state",
            )

        payload = self._read_state(
            self._transport.get_joint_state,
            timeout=timeout_value,
            max_age=max_age,
        )
        return state_to_servoj(
            _state_values(
                payload,
                width=JointIndex.STATE_DIM,
                name="measured joint state",
            )
        )

    @staticmethod
    def _read_state(getter, *, timeout: float, max_age: Optional[float]) -> dict:
        if max_age is not None and _supports_keyword(getter, "max_age"):
            return getter(timeout=timeout, max_age=max_age)
        return getter(timeout=timeout)

    def reset(self, measured: np.ndarray) -> None:
        self._interpolator.reset(
            validate_finite_vector(
                measured,
                width=self.target_dim,
                name="measured ServoJ target",
            )
        )

    def set_destination(self, target: np.ndarray, eta: float) -> None:
        self._interpolator.set_destination(
            validate_finite_vector(
                target,
                width=self.target_dim,
                name="joint target",
            ),
            eta,
        )

    def publish_current(self) -> None:
        target = validate_finite_vector(
            self._interpolator.current(),
            width=self.target_dim,
            name="interpolated joint target",
        )
        self._transport.send_joint_cmd(target)


class ServoPStrategy:
    """Cartesian strategy for two XYZ + WXYZ arm poses."""

    __slots__ = ("_transport", "_interpolator")

    mode: ClassVar[Literal["servop"]] = "servop"
    target_dim: ClassVar[int] = 14

    def __init__(
        self,
        transport,
        interpolator: JointInterpolator | None = None,
    ) -> None:
        for method_name in ("request_pose_state", "get_pose_state", "send_pose_cmd"):
            if not callable(getattr(transport, method_name, None)):
                raise TypeError(
                    f"servop transport requires callable {method_name}()"
                )
        if interpolator is not None and not isinstance(interpolator, PoseInterpolator):
            raise ValueError("servop mode requires a PoseInterpolator")
        self._transport = transport
        self._interpolator = interpolator or PoseInterpolator()

    def read_measured_target(
        self,
        timeout: float,
        max_age: Optional[float] = None,
    ) -> np.ndarray:
        """Request and wait for the pose correlated with this specific query."""
        payload = _read_correlated_pose_payload(
            self._transport,
            timeout=timeout,
            max_age=max_age,
        )
        return _state_values(
            payload,
            width=self.target_dim,
            name="measured pose state",
        )

    def reset(self, measured: np.ndarray) -> None:
        self._interpolator.reset(
            validate_finite_vector(
                measured,
                width=self.target_dim,
                name="measured pose target",
            )
        )

    def set_destination(self, target: np.ndarray, eta: float) -> None:
        self._interpolator.set_destination(
            validate_finite_vector(
                target,
                width=self.target_dim,
                name="pose target",
            ),
            eta,
        )

    def publish_current(self) -> None:
        target = validate_finite_vector(
            self._interpolator.current(),
            width=self.target_dim,
            name="interpolated pose target",
        )
        self._transport.send_pose_cmd(target[:7], target[7:])


__all__ = [
    "ArmMode",
    "ArmStrategy",
    "ServoJStrategy",
    "ServoPStrategy",
]
