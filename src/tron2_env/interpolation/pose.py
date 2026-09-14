"""Cartesian pose interpolation for two robot arms."""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

_POSE_SHAPE = (14,)
_LEFT_XYZ = slice(0, 3)
_LEFT_QUATERNION = slice(3, 7)
_RIGHT_XYZ = slice(7, 10)
_RIGHT_QUATERNION = slice(10, 14)
_QUATERNION_SLICES = (_LEFT_QUATERNION, _RIGHT_QUATERNION)
_MIN_QUATERNION_NORM = 1e-8
_NLERP_DOT_THRESHOLD = 0.9995


def _normalize_quaternion(quaternion: np.ndarray, *, side: str) -> np.ndarray:
    """Return a unit quaternion while avoiding overflow in the norm."""

    scale = float(np.max(np.abs(quaternion)))
    if scale == 0.0:
        raise ValueError(f"{side} quaternion norm must be >= {_MIN_QUATERNION_NORM}")

    scaled = quaternion / scale
    scaled_norm = float(np.linalg.norm(scaled))
    if scale < _MIN_QUATERNION_NORM / scaled_norm:
        raise ValueError(f"{side} quaternion norm must be >= {_MIN_QUATERNION_NORM}")
    return scaled / scaled_norm


def validate_pose(pose: np.ndarray, *, name: str = "pose") -> np.ndarray:
    """Return an owned dual-arm pose with normalized valid WXYZ quaternions."""
    try:
        result = np.asarray(pose, dtype=np.float64).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric finite values") from exc
    if result.shape != _POSE_SHAPE:
        raise ValueError(f"{name} shape must be {_POSE_SHAPE}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")

    result[_LEFT_QUATERNION] = _normalize_quaternion(
        result[_LEFT_QUATERNION], side="left"
    )
    result[_RIGHT_QUATERNION] = _normalize_quaternion(
        result[_RIGHT_QUATERNION], side="right"
    )
    return result


def _slerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-path spherical interpolation between unit WXYZ quaternions."""

    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > _NLERP_DOT_THRESHOLD:
        blended = (1.0 - alpha) * start + alpha * end
    else:
        theta = float(np.arccos(dot))
        sin_theta = float(np.sin(theta))
        start_weight = np.sin((1.0 - alpha) * theta) / sin_theta
        end_weight = np.sin(alpha * theta) / sin_theta
        blended = start_weight * start + end_weight * end

    # The input validation and shortest-path sign correction keep this norm
    # away from zero. Renormalising also removes floating-point drift.
    norm = float(np.linalg.norm(blended))
    return blended / norm


class PoseInterpolator:
    """Interpolate fixed dual-arm Cartesian poses.

    A pose is always represented as 14 values::

        [left x, y, z, qw, qx, qy, qz,
         right x, y, z, qw, qx, qy, qz]

    Translation is interpolated linearly and orientation uses shortest-path
    spherical interpolation independently for the left and right arms.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t_start = 0.0
        self._t_end = 0.0
        self._pose_start: Optional[np.ndarray] = None
        self._pose_end: Optional[np.ndarray] = None

    def reset(self, pose: np.ndarray) -> None:
        """Replace the trajectory with a stationary, validated pose."""

        pose_array = validate_pose(pose, name="pose")
        now = time.perf_counter()
        with self._lock:
            self._t_start = now
            self._t_end = now
            self._pose_start = pose_array
            self._pose_end = pose_array.copy()

    def set_destination(
        self,
        target: np.ndarray,
        eta: float,
        *,
        now: Optional[float] = None,
    ) -> None:
        """Start a trajectory to ``target`` lasting ``eta`` seconds."""

        target_array = validate_pose(target, name="target")
        with self._lock:
            if now is None:
                now = time.perf_counter()
            if self._pose_end is None:
                # Match LinearInterpolator: without a source pose, the first
                # destination initializes the interpolator immediately.
                self._pose_start = target_array.copy()
                self._pose_end = target_array.copy()
                self._t_start = now
                self._t_end = now
                return

            pose_start = self._current_unlocked(now).copy()
            for quaternion_slice in _QUATERNION_SLICES:
                if (
                    np.dot(
                        pose_start[quaternion_slice], target_array[quaternion_slice]
                    )
                    < 0.0
                ):
                    target_array[quaternion_slice] *= -1.0

            self._pose_start = pose_start
            self._pose_end = target_array
            self._t_start = now
            self._t_end = now + max(eta, 0.0)

    def current(self, t: Optional[float] = None) -> np.ndarray:
        """Return the interpolated pose at time ``t`` (or now)."""

        with self._lock:
            return self._current_unlocked(t)

    def at_destination(self, t: Optional[float] = None) -> bool:
        """Return whether the active trajectory has reached its end time."""

        with self._lock:
            if t is None:
                t = time.perf_counter()
            return t >= self._t_end

    def _current_unlocked(self, t: Optional[float]) -> np.ndarray:
        if self._pose_start is None or self._pose_end is None:
            raise RuntimeError(
                "interpolator not initialised — call reset() or set_destination()"
            )
        if t is None:
            t = time.perf_counter()

        span = self._t_end - self._t_start
        if span <= 0.0 or t >= self._t_end:
            return self._pose_end.copy()
        if t <= self._t_start:
            return self._pose_start.copy()

        alpha = (t - self._t_start) / span
        output = np.empty(_POSE_SHAPE, dtype=np.float64)
        output[_LEFT_XYZ] = (
            (1.0 - alpha) * self._pose_start[_LEFT_XYZ]
            + alpha * self._pose_end[_LEFT_XYZ]
        )
        output[_RIGHT_XYZ] = (
            (1.0 - alpha) * self._pose_start[_RIGHT_XYZ]
            + alpha * self._pose_end[_RIGHT_XYZ]
        )
        for quaternion_slice in _QUATERNION_SLICES:
            output[quaternion_slice] = _slerp(
                self._pose_start[quaternion_slice],
                self._pose_end[quaternion_slice],
                alpha,
            )
        return output
