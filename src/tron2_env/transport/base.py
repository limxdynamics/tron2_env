"""RobotTransport abstraction — the bottom layer of the control stack.

A transport owns the connection to the robot and exposes ONLY single-shot
primitives:

  * ``send_joint_cmd(q[16])``                — send one joint setpoint
  * ``get_joint_state(timeout)``             — read latest 18-dim state
  * ``get_head_position()``                  — read latest head [pitch, yaw]
  * ``set_gripper(left, right)``             — drive grippers
  * ``wait_until_reached(target, tol, t)``   — blocking polling helper
  * ``disconnect`` / ``is_connected``        — lifecycle

No rate limiting, no interpolation, no publish loop — those live in
``tron2_env.motion.MotionController``. This keeps transports trivially
testable (mock socket / fake transport) and lets the same publish loop drive any
backend.

Concrete implementations:
  * ``tron2_env.transport.websocket.WebsocketTransport`` — JSON over ws://...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class LegacyInitPermissions:
    """Optional legacy WebSocket bring-up behavior outside pose initialization."""

    limx_gripper_open: bool = True

    def __post_init__(self) -> None:
        if type(self.limx_gripper_open) is not bool:
            raise TypeError("limx_gripper_open must be a bool")


@runtime_checkable
class ConnectionStateTransport(Protocol):
    """Optional main-session readiness notifications for reconnecting transports."""

    def register_connection_state_callback(
        self,
        callback: Callable[[bool, int], None],
    ) -> None:
        """Report ``(ready, generation)`` and future connection transitions."""

    def unregister_connection_state_callback(self) -> None:
        """Remove the previously registered callback."""


@runtime_checkable
class ArmHeadStateTransport(Protocol):
    """Optional capability for independently sampled 16D arm/head state."""

    def get_arm_head_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
        expected_guid: Optional[str] = None,
    ) -> dict:
        """Return the latest arm/head state without waiting for a gripper."""


@runtime_checkable
class GripperStateTransport(Protocol):
    """Optional capability for independently sampled two-gripper state."""

    def get_gripper_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> dict:
        """Return the latest normalized left/right gripper state."""


@runtime_checkable
class CartesianServoTransport(Protocol):
    """Optional capability for Cartesian servo commands and pose feedback."""

    def send_pose_cmd(self, left: np.ndarray, right: np.ndarray) -> None:
        """Send paired poses as ``[x, y, z, qw, qx, qy, qz]``."""

    def request_pose_state(self, timeout: float = 1.0) -> str:
        """Request one canonical pose sample and return its correlation GUID."""

    def get_pose_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
        expected_guid: Optional[str] = None,
    ) -> dict:
        """Return a canonical pose, optionally waiting for one correlation GUID."""


@runtime_checkable
class BrainCo2Transport(Protocol):
    """Optional capability for paired BrainCo2 hand command and feedback."""

    def set_brainco2_position(
        self,
        left: np.ndarray,
        right: np.ndarray,
        command_time: np.ndarray,
    ) -> str:
        """Send raw paired 6D positions and return the command GUID."""

    def request_brainco2_state(self) -> str:
        """Request one paired hand-state sample and return its GUID."""

    def get_brainco2_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> dict:
        """Return the latest validated 12D left/right hand state."""

    def raise_brainco2_fault(self) -> None:
        """Raise a previously recorded asynchronous command failure."""


@runtime_checkable
class BrainCo2SessionRecoveryTransport(Protocol):
    """Optional recovery gate for commands whose ACK was lost with a session."""

    def clear_brainco2_session_uncertainty(self, generation: int) -> None:
        """Clear only session uncertainty after fresh state from ``generation``."""


@runtime_checkable
class RobotTransport(Protocol):
    """Single-shot send/recv primitives. No internal pacing."""

    def send_joint_cmd(self, q: np.ndarray) -> None:
        """Send a 16-dim joint setpoint [L_arm(7), R_arm(7), head(2)].

        Non-blocking; the actual on-the-wire send may be async. Raises
        ``CommandError`` if the transport is disconnected or the array is
        the wrong shape.
        """

    def get_joint_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> dict:
        """Return the latest 18-dim state.

        Shape::

            {
                "timestamp": int (ms),
                "states": list[float] (18),  # [L_arm(7), L_grip, R_arm(7), R_grip, head(2)]
                ...
            }

        Raises ``StateError`` on timeout.
        """

    def get_head_position(self) -> np.ndarray:
        """Return the latest 2-dim head [pitch, yaw] without dequeuing state."""

    def set_gripper(self, left_opening: float, right_opening: float) -> None:
        """Set gripper opening 0..100 for left/right."""

    def wait_until_reached(
        self,
        target_joints,
        tolerance: float = 0.05,
        timeout: float = 10.0,
    ) -> bool:
        """Block until arm joints are within ``tolerance`` rad of ``target_joints`` (14-dim)."""

    def disconnect(self) -> None:
        """Close the connection and stop any internal threads."""

    def is_connected(self) -> bool: ...

    def __enter__(self): ...

    def __exit__(self, exc_type, exc_val, exc_tb): ...
