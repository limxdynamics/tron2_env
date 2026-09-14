"""MotionController — orchestrates transport + interpolator + publish loop.

This is the concrete object the environment talks to. It owns:

  * a :class:`RobotTransport`     — sends single frames to the robot
  * a :class:`JointInterpolator`  — produces smooth ``current(t)`` between waypoints
  * a daemon publish thread       — drives the transport at ``publish_rate`` Hz

The public package uses the high-level WebSocket transport.

Call ``command_joints(target)`` to retarget (non-blocking). The publish
thread keeps the robot at the latest target indefinitely thanks to PD on
the robot side.
"""

from __future__ import annotations

import copy
import inspect
import logging
import threading
import time
from contextlib import contextmanager, nullcontext
from typing import Optional

import numpy as np

from tron2_env.config import Tron2Config
from tron2_env.end_effector import EndEffectorProvider, create_end_effector_provider
from tron2_env.errors import CommandError, StateError
from tron2_env.interpolation import JointInterpolator, validate_pose
from tron2_env.joints import JointIndex
from tron2_env.layout import BrainCo2Config, PhysicalLayout, RobotModules
from tron2_env.motion.arm_strategy import (
    ArmStrategy,
    ServoJStrategy,
    ServoPStrategy,
    _read_correlated_pose_payload,
    state_to_servoj,
    validate_finite_vector,
)
from tron2_env.motion.head_scheduler import HeadCommandScheduler
from tron2_env.transport import (
    LegacyInitPermissions,
    RobotTransport,
    WebsocketTransport,
)
from tron2_env.util import RateLimiter

logger = logging.getLogger(__name__)


def _positive_number(value, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _supports_keyword(callable_, keyword: str) -> bool:
    """Return whether an introspectable legacy hook accepts ``keyword``."""
    try:
        parameters = inspect.signature(callable_).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        (
            parameter.name == keyword
            and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
        )
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


class MotionController:
    """Own one immutable arm mode and its single publication loop."""

    def __init__(
        self,
        transport: RobotTransport,
        interpolator: Optional[JointInterpolator] = None,
        publish_rate: float = 300.0,
        eta_default: float = 1.0 / 30.0,
        arm_mode: str = "servoj",
        arm_strategy: Optional[ArmStrategy] = None,
        head_scheduler: Optional[HeadCommandScheduler] = None,
        end_effector_provider: Optional[EndEffectorProvider] = None,
        state_max_age: float = 0.5,
    ) -> None:
        self._transport = transport
        self._publish_rate = _positive_number(publish_rate, name="publish_rate")
        self._eta_default = _positive_number(eta_default, name="eta_default")
        self._state_max_age = _positive_number(state_max_age, name="state_max_age")

        if arm_mode not in {"servoj", "servop"}:
            raise ValueError("arm_mode must be 'servoj' or 'servop'")
        if arm_strategy is None:
            if arm_mode == "servoj":
                arm_strategy = ServoJStrategy(transport, interpolator)
            else:
                arm_strategy = ServoPStrategy(transport, interpolator)
        else:
            self._validate_strategy(arm_strategy, expected_mode=arm_mode)
            if interpolator is not None:
                raise ValueError(
                    "interpolator cannot be supplied together with arm_strategy"
                )

        self._arm_strategy = arm_strategy
        self._arm_mode = arm_mode
        if head_scheduler is not None and arm_mode != "servop":
            raise ValueError("head_scheduler is only supported in servop mode")
        self._head_scheduler = head_scheduler
        if end_effector_provider is not None and not isinstance(
            end_effector_provider,
            EndEffectorProvider,
        ):
            raise TypeError("end_effector_provider must implement EndEffectorProvider")
        self._end_effector_provider = end_effector_provider

        self._shutdown = threading.Event()
        self._loop_wakeup = threading.Event()
        self._publish_allowed = threading.Event()
        self._pause_condition = threading.Condition()
        self._pause_generation = 0
        self._paused_generation = -1
        self._commands_in_flight = 0
        self._optional_pause_lock = threading.Lock()
        self._command_gate_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._publish_thread: Optional[threading.Thread] = None
        self._started = False
        self._disconnect_requested = False
        self._disconnected = False
        self._scheduler_closed = head_scheduler is None
        self._head_callback_unregistered = head_scheduler is None
        self._connection_callback_registered = False
        self._connection_callback_unregistered = True
        self._end_effector_closed = end_effector_provider is None
        self._head_pause_complete = head_scheduler is None
        self._end_effector_pause_complete = end_effector_provider is None
        self._publish_thread_stopped = False
        self._transport_disconnected = False
        self._session_tracking_enabled = False
        self._transport_session_ready = True
        self._transport_session_generation: Optional[int] = None
        self._accepted_session_generation: Optional[int] = None
        self._last_transport_session_state: Optional[tuple[bool, int]] = None
        self._connection_fault_epoch = 0
        self._reconnect_required = False

        register_connection_callback = getattr(
            self._transport,
            "register_connection_state_callback",
            None,
        )
        unregister_connection_callback = getattr(
            self._transport,
            "unregister_connection_state_callback",
            None,
        )
        if callable(register_connection_callback) and callable(
            unregister_connection_callback
        ):
            self._session_tracking_enabled = True
            self._transport_session_ready = False
            register_connection_callback(self._handle_transport_connection_state)
            self._connection_callback_registered = True
            self._connection_callback_unregistered = False

    def _handle_transport_connection_state(
        self,
        ready: bool,
        generation: int,
    ) -> None:
        """Latch reconnect recovery without running component hooks in the callback."""
        try:
            generation_value = int(generation)
        except (TypeError, ValueError, OverflowError):
            logger.error("ignoring invalid transport session generation: %r", generation)
            return
        state = (bool(ready), generation_value)
        with self._command_gate_lock:
            with self._pause_condition:
                if state == self._last_transport_session_state:
                    return
                self._last_transport_session_state = state
                self._transport_session_ready = state[0]
                self._transport_session_generation = generation_value

                # Before the controller starts there is no command history to
                # invalidate. Let start() seed from the first ready generation;
                # its own generation check rejects a transition during seeding.
                if not self._started:
                    self._accepted_session_generation = (
                        generation_value if state[0] else None
                    )
                    self._reconnect_required = False
                    return
                if self._disconnect_requested or self._disconnected:
                    return

                session_invalid = (
                    not state[0]
                    or self._accepted_session_generation is None
                    or generation_value != self._accepted_session_generation
                    or self._reconnect_required
                )
                if not session_invalid:
                    return

                begin_recovery = not self._reconnect_required
                self._reconnect_required = True
                self._connection_fault_epoch += 1
                if self._publish_allowed.is_set():
                    self._pause_generation += 1
                    self._publish_allowed.clear()
                if begin_recovery:
                    if self._head_scheduler is not None:
                        self._head_pause_complete = False
                    if self._end_effector_provider is not None:
                        self._end_effector_pause_complete = False
        self._loop_wakeup.set()

    def _raise_if_command_blocked(self) -> None:
        if self._disconnect_requested or self._disconnected:
            raise RuntimeError(
                "controller is disconnecting or disconnected; command rejected"
            )
        if not self._started:
            raise RuntimeError("controller has not started; command rejected")
        with self._pause_condition:
            if self._reconnect_required:
                raise RuntimeError(
                    "transport reconnected; command rejected until "
                    "resume_from_state() reads fresh measured state"
                )
            if not self._publish_allowed.is_set():
                raise RuntimeError("controller is paused; command rejected")

    @contextmanager
    def _ordinary_command_gate(self):
        """Admit one command without holding controller locks during its I/O."""
        with self._command_gate_lock:
            self._raise_if_command_blocked()
            with self._pause_condition:
                self._commands_in_flight += 1
                command_generation = self._accepted_session_generation
        try:
            with self._transport_command_generation(command_generation):
                yield
        finally:
            self._finish_command_in_flight()

    def _transport_command_generation(self, generation: Optional[int]):
        if not self._session_tracking_enabled:
            return nullcontext()
        bind_generation = getattr(
            self._transport,
            "bind_command_generation",
            None,
        )
        if not callable(bind_generation):
            return nullcontext()
        return bind_generation(generation)

    def _read_fresh_measured_arm_target(self, *, timeout: float) -> np.ndarray:
        """Read a seed while enforcing the controller's freshness bound when supported."""
        read_measured = self._arm_strategy.read_measured_target
        if _supports_keyword(read_measured, "max_age"):
            return read_measured(
                timeout=timeout,
                max_age=self._state_max_age,
            )
        return read_measured(timeout=timeout)

    def _read_end_effector_state(
        self,
        *,
        timeout: float,
        max_age: Optional[float],
    ) -> tuple[np.ndarray, dict]:
        """Read provider state without extending the public provider protocol."""
        if self._end_effector_provider is None:
            raise StateError("no end-effector provider is configured")
        read_state = self._end_effector_provider.read_state
        if max_age is not None and _supports_keyword(read_state, "max_age"):
            return read_state(timeout=timeout, max_age=max_age)
        return read_state(timeout=timeout)

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Seed from measured state before launching the publish loop.

        Reading first is mandatory: otherwise the loop could broadcast an
        uninitialised target and move the robot unexpectedly.
        """
        with self._lifecycle_lock:
            if self._started:
                return
            if self._disconnect_requested:
                raise RuntimeError("cannot start a disconnected controller")
            with self._pause_condition:
                if (
                    self._session_tracking_enabled
                    and not self._transport_session_ready
                ):
                    raise RuntimeError(
                        "cannot start before the transport session is ready"
                    )
                if self._reconnect_required:
                    raise RuntimeError(
                        "cannot start until reconnect is recovered with "
                        "resume_from_state()"
                    )
                start_session_generation = self._transport_session_generation
                start_fault_epoch = self._connection_fault_epoch
            try:
                measured = self._read_fresh_measured_arm_target(timeout=2.0)
                self._arm_strategy.reset(measured)
            except StateError as exc:
                state_name = "joint" if self.arm_mode == "servoj" else "pose"
                raise RuntimeError(
                    f"MotionController.start: could not read initial {state_name} state — "
                    "is the transport connected and producing state?"
                ) from exc
            if self._end_effector_provider is not None:
                try:
                    self._read_end_effector_state(
                        timeout=2.0,
                        max_age=self._state_max_age,
                    )
                except StateError as exc:
                    raise RuntimeError(
                        "MotionController.start: could not read initial end-effector state"
                    ) from exc

            with self._command_gate_lock:
                if self._disconnect_requested or self._disconnected:
                    raise RuntimeError(
                        "controller disconnected while MotionController was starting"
                    )
                self._shutdown.clear()
                self._loop_wakeup.clear()
                with self._pause_condition:
                    if self._session_tracking_enabled and (
                        not self._transport_session_ready
                        or self._transport_session_generation
                        != start_session_generation
                        or self._connection_fault_epoch != start_fault_epoch
                        or self._reconnect_required
                    ):
                        raise RuntimeError(
                            "transport session changed while MotionController was starting"
                        )
                thread = threading.Thread(
                    target=self._publish_loop,
                    daemon=True,
                    name="MotionController-publish",
                )
                self._publish_thread = thread
                self._started = True
                try:
                    thread.start()
                except BaseException:
                    self._publish_allowed.clear()
                    self._started = False
                    self._publish_thread = None
                    raise
                self._publish_allowed.set()
        logger.info(
            "MotionController started: arm_mode=%s publish_rate=%.1fHz eta_default=%.1fms",
            self.arm_mode,
            self._publish_rate,
            self._eta_default * 1000.0,
        )

    def disconnect(self) -> None:
        with self._cleanup_lock:
            # Commit shutdown without waiting for lifecycle work or an admitted
            # command.  The transport close below is what can release blocked
            # provider/publish I/O, so it must not sit behind those operations.
            with self._command_gate_lock:
                if self._disconnected:
                    return
                self._disconnect_requested = True
                self._shutdown.set()
                self._publish_allowed.set()
                self._loop_wakeup.set()
                thread = self._publish_thread
                with self._pause_condition:
                    self._pause_condition.notify_all()

            errors: list[BaseException] = []
            transport_errors: list[BaseException] = []

            def attempt(flag_name: str, label: str, action, *, error_sink=None) -> None:
                if getattr(self, flag_name):
                    return
                try:
                    action()
                except BaseException as exc:
                    logger.exception("MotionController cleanup failed: %s", label)
                    (errors if error_sink is None else error_sink).append(exc)
                else:
                    setattr(self, flag_name, True)

            # Closing the transport first releases a command or publish call
            # blocked in socket/SDK I/O.  Keep its error last to preserve the
            # established best-effort cleanup error precedence.
            attempt(
                "_transport_disconnected",
                "transport disconnect",
                self._transport.disconnect,
                error_sink=transport_errors,
            )

            unregister_connection = getattr(
                self._transport,
                "unregister_connection_state_callback",
                None,
            )
            if self._connection_callback_registered and callable(
                unregister_connection
            ):
                attempt(
                    "_connection_callback_unregistered",
                    "connection callback unregister",
                    unregister_connection,
                )
            else:
                self._connection_callback_unregistered = True

            if self._head_scheduler is not None:
                attempt(
                    "_scheduler_closed",
                    "head scheduler close",
                    self._head_scheduler.close,
                )

            unregister = getattr(
                self._transport,
                "unregister_head_response_callback",
                None,
            )
            if callable(unregister):
                attempt(
                    "_head_callback_unregistered",
                    "head callback unregister",
                    unregister,
                )
            else:
                self._head_callback_unregistered = True

            if self._end_effector_provider is not None:
                attempt(
                    "_end_effector_closed",
                    "end-effector provider close",
                    self._end_effector_provider.close,
                )

            if not self._publish_thread_stopped:
                if (
                    thread is None
                    or thread is threading.current_thread()
                    or not thread.is_alive()
                ):
                    self._publish_thread_stopped = True
                else:
                    try:
                        thread.join(timeout=1.0)
                    except BaseException as exc:
                        logger.exception(
                            "MotionController cleanup failed: publish thread join"
                        )
                        errors.append(exc)
                    else:
                        if thread.is_alive():
                            error = RuntimeError(
                                "MotionController publish thread did not stop"
                            )
                            logger.error("%s", error)
                            errors.append(error)
                        else:
                            self._publish_thread_stopped = True

            cleanup_complete = all(
                (
                    self._scheduler_closed,
                    self._head_callback_unregistered,
                    self._connection_callback_unregistered,
                    self._end_effector_closed,
                    self._publish_thread_stopped,
                    self._transport_disconnected,
                )
            )
            with self._lifecycle_lock:
                with self._command_gate_lock:
                    if self._publish_thread_stopped:
                        self._started = False
                    self._disconnected = cleanup_complete

            errors.extend(transport_errors)
            if errors:
                raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # ------------------------------------------------------------ public API

    @property
    def arm_mode(self) -> str:
        """The arm servo mode selected permanently at construction."""
        return self._arm_mode

    def preflight_arm(
        self,
        left: np.ndarray,
        right: np.ndarray,
        head: Optional[np.ndarray] = None,
        eta: Optional[float] = None,
    ) -> None:
        """Validate an arm/head target without changing any command destination."""
        with self._ordinary_command_gate():
            self._validate_arm_command(left, right, head=head, eta=eta)

    def command_arm(
        self,
        left: np.ndarray,
        right: np.ndarray,
        head: Optional[np.ndarray] = None,
        eta: Optional[float] = None,
    ) -> None:
        """Retarget two seven-value arm components and an optional head."""
        with self._ordinary_command_gate():
            if self.arm_mode == "servoj" and head is None:
                head = self._transport.get_head_position()
            left_target, right_target, head_target, eta_value = (
                self._validate_arm_command(left, right, head=head, eta=eta)
            )

            if self.arm_mode == "servoj":
                assert head_target is not None
                self._arm_strategy.set_destination(
                    np.concatenate((left_target, right_target, head_target)),
                    eta_value,
                )
                return

            self._arm_strategy.set_destination(
                np.concatenate((left_target, right_target)),
                eta_value,
            )
            if head_target is not None:
                self._require_head_scheduler().submit(head_target)

    def command_joints(
        self,
        target: np.ndarray,
        eta: Optional[float] = None,
    ) -> None:
        """Compatibility API for one complete 16D ServoJ target."""
        if self.arm_mode != "servoj":
            raise RuntimeError("command_joints is unavailable in servop mode")
        joint_target = validate_finite_vector(
            target,
            width=JointIndex.SERVOJ_DIM,
            name="joint target",
        )
        self.command_arm(
            joint_target[:7],
            joint_target[7:14],
            head=joint_target[14:16],
            eta=eta,
        )

    def command_poses(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        eta: Optional[float] = None,
    ) -> None:
        """Compatibility API for paired 7D ServoP targets."""
        if self.arm_mode != "servop":
            raise RuntimeError("command_poses is unavailable in servoj mode")
        self.command_arm(left_pose, right_pose, eta=eta)

    def command_head(self, target: np.ndarray) -> None:
        """Submit a head target through the optional ServoP scheduler."""
        with self._ordinary_command_gate():
            head_target, scheduler = self._validate_head_command(target)
            scheduler.submit(head_target)

    def preflight_head(self, target: np.ndarray) -> None:
        """Validate a standalone ServoP head target without submitting it."""
        with self._ordinary_command_gate():
            self._validate_head_command(target)

    def raise_if_faulted(self) -> None:
        """Raise retained asynchronous faults before any new policy command."""
        with self._ordinary_command_gate():
            if self._head_scheduler is not None:
                self._head_scheduler.raise_if_faulted()
            if self._end_effector_provider is not None:
                self._end_effector_provider.raise_if_faulted()

    def command_end_effector(self, left: np.ndarray, right: np.ndarray) -> None:
        """Route paired end-effector targets through the selected provider."""
        with self._ordinary_command_gate():
            left_target, right_target = self._validate_end_effector_command(
                left,
                right,
            )
            if self._end_effector_provider is not None:
                self._end_effector_provider.command(left_target, right_target)
                return

            # Legacy controllers intentionally have no provider. Keep their
            # normalized gripper command path available.
            left_opening = float(left_target[0])
            right_opening = float(right_target[0])
            self._transport.set_gripper(left_opening * 100.0, right_opening * 100.0)

    def preflight_end_effector(self, left: np.ndarray, right: np.ndarray) -> None:
        """Run the provider's side-effect-free validation without commanding it."""
        with self._ordinary_command_gate():
            self._validate_end_effector_command(left, right)

    def get_arm_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> tuple[np.ndarray, dict]:
        """Return two seven-value arm states plus provider metadata."""
        timeout_value = _positive_number(timeout, name="timeout")
        effective_max_age = self._resolve_max_age(max_age)
        if self.arm_mode == "servop":
            payload = self._read_correlated_pose_state(
                timeout=timeout_value,
                max_age=effective_max_age,
            )
            values = self._state_values(payload, width=14, name="arm pose state")
            return values, self._payload_metadata(payload)

        getter = getattr(self._transport, "get_arm_head_state", None)
        if callable(getter):
            payload = getter(timeout=timeout_value, max_age=effective_max_age)
            values = self._state_values(
                payload,
                width=JointIndex.SERVOJ_DIM,
                name="arm/head state",
            )
            return values[: JointIndex.TOTAL_ARM_DIM], self._payload_metadata(payload)

        payload = self._read_combined_state(
            timeout=timeout_value,
            max_age=effective_max_age,
        )
        values = self._state_values(
            payload,
            width=JointIndex.STATE_DIM,
            name="legacy joint state",
        )
        arms = np.concatenate(
            (values[JointIndex.LEFT_ARM], values[JointIndex.RIGHT_ARM])
        )
        return arms, self._payload_metadata(payload)

    def get_head_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> tuple[np.ndarray, dict]:
        """Return fresh head feedback from the independent arm/head snapshot."""
        timeout_value = _positive_number(timeout, name="timeout")
        effective_max_age = self._resolve_max_age(max_age)
        getter = getattr(self._transport, "get_arm_head_state", None)
        if callable(getter):
            payload = getter(timeout=timeout_value, max_age=effective_max_age)
            values = self._state_values(
                payload,
                width=JointIndex.SERVOJ_DIM,
                name="arm/head state",
            )
            return values[JointIndex.TOTAL_ARM_DIM :], self._payload_metadata(payload)

        payload = self._read_combined_state(
            timeout=timeout_value,
            max_age=effective_max_age,
        )
        values = self._state_values(
            payload,
            width=JointIndex.STATE_DIM,
            name="legacy joint state",
        )
        return values[JointIndex.HEAD], self._payload_metadata(payload)

    def get_end_effector_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> tuple[np.ndarray, dict]:
        """Return paired end-effector state from the selected provider."""
        timeout_value = _positive_number(timeout, name="timeout")
        effective_max_age = self._resolve_max_age(max_age)
        if self._end_effector_provider is not None:
            return self._read_end_effector_state(
                timeout=timeout_value,
                max_age=effective_max_age,
            )

        payload = self._read_combined_state(
            timeout=timeout_value,
            max_age=effective_max_age,
        )
        values = self._state_values(
            payload,
            width=JointIndex.STATE_DIM,
            name="legacy joint state",
        )
        grippers = values[[JointIndex.LEFT_GRIPPER, JointIndex.RIGHT_GRIPPER]]
        return grippers, self._payload_metadata(payload)

    def pause(self, timeout: float = 1.0) -> None:
        """Stop all arm/head publication and wait for loop acknowledgement."""
        timeout_value = _positive_number(timeout, name="timeout")
        if timeout_value > threading.TIMEOUT_MAX:
            raise ValueError("timeout must not exceed threading.TIMEOUT_MAX")
        deadline = time.monotonic() + timeout_value
        with self._lifecycle_lock:
            with self._command_gate_lock:
                if self._disconnect_requested or self._disconnected:
                    raise RuntimeError(
                        "cannot pause a disconnecting or disconnected controller"
                    )
                if not self._started:
                    raise RuntimeError("cannot pause a controller that has not started")
                with self._pause_condition:
                    if (
                        not self._publish_allowed.is_set()
                        and self._paused_generation == self._pause_generation
                        and self._commands_in_flight == 0
                        and self._head_pause_complete
                        and self._end_effector_pause_complete
                    ):
                        return
                    if self._publish_allowed.is_set():
                        self._pause_generation += 1
                        self._publish_allowed.clear()
                    pause_generation = self._pause_generation
            self._loop_wakeup.set()
            with self._pause_condition:
                paused = self._pause_condition.wait_for(
                    lambda: self._disconnect_requested
                    or (
                        self._paused_generation == pause_generation
                        and self._commands_in_flight == 0
                    ),
                    timeout=min(
                        threading.TIMEOUT_MAX,
                        max(0.0, deadline - time.monotonic()),
                    ),
                )
                disconnect_requested = self._disconnect_requested
            if disconnect_requested:
                raise RuntimeError(
                    "cannot pause a disconnecting or disconnected controller"
                )
            if not paused:
                raise TimeoutError(
                    "MotionController publish loop did not pause in time"
                )
            errors = self._pause_optional_components()
            if errors:
                raise errors[0]

    def resume_from_state(self, state18: Optional[np.ndarray] = None) -> None:
        """Discard the old target and resume publication from measured state."""
        with self._lifecycle_lock:
            with self._command_gate_lock:
                if self._disconnect_requested or self._disconnected:
                    raise RuntimeError(
                        "cannot resume a disconnecting or disconnected controller"
                    )
                with self._pause_condition:
                    if self._publish_allowed.is_set():
                        raise RuntimeError(
                            "resume_from_state requires a fully paused controller"
                        )
                    if self._commands_in_flight != 0:
                        raise RuntimeError(
                            "resume_from_state cannot proceed while commands are still "
                            "in flight; retry after the controller is fully paused"
                        )
                    if (
                        self._paused_generation != self._pause_generation
                        or not self._head_pause_complete
                        or not self._end_effector_pause_complete
                    ):
                        raise RuntimeError(
                            "resume_from_state requires a fully paused controller"
                        )
                    if (
                        self._session_tracking_enabled
                        and not self._transport_session_ready
                    ):
                        raise RuntimeError(
                            "resume_from_state requires a ready transport session"
                        )
                    reconnect_recovery = self._reconnect_required
                    resume_generation = self._transport_session_generation
                    resume_fault_epoch = self._connection_fault_epoch
                    resume_pause_generation = self._pause_generation

            if (
                self.arm_mode == "servoj"
                and state18 is not None
                and not reconnect_recovery
            ):
                measured = state_to_servoj(state18)
            else:
                # ServoP deliberately ignores a legacy argument: it can only
                # resume safely from a fresh, correlated Cartesian sample.
                # Reconnect recovery applies the same rule to ServoJ because a
                # caller-provided state may belong to the lost session.
                measured = self._read_fresh_measured_arm_target(timeout=2.0)

            if reconnect_recovery:
                if self._head_scheduler is not None:
                    self.get_head_state(
                        timeout=2.0,
                        max_age=self._state_max_age,
                    )
                if self._end_effector_provider is not None:
                    self._read_end_effector_state(
                        timeout=2.0,
                        max_age=self._state_max_age,
                    )

            # Resetting a strategy or resuming a component may execute user or
            # SDK code.  Keep all such hooks outside the command gate so a
            # concurrent disconnect can commit shutdown immediately.
            self._arm_strategy.reset(measured)

            with self._command_gate_lock:
                if self._disconnect_requested or self._disconnected:
                    raise RuntimeError(
                        "controller disconnected while reading resume state"
                    )
                with self._pause_condition:
                    if self._session_tracking_enabled and (
                        not self._transport_session_ready
                        or self._transport_session_generation != resume_generation
                        or self._connection_fault_epoch != resume_fault_epoch
                    ):
                        raise StateError(
                            "transport session changed while reading resume state"
                        )
                    if (
                        self._publish_allowed.is_set()
                        or self._pause_generation != resume_pause_generation
                        or self._paused_generation != resume_pause_generation
                        or self._commands_in_flight != 0
                        or not self._head_pause_complete
                        or not self._end_effector_pause_complete
                    ):
                        raise RuntimeError(
                            "controller pause state changed while reading resume state"
                        )
                    if self._end_effector_provider is not None:
                        self._end_effector_pause_complete = False
                    if self._head_scheduler is not None:
                        self._head_pause_complete = False

            def pause_provider_after_failure() -> None:
                try:
                    self._end_effector_provider.pause()
                except BaseException:
                    logger.exception(
                        "MotionController resume compensation failed: "
                        "end-effector provider pause"
                    )
                else:
                    with self._pause_condition:
                        self._end_effector_pause_complete = True
                        self._pause_condition.notify_all()

            def pause_head_after_failure(*, for_reconnect: bool) -> None:
                pause_head = self._head_scheduler.pause
                if for_reconnect:
                    pause_head = getattr(
                        self._head_scheduler,
                        "pause_for_reconnect",
                        pause_head,
                    )
                try:
                    pause_head()
                except BaseException:
                    logger.exception(
                        "MotionController resume compensation failed: "
                        "head scheduler pause"
                    )
                else:
                    with self._pause_condition:
                        self._head_pause_complete = True
                        self._pause_condition.notify_all()

            provider_resumed = False
            if self._end_effector_provider is not None:
                resume_provider = getattr(
                    self._end_effector_provider,
                    "resume",
                    None,
                )
                if callable(resume_provider):
                    try:
                        resume_provider()
                    except BaseException:
                        pause_provider_after_failure()
                        raise
                    provider_resumed = True

            head_resumed = False
            if self._head_scheduler is not None:
                try:
                    self._head_scheduler.resume()
                except BaseException:
                    pause_head_after_failure(for_reconnect=reconnect_recovery)
                    if provider_resumed:
                        pause_provider_after_failure()
                    raise
                head_resumed = True

            def pause_resumed_components() -> None:
                if head_resumed:
                    pause_head_after_failure(for_reconnect=True)
                if provider_resumed:
                    pause_provider_after_failure()

            if reconnect_recovery and self._end_effector_provider is not None:
                recover_provider = getattr(
                    self._end_effector_provider,
                    "recover_after_reconnect",
                    None,
                )
                if callable(recover_provider):
                    try:
                        recover_provider(resume_generation)
                    except BaseException:
                        pause_resumed_components()
                        raise

            resume_failure: Optional[BaseException] = None
            with self._command_gate_lock:
                with self._pause_condition:
                    if self._disconnect_requested or self._disconnected:
                        resume_failure = RuntimeError(
                            "controller disconnected while resuming"
                        )
                    elif self._session_tracking_enabled and (
                        not self._transport_session_ready
                        or self._transport_session_generation != resume_generation
                        or self._connection_fault_epoch != resume_fault_epoch
                    ):
                        resume_failure = StateError(
                            "transport session changed while resuming"
                        )
                    elif (
                        self._publish_allowed.is_set()
                        or self._pause_generation != resume_pause_generation
                        or self._paused_generation != resume_pause_generation
                        or self._commands_in_flight != 0
                    ):
                        resume_failure = StateError(
                            "controller pause state changed while resuming"
                        )
                    elif reconnect_recovery:
                        self._accepted_session_generation = resume_generation
                        self._reconnect_required = False
                    if resume_failure is None:
                        self._head_pause_complete = self._head_scheduler is None
                        self._end_effector_pause_complete = (
                            self._end_effector_provider is None
                        )
                        self._publish_allowed.set()

            if resume_failure is not None:
                pause_resumed_components()
                raise resume_failure

    @property
    def is_paused(self) -> bool:
        return not self._publish_allowed.is_set()

    def set_gripper(self, left_opening: float, right_opening: float) -> None:
        with self._ordinary_command_gate():
            provider = self._end_effector_provider
            if provider is None:
                self._transport.set_gripper(left_opening, right_opening)
                return
            if provider.name != "gripper":
                raise RuntimeError(
                    "set_gripper is unavailable for the selected end-effector provider"
                )
            try:
                percentages = np.asarray(
                    [left_opening, right_opening],
                    dtype=np.float64,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise CommandError("gripper percentages must be finite scalars") from exc
            if percentages.shape != (2,) or not np.all(np.isfinite(percentages)):
                raise CommandError("gripper percentages must be finite scalars")
            normalized = np.clip(percentages, 0.0, 100.0) / 100.0
            provider.command(normalized[:1], normalized[1:])

    def get_chassis_lifter_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> dict:
        getter = getattr(self._transport, "get_chassis_lifter_state", None)
        if getter is None:
            raise StateError("The selected transport does not support chassis/lifter feedback")
        timeout_value = _positive_number(timeout, name="timeout")
        max_age_value = (
            None if max_age is None else _positive_number(max_age, name="max_age")
        )
        payload = self._call_state_getter(
            getter,
            timeout=timeout_value,
            max_age=max_age_value,
            name="chassis/lifter state",
            received_at_key=None,
        )
        self._raise_if_received_at_stale(
            payload,
            key="chassis_received_at",
            max_age=max_age_value,
            name="chassis state",
        )
        self._raise_if_received_at_stale(
            payload,
            key="lifter_received_at",
            max_age=max_age_value,
            name="lifter state",
        )
        for key in ("linear_velocity", "angular_velocity", "lifter_state"):
            self._finite_state_scalar(payload, key=key, name="chassis/lifter state")
        return payload

    def preflight_chassis(self, linear_x: float, angular_z: float) -> None:
        """Validate a chassis command without invoking the transport."""
        with self._ordinary_command_gate():
            self._validate_chassis_command(linear_x, angular_z)

    def command_chassis(self, linear_x: float, angular_z: float) -> None:
        with self._ordinary_command_gate():
            command, linear_value, angular_value = self._validate_chassis_command(
                linear_x,
                angular_z,
            )
            command(linear_value, angular_value)

    def stop_chassis(self) -> None:
        stop = getattr(self._transport, "stop_chassis", None)
        if stop is None:
            raise RuntimeError("The selected transport does not support chassis commands")
        stop()

    def stream_lifter_position(
        self,
        position_mm: float,
        velocity_mm_s: float = 80.0,
    ) -> None:
        """Refresh the lifter's absolute position stream target."""
        with self._ordinary_command_gate():
            command, position_value, velocity_value = (
                self._validate_lifter_stream_command(
                    position_mm,
                    velocity_mm_s,
                )
            )
            command(position_value, velocity_value)

    def preflight_lifter_stream_position(
        self,
        position_mm: float,
        velocity_mm_s: float = 80.0,
    ) -> None:
        """Validate a lifter stream refresh without invoking the transport."""
        with self._ordinary_command_gate():
            self._validate_lifter_stream_command(position_mm, velocity_mm_s)

    def stop_lifter_stream(self) -> None:
        """Stop lifter streaming even after ordinary policy commands are paused."""
        stop = getattr(self._transport, "stop_lifter_stream", None)
        if stop is None:
            raise RuntimeError(
                "The selected transport does not support lifter stream commands"
            )
        stop()

    def get_joint_states(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> dict:
        timeout_value = _positive_number(timeout, name="timeout")
        max_age_value = (
            None if max_age is None else _positive_number(max_age, name="max_age")
        )
        return self._read_combined_state(
            timeout=timeout_value,
            max_age=max_age_value,
        )

    def find_nearest_state(self, target_timestamp_s: float):
        """Find the queued state closest to target_timestamp_s (non-destructive).

        Returns ``None`` if the underlying transport doesn't support search or
        if the queue is empty. Useful for image/state alignment in legacy obs.
        """
        finder = getattr(self._transport, "find_nearest_state", None)
        if finder is None:
            return None
        return finder(target_timestamp_s)

    def get_head_position(self) -> np.ndarray:
        return self._transport.get_head_position()

    def wait_until_reached(self, *args, **kwargs) -> bool:
        return self._transport.wait_until_reached(*args, **kwargs)

    def is_connected(self) -> bool:
        return self._transport.is_connected()

    @property
    def transport(self) -> RobotTransport:
        """Exposes the underlying transport for backend-specific operations (e.g. movej during reset)."""
        return self._transport

    # ----------------------------------------------------------------- loop

    def _publish_loop(self) -> None:
        rate = RateLimiter(self._publish_rate)
        while True:
            with self._command_gate_lock:
                if self._shutdown.is_set():
                    return
                with self._pause_condition:
                    publish_allowed = self._publish_allowed.is_set()
                    pause_generation = self._pause_generation
                    if publish_allowed:
                        self._commands_in_flight += 1
                        publish_generation = self._accepted_session_generation
            if publish_allowed:
                try:
                    with self._transport_command_generation(publish_generation):
                        self._arm_strategy.publish_current()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("arm publish loop: %s", exc)
                finally:
                    self._finish_command_in_flight()

                tick_allowed = False
                if self._head_scheduler is not None:
                    with self._command_gate_lock:
                        with self._pause_condition:
                            tick_allowed = (
                                not self._shutdown.is_set()
                                and self._publish_allowed.is_set()
                                and self._pause_generation == pause_generation
                            )
                            if tick_allowed:
                                self._commands_in_flight += 1
                                tick_generation = (
                                    self._accepted_session_generation
                                )
                    if tick_allowed:
                        try:
                            with self._transport_command_generation(tick_generation):
                                self._head_scheduler.tick()
                        except Exception as exc:  # noqa: BLE001
                            # Keep ticking the arm even if scheduler diagnostics fail.
                            logger.warning("head scheduler tick: %s", exc)
                        finally:
                            self._finish_command_in_flight()
            if not publish_allowed:
                with self._pause_condition:
                    commands_quiescent = self._commands_in_flight == 0
                    reconnect_pause = self._reconnect_required
                if not commands_quiescent:
                    with self._pause_condition:
                        self._pause_condition.wait_for(
                            lambda: self._commands_in_flight == 0
                            or self._shutdown.is_set()
                            or self._publish_allowed.is_set()
                            or self._pause_generation != pause_generation,
                            timeout=0.05,
                        )
                    continue
                if reconnect_pause:
                    for error in self._pause_optional_components():
                        logger.warning(
                            "MotionController reconnect pause hook failed: %s",
                            error,
                        )
                with self._pause_condition:
                    if (
                        not self._publish_allowed.is_set()
                        and self._pause_generation == pause_generation
                        and self._commands_in_flight == 0
                    ):
                        self._paused_generation = max(
                            self._paused_generation,
                            pause_generation,
                        )
                        self._pause_condition.notify_all()
                if self._shutdown.is_set():
                    return
                self._publish_allowed.wait(timeout=0.05)
                rate.reset()
                continue
            self._loop_wakeup.clear()
            if self._shutdown.is_set() or not self._publish_allowed.is_set():
                continue
            rate.sleep(self._loop_wakeup)

    def _finish_command_in_flight(self) -> None:
        with self._pause_condition:
            self._commands_in_flight -= 1
            self._pause_condition.notify_all()

    # ---------------------------------------------------------------- helpers

    def _pause_optional_components(self) -> list[BaseException]:
        """Pause optional command producers without involving the WS callback."""
        errors: list[BaseException] = []
        with self._optional_pause_lock:
            with self._pause_condition:
                pause_head = (
                    self._head_scheduler is not None
                    and not self._head_pause_complete
                )
                pause_provider = (
                    self._end_effector_provider is not None
                    and not self._end_effector_pause_complete
                )
                reconnect_pause = self._reconnect_required

            if pause_head:
                pause_head_action = self._head_scheduler.pause
                if reconnect_pause:
                    pause_head_action = getattr(
                        self._head_scheduler,
                        "pause_for_reconnect",
                        pause_head_action,
                    )
                try:
                    pause_head_action()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    with self._pause_condition:
                        if reconnect_pause or not self._reconnect_required:
                            self._head_pause_complete = True

            if pause_provider:
                try:
                    self._end_effector_provider.pause()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    with self._pause_condition:
                        self._end_effector_pause_complete = True
        return errors

    @staticmethod
    def _validate_strategy(strategy: ArmStrategy, *, expected_mode: str) -> None:
        mode = getattr(strategy, "mode", None)
        target_dim = getattr(strategy, "target_dim", None)
        expected_dim = JointIndex.SERVOJ_DIM if expected_mode == "servoj" else 14
        if mode != expected_mode or target_dim != expected_dim:
            raise ValueError(
                "arm_strategy mode/target_dim must match the selected arm_mode"
            )
        for method_name in (
            "read_measured_target",
            "reset",
            "set_destination",
            "publish_current",
        ):
            if not callable(getattr(strategy, method_name, None)):
                raise TypeError(f"arm_strategy requires callable {method_name}()")

    def _resolve_eta(self, eta: Optional[float]) -> float:
        if eta is None:
            return self._eta_default
        return _positive_number(eta, name="eta")

    def _validate_arm_command(
        self,
        left,
        right,
        *,
        head,
        eta,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float]:
        left_target = validate_finite_vector(left, width=7, name="left arm target")
        right_target = validate_finite_vector(right, width=7, name="right arm target")
        eta_value = self._resolve_eta(eta)

        if self.arm_mode == "servop":
            pose_target = validate_pose(
                np.concatenate((left_target, right_target)),
                name="pose target",
            )
            left_target = pose_target[:7]
            right_target = pose_target[7:]

        head_target = None
        if head is not None:
            if self.arm_mode == "servop":
                head_target, _scheduler = self._validate_head_command(head)
            else:
                head_target = validate_finite_vector(
                    head,
                    width=JointIndex.HEAD_DIM,
                    name="head target",
                )
        return left_target, right_target, head_target, eta_value

    def _validate_head_command(
        self,
        target,
    ) -> tuple[np.ndarray, HeadCommandScheduler]:
        if self.arm_mode != "servop":
            raise RuntimeError("command_head is unavailable in servoj mode")
        head_target = validate_finite_vector(
            target,
            width=JointIndex.HEAD_DIM,
            name="head target",
        )
        scheduler = self._require_head_scheduler()
        scheduler.raise_if_faulted()
        return head_target, scheduler

    def _validate_end_effector_command(
        self,
        left,
        right,
    ) -> tuple[np.ndarray, np.ndarray]:
        provider = self._end_effector_provider
        if provider is None:
            return (
                np.array(
                    [self._legacy_gripper_opening(left, side="left")],
                    dtype=np.float64,
                ),
                np.array(
                    [self._legacy_gripper_opening(right, side="right")],
                    dtype=np.float64,
                ),
            )

        width = provider.width_per_side
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise CommandError("end-effector provider has invalid width_per_side")
        try:
            left_target = validate_finite_vector(
                left,
                width=width,
                name="left end-effector target",
            )
            right_target = validate_finite_vector(
                right,
                width=width,
                name="right end-effector target",
            )
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        if provider.name == "gripper" and (
            np.any((left_target < 0.0) | (left_target > 1.0))
            or np.any((right_target < 0.0) | (right_target > 1.0))
        ):
            raise CommandError("gripper targets must contain values in [0, 1]")

        validated = provider.validate_command(
            left_target.copy(),
            right_target.copy(),
        )
        try:
            validated_left, validated_right = validated
            validated_left = validate_finite_vector(
                validated_left,
                width=width,
                name="validated left end-effector target",
            )
            validated_right = validate_finite_vector(
                validated_right,
                width=width,
                name="validated right end-effector target",
            )
        except (TypeError, ValueError) as exc:
            raise CommandError(
                "end-effector provider validate_command() must return two finite "
                f"shape ({width},) vectors"
            ) from exc
        return validated_left, validated_right

    def _validate_chassis_command(self, linear_x, angular_z):
        command = getattr(self._transport, "command_chassis", None)
        if not callable(command):
            raise RuntimeError("The selected transport does not support chassis commands")
        try:
            linear_value = float(linear_x)
            angular_value = float(angular_z)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CommandError("chassis command values must be finite") from exc
        if not np.isfinite(linear_value) or not np.isfinite(angular_value):
            raise CommandError("chassis command values must be finite")
        return command, linear_value, angular_value

    def _validate_lifter_stream_command(self, position_mm, velocity_mm_s):
        command = getattr(self._transport, "stream_lifter_position", None)
        if not callable(command):
            raise RuntimeError(
                "The selected transport does not support lifter stream commands"
            )
        try:
            position_value = float(position_mm)
            velocity_value = float(velocity_mm_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CommandError(
                "lifter stream position/velocity must be finite numeric values"
            ) from exc
        if (
            not np.isfinite(position_value)
            or not np.isfinite(velocity_value)
            or velocity_value <= 0.0
        ):
            raise CommandError(
                "lifter stream position/velocity must be finite, with velocity > 0"
            )
        return command, position_value, velocity_value

    @staticmethod
    def _finite_state_scalar(payload: dict, *, key: str, name: str) -> float:
        try:
            value = payload[key]
            array = np.asarray(value)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise StateError(f"{name} has invalid {key}") from exc
        if (
            array.shape != ()
            or not (
                np.issubdtype(array.dtype, np.integer)
                or np.issubdtype(array.dtype, np.floating)
            )
            or not bool(np.isfinite(array))
        ):
            raise StateError(f"{name} has invalid {key}")
        return float(array)

    def _require_head_scheduler(self) -> HeadCommandScheduler:
        if self._head_scheduler is None:
            raise RuntimeError("servop head command requires a configured head scheduler")
        return self._head_scheduler

    def _resolve_max_age(self, max_age: Optional[float]) -> float:
        if max_age is None:
            return self._state_max_age
        return _positive_number(max_age, name="max_age")

    @staticmethod
    def _payload_metadata(payload) -> dict:
        if not isinstance(payload, dict):
            raise StateError("state response must be a mapping")
        return copy.deepcopy({key: value for key, value in payload.items() if key != "states"})

    @staticmethod
    def _state_values(payload, *, width: int, name: str) -> np.ndarray:
        try:
            values = payload["states"]
        except (KeyError, TypeError) as exc:
            raise StateError(f"{name} response has no valid 'states' vector") from exc
        try:
            return validate_finite_vector(values, width=width, name=name)
        except ValueError as exc:
            raise StateError(str(exc)) from exc

    def _read_correlated_pose_state(self, *, timeout: float, max_age: float) -> dict:
        return _read_correlated_pose_payload(
            self._transport,
            timeout=timeout,
            max_age=max_age,
        )

    def _read_combined_state(
        self,
        *,
        timeout: float,
        max_age: Optional[float],
    ) -> dict:
        return self._call_state_getter(
            self._transport.get_joint_state,
            timeout=timeout,
            max_age=max_age,
            name="legacy joint state",
        )

    @staticmethod
    def _call_state_getter(
        getter,
        *,
        timeout: float,
        max_age: Optional[float],
        name: str,
        received_at_key: Optional[str] = "received_at",
    ) -> dict:
        if max_age is not None and _supports_keyword(getter, "max_age"):
            payload = getter(timeout=timeout, max_age=max_age)
        else:
            payload = getter(timeout=timeout)
        if not isinstance(payload, dict):
            raise StateError(f"{name} response must be a mapping")
        if received_at_key is not None:
            MotionController._raise_if_received_at_stale(
                payload,
                key=received_at_key,
                max_age=max_age,
                name=name,
            )
        return payload

    @staticmethod
    def _raise_if_received_at_stale(
        payload: dict,
        *,
        key: str,
        max_age: Optional[float],
        name: str,
    ) -> None:
        if max_age is None:
            return
        try:
            received_at_value = float(payload[key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise StateError(f"{name} has invalid received_at") from exc
        if not np.isfinite(received_at_value):
            raise StateError(f"{name} has invalid received_at")
        if time.monotonic() - received_at_value > max_age:
            raise StateError(f"{name} is stale")

    @staticmethod
    def _legacy_gripper_opening(value, *, side: str) -> float:
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CommandError(f"{side} gripper must be one finite value") from exc
        if vector.shape != (1,):
            raise CommandError(f"{side} gripper must be one finite value")
        opening = float(vector[0])
        if not np.isfinite(opening):
            raise CommandError(f"{side} gripper must be one finite value")
        return float(np.clip(opening, 0.0, 1.0))

    @staticmethod
    def _state_to_servoj(state18: np.ndarray) -> np.ndarray:
        """Convert an 18-dim ``[L_arm, L_grip, R_arm, R_grip, head]`` state
        into the 16-dim ``[L_arm, R_arm, head]`` setpoint vector."""
        return state_to_servoj(state18)


# ---------------------------------------------------------------------- factory


def create_motion_controller(
    config: Tron2Config,
    backend: str = "websocket",
    publish_rate: float = 300.0,
    eta_default: float = 1.0 / 30.0,
    interpolator: Optional[JointInterpolator] = None,
    enable_chassis_lifter: bool = False,
    auxiliary_polling_rate: float = 20.0,
    lifter_state_source: str = "raw_q",
    modules: Optional[RobotModules] = None,
    head_command_interval: Optional[float] = None,
    head_move_duration: Optional[float] = None,
    head_ack_timeout: float = 1.0,
    brainco2_config: Optional[BrainCo2Config] = None,
    state_max_age: float = 0.5,
) -> MotionController:
    """Build a started MotionController with the requested backend.

    Args:
        config: Robot connection + bring-up params.
        backend: must be ``"websocket"`` in the public package.
        publish_rate: Hz at which the publish loop drives the transport.
        eta_default: default time-to-target for ``command_joints`` (= 1/fps).
        interpolator: override the default ``LinearInterpolator``.
        enable_chassis_lifter: enable the websocket chassis/lifter protocol.
        auxiliary_polling_rate: feedback request rate for chassis/lifter state.
        lifter_state_source: ``raw_q`` or absolute ``position_mm`` feedback.
        modules: immutable robot-wide module selection; legacy default is ServoJ.
        head_command_interval: minimum ServoP head-send interval. ``None`` uses ETA.
        head_move_duration: ServoP head movement duration. ``None`` uses ETA.
        head_ack_timeout: timeout for a ServoP head acknowledgement.
        brainco2_config: provider configuration required by BrainCo2 modules.
        state_max_age: maximum accepted age for modular state snapshots.
    """
    # Validate selections before constructing hardware or background workers.
    publish_rate = _positive_number(publish_rate, name="publish_rate")
    eta_default = _positive_number(eta_default, name="eta_default")
    state_max_age = _positive_number(state_max_age, name="state_max_age")
    if modules is None:
        selected_modules = RobotModules()
    elif isinstance(modules, RobotModules):
        selected_modules = modules
        # Explicit module selection is authoritative. The legacy flag remains
        # available only for callers that do not supply RobotModules.
        enable_chassis_lifter = selected_modules.mobile_base
    else:
        raise TypeError("modules must be a RobotModules instance or None")

    for timing_name, timing_value in (
        ("head_command_interval", head_command_interval),
        ("head_move_duration", head_move_duration),
    ):
        if timing_value is not None:
            _positive_number(timing_value, name=timing_name)
    head_ack_timeout = _positive_number(head_ack_timeout, name="head_ack_timeout")

    if backend != "websocket":
        raise ValueError(
            f"unknown control backend: {backend!r} (public package supports only 'websocket')"
        )

    if selected_modules.end_effector == "brainco2":
        if not isinstance(brainco2_config, BrainCo2Config):
            raise ValueError("BrainCo2 modules require BrainCo2Config")
    elif brainco2_config is not None:
        raise ValueError(
            "BrainCo2Config is only valid when the BrainCo2 end effector is selected"
        )

    websocket_kwargs = {
        "enable_chassis_lifter": enable_chassis_lifter,
        "auxiliary_polling_rate": auxiliary_polling_rate,
        "lifter_state_source": lifter_state_source,
    }
    if modules is not None:
        websocket_kwargs.update({
            "poll_joint_state": (
                selected_modules.arm == "servoj" or selected_modules.head
            ),
            "poll_gripper_state": (
                selected_modules.end_effector == "gripper"
            ),
        })
        websocket_kwargs["legacy_init_permissions"] = LegacyInitPermissions(
            limx_gripper_open=selected_modules.end_effector == "gripper",
        )
    transport: RobotTransport = WebsocketTransport(
        config,
        **websocket_kwargs,
    )

    scheduler: Optional[HeadCommandScheduler] = None
    end_effector_provider: Optional[EndEffectorProvider] = None
    callback_registered = False
    controller: Optional[MotionController] = None
    try:
        if modules is not None:
            end_effector_provider = create_end_effector_provider(
                selected_modules.end_effector,
                transport,
                brainco2_config,
            )
            expected_width = PhysicalLayout.from_modules(
                selected_modules
            ).end_effector_width
            if end_effector_provider.width_per_side != expected_width:
                raise ValueError(
                    "end-effector provider width does not match the selected layout"
                )
        if selected_modules.arm == "servop" and selected_modules.head:
            sender = getattr(transport, "send_head_move", None)
            register = getattr(transport, "register_head_response_callback", None)
            if not callable(sender) or not callable(register):
                raise TypeError(
                    "servop head control requires head send and response callback capabilities"
                )
            scheduler = HeadCommandScheduler(
                sender=sender,
                min_interval=(
                    eta_default
                    if head_command_interval is None
                    else head_command_interval
                ),
                move_duration=(
                    eta_default if head_move_duration is None else head_move_duration
                ),
                ack_timeout=head_ack_timeout,
            )
            register(scheduler.handle_response)
            callback_registered = True

        controller = MotionController(
            transport=transport,
            interpolator=interpolator,
            publish_rate=publish_rate,
            eta_default=eta_default,
            arm_mode=selected_modules.arm,
            head_scheduler=scheduler,
            end_effector_provider=end_effector_provider,
            state_max_age=state_max_age,
        )
        controller.start()
        return controller
    except BaseException:
        def rollback(label: str, action) -> None:
            try:
                action()
            except BaseException:
                logger.exception("MotionController factory rollback failed: %s", label)

        if controller is not None:
            rollback("controller disconnect", controller.disconnect)
        else:
            if end_effector_provider is not None:
                rollback("end-effector provider close", end_effector_provider.close)
            if scheduler is not None:
                rollback("head scheduler close", scheduler.close)
            if callback_registered:
                unregister = getattr(
                    transport,
                    "unregister_head_response_callback",
                    None,
                )
                if callable(unregister):
                    rollback("head callback unregister", unregister)
            rollback("transport disconnect", transport.disconnect)
        raise
