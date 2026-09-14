"""WebsocketTransport — high-level ws://<ip>:5000 JSON control.

Migrated from the original ``tron2_env.robot.Tron2`` class. Changes vs. the
old implementation:

* Renamed to make the abstraction layer explicit (``RobotTransport``).
* Single-shot ``send_joint_cmd`` replaces public ``servoj``; the internal
  ``servoj_rate_limiter`` is removed — pacing now lives in
  ``MotionController._publish_loop``.
* Added ``get_head_position`` so callers no longer reach into private state.
* Chassis velocity and lifter feedback/control primitives are enabled only
  when the deployment layout contains the ``chassis`` group.
* ``movej`` / ``move_head`` / ``wait_until_reached`` / ``set_gripper``
  retained — used by ``_move_to_init_pose`` and ``env.reset``.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import socket
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from numbers import Real
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import websocket

from tron2_env.config import Tron2Config
from tron2_env.errors import CommandError, StateError
from tron2_env.joints import JointIndex
from tron2_env.transport.base import LegacyInitPermissions
from tron2_env.transport.snapshot import LatestSnapshot

logger = logging.getLogger(__name__)

_WEBSOCKET_RECONNECT_DELAY_SECONDS = 1.0
_POSE_QUERY_GUID_PREFIX = "tron2_env:canonical_pose:"
_LIFTER_FEEDBACK_TIMEOUT_SECONDS = 1.0


def _validated_polling_period(value, *, name: str) -> tuple[float, float]:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite positive number")
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    try:
        period = 1.0 / rate
    except OverflowError as exc:
        raise ValueError(f"{name} does not produce a usable polling period") from exc
    if (
        not np.isfinite(period)
        or period <= 0.0
        or period > threading.TIMEOUT_MAX
    ):
        raise ValueError(f"{name} does not produce a usable polling period")
    return rate, period


class WebsocketTransport:
    """Tron2 control via the ws://<ip>:5000 JSON protocol.

    Implements :class:`tron2_env.transport.base.RobotTransport`.

    Example:
        >>> config = Tron2Config(robot_ip="ROBOT_IP")
        >>> with WebsocketTransport(config) as t:
        ...     t.send_joint_cmd(np.zeros(16))
        ...     state = t.get_joint_state()
    """

    SERVOJ_DIM = JointIndex.SERVOJ_DIM

    def __init__(
        self,
        config: Optional[Tron2Config] = None,
        *,
        enable_chassis_lifter: bool = False,
        auxiliary_polling_rate: float = 20.0,
        lifter_state_source: str = "raw_q",
        feedback_clock: Optional[Callable[[], float]] = None,
        legacy_init_permissions: LegacyInitPermissions = LegacyInitPermissions(),
        poll_joint_state: bool = True,
        poll_gripper_state: bool = True,
    ) -> None:
        if not isinstance(legacy_init_permissions, LegacyInitPermissions):
            raise TypeError("legacy_init_permissions must be LegacyInitPermissions")
        if type(poll_joint_state) is not bool:
            raise TypeError("poll_joint_state must be a bool")
        if type(poll_gripper_state) is not bool:
            raise TypeError("poll_gripper_state must be a bool")
        self.config = config or Tron2Config()
        try:
            connection_timeout = float(self.config.connection_timeout)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "connection_timeout must be finite and positive"
            ) from exc
        if not np.isfinite(connection_timeout) or connection_timeout <= 0.0:
            raise ValueError("connection_timeout must be finite and positive")
        if connection_timeout > threading.TIMEOUT_MAX:
            raise ValueError(
                "connection_timeout must not exceed threading.TIMEOUT_MAX"
            )
        polling_rate, main_feedback_period = _validated_polling_period(
            self.config.polling_rate,
            name="polling_rate",
        )
        auxiliary_polling_rate, auxiliary_feedback_period = (
            _validated_polling_period(
                auxiliary_polling_rate,
                name="auxiliary_polling_rate",
            )
        )
        self.legacy_init_permissions = legacy_init_permissions
        self.poll_joint_state = poll_joint_state
        self.poll_gripper_state = poll_gripper_state
        self.enable_chassis_lifter = bool(enable_chassis_lifter)
        self.polling_rate = polling_rate
        self.auxiliary_polling_rate = auxiliary_polling_rate
        self.lifter_state_source = str(lifter_state_source)
        self._feedback_clock = feedback_clock or time.monotonic
        if not callable(self._feedback_clock):
            raise TypeError("feedback_clock must be callable")
        if self.lifter_state_source not in {"raw_q", "position_mm"}:
            raise ValueError("lifter_state_source must be 'raw_q' or 'position_mm'")
        self._setup_logger()

        # ws connection
        self.accid: Optional[str] = None
        self.ws_client: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.connected = False
        self._session_ready = threading.Event()
        self._connection_state_lock = threading.RLock()
        self._connection_transition_lock = threading.RLock()
        self._message_dispatch_lock = threading.RLock()
        self._session_generation = 0
        self._abort_socket_lock = threading.Lock()
        self._main_abort_socket = None
        self._command_generation_local = threading.local()
        self._connection_state_callback: Optional[
            Callable[[bool, int], None]
        ] = None
        self._lifter_command_accid: Optional[str] = None
        self._lifter_command_ws_client: Optional[websocket.WebSocketApp] = None
        self._lifter_command_ws_thread: Optional[threading.Thread] = None
        self._lifter_connection_state_lock = threading.RLock()
        self._lifter_command_socket_open = False
        self._lifter_command_connected = False
        self._lifter_command_ready = threading.Event()
        self._lifter_command_session_generation = 0
        self._lifter_abort_socket = None
        self.should_exit = False

        # state buffers (timestamp = client wall clock ms, robot_timestamp = server's own ts)
        self._arm_head_snapshot = LatestSnapshot(
            width=JointIndex.SERVOJ_DIM,
            name="arm/head",
        )
        self._gripper_snapshot = LatestSnapshot(width=2, name="gripper")
        self._pose_snapshot = LatestSnapshot(width=14, name="pose")
        self._brainco2_snapshot = LatestSnapshot(width=12, name="BrainCo2")
        self.joint_states: Dict = {
            "timestamp": -1,
            "robot_timestamp": -1,
            "states": [-1.0] * JointIndex.STATE_DIM,
            "joint_updated": False,
            "gripper_updated": False,
        }
        self.ee_pose_states: Dict = {
            "timestamp": -1,
            "left_position": [-1.0, -1.0, -1.0],
            "left_quat": [-1.0, -1.0, -1.0, -1.0],
            "right_position": [-1.0, -1.0, -1.0],
            "right_quat": [-1.0, -1.0, -1.0, -1.0],
        }
        self.chassis_states: Dict = {
            "timestamp": -1,
            "received_at": None,
            "linear_velocity": 0.0,
            "angular_velocity": 0.0,
            "valid": False,
        }
        self.lifter_states: Dict = {
            "timestamp": -1,
            "received_at": None,
            "value": 0.0,
            "valid": False,
        }
        self._legacy_joint_received_at: Optional[float] = None
        self._legacy_gripper_received_at: Optional[float] = None
        self.joint_state_queue: deque = deque(maxlen=self.config.state_queue_maxlen)
        self.ee_pose_queue: deque = deque(maxlen=self.config.state_queue_maxlen)
        self._queue_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._lifter_feedback_lock = threading.Lock()
        self._lifter_feedback_in_flight: Optional[str] = None
        self._lifter_feedback_deadline: Optional[float] = None
        self._pose_query_lock = threading.Lock()
        self._pose_query_in_flight: Optional[str] = None
        self._pose_query_deadline: Optional[float] = None
        self._head_response_callback_lock = threading.Lock()
        self._head_response_callback: Optional[
            Callable[[Optional[str], object], None]
        ] = None
        self._brainco2_command_lock = threading.RLock()
        self._brainco2_command_guids: dict[str, float] = {}
        self._brainco2_command_ack_timeout = 1.0
        self._brainco2_fault_message: Optional[str] = None
        self._brainco2_session_uncertain = False
        self._brainco2_query_lock = threading.RLock()
        self._brainco2_query_in_flight: Optional[str] = None
        self._brainco2_query_deadline: Optional[float] = None
        self._brainco2_state_generation: Optional[int] = None
        self._brainco2_query_timeout = 1.0
        self._brainco2_polling_enabled = False
        self._brainco2_polling_period = 0.0
        self._brainco2_next_poll_at: Optional[float] = None
        scheduler_start = float(self._feedback_clock())
        if not np.isfinite(scheduler_start):
            raise ValueError("feedback clock must return a finite value")
        self._main_feedback_period = main_feedback_period
        self._main_feedback_next_at = scheduler_start
        self._auxiliary_feedback_period = auxiliary_feedback_period
        self._auxiliary_feedback_next_at = scheduler_start
        self._feedback_wakeup = threading.Event()
        self._reconnect_wakeup = threading.Event()
        self._polling_thread: Optional[threading.Thread] = None

        # bring-up
        try:
            self._connect()
            if self.enable_chassis_lifter:
                self._connect_lifter_command_channel()
            readiness_deadline = time.monotonic() + connection_timeout
            if not self._session_ready.wait(
                timeout=max(0.0, readiness_deadline - time.monotonic())
            ):
                raise StateError(
                    "WebsocketTransport session was not ready within "
                    f"{connection_timeout}s "
                    "(socket open plus notify_robot_info required)"
                )
            if self.enable_chassis_lifter and not self._lifter_command_ready.wait(
                timeout=max(0.0, readiness_deadline - time.monotonic())
            ):
                raise StateError(
                    "WebsocketTransport lifter command session was not ready within "
                    f"{connection_timeout}s "
                    "(socket open plus notify_robot_info required)"
                )
            self._second_joint = [
                0.000999913, -0.00449967, 1.482, -1.57, 0.0036, 0.00289989,
                -0.00160009, 0.0415001, 0.1279, -1.4808, -1.57, -0.00739986,
                0.0151, -0.0624998,
            ]

            if self.config.init_joints is not None or self.config.init_head is not None:
                self._move_to_init_pose()
                self._invalidate_main_feedback_after_init()
            self._start_polling_thread()
        except BaseException:
            try:
                self.disconnect()
            except BaseException:
                self.logger.exception(
                    "failed to disconnect after WebsocketTransport bring-up failure"
                )
            raise

    # ------------------------------------------------------------------ logger

    def _setup_logger(self) -> None:
        self.logger = logger

    # --------------------------------------------------------------- websocket

    @staticmethod
    def _new_guid() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _validate_public_wait_timeout(timeout: float) -> float:
        if isinstance(timeout, (bool, np.bool_)) or not isinstance(
            timeout,
            (Real, np.integer, np.floating),
        ):
            raise TypeError("timeout must be a real number")
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "timeout must be finite and non-negative"
            ) from exc
        if not np.isfinite(timeout_value) or timeout_value < 0.0:
            raise ValueError("timeout must be finite and non-negative")
        if timeout_value > threading.TIMEOUT_MAX:
            raise ValueError("timeout must not exceed threading.TIMEOUT_MAX")
        return timeout_value

    @staticmethod
    def _new_pose_query_guid() -> str:
        return f"{_POSE_QUERY_GUID_PREFIX}{uuid.uuid4()}"

    @staticmethod
    def _is_pose_query_guid(guid) -> bool:
        return isinstance(guid, str) and guid.startswith(_POSE_QUERY_GUID_PREFIX)

    @contextmanager
    def bind_command_generation(self, generation: Optional[int]):
        """Bind controller sends on this thread to one main-session generation."""
        missing = object()
        previous = getattr(
            self._command_generation_local,
            "expected_generation",
            missing,
        )
        self._command_generation_local.expected_generation = generation
        try:
            yield
        finally:
            if previous is missing:
                del self._command_generation_local.expected_generation
            else:
                self._command_generation_local.expected_generation = previous

    def _bound_command_generation(self) -> Optional[int]:
        return getattr(
            self._command_generation_local,
            "expected_generation",
            None,
        )

    def _send_request(
        self,
        title: str,
        data: Optional[Dict] = None,
        *,
        guid: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> bool:
        if expected_generation is None:
            expected_generation = self._bound_command_generation()
        with self._connection_state_lock:
            client = self.ws_client
            accid = self.accid
            ready = self.connected and self._session_ready.is_set()
            generation_matches = (
                expected_generation is None
                or self._session_generation == expected_generation
            )
            if (
                not client
                or not ready
                or not generation_matches
                or not isinstance(accid, str)
                or not accid
            ):
                self.logger.warning("ws not connected, dropping: %s", title)
                return False
            try:
                message = {
                    "accid": accid,
                    "title": title,
                    "timestamp": int(time.time() * 1000),
                    "guid": guid or self._new_guid(),
                    "data": data or {},
                }
                client.send(json.dumps(message))
                return True
            except Exception as exc:
                self.logger.error("send failed (%s): %s", title, exc)
                return False

    def _send_lifter_request(
        self,
        title: str,
        data: Optional[Dict] = None,
        *,
        guid: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> bool:
        if not self.enable_chassis_lifter:
            self.logger.warning("lifter stream channel is disabled, dropping: %s", title)
            return False
        if expected_generation is None:
            expected_generation = self._bound_command_generation()
        with self._connection_state_lock:
            if (
                not self.connected
                or not self._session_ready.is_set()
                or (
                    expected_generation is not None
                    and self._session_generation != expected_generation
                )
            ):
                self.logger.warning("main session not ready, dropping: %s", title)
                return False
        # A lifter send can block on socket I/O.  Do not hold the main-session
        # lock across it: feedback and arm/chassis sends must keep using the
        # main socket while the dedicated channel is congested.
        with self._lifter_connection_state_lock:
            # Lifecycle transitions never take the lifter connection lock, so
            # taking the transition lock here cannot invert their lock order.
            # It keeps a generation change from racing the final validation and
            # the dedicated send without serializing ordinary main-socket I/O.
            with self._connection_transition_lock:
                with self._connection_state_lock:
                    if (
                        not self.connected
                        or not self._session_ready.is_set()
                        or (
                            expected_generation is not None
                            and self._session_generation != expected_generation
                        )
                    ):
                        self.logger.warning("main session not ready, dropping: %s", title)
                        return False
                client = self._lifter_command_ws_client
                accid = self._lifter_command_accid
                if not client or not self._lifter_command_connected or not accid:
                    self.logger.warning(
                        "lifter command ws not connected, dropping: %s",
                        title,
                    )
                    return False
                try:
                    message = {
                        "accid": accid,
                        "title": title,
                        "timestamp": int(time.time() * 1000),
                        "guid": guid or self._new_guid(),
                        "data": data or {},
                    }
                    client.send(json.dumps(message))
                    return True
                except Exception as exc:
                    self.logger.error(
                        "lifter command send failed (%s): %s",
                        title,
                        exc,
                    )
                    return False

    def _on_open(self, ws) -> None:
        self.logger.info("connected: %s:%s", self.config.robot_ip, self.config.port)
        with self._connection_transition_lock:
            with self._message_dispatch_lock:
                with self._connection_state_lock:
                    if ws is not self.ws_client:
                        return
                    self._session_generation += 1
                    generation = self._session_generation
                    with self._abort_socket_lock:
                        self._main_abort_socket = (
                            generation,
                            getattr(ws, "sock", None),
                        )
                    self.connected = True
                    self.accid = None
                    self._session_ready.clear()
                self._notify_connection_state(False, generation)
        self._feedback_wakeup.set()

    def _on_message(self, ws, message: str) -> None:
        try:
            root = json.loads(message)
            title = root.get("title", "")
            if title == "notify_robot_info":
                self._handle_robot_info(ws, root)
                return
            with self._message_dispatch_lock:
                with self._connection_state_lock:
                    if (
                        (ws is not None and ws is not self.ws_client)
                        or not self.connected
                        or not self._session_ready.is_set()
                    ):
                        self.logger.debug("ignoring %s before session readiness", title)
                        return
                if title == "response_get_joint_state":
                    self._handle_joint_state(root)
                elif title == "response_get_limx_2fclaw_state":
                    self._handle_gripper_state(root)
                elif title == "response_get_servo_pose":
                    self._handle_pose_state(root)
                elif title == "response_get_move_pose":
                    if not self._handle_pose_state(root):
                        self._handle_ee_pose(root)
                elif title == "response_moveh":
                    self._handle_head_move_response(root)
                elif title == "response_set_brainco2_hand_cmd":
                    self._handle_brainco2_command_response(root)
                elif title == "response_get_brainco2_hand_state":
                    self._handle_brainco2_state(root)
                elif title == "response_chassis_state":
                    self._handle_chassis_state(root)
                elif title == "response_get_lifter_position":
                    self._handle_lifter_position(root)
                elif title == "response_lifter_state":
                    self._handle_lifter_raw_state(root)
                elif title not in {
                    "response_servoj",
                    "response_set_limx_2fclaw_cmd",
                    "response_movej",
                    "response_chassis_move",
                }:
                    self.logger.debug("rx: %s", title)
        except json.JSONDecodeError:
            self.logger.error("bad json: %s", message)
        except Exception as exc:
            self.logger.error("on_message error: %s", exc)

    def _handle_robot_info(self, ws, root: Dict) -> None:
        """Mark one socket generation ready only after a valid robot identity."""
        accid = root.get("accid")
        if not isinstance(accid, str) or not accid.strip():
            self.logger.warning("invalid notify_robot_info identity: %s", accid)
            return
        now = float(self._feedback_clock())
        if not np.isfinite(now):
            self._reset_main_connection_state()
            raise ValueError("feedback clock must return a finite value")
        with self._connection_transition_lock:
            with self._message_dispatch_lock:
                with self._connection_state_lock:
                    if (
                        (ws is not None and ws is not self.ws_client)
                        or not self.connected
                    ):
                        return
                    if self._session_ready.is_set():
                        if self.accid != accid:
                            self.logger.warning(
                                "ignoring robot identity change within one session: %s",
                                accid,
                            )
                        return
                    self.accid = accid
                    generation = self._session_generation

                self._main_feedback_next_at = now
                self._auxiliary_feedback_next_at = now
                with self._brainco2_query_lock:
                    if self._brainco2_polling_enabled:
                        self._brainco2_next_poll_at = now

        # Let consumers drain old-generation commands before the new generation
        # becomes sendable.  Do not hold transport transition/dispatch locks:
        # shutdown must remain able to abort the old socket while a consumer waits.
        self._notify_connection_state(True, generation)

        with self._connection_transition_lock:
            with self._message_dispatch_lock:
                with self._brainco2_query_lock:
                    with self._connection_state_lock:
                        if (
                            ws is not self.ws_client
                            or not self.connected
                            or self._session_generation != generation
                            or self.accid != accid
                        ):
                            return
                        if self._brainco2_polling_enabled:
                            self._brainco2_next_poll_at = now
                        self._session_ready.set()
        self._feedback_wakeup.set()

    def _handle_joint_state(self, root: Dict) -> None:
        data = root.get("data", {})
        if not isinstance(data, dict) or data.get("result") not in (None, "success"):
            self.logger.warning("invalid joint state response: %s", data)
            return
        try:
            joint_q = np.asarray(data.get("q", []), dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            self.logger.warning("invalid joint state response: %s", data)
            return
        if joint_q.shape != (JointIndex.SERVOJ_DIM,) or not np.all(
            np.isfinite(joint_q)
        ):
            self.logger.warning("invalid joint state response: %s", data)
            return

        # 用客户端 wall clock 作为统一时间基准 (与 camera.py 中的 time.time() 一致)
        # 机器人原始时间戳保留在 robot_timestamp 字段供调试。
        client_ts_ms = int(time.time() * 1000)
        received_at = time.monotonic()
        robot_timestamp = root.get("timestamp", -1)
        metadata = self._state_response_metadata(
            root,
            data,
            excluded_data={"q"},
            client_timestamp=client_ts_ms,
        )
        with self._state_lock:
            self._arm_head_snapshot.update(
                joint_q,
                robot_timestamp=robot_timestamp,
                received_at=received_at,
                metadata=metadata,
            )
            self.joint_states["timestamp"] = client_ts_ms
            self.joint_states["robot_timestamp"] = robot_timestamp
            self.joint_states["states"][JointIndex.LEFT_ARM] = joint_q[
                : JointIndex.ARM_JOINT_DIM
            ].tolist()
            self.joint_states["states"][JointIndex.RIGHT_ARM] = joint_q[
                JointIndex.ARM_JOINT_DIM : JointIndex.TOTAL_ARM_DIM
            ].tolist()
            self.joint_states["states"][JointIndex.HEAD_PITCH] = float(joint_q[14])
            self.joint_states["states"][JointIndex.HEAD_YAW] = float(joint_q[15])
            self._legacy_joint_received_at = received_at
            self.joint_states["joint_updated"] = True
            self._try_commit_state()

    def _handle_gripper_state(self, root: Dict) -> None:
        data = root.get("data", {})
        if not isinstance(data, dict) or data.get("result") not in (None, "success"):
            self.logger.warning("invalid gripper state response: %s", data)
            return
        try:
            raw_openings = np.asarray(
                [data["left_opening"], data["right_opening"]],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            self.logger.warning("invalid gripper state response: %s", data)
            return
        if (
            raw_openings.shape != (2,)
            or not np.all(np.isfinite(raw_openings))
            or np.any(raw_openings < 0.0)
            or np.any(raw_openings > 100.0)
        ):
            self.logger.warning("invalid gripper state response: %s", data)
            return

        grippers = raw_openings / 100.0
        client_ts_ms = int(time.time() * 1000)
        received_at = time.monotonic()
        robot_timestamp = root.get("timestamp", -1)
        metadata = self._state_response_metadata(
            root,
            data,
            excluded_data={"left_opening", "right_opening"},
            client_timestamp=client_ts_ms,
        )
        with self._state_lock:
            self._gripper_snapshot.update(
                grippers,
                robot_timestamp=robot_timestamp,
                received_at=received_at,
                metadata=metadata,
            )
            self.joint_states["states"][JointIndex.LEFT_GRIPPER] = float(grippers[0])
            self.joint_states["states"][JointIndex.RIGHT_GRIPPER] = float(grippers[1])
            self._legacy_gripper_received_at = received_at
            self.joint_states["gripper_updated"] = True
            self._try_commit_state()

    @staticmethod
    def _state_response_metadata(
        root: Dict,
        data: Dict,
        *,
        excluded_data: set[str],
        client_timestamp: int,
    ) -> Dict:
        """Copy non-state response fields for diagnostics and later assembly."""
        metadata = {
            key: copy.deepcopy(value)
            for key, value in root.items()
            if key not in {"title", "timestamp", "data", "states"}
        }
        metadata.update({
            key: copy.deepcopy(value)
            for key, value in data.items()
            if key not in excluded_data | {"timestamp", "states"}
        })
        if "timestamp" in data:
            metadata["data_timestamp"] = copy.deepcopy(data["timestamp"])
        metadata["client_timestamp"] = client_timestamp
        return metadata

    def _handle_ee_pose(self, root: Dict) -> None:
        data = root.get("data", {})
        pose = {
            "timestamp": data.get("timestamp", root.get("timestamp", -1)),
            "left_position": data.get("left_position", [-1.0, -1.0, -1.0]),
            "left_quat": data.get("left_quat", [-1.0, -1.0, -1.0, -1.0]),
            "right_position": data.get("right_position", [-1.0, -1.0, -1.0]),
            "right_quat": data.get("right_quat", [-1.0, -1.0, -1.0, -1.0]),
            "result": data.get("result"),
        }
        self.ee_pose_states = pose
        with self._queue_lock:
            self.ee_pose_queue.append(pose.copy())

    def _handle_pose_state(self, root: Dict) -> bool:
        """Consume a correlated canonical pose response, if one is pending."""
        guid = root.get("guid")
        with self._pose_query_lock:
            if not self._is_pose_query_guid(guid):
                return False
            if guid != self._pose_query_in_flight:
                return True
            try:
                deadline = self._pose_query_deadline
                if deadline is not None:
                    now = float(self._feedback_clock())
                    if not np.isfinite(now) or now >= deadline:
                        self.logger.warning("expired pose state response: %s", guid)
                        return True
                data = root.get("data", {})
                if not isinstance(data, dict) or data.get("result") != "success":
                    self.logger.warning("invalid pose state response: %s", data)
                    return True
                try:
                    left = self._normalize_pose(
                        data.get("left_position", []),
                        data.get("left_quat", []),
                    )
                    right = self._normalize_pose(
                        data.get("right_position", []),
                        data.get("right_quat", []),
                    )
                except ValueError:
                    self.logger.warning("invalid pose state response: %s", data)
                    return True

                client_ts_ms = int(time.time() * 1000)
                metadata = self._state_response_metadata(
                    root,
                    data,
                    excluded_data={
                        "left_position",
                        "left_quat",
                        "right_position",
                        "right_quat",
                    },
                    client_timestamp=client_ts_ms,
                )
                self._pose_snapshot.update(
                    np.concatenate((left, right)),
                    robot_timestamp=copy.deepcopy(
                        data.get("timestamp", root.get("timestamp", -1))
                    ),
                    received_at=time.monotonic(),
                    metadata=metadata,
                )
                return True
            finally:
                self._pose_query_in_flight = None
                self._pose_query_deadline = None
                self._feedback_wakeup.set()

    def _handle_head_move_response(self, root: Dict) -> None:
        """Forward a move-head acknowledgement to the optional scheduler."""
        data = root.get("data", {})
        result = data.get("result") if isinstance(data, dict) else None
        with self._head_response_callback_lock:
            callback = self._head_response_callback
        if callback is None:
            return
        try:
            callback(root.get("guid"), result)
        except Exception:
            self.logger.exception("head response callback failed")

    def _handle_brainco2_command_response(self, root: Dict) -> bool:
        """Consume a matching hand-command ACK and retain asynchronous faults."""
        guid = root.get("guid")
        with self._brainco2_command_lock:
            self._expire_brainco2_commands_locked(float(self._feedback_clock()))
            if guid is None or guid not in self._brainco2_command_guids:
                return False
            self._brainco2_command_guids.pop(guid)
            self._feedback_wakeup.set()
            data = root.get("data", {})
            if not isinstance(data, dict) or data.get("result") != "success":
                if self._brainco2_fault_message is None:
                    detail = (
                        data.get("message", data.get("result", "invalid ACK"))
                        if isinstance(data, dict)
                        else "invalid ACK"
                    )
                    self._brainco2_fault_message = (
                        f"BrainCo2 command failed: {detail}"
                    )
            return True

    def _expire_brainco2_commands_locked(self, now: float) -> None:
        expired = [
            guid
            for guid, deadline in self._brainco2_command_guids.items()
            if now >= deadline
        ]
        for guid in expired:
            self._brainco2_command_guids.pop(guid, None)
        if expired and self._brainco2_fault_message is None:
            self._brainco2_fault_message = "BrainCo2 acknowledgement timed out"

    def _handle_brainco2_state(self, root: Dict) -> bool:
        """Consume one correlated, complete paired BrainCo2 state response."""
        guid = root.get("guid")
        generation = self._session_generation
        with self._brainco2_query_lock:
            if guid is None or guid != self._brainco2_query_in_flight:
                return False
            deadline = self._brainco2_query_deadline
            if deadline is not None and float(self._feedback_clock()) >= deadline:
                self._brainco2_query_in_flight = None
                self._brainco2_query_deadline = None
                self._feedback_wakeup.set()
                return True
            try:
                data = root.get("data", {})
                if not isinstance(data, dict) or data.get("result") != "success":
                    self.logger.warning("invalid BrainCo2 state response: %s", data)
                    return True
                try:
                    left = self._finite_brainco2_vector(data.get("left_pos", []))
                    right = self._finite_brainco2_vector(data.get("right_pos", []))
                except ValueError:
                    self.logger.warning("invalid BrainCo2 state response: %s", data)
                    return True

                received_at = time.monotonic()
                client_ts_ms = int(time.time() * 1000)
                robot_timestamp = copy.deepcopy(
                    data.get("timestamp", root.get("timestamp", -1))
                )
                metadata = self._state_response_metadata(
                    root,
                    data,
                    excluded_data={"left_pos", "right_pos"},
                    client_timestamp=client_ts_ms,
                )
                metadata["robot_timestamp"] = copy.deepcopy(robot_timestamp)
                metadata["received_at"] = received_at
                metadata["session_generation"] = generation
                self._brainco2_snapshot.update(
                    np.concatenate((left, right)),
                    robot_timestamp=robot_timestamp,
                    received_at=received_at,
                    metadata=metadata,
                )
                self._brainco2_state_generation = generation
                return True
            finally:
                self._brainco2_query_in_flight = None
                self._brainco2_query_deadline = None
                self._feedback_wakeup.set()

    @staticmethod
    def _finite_brainco2_vector(value) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("BrainCo2 value must be finite shape (6,)") from exc
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError("BrainCo2 value must be finite shape (6,)")
        return vector

    @staticmethod
    def _finite_state_scalar(value, *, name: str) -> float:
        """Parse one real numeric state scalar without accepting strings/bools."""
        try:
            scalar = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite numeric data") from exc
        if (
            scalar.shape != ()
            or not (
                np.issubdtype(scalar.dtype, np.integer)
                or np.issubdtype(scalar.dtype, np.floating)
            )
            or not bool(np.isfinite(scalar))
        ):
            raise ValueError(f"{name} must be finite numeric data")
        return float(scalar)

    @staticmethod
    def _normalize_pose(position, quaternion) -> np.ndarray:
        """Validate and combine XYZ with a normalized WXYZ quaternion."""
        try:
            position_array = np.asarray(position, dtype=np.float64)
            quaternion_array = np.asarray(quaternion, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("pose must contain finite XYZ and WXYZ values") from exc
        if (
            position_array.shape != (3,)
            or quaternion_array.shape != (4,)
            or not np.all(np.isfinite(position_array))
            or not np.all(np.isfinite(quaternion_array))
        ):
            raise ValueError("pose must contain finite XYZ and WXYZ values")
        quaternion_scale = float(np.max(np.abs(quaternion_array)))
        if quaternion_scale == 0.0:
            raise ValueError("pose quaternion norm must be at least 1e-8")
        scaled_quaternion = quaternion_array / quaternion_scale
        scaled_norm = float(np.linalg.norm(scaled_quaternion))
        if quaternion_scale < 1e-8 / scaled_norm:
            raise ValueError("pose quaternion norm must be at least 1e-8")
        return np.concatenate((position_array, scaled_quaternion / scaled_norm))

    def _handle_chassis_state(self, root: Dict) -> None:
        data = root.get("data", {})
        if not isinstance(data, dict) or data.get("result") not in (None, "success"):
            self.logger.warning("invalid chassis state response: %s", data)
            return
        values = data.get("data", [])
        try:
            if not isinstance(values, (list, tuple, np.ndarray)):
                raise ValueError("chassis state must be a vector")
            linear_velocity = self._finite_state_scalar(
                values[0],
                name="chassis linear velocity",
            )
            angular_velocity = self._finite_state_scalar(
                values[1],
                name="chassis angular velocity",
            )
            received_at = self._finite_state_scalar(
                time.monotonic(),
                name="chassis receive timestamp",
            )
        except (IndexError, ValueError):
            self.logger.warning("invalid chassis state response: %s", data)
            return
        with self._state_lock:
            self.chassis_states = {
                "timestamp": data.get("timestamp", root.get("timestamp", -1)),
                "received_at": received_at,
                "linear_velocity": linear_velocity,
                "angular_velocity": angular_velocity,
                "valid": True,
            }

    def _handle_lifter_position(self, root: Dict) -> None:
        self._complete_lifter_feedback_request(root)
        data = root.get("data", {})
        if (
            not isinstance(data, dict)
            or data.get("result") not in (None, "success")
            or "position" not in data
        ):
            self.logger.warning("invalid lifter position response: %s", data)
            return
        try:
            position = self._finite_state_scalar(
                data["position"],
                name="lifter position",
            )
            received_at = self._finite_state_scalar(
                time.monotonic(),
                name="lifter receive timestamp",
            )
        except ValueError:
            self.logger.warning("invalid lifter position response: %s", data)
            return
        with self._state_lock:
            self.lifter_states = {
                "timestamp": data.get("timestamp", root.get("timestamp", -1)),
                "received_at": received_at,
                "value": position,
                "valid": True,
            }

    def _handle_lifter_raw_state(self, root: Dict) -> None:
        self._complete_lifter_feedback_request(root)
        data = root.get("data", {})
        if not isinstance(data, dict) or data.get("result") not in (None, "success"):
            self.logger.warning("invalid raw lifter state response: %s", data)
            return
        q = data.get("q", [])
        try:
            if not isinstance(q, (list, tuple, np.ndarray)):
                raise ValueError("raw lifter state must be a vector")
            position = self._finite_state_scalar(q[0], name="raw lifter state")
            received_at = self._finite_state_scalar(
                time.monotonic(),
                name="lifter receive timestamp",
            )
        except (IndexError, ValueError):
            self.logger.warning("invalid raw lifter state response: %s", data)
            return
        with self._state_lock:
            self.lifter_states = {
                "timestamp": data.get("timestamp", root.get("timestamp", -1)),
                "received_at": received_at,
                "value": position,
                "valid": True,
            }

    def _handle_lifter_stream_response(self, root: Dict, *, failed: bool) -> None:
        """Log a streaming acknowledgement without changing command state."""
        data = root.get("data", {})
        result = data.get("result") if isinstance(data, dict) else None
        if failed or result != "success":
            self.logger.error(
                "lifter stream command failed (guid=%s): %s",
                root.get("guid"),
                data,
            )

    def _try_commit_state(self) -> None:
        """Commit to queue only when both joint and gripper frames have arrived."""
        if (
            self.joint_states["joint_updated"]
            and self.joint_states["gripper_updated"]
            and self.joint_states["states"][JointIndex.LEFT_ARM_START] != -1
            and self.joint_states["timestamp"] != -1
            and self._legacy_joint_received_at is not None
            and self._legacy_gripper_received_at is not None
        ):
            state = copy.deepcopy(self.joint_states)
            state["received_at"] = min(
                self._legacy_joint_received_at,
                self._legacy_gripper_received_at,
            )
            with self._queue_lock:
                self.joint_state_queue.append(state)
            self.joint_states["joint_updated"] = False
            self.joint_states["gripper_updated"] = False

    def _on_close(self, ws, status_code, msg) -> None:
        self.logger.warning("ws closed: %s - %s", status_code, msg)
        self._reset_main_connection_state()

    def _reset_main_connection_state(self) -> None:
        """Clear state tied to one main-websocket connection attempt."""
        with self._connection_transition_lock:
            with self._connection_state_lock:
                was_active = self.connected or self._session_ready.is_set()
                self.connected = False
                self.accid = None
                self._session_ready.clear()
                generation = self._session_generation

            if was_active:
                self._notify_connection_state(False, generation)

            # An admitted message holds this lock through its state commit.
            # Marking the session unready first prevents any later message from
            # entering while this reset waits for an older handler to finish.
            with self._message_dispatch_lock:
                with self._pose_query_lock:
                    self._pose_query_in_flight = None
                    self._pose_query_deadline = None
                with self._brainco2_query_lock:
                    self._brainco2_query_in_flight = None
                    self._brainco2_query_deadline = None
                    self._brainco2_next_poll_at = None
                    self._brainco2_state_generation = None
                with self._brainco2_command_lock:
                    self._expire_brainco2_commands_locked(
                        float(self._feedback_clock())
                    )
                    if self._brainco2_command_guids:
                        self._brainco2_session_uncertain = True
                    self._brainco2_command_guids.clear()

                with self._state_lock:
                    self._arm_head_snapshot.clear()
                    self._gripper_snapshot.clear()
                    self._pose_snapshot.clear()
                    self._brainco2_snapshot.clear()
                    self.joint_states = {
                        "timestamp": -1,
                        "robot_timestamp": -1,
                        "states": [-1.0] * JointIndex.STATE_DIM,
                        "joint_updated": False,
                        "gripper_updated": False,
                    }
                    self.ee_pose_states = {
                        "timestamp": -1,
                        "left_position": [-1.0, -1.0, -1.0],
                        "left_quat": [-1.0, -1.0, -1.0, -1.0],
                        "right_position": [-1.0, -1.0, -1.0],
                        "right_quat": [-1.0, -1.0, -1.0, -1.0],
                    }
                    self.chassis_states = {
                        "timestamp": -1,
                        "received_at": None,
                        "linear_velocity": 0.0,
                        "angular_velocity": 0.0,
                        "valid": False,
                    }
                    self.lifter_states = {
                        "timestamp": -1,
                        "received_at": None,
                        "value": 0.0,
                        "valid": False,
                    }
                    self._legacy_joint_received_at = None
                    self._legacy_gripper_received_at = None
                    with self._queue_lock:
                        self.joint_state_queue.clear()
                        self.ee_pose_queue.clear()

                with self._lifter_feedback_lock:
                    self._lifter_feedback_in_flight = None
                    self._lifter_feedback_deadline = None
        self._feedback_wakeup.set()

    def register_connection_state_callback(
        self,
        callback: Callable[[bool, int], None],
    ) -> None:
        """Register the sole non-blocking main-session state callback."""
        if not callable(callback):
            raise TypeError("connection state callback must be callable")
        with self._connection_transition_lock:
            with self._connection_state_lock:
                self._connection_state_callback = callback
                ready = self.connected and self._session_ready.is_set()
                generation = self._session_generation
            self._invoke_connection_state_callback(callback, ready, generation)

    def unregister_connection_state_callback(self) -> None:
        """Remove the current main-session callback, if any."""
        with self._connection_transition_lock:
            with self._connection_state_lock:
                self._connection_state_callback = None

    def _notify_connection_state(self, ready: bool, generation: int) -> None:
        with self._connection_state_lock:
            callback = self._connection_state_callback
        if callback is not None:
            self._invoke_connection_state_callback(callback, ready, generation)

    def _invoke_connection_state_callback(
        self,
        callback: Callable[[bool, int], None],
        ready: bool,
        generation: int,
    ) -> None:
        try:
            callback(bool(ready), int(generation))
        except Exception:
            self.logger.exception("connection state callback failed")

    def _on_error(self, ws, error) -> None:
        self.logger.error("ws error: %s", error)

    def _on_lifter_command_open(self, ws) -> None:
        self.logger.info(
            "lifter command connected: %s:%s",
            self.config.robot_ip,
            self.config.port,
        )
        with self._lifter_connection_state_lock:
            if ws is not self._lifter_command_ws_client:
                return
            self._lifter_command_session_generation += 1
            with self._abort_socket_lock:
                self._lifter_abort_socket = (
                    self._lifter_command_session_generation,
                    getattr(ws, "sock", None),
                )
            self._lifter_command_socket_open = True
            self._lifter_command_connected = False
            self._lifter_command_accid = None
            self._lifter_command_ready.clear()

    def _on_lifter_command_message(self, ws, message: str) -> None:
        try:
            root = json.loads(message)
            title = root.get("title", "")
            if title == "notify_robot_info":
                accid = root.get("accid")
                if not isinstance(accid, str) or not accid.strip():
                    self.logger.warning(
                        "invalid lifter command robot identity: %s",
                        accid,
                    )
                    return
                with self._lifter_connection_state_lock:
                    if (
                        ws is not self._lifter_command_ws_client
                        or not self._lifter_command_socket_open
                    ):
                        return
                    self._lifter_command_accid = accid
                    self._lifter_command_connected = True
                    self._lifter_command_ready.set()
                return
            with self._lifter_connection_state_lock:
                if (
                    ws is not self._lifter_command_ws_client
                    or not self._lifter_command_connected
                ):
                    return
            if title in {
                "response_lifter_stream_position",
                "response_lifter_stream_velocity",
            }:
                self._handle_lifter_stream_response(root, failed=False)
            elif title in {
                "fail_lifter_stream_position",
                "fail_lifter_stream_velocity",
            }:
                self._handle_lifter_stream_response(root, failed=True)
            else:
                self.logger.debug("lifter command rx: %s", title)
        except json.JSONDecodeError:
            self.logger.error("bad lifter command json: %s", message)
        except Exception as exc:
            self.logger.error("lifter command on_message error: %s", exc)

    def _on_lifter_command_close(self, ws, status_code, msg) -> None:
        self.logger.warning(
            "lifter command ws closed: %s - %s", status_code, msg
        )
        self._reset_lifter_command_connection_state()

    def _reset_lifter_command_connection_state(self) -> None:
        """Clear state tied to one lifter-websocket connection attempt."""
        with self._lifter_connection_state_lock:
            self._lifter_command_socket_open = False
            self._lifter_command_connected = False
            self._lifter_command_accid = None
            self._lifter_command_ready.clear()

    def _on_lifter_command_error(self, ws, error) -> None:
        self.logger.error("lifter command ws error: %s", error)

    def _connect(self) -> None:
        url = f"ws://{self.config.robot_ip}:{self.config.port}"
        self.logger.info("connecting: %s", url)
        self.ws_client = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error,
        )
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()

    def _run_websocket(self) -> None:
        self._run_websocket_attempts(
            self.ws_client,
            self._reset_main_connection_state,
            "ws",
        )

    def _connect_lifter_command_channel(self) -> None:
        url = f"ws://{self.config.robot_ip}:{self.config.port}"
        self.logger.info("connecting lifter command channel: %s", url)
        self._lifter_command_ws_client = websocket.WebSocketApp(
            url,
            on_open=self._on_lifter_command_open,
            on_message=self._on_lifter_command_message,
            on_close=self._on_lifter_command_close,
            on_error=self._on_lifter_command_error,
        )
        self._lifter_command_ws_thread = threading.Thread(
            target=self._run_lifter_command_websocket,
            daemon=True,
        )
        self._lifter_command_ws_thread.start()

    def _run_lifter_command_websocket(self) -> None:
        self._run_websocket_attempts(
            self._lifter_command_ws_client,
            self._reset_lifter_command_connection_state,
            "lifter command ws",
        )

    def _run_websocket_attempts(
        self,
        client,
        reset_connection_state: Callable[[], None],
        label: str,
    ) -> None:
        """Run complete websocket attempts with interruptible retry backoff."""
        while not self.should_exit:
            try:
                client.run_forever()
            except Exception as exc:
                self.logger.error("%s attempt failed: %s", label, exc)
            finally:
                reset_connection_state()
            if self.should_exit:
                return
            if self._reconnect_wakeup.wait(
                _WEBSOCKET_RECONNECT_DELAY_SECONDS
            ):
                return

    # ---------------------------------------------------------------- polling

    def _start_polling_thread(self) -> None:
        self._polling_thread = threading.Thread(
            target=self._poll_feedback,
            daemon=True,
        )
        self._polling_thread.start()
        self.logger.info("state polling started (%s Hz)", self.polling_rate)

    def _poll_feedback(self) -> None:
        while not self.should_exit:
            self._feedback_wakeup.clear()
            self._poll_feedback_once()
            if self.should_exit:
                break
            deadline = self._next_feedback_deadline()
            if deadline is None:
                self._feedback_wakeup.wait(None)
            else:
                now = float(self._feedback_clock())
                wait_timeout = min(
                    threading.TIMEOUT_MAX,
                    max(0.0, deadline - now),
                )
                self._feedback_wakeup.wait(wait_timeout)

    def _poll_feedback_once(self) -> None:
        """Poll each feedback source only when its independent gate is due."""
        now = float(self._feedback_clock())
        if not np.isfinite(now):
            raise ValueError("feedback clock must return a finite value")
        if self.should_exit:
            return
        with self._pose_query_lock:
            self._expire_pose_query_locked(now)
        main_feedback_enabled = self.poll_joint_state or self.poll_gripper_state
        if not self._session_ready.is_set():
            if (
                main_feedback_enabled
                and now + 1e-12 >= self._main_feedback_next_at
            ):
                self._main_feedback_next_at = self._advance_feedback_deadline(
                    self._main_feedback_next_at,
                    self._main_feedback_period,
                    now,
                )
            if now + 1e-12 >= self._auxiliary_feedback_next_at:
                self._auxiliary_feedback_next_at = self._advance_feedback_deadline(
                    self._auxiliary_feedback_next_at,
                    self._auxiliary_feedback_period,
                    now,
                )
            return
        if (
            main_feedback_enabled
            and now + 1e-12 >= self._main_feedback_next_at
        ):
            if self.poll_joint_state:
                self._send_request("request_get_joint_state")
            if self.poll_gripper_state:
                self._send_request("request_get_limx_2fclaw_state")
            self._main_feedback_next_at = self._advance_feedback_deadline(
                self._main_feedback_next_at,
                self._main_feedback_period,
                now,
            )
        if (
            self.enable_chassis_lifter
            and now + 1e-12 >= self._auxiliary_feedback_next_at
        ):
            self._send_request("request_chassis_state")
            self._request_lifter_feedback()
            self._auxiliary_feedback_next_at = self._advance_feedback_deadline(
                self._auxiliary_feedback_next_at,
                self._auxiliary_feedback_period,
                now,
            )
        self.poll_enabled_state(now=now)

    @staticmethod
    def _advance_feedback_deadline(deadline: float, period: float, now: float) -> float:
        """Advance a periodic gate past ``now`` without burst catch-up sends."""
        periods = int(max(0.0, now - deadline) // period) + 1
        return deadline + periods * period

    def _next_feedback_deadline(self) -> Optional[float]:
        """Return the next useful wake-up, excluding already-blocked poll slots."""
        deadlines = []
        if self.poll_joint_state or self.poll_gripper_state:
            deadlines.append(self._main_feedback_next_at)
        if self.enable_chassis_lifter:
            deadlines.append(self._auxiliary_feedback_next_at)
        with self._pose_query_lock:
            if (
                self._pose_query_in_flight is not None
                and self._pose_query_deadline is not None
            ):
                deadlines.append(self._pose_query_deadline)
        with self._brainco2_command_lock:
            deadlines.extend(self._brainco2_command_guids.values())
        with self._brainco2_query_lock:
            if self._brainco2_query_in_flight is not None:
                if self._brainco2_query_deadline is not None:
                    deadlines.append(self._brainco2_query_deadline)
            elif (
                self._brainco2_polling_enabled
                and self._brainco2_next_poll_at is not None
            ):
                deadlines.append(self._brainco2_next_poll_at)
        return min(deadlines) if deadlines else None

    def _expire_pose_query_locked(self, now: float) -> None:
        deadline = self._pose_query_deadline
        if (
            self._pose_query_in_flight is not None
            and deadline is not None
            and now >= deadline
        ):
            self._pose_query_in_flight = None
            self._pose_query_deadline = None
            self._feedback_wakeup.set()

    def enable_brainco2_polling(self, rate: float, query_timeout: float) -> None:
        """Enable BrainCo2 feedback on the existing transport polling loop."""
        try:
            rate_value = float(rate)
            timeout_value = float(query_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("BrainCo2 polling rate and query timeout must be positive") from exc
        if (
            not np.isfinite(rate_value)
            or rate_value <= 0.0
            or not np.isfinite(timeout_value)
            or timeout_value <= 0.0
        ):
            raise ValueError("BrainCo2 polling rate and query timeout must be positive")
        now = float(self._feedback_clock())
        if not np.isfinite(now):
            raise ValueError("feedback clock must return a finite value")
        with self._brainco2_query_lock:
            self._brainco2_polling_period = 1.0 / rate_value
            self._brainco2_query_timeout = timeout_value
            self._brainco2_next_poll_at = (
                now if self._session_ready.is_set() else None
            )
            self._brainco2_polling_enabled = True
            if self._brainco2_query_in_flight is not None:
                self._brainco2_query_deadline = now + timeout_value
        with self._brainco2_command_lock:
            self._brainco2_command_ack_timeout = timeout_value
        self._feedback_wakeup.set()

    def disable_brainco2_polling(self) -> None:
        """Stop BrainCo2 polling and invalidate any outstanding state query."""
        with self._brainco2_query_lock:
            self._brainco2_polling_enabled = False
            self._brainco2_next_poll_at = None
            self._brainco2_query_in_flight = None
            self._brainco2_query_deadline = None
        self._feedback_wakeup.set()

    def poll_enabled_state(self, now: Optional[float] = None) -> None:
        """Advance explicitly enabled optional feedback sources once."""
        current = float(self._feedback_clock()) if now is None else float(now)
        if not np.isfinite(current):
            raise ValueError("feedback clock must return a finite value")
        with self._brainco2_command_lock:
            self._expire_brainco2_commands_locked(current)
        with self._brainco2_query_lock:
            self._expire_brainco2_query_locked(current)
            if (
                not self._brainco2_polling_enabled
                or self.should_exit
                or not self._session_ready.is_set()
            ):
                return
            if self._brainco2_query_in_flight is not None:
                return
            next_poll = self._brainco2_next_poll_at
            if next_poll is None or current + 1e-12 < next_poll:
                return
            self._brainco2_next_poll_at = self._advance_feedback_deadline(
                next_poll,
                self._brainco2_polling_period,
                current,
            )
            try:
                self.request_brainco2_state()
            except StateError as exc:
                self.logger.warning("BrainCo2 state request failed: %s", exc)

    def _expire_brainco2_query_locked(self, now: float) -> None:
        deadline = self._brainco2_query_deadline
        if (
            self._brainco2_query_in_flight is not None
            and deadline is not None
            and now >= deadline
        ):
            self._brainco2_query_in_flight = None
            self._brainco2_query_deadline = None

    def _request_lifter_feedback(self) -> None:
        title = (
            "request_lifter_state"
            if self.lifter_state_source == "raw_q"
            else "request_get_lifter_position"
        )
        now = float(self._feedback_clock())
        if not np.isfinite(now):
            self.logger.error("feedback clock returned a non-finite value")
            return
        with self._lifter_feedback_lock:
            if (
                self._lifter_feedback_in_flight is not None
                and self._lifter_feedback_deadline is not None
                and now >= self._lifter_feedback_deadline
            ):
                self._lifter_feedback_in_flight = None
                self._lifter_feedback_deadline = None
            if self._lifter_feedback_in_flight is not None:
                return
            guid = self._new_guid()
            self._lifter_feedback_in_flight = guid
            self._lifter_feedback_deadline = now + _LIFTER_FEEDBACK_TIMEOUT_SECONDS
            if not self._send_request(title, guid=guid):
                self._lifter_feedback_in_flight = None
                self._lifter_feedback_deadline = None

    def _complete_lifter_feedback_request(self, root: Dict) -> None:
        with self._lifter_feedback_lock:
            if root.get("guid") == self._lifter_feedback_in_flight:
                self._lifter_feedback_in_flight = None
                self._lifter_feedback_deadline = None

    # ----------------------------------------------------------- RobotTransport

    def send_joint_cmd(self, q: Union[List[float], np.ndarray]) -> None:
        """Send one 16-dim servoj setpoint (no rate limiting — caller paces)."""
        if isinstance(q, np.ndarray):
            q = q.tolist()
        if len(q) != self.SERVOJ_DIM:
            raise CommandError(f"expected {self.SERVOJ_DIM}-dim q, got {len(q)}")
        if not self._send_request("request_servoj", {"q": q, "filter_ratio": 1.0}):
            raise CommandError("servoj send failed")

    def send_pose_cmd(
        self,
        left: Union[List[float], np.ndarray],
        right: Union[List[float], np.ndarray],
    ) -> None:
        """Send paired Cartesian poses in XYZ + WXYZ order."""
        poses = {}
        for key, value in (("left_pos", left), ("right_pos", right)):
            try:
                pose = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise CommandError(f"{key} must be a finite shape (7,) pose") from exc
            if pose.shape != (7,):
                raise CommandError(f"{key} must be a finite shape (7,) pose")
            try:
                poses[key] = self._normalize_pose(pose[:3], pose[3:]).tolist()
            except ValueError as exc:
                raise CommandError(f"invalid {key}: {exc}") from exc

        if not self._send_request("request_servop", poses):
            raise CommandError("servop send failed")

    def send_head_move(
        self,
        target: Union[List[float], np.ndarray],
        duration: float,
        guid: str,
    ) -> bool:
        """Send one correlated two-joint head move for the ServoP scheduler."""
        try:
            joints = np.asarray(target, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CommandError("head target must be finite shape (2,)") from exc
        if joints.shape != (JointIndex.HEAD_DIM,) or not np.all(np.isfinite(joints)):
            raise CommandError("head target must be finite shape (2,)")
        try:
            move_duration = float(duration)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CommandError("head move duration must be finite and positive") from exc
        if not np.isfinite(move_duration) or move_duration <= 0.0:
            raise CommandError("head move duration must be finite and positive")
        if not isinstance(guid, str) or not guid.strip():
            raise CommandError("head move GUID must be a non-empty string")
        return self._send_request(
            "request_moveh",
            {"joint": joints.tolist(), "time": move_duration},
            guid=guid,
        )

    def register_head_response_callback(
        self,
        callback: Callable[[Optional[str], object], None],
    ) -> None:
        """Register the sole callback for correlated move-head responses."""
        if not callable(callback):
            raise TypeError("head response callback must be callable")
        with self._head_response_callback_lock:
            self._head_response_callback = callback

    def unregister_head_response_callback(self) -> None:
        """Remove the current move-head response callback, if any."""
        with self._head_response_callback_lock:
            self._head_response_callback = None

    def request_pose_state(self, timeout: float = 1.0) -> str:
        """Issue at most one correlated canonical pose query at a time."""
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("timeout must be finite and positive") from exc
        if not np.isfinite(timeout_value) or timeout_value <= 0.0:
            raise ValueError("timeout must be finite and positive")
        if timeout_value > threading.TIMEOUT_MAX:
            raise ValueError("timeout must not exceed threading.TIMEOUT_MAX")
        with self._pose_query_lock:
            now = float(self._feedback_clock())
            if not np.isfinite(now):
                raise StateError("feedback clock returned a non-finite value")
            self._expire_pose_query_locked(now)
            if self._pose_query_in_flight is not None:
                return self._pose_query_in_flight
            deadline = now + timeout_value
            if not np.isfinite(deadline):
                raise ValueError("pose query deadline must be finite")
            guid = self._new_pose_query_guid()
            self._pose_query_in_flight = guid
            self._pose_query_deadline = deadline
            try:
                sent = self._send_request(
                    "request_get_servo_pose",
                    {},
                    guid=guid,
                )
            except Exception as exc:
                if self._pose_query_in_flight == guid:
                    self._pose_query_in_flight = None
                    self._pose_query_deadline = None
                self._feedback_wakeup.set()
                raise StateError("request_pose_state request failed") from exc
            if not sent:
                if self._pose_query_in_flight == guid:
                    self._pose_query_in_flight = None
                    self._pose_query_deadline = None
                self._feedback_wakeup.set()
                raise StateError("request_pose_state request failed")
            self._feedback_wakeup.set()
            return guid

    def set_brainco2_position(
        self,
        left: Union[List[float], np.ndarray],
        right: Union[List[float], np.ndarray],
        command_time: Union[List[float], np.ndarray],
    ) -> str:
        """Send paired raw BrainCo2 positions with mode-one position control."""
        with self._brainco2_command_lock:
            now = float(self._feedback_clock())
            self._expire_brainco2_commands_locked(now)
            if self._brainco2_fault_message is not None:
                raise CommandError(self._brainco2_fault_message)
            if self._brainco2_session_uncertain:
                raise CommandError(
                    "BrainCo2 session lost with unacknowledged command"
                )

            values = {}
            for name, value in (
                ("left_pos", left),
                ("right_pos", right),
                ("command_time", command_time),
            ):
                try:
                    values[name] = self._finite_brainco2_vector(value).tolist()
                except ValueError as exc:
                    raise CommandError(
                        f"{name} must be a finite shape (6,) BrainCo2 vector"
                    ) from exc

            data = {
                "left_mode": 1,
                "left_pos": values["left_pos"],
                "left_time": values["command_time"],
                "right_mode": 1,
                "right_pos": values["right_pos"],
                "right_time": values["command_time"].copy(),
            }
            guid = self._new_guid()
            self._brainco2_command_guids[guid] = (
                now + self._brainco2_command_ack_timeout
            )
            try:
                sent = self._send_request(
                    "request_set_brainco2_hand_cmd",
                    data,
                    guid=guid,
                )
            except Exception as exc:
                self._brainco2_command_guids.pop(guid, None)
                raise CommandError("BrainCo2 command send failed") from exc
            if not sent:
                self._brainco2_command_guids.pop(guid, None)
                raise CommandError("BrainCo2 command send failed")
            self._feedback_wakeup.set()
            return guid

    def request_brainco2_state(self) -> str:
        """Issue at most one correlated BrainCo2 state query at a time."""
        with self._brainco2_query_lock:
            now = float(self._feedback_clock())
            if not np.isfinite(now):
                raise StateError("feedback clock returned a non-finite value")
            self._expire_brainco2_query_locked(now)
            if self._brainco2_query_in_flight is not None:
                return self._brainco2_query_in_flight
            guid = self._new_guid()
            self._brainco2_query_in_flight = guid
            self._brainco2_query_deadline = now + self._brainco2_query_timeout
            try:
                sent = self._send_request(
                    "request_get_brainco2_hand_state",
                    {},
                    guid=guid,
                )
            except Exception as exc:
                if self._brainco2_query_in_flight == guid:
                    self._brainco2_query_in_flight = None
                    self._brainco2_query_deadline = None
                raise StateError("request_brainco2_state request failed") from exc
            if not sent:
                if self._brainco2_query_in_flight == guid:
                    self._brainco2_query_in_flight = None
                    self._brainco2_query_deadline = None
                raise StateError("request_brainco2_state request failed")
            self._feedback_wakeup.set()
            return guid

    def get_brainco2_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> Dict:
        """Return the latest complete paired BrainCo2 position snapshot."""
        item = self._brainco2_snapshot.get(timeout=timeout, max_age=max_age)
        return {
            "timestamp": copy.deepcopy(item.robot_timestamp),
            "states": item.values.tolist(),
            "metadata": copy.deepcopy(item.metadata),
        }

    def raise_brainco2_fault(self) -> None:
        """Raise a retained asynchronous BrainCo2 command failure, if any."""
        with self._brainco2_command_lock:
            self._expire_brainco2_commands_locked(float(self._feedback_clock()))
            fault_message = self._brainco2_fault_message
            session_uncertain = self._brainco2_session_uncertain
        if fault_message is not None:
            raise CommandError(fault_message)
        if session_uncertain:
            raise CommandError("BrainCo2 session lost with unacknowledged command")

    def clear_brainco2_session_uncertainty(self, generation: int) -> None:
        """Clear only lost-ACK uncertainty after fresh state in one ready session."""
        with self._brainco2_command_lock:
            if self._brainco2_fault_message is not None:
                raise CommandError(self._brainco2_fault_message)
            try:
                expected_generation = int(generation)
            except (TypeError, ValueError, OverflowError) as exc:
                raise StateError("invalid BrainCo2 recovery generation") from exc
            with self._brainco2_query_lock:
                recovery_error = (
                    "BrainCo2 recovery requires fresh BrainCo2 state from the "
                    "current ready session"
                )
                try:
                    snapshot = self._brainco2_snapshot.get(
                        timeout=0.0,
                        max_age=self._brainco2_query_timeout,
                    )
                except StateError as exc:
                    raise StateError(recovery_error) from exc
                snapshot_generation = snapshot.metadata.get(
                    "session_generation"
                )
                if (
                    self._brainco2_state_generation != expected_generation
                    or snapshot_generation != expected_generation
                ):
                    raise StateError(recovery_error)
                with self._connection_state_lock:
                    if not (
                        self.connected
                        and self._session_ready.is_set()
                        and self._session_generation == expected_generation
                    ):
                        raise StateError(recovery_error)
                    self._brainco2_session_uncertain = False

    def get_joint_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> Dict:
        timeout_value = self._validate_public_wait_timeout(timeout)
        max_age_value = None if max_age is None else float(max_age)
        if max_age_value is not None and (
            not np.isfinite(max_age_value) or max_age_value < 0.0
        ):
            raise ValueError("max_age must be finite and non-negative")
        deadline = time.monotonic() + timeout_value
        while True:
            with self._queue_lock:
                if self.joint_state_queue:
                    states = self.joint_state_queue.popleft()
                    if len(states["states"]) != JointIndex.STATE_DIM:
                        raise StateError(
                            f"bad state dim: expected {JointIndex.STATE_DIM}, got {len(states['states'])}"
                        )
                    received_at = states.get("received_at")
                    if received_at is None:
                        raise StateError("joint state has no receive timestamp")
                    if (
                        max_age_value is not None
                        and time.monotonic() - received_at > max_age_value
                    ):
                        raise StateError("joint state is stale")
                    return states
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.001, remaining))
        raise StateError(f"get_joint_state timed out ({timeout}s)")

    def get_arm_head_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
        expected_guid: Optional[str] = None,
    ) -> Dict:
        """Return the latest valid 16D arm/head frame independently."""
        if expected_guid is not None and (
            not isinstance(expected_guid, str) or not expected_guid.strip()
        ):
            raise ValueError("expected_guid must be a non-empty string")
        predicate = None
        if expected_guid is not None:
            def matches_expected_guid(item):
                return item.metadata.get("guid") == expected_guid

            predicate = matches_expected_guid
        item = self._arm_head_snapshot.get(
            timeout=timeout,
            max_age=max_age,
            predicate=predicate,
        )
        return {
            "timestamp": item.robot_timestamp,
            "states": item.values.tolist(),
            **item.metadata,
        }

    def get_gripper_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> Dict:
        """Return the latest valid normalized left/right gripper frame."""
        item = self._gripper_snapshot.get(timeout=timeout, max_age=max_age)
        return {
            "timestamp": item.robot_timestamp,
            "states": item.values.tolist(),
            **item.metadata,
        }

    def get_pose_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
        expected_guid: Optional[str] = None,
    ) -> Dict:
        """Return the latest valid canonical 14D Cartesian pose frame."""
        if expected_guid is not None and (
            not isinstance(expected_guid, str) or not expected_guid.strip()
        ):
            raise ValueError("expected_guid must be a non-empty string")
        predicate = None
        if expected_guid is not None:
            def matches_expected_guid(item):
                return item.metadata.get("guid") == expected_guid

            predicate = matches_expected_guid
        item = self._pose_snapshot.get(
            timeout=timeout,
            max_age=max_age,
            predicate=predicate,
        )
        return {
            "timestamp": copy.deepcopy(item.robot_timestamp),
            "states": item.values.tolist(),
            **item.metadata,
        }

    def get_ee_poses(
        self,
        timeout: float = 1.0,
        *,
        expected_generation: Optional[int] = None,
    ) -> Dict:
        """Request and return the latest end-effector poses."""
        timeout_value = self._validate_public_wait_timeout(timeout)
        if expected_generation is not None:
            self._require_init_session(expected_generation)
        with self._queue_lock:
            self.ee_pose_queue.clear()
        if not self._send_request(
            "request_get_move_pose",
            expected_generation=expected_generation,
        ):
            raise StateError("get_ee_poses request failed")
        if expected_generation is not None:
            self._require_init_session(expected_generation)

        deadline = time.monotonic() + timeout_value
        while time.monotonic() < deadline:
            if expected_generation is not None:
                self._require_init_session(expected_generation)
            with self._queue_lock:
                if self.ee_pose_queue:
                    pose = self.ee_pose_queue.popleft()
                    result = pose.get("result")
                    if result not in (None, "success"):
                        raise StateError(f"get_ee_poses failed: {result}")
                    if expected_generation is not None:
                        self._require_init_session(expected_generation)
                    return pose
            time.sleep(0.001)
        raise StateError(f"get_ee_poses timed out ({timeout}s)")

    def find_nearest_state(self, target_timestamp_s: float) -> Optional[Dict]:
        """Find the state in the queue whose timestamp is closest to target.

        Non-destructive: does NOT consume the queue. Returns None if the queue
        is empty. The timestamp field in state dicts is in milliseconds.
        """
        with self._queue_lock:
            if not self.joint_state_queue:
                return None
            target_ms = target_timestamp_s * 1000.0
            best = None
            best_diff = float("inf")
            for state in self.joint_state_queue:
                diff = abs(state["timestamp"] - target_ms)
                if diff < best_diff:
                    best_diff = diff
                    best = state
            return copy.deepcopy(best) if best is not None else None

    def get_head_position(self) -> np.ndarray:
        """Return latest head [pitch, yaw] without dequeuing state."""
        with self._state_lock:
            states = self.joint_states["states"]
            if len(states) < JointIndex.STATE_DIM:
                return np.array([0.0, 0.0])
            return np.array(states[JointIndex.HEAD], dtype=np.float64)

    def set_gripper(
        self,
        left_opening: float = 0.0,
        right_opening: float = 0.0,
        left_speed: float = 100.0,
        left_force: float = 50.0,
        right_speed: float = 100.0,
        right_force: float = 50.0,
    ) -> None:
        data = {
            "left_opening": int(np.clip(left_opening, 0, 100)),
            "left_speed": int(np.clip(left_speed, 0, 100)),
            "left_force": int(np.clip(left_force, 0, 100)),
            "right_opening": int(np.clip(right_opening, 0, 100)),
            "right_speed": int(np.clip(right_speed, 0, 100)),
            "right_force": int(np.clip(right_force, 0, 100)),
        }
        if not self._send_request("request_set_limx_2fclaw_cmd", data):
            raise CommandError("gripper command send failed")

    def get_chassis_lifter_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> Dict:
        """Return planar chassis velocity plus the configured lifter feedback scalar."""
        timeout_value = self._validate_public_wait_timeout(timeout)
        if not self.enable_chassis_lifter:
            raise StateError("chassis/lifter feedback is not enabled for this transport")
        max_age_value = None if max_age is None else float(max_age)
        if max_age_value is not None and (
            not np.isfinite(max_age_value) or max_age_value < 0.0
        ):
            raise ValueError("max_age must be finite and non-negative")
        deadline = time.monotonic() + timeout_value
        while True:
            with self._state_lock:
                if self.chassis_states["valid"] and self.lifter_states["valid"]:
                    try:
                        linear_velocity = self._finite_state_scalar(
                            self.chassis_states["linear_velocity"],
                            name="chassis linear velocity",
                        )
                        angular_velocity = self._finite_state_scalar(
                            self.chassis_states["angular_velocity"],
                            name="chassis angular velocity",
                        )
                        lifter_state = self._finite_state_scalar(
                            self.lifter_states["value"],
                            name="lifter state",
                        )
                        chassis_received_at = self._finite_state_scalar(
                            self.chassis_states["received_at"],
                            name="chassis receive timestamp",
                        )
                        lifter_received_at = self._finite_state_scalar(
                            self.lifter_states["received_at"],
                            name="lifter receive timestamp",
                        )
                        now = self._finite_state_scalar(
                            time.monotonic(),
                            name="local receive clock",
                        )
                    except (KeyError, ValueError) as exc:
                        raise StateError("chassis/lifter state contains invalid values") from exc
                    if (
                        max_age_value is not None
                        and now - chassis_received_at > max_age_value
                    ):
                        raise StateError("chassis state is stale")
                    if (
                        max_age_value is not None
                        and now - lifter_received_at > max_age_value
                    ):
                        raise StateError("lifter state is stale")
                    return {
                        "linear_velocity": linear_velocity,
                        "angular_velocity": angular_velocity,
                        "lifter_state": lifter_state,
                        "chassis_timestamp": self.chassis_states["timestamp"],
                        "lifter_timestamp": self.lifter_states["timestamp"],
                        "chassis_received_at": chassis_received_at,
                        "lifter_received_at": lifter_received_at,
                    }
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.001, remaining))
        raise StateError(f"get_chassis_lifter_state timed out ({timeout}s)")

    def command_chassis(self, linear_x: float, angular_z: float) -> None:
        linear_x = float(linear_x)
        angular_z = float(angular_z)
        if not np.isfinite(linear_x) or not np.isfinite(angular_z):
            raise CommandError("chassis command values must be finite")
        data = {
            "x": float(np.clip(linear_x, -1.0, 1.0)),
            "yaw": float(np.clip(angular_z, -1.0, 1.0)),
        }
        if not self._send_request("request_chassis_move", data):
            raise CommandError("chassis command send failed")

    def stop_chassis(self) -> None:
        self.command_chassis(0.0, 0.0)

    def stream_lifter_position(
        self,
        position_mm: float,
        velocity_mm_s: float = 80.0,
    ) -> None:
        """Send one lifter setpoint frame on the dedicated stream channel."""
        try:
            position = float(position_mm)
            velocity = float(velocity_mm_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CommandError("lifter stream position and velocity must be finite") from exc
        if not np.isfinite(position):
            raise CommandError(f"lifter stream position must be finite mm, got {position}")
        if not np.isfinite(velocity) or velocity <= 0.0:
            raise CommandError(
                f"lifter stream velocity must be finite and positive mm/s, got {velocity}"
            )
        if not self._send_lifter_request(
            "request_lifter_stream_position",
            {"position": round(position, 1), "velocity": velocity},
            expected_generation=self._bound_command_generation(),
        ):
            raise CommandError("lifter stream position command send failed")

    def stop_lifter_stream(self) -> None:
        """Stop lifter streaming without waiting for an acknowledgement."""
        if not self._send_lifter_request(
            "request_lifter_stream_velocity",
            {"velocity": 0.0},
            expected_generation=self._bound_command_generation(),
        ):
            raise CommandError("lifter stream stop command send failed")

    def wait_until_reached(
        self,
        target_joints: Union[List[float], np.ndarray],
        tolerance: float = 0.05,
        timeout: float = 10.0,
    ) -> bool:
        timeout_value = self._validate_public_wait_timeout(timeout)
        if isinstance(target_joints, np.ndarray):
            target_joints = target_joints.tolist()
        target = np.array(target_joints)
        deadline = time.monotonic() + timeout_value
        while time.monotonic() < deadline:
            try:
                states = self.get_joint_state(timeout=1.0)
                current = states["states"]
                arm = np.array(current[JointIndex.LEFT_ARM] + current[JointIndex.RIGHT_ARM])
                diff = arm - target
                err = float(np.max(np.abs(diff)))
                if err < tolerance:
                    self.logger.info("reached target (err=%.4f)", err)
                    return True
                time.sleep(0.1)
            except StateError:
                self.logger.warning("state read failed; retrying")
                continue
        self.logger.warning("wait_until_reached timed out (%.1fs)", timeout_value)
        return False

    def is_connected(self) -> bool:
        with self._connection_state_lock:
            return self.connected and self._session_ready.is_set()

    def _try_stop_chassis_before_disconnect(
        self,
        attempt_started: Optional[threading.Event] = None,
    ) -> None:
        """Send a best-effort stop without waiting behind an in-flight send."""
        if not self.enable_chassis_lifter:
            if attempt_started is not None:
                attempt_started.set()
            return
        acquired = self._connection_state_lock.acquire(blocking=False)
        if not acquired:
            self.logger.warning(
                "skipping chassis stop because another main-channel send is in flight"
            )
            if attempt_started is not None:
                attempt_started.set()
            return
        try:
            if not self.connected or not self._session_ready.is_set():
                return
            if attempt_started is not None:
                attempt_started.set()
            try:
                self.stop_chassis()
            except CommandError:
                self.logger.exception("failed to stop chassis before disconnect")
        finally:
            if attempt_started is not None:
                attempt_started.set()
            self._connection_state_lock.release()

    def _abort_websocket_client(self, client, cached_socket) -> None:
        """Best-effort low-level abort that never waits for the send lock."""
        websocket_socket = getattr(client, "sock", None)
        if websocket_socket is None:
            websocket_socket = cached_socket
        if websocket_socket is None:
            return

        abort = getattr(websocket_socket, "abort", None)
        if callable(abort):
            try:
                abort()
                return
            except BaseException:
                self.logger.exception("failed to abort websocket during disconnect")

        raw_socket = getattr(websocket_socket, "sock", websocket_socket)
        shutdown = getattr(raw_socket, "shutdown", None)
        if callable(shutdown):
            shutdown_args = (socket.SHUT_RDWR,)
            try:
                try:
                    signature = inspect.signature(shutdown)
                except (TypeError, ValueError):
                    pass
                else:
                    try:
                        signature.bind(socket.SHUT_RDWR)
                    except TypeError:
                        try:
                            signature.bind()
                        except TypeError:
                            pass
                        else:
                            shutdown_args = ()
                shutdown(*shutdown_args)
            except BaseException:
                self.logger.exception(
                    "failed to shut down websocket during disconnect"
                )

    def disconnect(self) -> None:
        self.logger.info("disconnecting")
        self.should_exit = True
        self._reconnect_wakeup.set()
        self._feedback_wakeup.set()

        # Preserve the safety stop when possible.  Run it independently so a
        # newly blocked stop send also cannot postpone the socket abort.
        chassis_stop_thread = None
        chassis_stop_started = None
        if self.enable_chassis_lifter:
            chassis_stop_started = threading.Event()
            chassis_stop_thread = threading.Thread(
                target=self._try_stop_chassis_before_disconnect,
                args=(chassis_stop_started,),
                daemon=True,
                name="WebsocketTransport-chassis-stop",
            )
            chassis_stop_thread.start()
            chassis_stop_started.wait()
            chassis_stop_thread.join(timeout=0.05)

        clients = tuple(
            client
            for client in (self._lifter_command_ws_client, self.ws_client)
            if client is not None
        )
        with self._abort_socket_lock:
            main_abort_socket = (
                self._main_abort_socket[1]
                if self._main_abort_socket is not None
                and self._main_abort_socket[0] == self._session_generation
                else None
            )
            lifter_abort_socket = (
                self._lifter_abort_socket[1]
                if self._lifter_abort_socket is not None
                and self._lifter_abort_socket[0]
                == self._lifter_command_session_generation
                else None
            )
        cached_sockets = {
            id(self.ws_client): main_abort_socket,
            id(self._lifter_command_ws_client): lifter_abort_socket,
        }
        for client in clients:
            self._abort_websocket_client(client, cached_sockets.get(id(client)))

        close_errors: list[BaseException] = []
        # A dedicated lifter send holds the main lock too.  Abort that socket
        # first so main-channel close callbacks cannot wait behind it forever.
        for client in clients:
            try:
                client.close()
            except BaseException as exc:
                self.logger.exception("failed to close websocket during disconnect")
                close_errors.append(exc)

        # These operations take locks also used by send paths, so do them only
        # after socket close has released any blocked I/O.
        self.disable_brainco2_polling()
        current_thread = threading.current_thread()
        thread_errors: list[BaseException] = []

        def join_thread(thread, label: str) -> None:
            if (
                thread is None
                or thread is current_thread
                or not thread.is_alive()
            ):
                return
            try:
                thread.join(timeout=2.0)
            except BaseException as exc:
                self.logger.exception("failed to join %s during disconnect", label)
                thread_errors.append(exc)
                return
            if thread.is_alive():
                error = RuntimeError(f"{label} did not stop during disconnect")
                self.logger.error("%s", error)
                thread_errors.append(error)

        join_thread(chassis_stop_thread, "chassis stop thread")
        join_thread(self.ws_thread, "websocket thread")
        join_thread(self._polling_thread, "polling thread")
        join_thread(self._lifter_command_ws_thread, "lifter websocket thread")
        if thread_errors:
            if close_errors:
                raise close_errors[0]
            raise thread_errors[0]
        self._reset_main_connection_state()
        self._reset_lifter_command_connection_state()
        self.logger.info("disconnected")
        if close_errors:
            raise close_errors[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def __del__(self):
        if (
            getattr(self, "connected", False)
            or getattr(self, "_lifter_command_connected", False)
        ):
            try:
                self.disconnect()
            except Exception:
                pass

    # ----------------------------------------------------- ws-specific helpers
    # These are NOT part of RobotTransport. They are used by ``_move_to_init_pose``
    # and the environment's reset path.

    def movej(self, joint_positions: Union[List[float], np.ndarray], move_time: float = 2.0) -> None:
        """Interpolated joint-space motion (ws controller does the trajectory)."""
        if isinstance(joint_positions, np.ndarray):
            joint_positions = joint_positions.tolist()
        if len(joint_positions) != JointIndex.MOVEJ_DIM:
            raise CommandError(f"expected {JointIndex.MOVEJ_DIM}-dim joints, got {len(joint_positions)}")
        if not self._send_request("request_movej", {"joint": joint_positions, "time": move_time}):
            raise CommandError("movej send failed")
        self.logger.debug("movej sent (time=%.2fs)", move_time)

    def move_head(self, head_joint: Union[List[float], np.ndarray], move_time: float = 5.0) -> None:
        """Interpolated head motion."""
        if isinstance(head_joint, np.ndarray):
            head_joint = head_joint.tolist()
        if len(head_joint) != JointIndex.HEAD_DIM:
            raise CommandError(f"expected {JointIndex.HEAD_DIM}-dim head, got {len(head_joint)}")
        if not self._send_request(
            "request_moveh",
            {"joint": head_joint, "time": move_time},
        ):
            raise CommandError("move_head send failed")
        self.logger.debug("move_head sent: %s", head_joint)

    def _joint_pose_needs_second_joint(self, current: List[float]) -> bool:
        return (
            abs(current[0]) < 0.1
            and abs(current[7]) < 0.1
            and abs(current[3]) < 0.2
            and abs(current[10]) < 0.2
        )

    def _ee_pose_needs_second_joint(self, ee_pose: Dict) -> bool:
        threshold = self.config.init_ee_z_min
        if threshold is None:
            return False

        z_values = []
        for key in ("left_position", "right_position"):
            position = ee_pose.get(key, [])
            if len(position) >= 3:
                try:
                    z_values.append(float(position[2]))
                except (TypeError, ValueError):
                    continue
        if not z_values:
            raise StateError("bad ee pose: missing left/right z position")

        min_z = min(z_values)
        if min_z < float(threshold):
            self.logger.info(
                "ee z %.4f below init threshold %.4f; routing via second joint",
                min_z,
                threshold,
            )
            return True
        return False

    def _bind_init_session(self) -> int:
        with self._connection_state_lock:
            generation = self._session_generation
            self._require_init_session(generation)
            return generation

    def _require_init_session(self, generation: int) -> None:
        with self._connection_state_lock:
            if (
                not self.connected
                or not self._session_ready.is_set()
                or self._session_generation != generation
            ):
                raise StateError("main WebSocket session changed during initialization")

    def _read_during_init(self, generation: int, reader):
        self._require_init_session(generation)
        value = reader()
        self._require_init_session(generation)
        return value

    def _send_during_init(self, generation: int, sender) -> None:
        with self._connection_state_lock:
            self._require_init_session(generation)
            sender()
            self._require_init_session(generation)

    def _sleep_during_init(self, generation: int, duration: float) -> None:
        self._require_init_session(generation)
        time.sleep(duration)
        self._require_init_session(generation)

    def _invalidate_main_feedback_after_init(self) -> None:
        """Discard pre-motion feedback before runtime polling seeds controllers."""
        with self._message_dispatch_lock:
            with self._state_lock:
                self._arm_head_snapshot.clear()
                self._gripper_snapshot.clear()
                self.joint_states = {
                    "timestamp": -1,
                    "robot_timestamp": -1,
                    "states": [-1.0] * JointIndex.STATE_DIM,
                    "joint_updated": False,
                    "gripper_updated": False,
                }
                self._legacy_joint_received_at = None
                self._legacy_gripper_received_at = None
                with self._queue_lock:
                    self.joint_state_queue.clear()

    def _move_to_init_pose(self) -> None:
        """Bring the robot to the init pose declared in config."""
        self.logger.info("moving to init pose")
        generation = self._bind_init_session()
        permissions = self.legacy_init_permissions
        if self.config.init_joints is not None:
            joint_state_guid = self._new_guid()

            def request_joint_state() -> None:
                if not self._send_request(
                    "request_get_joint_state",
                    guid=joint_state_guid,
                    expected_generation=generation,
                ):
                    raise StateError(
                        "joint state request failed during initialization"
                    )

            self._send_during_init(generation, request_joint_state)
            states = self._read_during_init(
                generation,
                lambda: self.get_arm_head_state(
                    timeout=1.0,
                    expected_guid=joint_state_guid,
                ),
            )
            current = states["states"]
            # If lying close to home, swing through a known intermediate first to
            # avoid the controller picking a weird shortest path.
            need_second_joint = self._joint_pose_needs_second_joint(current)
            if not need_second_joint and self.config.init_ee_z_min is not None:
                ee_pose = self._read_during_init(
                    generation,
                    lambda: self.get_ee_poses(
                        timeout=1.0,
                        expected_generation=generation,
                    ),
                )
                need_second_joint = self._ee_pose_needs_second_joint(ee_pose)
            if need_second_joint:
                self._send_during_init(
                    generation,
                    lambda: self.movej(self._second_joint, move_time=2.0),
                )
                self._sleep_during_init(generation, 2.0)
            self._send_during_init(
                generation,
                lambda: self.movej(self.config.init_joints, move_time=2.0),
            )
            # 2.1.5 controller bug: movej state is clobbered by move_head.
            self._sleep_during_init(generation, 2.0)
        if self.config.init_head is not None:
            head_move_time = 1.0
            self._send_during_init(
                generation,
                lambda: self.move_head(
                    self.config.init_head,
                    move_time=head_move_time,
                ),
            )
            self._sleep_during_init(generation, head_move_time)
        if permissions.limx_gripper_open:
            self._send_during_init(
                generation,
                lambda: self.set_gripper(left_opening=100, right_opening=100),
            )
            self._sleep_during_init(generation, 2.0)
        self._require_init_session(generation)
        self.logger.info("init pose reached")
