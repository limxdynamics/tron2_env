"""Tests for WebsocketTransport — uses a fake websocket; no robot needed."""

from __future__ import annotations

import json
import logging
import socket
import sys
import threading
import time
import types
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

# Build the fake module first; we'll inject it both into sys.modules (covers
# fresh imports) and onto the already-imported transport module (covers cases
# where another test pulled the real ``websocket`` in earlier).
_fake_ws_module = types.ModuleType("websocket")


class _FakeApp:
    """Recorded sends + can inject incoming messages via .feed_message()."""

    instances: list = []

    def __init__(self, url, on_open=None, on_message=None, on_close=None, on_error=None):
        self.url = url
        self.on_open = on_open
        self.on_message = on_message
        self.on_close = on_close
        self.on_error = on_error
        self.sent: list = []
        self.closed = False
        _FakeApp.instances.append(self)

    def run_forever(self, reconnect=None):
        # Simulate "connection open" then sleep until close.
        if self.on_open:
            self.on_open(self)
        self.feed_message({
            "title": "notify_robot_info",
            "accid": "FAKE_ROBOT",
            "data": {},
        })
        while not self.closed:
            time.sleep(0.005)

    def send(self, message: str):
        self.sent.append(message)

    def close(self):
        self.closed = True
        if self.on_close:
            self.on_close(self, 0, "fake close")

    # helper for tests
    def feed_message(self, payload: dict):
        if self.on_message:
            self.on_message(self, json.dumps(payload))


class _ScriptedAttemptApp(_FakeApp):
    """Run one gated connection attempt per ``run_forever`` call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempt_count = 0
        self.reconnect_arguments = []
        self.raise_without_close = set()
        self.attempt_opened = {
            attempt: threading.Event() for attempt in range(1, 5)
        }
        self.finish_attempt = {
            attempt: threading.Event() for attempt in range(1, 5)
        }
        self.attempt_closed = {
            attempt: threading.Event() for attempt in range(1, 5)
        }
        self.allow_attempt_return = {
            attempt: threading.Event() for attempt in range(1, 5)
        }
        self.attempt_returned = {
            attempt: threading.Event() for attempt in range(1, 5)
        }

    def run_forever(self, reconnect=None):
        self.attempt_count += 1
        attempt = self.attempt_count
        self.reconnect_arguments.append(reconnect)
        if self.on_open:
            self.on_open(self)
        self.feed_message({
            "title": "notify_robot_info",
            "accid": f"FAKE_ROBOT_{attempt}",
            "data": {},
        })
        self.attempt_opened[attempt].set()

        self.finish_attempt[attempt].wait(timeout=2.0)
        if self.closed:
            self.attempt_returned[attempt].set()
            return
        if attempt in self.raise_without_close:
            self.attempt_returned[attempt].set()
            raise RuntimeError(f"scripted attempt {attempt} crash")
        if self.on_close:
            self.on_close(self, 1006, f"scripted attempt {attempt} loss")
        self.attempt_closed[attempt].set()
        self.allow_attempt_return[attempt].wait(timeout=2.0)
        self.attempt_returned[attempt].set()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for event in self.finish_attempt.values():
            event.set()
        for event in self.allow_attempt_return.values():
            event.set()
        if self.on_close:
            self.on_close(self, 0, "fake close")


class _RecordingStopEvent:
    """Record retry waits while retaining Event-compatible stop semantics."""

    def __init__(self):
        self._event = threading.Event()
        self.wait_started = threading.Event()
        self.wait_timeouts = []

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        self.wait_started.set()
        return self._event.wait(timeout)

    def set(self):
        self._event.set()


class _ControlledReadyApp(_FakeApp):
    """Expose deterministic gates for socket-open and robot identity."""

    created = threading.Event()
    allow_open = threading.Event()
    opened = threading.Event()
    allow_identity = threading.Event()
    identity_sent = threading.Event()

    @classmethod
    def reset_events(cls):
        cls.created = threading.Event()
        cls.allow_open = threading.Event()
        cls.opened = threading.Event()
        cls.allow_identity = threading.Event()
        cls.identity_sent = threading.Event()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).created.set()

    def run_forever(self, reconnect=None):
        if not type(self).allow_open.wait(timeout=1.0) or self.closed:
            return
        if self.on_open:
            self.on_open(self)
        type(self).opened.set()
        if not type(self).allow_identity.wait(timeout=1.0) or self.closed:
            return
        self.feed_message({
            "title": "notify_robot_info",
            "accid": "CONTROLLED_ROBOT",
            "data": {},
        })
        type(self).identity_sent.set()
        while not self.closed:
            time.sleep(0.005)


class _NoIdentityApp(_FakeApp):
    """Open the socket but deliberately never announce a robot identity."""

    opened = threading.Event()

    def run_forever(self, reconnect=None):
        if self.on_open:
            self.on_open(self)
        type(self).opened.set()
        while not self.closed:
            time.sleep(0.005)


class _DualReadyApp(_FakeApp):
    """Announce main identity immediately and gate dedicated identity."""

    main_identity_sent = threading.Event()
    lifter_opened = threading.Event()
    allow_lifter_identity = threading.Event()
    lifter_identity_sent = threading.Event()

    @classmethod
    def reset_events(cls):
        cls.main_identity_sent = threading.Event()
        cls.lifter_opened = threading.Event()
        cls.allow_lifter_identity = threading.Event()
        cls.lifter_identity_sent = threading.Event()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_lifter = len(_FakeApp.instances) == 2

    def run_forever(self, reconnect=None):
        if self.on_open:
            self.on_open(self)
        if self.is_lifter:
            type(self).lifter_opened.set()
            if not type(self).allow_lifter_identity.wait(timeout=1.0) or self.closed:
                return
            self.feed_message({
                "title": "notify_robot_info",
                "accid": "DEDICATED_ROBOT",
                "data": {},
            })
            type(self).lifter_identity_sent.set()
        else:
            self.feed_message({
                "title": "notify_robot_info",
                "accid": "MAIN_ROBOT",
                "data": {},
            })
            type(self).main_identity_sent.set()
        while not self.closed:
            time.sleep(0.005)

    def close(self):
        if self.is_lifter:
            type(self).allow_lifter_identity.set()
        super().close()


_fake_ws_module.WebSocketApp = _FakeApp
sys.modules["websocket"] = _fake_ws_module

# Force tron2_env.transport.websocket to bind to our fake even if it was
# already imported by another test (in which case sys.modules patch alone
# is ignored — the module already has its own ``websocket`` reference).
import tron2_env.transport.websocket as _ws_mod  # noqa: E402

_ws_mod.websocket = _fake_ws_module

from tron2_env.config import Tron2Config  # noqa: E402
from tron2_env.errors import CommandError, StateError  # noqa: E402
from tron2_env.joints import JointIndex  # noqa: E402
from tron2_env.transport import LegacyInitPermissions, WebsocketTransport  # noqa: E402


def test_default_init_ee_z_threshold_is_minus_point_six_meters():
    assert Tron2Config().init_ee_z_min == pytest.approx(-0.6)


def test_legacy_init_permissions_default_to_legacy_behavior_and_are_frozen():
    permissions = LegacyInitPermissions()

    assert permissions == LegacyInitPermissions(
        limx_gripper_open=True,
    )
    assert vars(permissions) == {"limx_gripper_open": True}
    with pytest.raises(FrozenInstanceError):
        permissions.limx_gripper_open = False


@pytest.mark.parametrize("removed_field", ["joint_move", "head_move"])
def test_legacy_init_permissions_reject_removed_motion_gates(removed_field):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        LegacyInitPermissions(**{removed_field: False})


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("poll_joint_state", 1),
        ("poll_joint_state", np.bool_(True)),
        ("poll_gripper_state", 0),
        ("poll_gripper_state", "yes"),
    ],
)
def test_feedback_polling_flags_require_strict_bools(keyword, value):
    with pytest.raises(TypeError, match=f"{keyword} must be a bool"):
        WebsocketTransport(Tron2Config(), **{keyword: value})


@pytest.mark.parametrize(
    ("timeout", "message"),
    [
        ("invalid", "finite and positive"),
        (0.0, "finite and positive"),
        (-1.0, "finite and positive"),
        (np.inf, "finite and positive"),
        (threading.TIMEOUT_MAX + 1.0, "threading.TIMEOUT_MAX"),
    ],
)
def test_constructor_rejects_invalid_connection_timeout_before_connect(
    monkeypatch,
    timeout,
    message,
):
    connection_attempts = []
    monkeypatch.setattr(
        WebsocketTransport,
        "_connect",
        lambda self: connection_attempts.append("main"),
    )
    monkeypatch.setattr(
        WebsocketTransport,
        "_connect_lifter_command_channel",
        lambda self: connection_attempts.append("lifter"),
    )

    with pytest.raises(ValueError, match=message):
        WebsocketTransport(
            Tron2Config(connection_timeout=timeout),
            enable_chassis_lifter=True,
        )

    assert connection_attempts == []


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, True])
@pytest.mark.parametrize("rate_name", ["polling_rate", "auxiliary_polling_rate"])
def test_constructor_rejects_invalid_polling_rates_before_connect(
    monkeypatch,
    rate_name,
    value,
):
    connection_attempts = []
    monkeypatch.setattr(
        WebsocketTransport,
        "_connect",
        lambda self: connection_attempts.append("main"),
    )
    monkeypatch.setattr(
        WebsocketTransport,
        "_connect_lifter_command_channel",
        lambda self: connection_attempts.append("lifter"),
    )
    config = Tron2Config()
    kwargs = {"enable_chassis_lifter": True}
    if rate_name == "polling_rate":
        config.polling_rate = value
    else:
        kwargs["auxiliary_polling_rate"] = value

    with pytest.raises((TypeError, ValueError), match=rate_name):
        WebsocketTransport(config, **kwargs)

    assert connection_attempts == []


_INVALID_PUBLIC_WAIT_TIMEOUTS = [
    pytest.param("0.1", TypeError, id="string"),
    pytest.param(True, TypeError, id="bool"),
    pytest.param(np.bool_(False), TypeError, id="numpy-bool"),
    pytest.param(None, TypeError, id="none"),
    pytest.param(0.1 + 0.0j, TypeError, id="complex"),
    pytest.param(np.nan, ValueError, id="nan"),
    pytest.param(np.inf, ValueError, id="infinity"),
    pytest.param(
        threading.TIMEOUT_MAX + 1.0,
        ValueError,
        id="above-threading-limit",
    ),
    pytest.param(-0.001, ValueError, id="negative"),
]


@pytest.fixture
def transport():
    _FakeApp.instances.clear()
    config = Tron2Config(robot_ip="127.0.0.1", polling_rate=1000.0)
    t = WebsocketTransport(config)
    # wait until on_open has fired
    deadline = time.time() + 1.0
    while not t.is_connected() and time.time() < deadline:
        time.sleep(0.01)
    # The polling thread spams ws messages — start each test from a clean send log.
    for app in _FakeApp.instances:
        app.sent.clear()
    yield t
    t.disconnect()


@pytest.fixture
def unpolled_transport(monkeypatch):
    """Connected transport whose feedback scheduler is driven explicitly."""
    _FakeApp.instances.clear()
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: None,
    )
    t = WebsocketTransport(Tron2Config(robot_ip="127.0.0.1", polling_rate=2.0))
    for app in _FakeApp.instances:
        app.sent.clear()
    yield t
    t.disconnect()


@pytest.fixture
def chassis_transport():
    _FakeApp.instances.clear()
    config = Tron2Config(robot_ip="127.0.0.1", polling_rate=1000.0)
    t = WebsocketTransport(
        config,
        enable_chassis_lifter=True,
        auxiliary_polling_rate=0.001,
    )
    deadline = time.time() + 1.0
    while (
        not (t.is_connected() and t._lifter_command_connected)
        and time.time() < deadline
    ):
        time.sleep(0.01)
    initial_feedback_guid = None
    while time.time() < deadline:
        with t._lifter_feedback_lock:
            initial_feedback_guid = t._lifter_feedback_in_flight
        if initial_feedback_guid is not None:
            break
        time.sleep(0.001)
    assert initial_feedback_guid is not None, "startup lifter feedback was not reserved"
    with t._lifter_feedback_lock:
        assert t._lifter_feedback_in_flight == initial_feedback_guid
        t._lifter_feedback_in_flight = None
        t._lifter_feedback_deadline = None
    for app in _FakeApp.instances:
        app.sent.clear()
    yield t
    t.disconnect()


def _main_ws_app() -> _FakeApp:
    assert _FakeApp.instances, "no fake ws created"
    return _FakeApp.instances[0]


def _lifter_ws_app() -> _FakeApp:
    assert len(_FakeApp.instances) == 2, "expected main and lifter fake ws connections"
    return _FakeApp.instances[1]


def _ws_app() -> _FakeApp:
    return _main_ws_app()


def _messages_from(app: _FakeApp, title: str) -> list[dict]:
    return [
        message
        for raw in app.sent
        if (message := json.loads(raw)).get("title") == title
    ]


def _sent_messages(title: str) -> list[dict]:
    return _messages_from(_ws_app(), title)


def _feed_joint_state(
    q,
    *,
    timestamp: int = 1000,
    result: str = "success",
    **metadata,
) -> None:
    _ws_app().feed_message({
        "title": "response_get_joint_state",
        "timestamp": timestamp,
        **metadata,
        "data": {"result": result, "q": q},
    })


def _feed_gripper_state(
    left,
    right,
    *,
    timestamp: int = 1001,
    result: str = "success",
    **metadata,
) -> None:
    _ws_app().feed_message({
        "title": "response_get_limx_2fclaw_state",
        "timestamp": timestamp,
        **metadata,
        "data": {
            "result": result,
            "left_opening": left,
            "right_opening": right,
        },
    })


def _new_snapshot(width: int, name: str):
    from tron2_env.transport.snapshot import LatestSnapshot

    return LatestSnapshot(width=width, name=name)


def _pose_response_data(**overrides) -> dict:
    data = {
        "result": "success",
        "timestamp": 1001,
        "left_position": [0.1, 0.2, 0.3],
        "left_quat": [1.0, 0.0, 0.0, 0.0],
        "right_position": [0.4, 0.5, 0.6],
        "right_quat": [1.0, 0.0, 0.0, 0.0],
    }
    data.update(overrides)
    return data


def _brainco_response_data(**overrides) -> dict:
    data = {
        "result": "success",
        "timestamp": 2001,
        "left_mode": 1,
        "left_names": [f"L{index}" for index in range(6)],
        "left_pos": [0, 1, 2, 3, 4, 5],
        "left_vel": [10, 11, 12, 13, 14, 15],
        "left_current": [20, 21, 22, 23, 24, 25],
        "left_time": [2.0] * 6,
        "right_mode": 1,
        "right_names": [f"R{index}" for index in range(6)],
        "right_pos": [6, 7, 8, 9, 10, 11],
        "right_vel": [16, 17, 18, 19, 20, 21],
        "right_current": [26, 27, 28, 29, 30, 31],
        "right_time": [2.0] * 6,
    }
    data.update(overrides)
    return data


def _feed_brainco_state(guid, **overrides) -> None:
    _ws_app().feed_message({
        "title": "response_get_brainco2_hand_state",
        "guid": guid,
        "timestamp": 2000,
        "data": _brainco_response_data(**overrides),
    })


def test_constructor_waits_for_socket_and_robot_identity_before_polling(monkeypatch):
    _FakeApp.instances.clear()
    _ControlledReadyApp.reset_events()
    monkeypatch.setattr(_fake_ws_module, "WebSocketApp", _ControlledReadyApp)
    polling_started = threading.Event()
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: polling_started.set(),
    )
    created = []
    errors = []
    constructor_returned = threading.Event()

    def construct():
        try:
            created.append(WebsocketTransport(Tron2Config(
                robot_ip="127.0.0.1",
                connection_timeout=0.5,
            )))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            constructor_returned.set()

    thread = threading.Thread(target=construct, daemon=True)
    thread.start()
    assert _ControlledReadyApp.created.wait(timeout=0.5)
    assert not constructor_returned.is_set()

    _ControlledReadyApp.allow_open.set()
    assert _ControlledReadyApp.opened.wait(timeout=0.5)
    returned_before_identity = constructor_returned.wait(timeout=0.2)
    polling_started_before_identity = polling_started.is_set()
    messages_before_identity = list(_FakeApp.instances[0].sent)

    _ControlledReadyApp.allow_identity.set()
    assert _ControlledReadyApp.identity_sent.wait(timeout=0.5)
    assert constructor_returned.wait(timeout=0.5)
    thread.join(timeout=0.5)

    try:
        assert not returned_before_identity
        assert not polling_started_before_identity
        assert messages_before_identity == []
        assert errors == []
        assert len(created) == 1
        assert created[0].accid == "CONTROLLED_ROBOT"
        assert created[0].is_connected()
        assert polling_started.is_set()
    finally:
        if created:
            created[0].disconnect()


def test_constructor_waits_for_dedicated_lifter_identity_before_returning(monkeypatch):
    _FakeApp.instances.clear()
    _DualReadyApp.reset_events()
    monkeypatch.setattr(_fake_ws_module, "WebSocketApp", _DualReadyApp)
    monkeypatch.setattr(WebsocketTransport, "_start_polling_thread", lambda self: None)
    created = []
    errors = []
    constructor_returned = threading.Event()

    def construct():
        try:
            created.append(WebsocketTransport(
                Tron2Config(robot_ip="127.0.0.1", connection_timeout=0.5),
                enable_chassis_lifter=True,
            ))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            constructor_returned.set()

    thread = threading.Thread(target=construct, daemon=True)
    thread.start()
    assert _DualReadyApp.main_identity_sent.wait(timeout=0.5)
    assert _DualReadyApp.lifter_opened.wait(timeout=0.5)
    assert not constructor_returned.wait(timeout=0.1)

    _DualReadyApp.allow_lifter_identity.set()
    assert _DualReadyApp.lifter_identity_sent.wait(timeout=0.5)
    assert constructor_returned.wait(timeout=0.5)
    thread.join(timeout=0.5)

    try:
        assert errors == []
        assert len(created) == 1
        assert created[0]._lifter_command_accid == "DEDICATED_ROBOT"
        assert created[0]._lifter_command_connected
        assert created[0]._lifter_command_ready.is_set()
    finally:
        if created:
            created[0].disconnect()
            assert not created[0]._lifter_command_ready.is_set()


def test_constructor_times_out_and_disconnects_without_dedicated_lifter_identity(
    monkeypatch,
):
    _FakeApp.instances.clear()
    _DualReadyApp.reset_events()
    monkeypatch.setattr(_fake_ws_module, "WebSocketApp", _DualReadyApp)
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: pytest.fail("polling started before dedicated identity was ready"),
    )

    with pytest.raises(StateError, match="lifter.*ready"):
        WebsocketTransport(
            Tron2Config(robot_ip="127.0.0.1", connection_timeout=0.02),
            enable_chassis_lifter=True,
        )

    assert _DualReadyApp.main_identity_sent.is_set()
    assert _DualReadyApp.lifter_opened.is_set()
    assert len(_FakeApp.instances) == 2
    assert all(app.closed for app in _FakeApp.instances)


def test_constructor_starts_polling_after_move_to_init_pose(monkeypatch):
    events = []

    def connect(self):
        self.connected = True
        self.accid = "FAKE_ROBOT"
        self._session_generation = 1
        self._session_ready.set()

    def initialize(self):
        events.append("init")
        self._arm_head_snapshot.update([0.0] * JointIndex.SERVOJ_DIM)
        self._gripper_snapshot.update([0.0, 0.0])
        self.joint_states.update({
            "timestamp": 123,
            "robot_timestamp": 456,
            "states": [0.0] * JointIndex.STATE_DIM,
            "joint_updated": True,
            "gripper_updated": True,
        })
        self._legacy_joint_received_at = 1.0
        self._legacy_gripper_received_at = 2.0
        self.joint_state_queue.append({"states": [0.0] * JointIndex.STATE_DIM})

    original_invalidate = WebsocketTransport._invalidate_main_feedback_after_init

    def invalidate(self):
        events.append("invalidate")
        original_invalidate(self)

    def start_polling(self):
        events.append("poll")
        with pytest.raises(StateError, match="arm/head state timed out"):
            self.get_arm_head_state(timeout=0.0)
        with pytest.raises(StateError, match="gripper state timed out"):
            self.get_gripper_state(timeout=0.0)
        assert self.joint_states == {
            "timestamp": -1,
            "robot_timestamp": -1,
            "states": [-1.0] * JointIndex.STATE_DIM,
            "joint_updated": False,
            "gripper_updated": False,
        }
        assert self._legacy_joint_received_at is None
        assert self._legacy_gripper_received_at is None
        assert list(self.joint_state_queue) == []

        self._handle_gripper_state({
            "timestamp": 124,
            "data": {
                "result": "success",
                "left_opening": 20,
                "right_opening": 30,
            },
        })
        assert list(self.joint_state_queue) == []
        self._handle_joint_state({
            "timestamp": 125,
            "data": {
                "result": "success",
                "q": [0.5] * JointIndex.SERVOJ_DIM,
            },
        })
        assert len(self.joint_state_queue) == 1

    monkeypatch.setattr(WebsocketTransport, "_connect", connect)
    monkeypatch.setattr(WebsocketTransport, "_move_to_init_pose", initialize)
    monkeypatch.setattr(
        WebsocketTransport,
        "_invalidate_main_feedback_after_init",
        invalidate,
    )
    monkeypatch.setattr(WebsocketTransport, "_start_polling_thread", start_polling)

    transport = WebsocketTransport(
        Tron2Config(init_joints=[0.0] * JointIndex.MOVEJ_DIM)
    )
    transport.connected = False

    assert events == ["init", "invalidate", "poll"]


def test_constructor_preserves_init_error_when_disconnect_fails(monkeypatch):
    init_error = RuntimeError("init failed")
    disconnect_error = RuntimeError("disconnect failed")
    events = []
    partial_transports = []
    logged_exceptions = []

    def setup_logger(self):
        self.logger = types.SimpleNamespace(
            exception=lambda *args, **kwargs: logged_exceptions.append(
                (args, kwargs)
            )
        )

    def connect(self):
        events.append("connect")
        self.connected = True
        self.accid = "FAKE_ROBOT"
        self._session_generation = 1
        self._session_ready.set()

    def initialize(self):
        events.append("init")
        partial_transports.append(self)
        raise init_error

    def disconnect(self):
        events.append("disconnect")
        self.connected = False
        raise disconnect_error

    monkeypatch.setattr(WebsocketTransport, "_setup_logger", setup_logger)
    monkeypatch.setattr(WebsocketTransport, "_connect", connect)
    monkeypatch.setattr(WebsocketTransport, "_move_to_init_pose", initialize)
    monkeypatch.setattr(WebsocketTransport, "disconnect", disconnect)
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: pytest.fail("polling started after initialization failed"),
    )

    try:
        with pytest.raises(RuntimeError) as exc_info:
            WebsocketTransport(
                Tron2Config(init_joints=[0.0] * JointIndex.MOVEJ_DIM)
            )

        assert exc_info.value is init_error
        assert events == ["connect", "init", "disconnect"]
        assert len(logged_exceptions) == 1
    finally:
        for transport in partial_transports:
            transport.connected = False


def test_constructor_disconnects_when_polling_start_fails(monkeypatch):
    polling_error = RuntimeError("polling start failed")
    events = []
    partial_transports = []

    def connect(self):
        events.append("connect")
        self.connected = True
        self.accid = "FAKE_ROBOT"
        self._session_generation = 1
        self._session_ready.set()

    def start_polling(self):
        events.append("poll")
        partial_transports.append(self)
        raise polling_error

    def disconnect(self):
        events.append("disconnect")
        self.connected = False

    monkeypatch.setattr(WebsocketTransport, "_connect", connect)
    monkeypatch.setattr(WebsocketTransport, "_start_polling_thread", start_polling)
    monkeypatch.setattr(WebsocketTransport, "disconnect", disconnect)

    try:
        with pytest.raises(RuntimeError) as exc_info:
            WebsocketTransport(Tron2Config())

        assert exc_info.value is polling_error
        assert events == ["connect", "poll", "disconnect"]
    finally:
        for transport in partial_transports:
            transport.connected = False


def test_constructor_times_out_when_socket_opens_without_robot_identity(monkeypatch):
    _FakeApp.instances.clear()
    _NoIdentityApp.opened = threading.Event()
    monkeypatch.setattr(_fake_ws_module, "WebSocketApp", _NoIdentityApp)
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: pytest.fail("polling started before session identity was ready"),
    )

    with pytest.raises(StateError, match="session.*ready"):
        WebsocketTransport(Tron2Config(
            robot_ip="127.0.0.1",
            connection_timeout=0.02,
        ))

    assert _NoIdentityApp.opened.is_set()
    assert _FakeApp.instances[0].closed


def test_send_joint_cmd_emits_servoj_message(transport):
    transport.send_joint_cmd(np.linspace(0.1, 1.6, 16))
    sent = [json.loads(m) for m in _ws_app().sent if json.loads(m).get("title") == "request_servoj"]
    assert sent, "no servoj message was sent"
    payload = sent[-1]["data"]
    assert len(payload["q"]) == 16
    assert payload["filter_ratio"] == 1.0


def test_send_joint_cmd_rejects_wrong_shape(transport):
    with pytest.raises(CommandError):
        transport.send_joint_cmd(np.zeros(14))


def test_send_head_move_emits_exact_moveh_payload_and_guid(transport):
    sent = transport.send_head_move(
        np.array([0.25, -0.5]),
        duration=0.125,
        guid="head-guid",
    )

    assert sent is True
    messages = _sent_messages("request_moveh")
    assert len(messages) == 1
    assert messages[0]["guid"] == "head-guid"
    assert messages[0]["data"] == {
        "joint": [0.25, -0.5],
        "time": 0.125,
    }


@pytest.mark.parametrize(
    "target,duration,guid",
    [
        ([0.0], 0.1, "guid"),
        ([[0.0, 0.0]], 0.1, "guid"),
        ([np.nan, 0.0], 0.1, "guid"),
        ([0.0, np.inf], 0.1, "guid"),
        ([0.0, 0.0], 0.0, "guid"),
        ([0.0, 0.0], np.nan, "guid"),
        ([0.0, 0.0], 0.1, ""),
        ([0.0, 0.0], 0.1, None),
    ],
)
def test_send_head_move_rejects_invalid_target_duration_or_guid(
    transport,
    target,
    duration,
    guid,
):
    with pytest.raises(CommandError):
        transport.send_head_move(target, duration, guid)

    assert not _sent_messages("request_moveh")


def test_head_response_callback_receives_guid_and_result(transport):
    received = []
    transport.register_head_response_callback(
        lambda guid, result: received.append((guid, result))
    )

    _ws_app().feed_message({
        "title": "response_moveh",
        "guid": "head-guid",
        "data": {"result": "failed"},
    })

    assert received == [("head-guid", "failed")]


def test_unregister_head_response_callback_stops_dispatch(transport):
    received = []
    transport.register_head_response_callback(
        lambda guid, result: received.append((guid, result))
    )
    transport.unregister_head_response_callback()

    _ws_app().feed_message({
        "title": "response_moveh",
        "guid": "head-guid",
        "data": {"result": "success"},
    })

    assert received == []


def test_head_response_callback_exception_is_logged_and_contained(
    transport,
    monkeypatch,
):
    logged = []

    def raising_callback(_guid, _result):
        raise RuntimeError("callback broke")

    transport.register_head_response_callback(raising_callback)
    monkeypatch.setattr(
        transport.logger,
        "exception",
        lambda message, *args: logged.append((message, args)),
    )

    _ws_app().feed_message({
        "title": "response_moveh",
        "guid": "head-guid",
        "data": {"result": "success"},
    })

    assert logged
    assert "head response callback failed" in logged[0][0]


def test_legacy_move_head_wire_behavior_remains_unchanged(transport):
    result = transport.move_head([0.1, -0.2], move_time=0.75)

    assert result is None
    messages = _sent_messages("request_moveh")
    assert len(messages) == 1
    assert messages[0]["data"] == {
        "joint": [0.1, -0.2],
        "time": 0.75,
    }

    _ws_app().feed_message({
        "title": "response_moveh",
        "guid": messages[0]["guid"],
        "data": {"result": "success"},
    })


def test_cartesian_pose_capability_protocol_is_optional(transport):
    from tron2_env.transport.base import CartesianServoTransport, RobotTransport

    assert isinstance(transport, CartesianServoTransport)
    assert "send_pose_cmd" not in RobotTransport.__dict__
    assert "request_pose_state" not in RobotTransport.__dict__
    assert "get_pose_state" not in RobotTransport.__dict__


def test_send_pose_cmd_emits_one_servop_message_in_wxyz_order(transport):
    left = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    right = np.array([0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 0.0])

    transport.send_pose_cmd(left, right)

    sent = _sent_messages("request_servop")
    assert len(sent) == 1
    assert sent[0]["data"] == {
        "left_pos": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        "right_pos": [0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 0.0],
    }
    assert not _sent_messages("request_servoj")


def test_send_pose_cmd_normalizes_each_wxyz_quaternion(transport):
    transport.send_pose_cmd(
        [0.1, 0.2, 0.3, 2.0, 0.0, 0.0, 0.0],
        [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, -4.0],
    )

    payload = _sent_messages("request_servop")[-1]["data"]
    assert payload["left_pos"] == [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    assert payload["right_pos"] == [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, -1.0]


def test_send_pose_cmd_normalizes_large_finite_quaternion_without_overflow(
    transport,
):
    transport.send_pose_cmd(
        [0.1, 0.2, 0.3, 1e308, 1e308, 1e308, 1e308],
        [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
    )

    payload = _sent_messages("request_servop")[-1]["data"]
    assert payload["left_pos"] == pytest.approx(
        [0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5]
    )


@pytest.mark.parametrize(
    "left,right",
    [
        ([0.0] * 6, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [[0.0] * 7]),
        ([np.nan, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, np.inf, 0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, 0.0]),
    ],
)
def test_send_pose_cmd_rejects_invalid_pose_inputs(transport, left, right):
    with pytest.raises(CommandError):
        transport.send_pose_cmd(left, right)

    assert not _sent_messages("request_servop")


def test_send_pose_cmd_raises_when_synchronous_send_fails(transport, monkeypatch):
    monkeypatch.setattr(transport, "_send_request", lambda *args, **kwargs: False)

    with pytest.raises(CommandError, match="servop send failed"):
        transport.send_pose_cmd(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )


def test_pose_query_uses_canonical_title_guid_and_one_in_flight(transport):
    first_guid = transport.request_pose_state()
    second_guid = transport.request_pose_state()

    sent = _sent_messages("request_get_servo_pose")
    assert second_guid == first_guid
    assert len(sent) == 1
    assert sent[0]["guid"] == first_guid
    assert sent[0]["data"] == {}
    assert not _sent_messages("request_get_move_pose")


def test_pose_query_reuses_unexpired_deadline_then_allocates_after_expiry(
    unpolled_transport,
    monkeypatch,
):
    now = [10.0]
    monkeypatch.setattr(unpolled_transport, "_feedback_clock", lambda: now[0])

    first_guid = unpolled_transport.request_pose_state(timeout=0.2)
    first_deadline = unpolled_transport._pose_query_deadline
    now[0] = 10.1
    assert unpolled_transport.request_pose_state(timeout=9.0) == first_guid
    assert unpolled_transport._pose_query_deadline == first_deadline

    now[0] = 10.2
    second_guid = unpolled_transport.request_pose_state(timeout=0.3)

    assert second_guid != first_guid
    assert unpolled_transport._pose_query_deadline == pytest.approx(10.5)
    assert [
        message["guid"] for message in _sent_messages("request_get_servo_pose")
    ] == [first_guid, second_guid]


def test_pose_query_rejects_timeout_above_threading_limit(unpolled_transport):
    with pytest.raises(ValueError, match="threading.TIMEOUT_MAX"):
        unpolled_transport.request_pose_state(timeout=threading.TIMEOUT_MAX + 1.0)

    assert unpolled_transport._pose_query_in_flight is None
    assert unpolled_transport._pose_query_deadline is None
    assert not _sent_messages("request_get_servo_pose")


@pytest.mark.parametrize("timeout", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_pose_query_requires_finite_positive_timeout(transport, timeout):
    with pytest.raises(ValueError, match="timeout must be finite and positive"):
        transport.request_pose_state(timeout=timeout)


@pytest.mark.parametrize("failure", [False, RuntimeError("send exploded")])
def test_pose_query_send_failure_clears_lease_and_wakes_scheduler(
    unpolled_transport,
    monkeypatch,
    failure,
):
    def fail_send(*args, **kwargs):
        if isinstance(failure, BaseException):
            raise failure
        return failure

    monkeypatch.setattr(unpolled_transport, "_send_request", fail_send)
    unpolled_transport._feedback_wakeup.clear()

    with pytest.raises(StateError, match="request_pose_state request failed"):
        unpolled_transport.request_pose_state(timeout=0.2)

    assert unpolled_transport._pose_query_in_flight is None
    assert unpolled_transport._pose_query_deadline is None
    assert unpolled_transport._feedback_wakeup.is_set()


def test_late_matching_pose_response_is_consumed_without_updating_snapshot(
    unpolled_transport,
    monkeypatch,
):
    now = [1.0]
    monkeypatch.setattr(unpolled_transport, "_feedback_clock", lambda: now[0])
    first_guid = unpolled_transport.request_pose_state(timeout=0.2)
    _ws_app().feed_message({
        "title": "response_get_servo_pose",
        "guid": first_guid,
        "data": _pose_response_data(timestamp=10),
    })
    original = unpolled_transport.get_pose_state(timeout=0.0)

    late_guid = unpolled_transport.request_pose_state(timeout=0.2)
    now[0] = 1.2
    _ws_app().feed_message({
        "title": "response_get_servo_pose",
        "guid": late_guid,
        "data": _pose_response_data(timestamp=20),
    })

    assert unpolled_transport.get_pose_state(timeout=0.0) == original
    assert unpolled_transport._pose_query_in_flight is None
    assert unpolled_transport._pose_query_deadline is None


def test_expired_canonical_move_pose_does_not_satisfy_new_legacy_wait(
    unpolled_transport,
    monkeypatch,
):
    now = [1.0]
    monkeypatch.setattr(unpolled_transport, "_feedback_clock", lambda: now[0])
    canonical_guid = unpolled_transport.request_pose_state(timeout=0.1)
    now[0] = 1.1
    unpolled_transport._poll_feedback_once()
    results = []
    errors = []

    def read_legacy_pose():
        try:
            results.append(unpolled_transport.get_ee_poses(timeout=0.5))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    legacy_reader = threading.Thread(target=read_legacy_pose, daemon=True)
    legacy_reader.start()
    deadline = time.time() + 0.5
    while not _sent_messages("request_get_move_pose") and time.time() < deadline:
        time.sleep(0.001)
    legacy_request = _sent_messages("request_get_move_pose")[-1]

    _ws_app().feed_message({
        "title": "response_get_move_pose",
        "guid": canonical_guid,
        "data": _pose_response_data(timestamp=10),
    })
    time.sleep(0.01)
    assert legacy_reader.is_alive()

    _ws_app().feed_message({
        "title": "response_get_move_pose",
        "guid": legacy_request["guid"],
        "data": _pose_response_data(timestamp=20),
    })
    legacy_reader.join(timeout=0.5)

    assert not legacy_reader.is_alive()
    assert errors == []
    assert results[0]["timestamp"] == 20


def test_duplicate_completed_canonical_move_pose_never_enters_legacy_queue(
    unpolled_transport,
):
    guid = unpolled_transport.request_pose_state(timeout=0.2)
    response = {
        "title": "response_get_move_pose",
        "guid": guid,
        "data": _pose_response_data(timestamp=10),
    }
    _ws_app().feed_message(response)
    assert list(unpolled_transport.ee_pose_queue) == []

    _ws_app().feed_message(response)

    assert list(unpolled_transport.ee_pose_queue) == []


def test_oldest_canonical_move_pose_never_enters_legacy_queue_after_many_queries(
    unpolled_transport,
):
    first_guid = None
    for timestamp in range(65):
        guid = unpolled_transport.request_pose_state(timeout=1.0)
        if first_guid is None:
            first_guid = guid
        _ws_app().feed_message({
            "title": "response_get_move_pose",
            "guid": guid,
            "data": _pose_response_data(timestamp=timestamp),
        })

    assert list(unpolled_transport.ee_pose_queue) == []
    assert unpolled_transport.get_pose_state(timeout=0.0)["timestamp"] == 64

    _ws_app().feed_message({
        "title": "response_get_move_pose",
        "guid": first_guid,
        "data": _pose_response_data(timestamp=999),
    })

    assert list(unpolled_transport.ee_pose_queue) == []
    assert unpolled_transport.get_pose_state(timeout=0.0)["timestamp"] == 64


def test_lost_pose_query_is_reissued_in_new_session_and_old_response_is_ignored(
    transport,
):
    app = _ws_app()
    first_guid = transport.request_pose_state()

    transport._on_close(app, 1006, "connection lost")
    assert transport._pose_query_in_flight is None
    assert transport._pose_query_deadline is None
    transport._on_open(app)
    app.feed_message({
        "title": "notify_robot_info",
        "accid": "RECONNECTED_ROBOT",
        "data": {},
    })
    second_guid = transport.request_pose_state()

    assert second_guid != first_guid
    assert [
        message["guid"] for message in _sent_messages("request_get_servo_pose")
    ] == [first_guid, second_guid]

    app.feed_message({
        "title": "response_get_move_pose",
        "guid": first_guid,
        "data": _pose_response_data(timestamp=1),
    })
    assert list(transport.ee_pose_queue) == []
    with pytest.raises(StateError, match="pose state timed out"):
        transport.get_pose_state(timeout=0.0)

    app.feed_message({
        "title": "response_get_servo_pose",
        "guid": second_guid,
        "data": _pose_response_data(timestamp=2),
    })
    assert transport.get_pose_state(timeout=0.1)["timestamp"] == 2


def test_pose_query_waits_for_failed_send_before_allocating_a_new_guid(
    transport,
    monkeypatch,
):
    first_send_entered = threading.Event()
    release_first_send = threading.Event()
    sent_guids = []

    def send_request(title, _data=None, *, guid=None):
        if title != "request_get_servo_pose":
            return True
        sent_guids.append(guid)
        if len(sent_guids) == 1:
            first_send_entered.set()
            assert release_first_send.wait(timeout=1.0)
            return False
        return True

    monkeypatch.setattr(transport, "_send_request", send_request)
    results = []
    errors = []

    def request_pose():
        try:
            results.append(transport.request_pose_state())
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=request_pose)
    second = threading.Thread(target=request_pose)
    first.start()
    assert first_send_entered.wait(timeout=1.0)
    second.start()
    release_first_send.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], StateError)
    assert len(results) == 1
    assert len(sent_guids) == 2
    assert results[0] == sent_guids[1]
    assert sent_guids[0] != sent_guids[1]


def test_pose_query_serializes_snapshot_update_before_next_guid(
    transport,
    monkeypatch,
):
    first_parse_entered = threading.Event()
    release_first_parse = threading.Event()
    second_request_ready = threading.Event()
    original_normalize = transport._normalize_pose
    normalize_calls = 0
    normalize_calls_lock = threading.Lock()

    def blocking_normalize(position, quaternion):
        nonlocal normalize_calls
        with normalize_calls_lock:
            call_index = normalize_calls
            normalize_calls += 1
        if call_index == 0:
            first_parse_entered.set()
            assert release_first_parse.wait(timeout=1.0)
        return original_normalize(position, quaternion)

    monkeypatch.setattr(transport, "_normalize_pose", blocking_normalize)
    first_guid = transport.request_pose_state()
    first_response = threading.Thread(
        target=lambda: _ws_app().feed_message({
            "title": "response_get_servo_pose",
            "guid": first_guid,
            "data": _pose_response_data(timestamp=1),
        })
    )
    first_response.start()
    assert first_parse_entered.wait(timeout=1.0)

    second_guid = []

    def request_second_pose():
        second_guid.append(transport.request_pose_state())
        second_request_ready.set()

    second_request = threading.Thread(target=request_second_pose)
    second_request.start()
    if second_request_ready.wait(timeout=0.05):
        _ws_app().feed_message({
            "title": "response_get_servo_pose",
            "guid": second_guid[0],
            "data": _pose_response_data(timestamp=2),
        })
        release_first_parse.set()
    else:
        release_first_parse.set()
        assert second_request_ready.wait(timeout=1.0)
        _ws_app().feed_message({
            "title": "response_get_servo_pose",
            "guid": second_guid[0],
            "data": _pose_response_data(timestamp=2),
        })

    first_response.join(timeout=1.0)
    second_request.join(timeout=1.0)
    assert transport.get_pose_state(timeout=0.1)["timestamp"] == 2


@pytest.mark.parametrize(
    "response_title",
    ["response_get_servo_pose", "response_get_move_pose"],
)
def test_pose_state_accepts_both_response_aliases_and_copies_metadata(
    transport,
    response_title,
):
    guid = transport.request_pose_state()
    _ws_app().feed_message({
        "title": response_title,
        "guid": guid,
        "timestamp": 999,
        "diagnostics": {"sources": ["controller"]},
        "data": _pose_response_data(
            left_quat=[2.0, 0.0, 0.0, 0.0],
            right_quat=[0.0, 0.0, 0.0, -4.0],
            details={"frames": ["left", "right"]},
        ),
    })

    first = transport.get_pose_state(timeout=0.1)
    assert first["timestamp"] == 1001
    assert first["states"] == pytest.approx([
        0.1,
        0.2,
        0.3,
        1.0,
        0.0,
        0.0,
        0.0,
        0.4,
        0.5,
        0.6,
        0.0,
        0.0,
        0.0,
        -1.0,
    ])
    assert first["guid"] == guid
    assert first["result"] == "success"
    assert first["data_timestamp"] == 1001
    assert first["diagnostics"] == {"sources": ["controller"]}
    assert first["details"] == {"frames": ["left", "right"]}

    first["diagnostics"]["sources"][0] = "mutated"
    first["details"]["frames"][0] = "mutated"
    second = transport.get_pose_state(timeout=0.0)
    assert second["diagnostics"] == {"sources": ["controller"]}
    assert second["details"] == {"frames": ["left", "right"]}


@pytest.mark.parametrize("response_guid", [None, "unmatched-guid"])
def test_missing_or_unmatched_pose_guid_never_updates_or_releases_query(
    transport,
    response_guid,
):
    expected_guid = transport.request_pose_state()
    response = {
        "title": "response_get_servo_pose",
        "timestamp": 999,
        "data": _pose_response_data(),
    }
    if response_guid is not None:
        response["guid"] = response_guid

    _ws_app().feed_message(response)

    with pytest.raises(StateError, match="pose state timed out"):
        transport.get_pose_state(timeout=0.0)
    assert transport.request_pose_state() == expected_guid
    assert len(_sent_messages("request_get_servo_pose")) == 1

    response["guid"] = expected_guid
    _ws_app().feed_message(response)
    assert transport.get_pose_state(timeout=0.1)["timestamp"] == 1001


@pytest.mark.parametrize(
    "invalid_data",
    [
        _pose_response_data(result="failed"),
        _pose_response_data(left_position=[0.1, 0.2]),
        _pose_response_data(left_quat=[0.0, 0.0, 0.0, 0.0]),
        _pose_response_data(right_position=[0.4, np.nan, 0.6]),
    ],
)
def test_invalid_pose_response_preserves_snapshot_and_releases_query(
    transport,
    invalid_data,
):
    first_guid = transport.request_pose_state()
    _ws_app().feed_message({
        "title": "response_get_servo_pose",
        "guid": first_guid,
        "data": _pose_response_data(timestamp=10),
    })
    original = transport.get_pose_state(timeout=0.1)

    invalid_guid = transport.request_pose_state()
    _ws_app().feed_message({
        "title": "response_get_servo_pose",
        "guid": invalid_guid,
        "data": invalid_data,
    })

    assert transport.get_pose_state(timeout=0.0) == original
    recovery_guid = transport.request_pose_state()
    assert recovery_guid != invalid_guid
    assert len(_sent_messages("request_get_servo_pose")) == 3


def test_pose_state_snapshot_honors_max_age(transport, monkeypatch):
    now = [10.0]
    monkeypatch.setattr(_ws_mod.time, "monotonic", lambda: now[0])
    guid = transport.request_pose_state()
    _ws_app().feed_message({
        "title": "response_get_servo_pose",
        "guid": guid,
        "data": _pose_response_data(),
    })

    now[0] = 10.25
    assert transport.get_pose_state(timeout=0.0, max_age=0.5)["timestamp"] == 1001
    now[0] = 10.75
    with pytest.raises(StateError, match="pose state is stale"):
        transport.get_pose_state(timeout=0.0, max_age=0.5)


def test_pose_state_waits_for_matching_guid_without_returning_cached_snapshot(
    transport,
):
    first_guid = transport.request_pose_state()
    _ws_app().feed_message({
        "title": "response_get_servo_pose",
        "guid": first_guid,
        "data": _pose_response_data(timestamp=1),
    })
    expected_guid = transport.request_pose_state()
    results = []
    errors = []

    def read_expected_pose():
        try:
            results.append(
                transport.get_pose_state(
                    timeout=0.5,
                    max_age=0.5,
                    expected_guid=expected_guid,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    reader = threading.Thread(target=read_expected_pose, daemon=True)
    reader.start()
    time.sleep(0.01)
    assert reader.is_alive()

    _ws_app().feed_message({
        "title": "response_get_servo_pose",
        "guid": expected_guid,
        "data": _pose_response_data(timestamp=2),
    })
    reader.join(timeout=0.5)

    assert not reader.is_alive()
    assert errors == []
    assert results[0]["guid"] == expected_guid
    assert results[0]["timestamp"] == 2


def test_brainco2_capability_protocol_is_optional(transport):
    from tron2_env.transport.base import BrainCo2Transport, RobotTransport

    assert isinstance(transport, BrainCo2Transport)
    assert "set_brainco2_position" not in RobotTransport.__dict__
    assert "request_brainco2_state" not in RobotTransport.__dict__
    assert "get_brainco2_state" not in RobotTransport.__dict__
    assert "raise_brainco2_fault" not in RobotTransport.__dict__


def test_set_brainco2_position_sends_exact_mode_one_raw_values(transport):
    left = np.array([-12.5, 0.25, 101.0, 4.0, 5.0, 6.0])
    right = np.array([16.0, 15.0, 14.0, 13.0, 12.0, 11.0])
    command_time = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    guid = transport.set_brainco2_position(left, right, command_time)

    sent = _sent_messages("request_set_brainco2_hand_cmd")
    assert len(sent) == 1
    assert sent[0]["guid"] == guid
    assert sent[0]["data"] == {
        "left_mode": 1,
        "left_pos": [-12.5, 0.25, 101.0, 4.0, 5.0, 6.0],
        "left_time": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "right_mode": 1,
        "right_pos": [16.0, 15.0, 14.0, 13.0, 12.0, 11.0],
        "right_time": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    }


@pytest.mark.parametrize(
    "left,right,command_time",
    [
        ([0.0] * 5, [0.0] * 6, [1.0] * 6),
        ([[0.0] * 6], [0.0] * 6, [1.0] * 6),
        ([0.0] * 6, [0.0] * 7, [1.0] * 6),
        ([0.0] * 6, [0.0] * 6, 1.0),
        ([0.0] * 6, [0.0] * 6, [1.0] * 5),
        ([0.0, 0.0, 0.0, 0.0, 0.0, np.nan], [0.0] * 6, [1.0] * 6),
        ([0.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.0, np.inf], [1.0] * 6),
        ([0.0] * 6, [0.0] * 6, [1.0, 1.0, 1.0, 1.0, 1.0, -np.inf]),
    ],
)
def test_set_brainco2_position_rejects_bad_shapes_and_nonfinite_values(
    transport,
    left,
    right,
    command_time,
):
    with pytest.raises(CommandError, match="finite shape \\(6,\\)"):
        transport.set_brainco2_position(left, right, command_time)

    assert not _sent_messages("request_set_brainco2_hand_cmd")


def test_brainco2_command_send_failure_raises_and_leaves_no_pending_guid(
    transport,
    monkeypatch,
):
    monkeypatch.setattr(transport, "_send_request", lambda *args, **kwargs: False)

    with pytest.raises(CommandError, match="BrainCo2 command send failed"):
        transport.set_brainco2_position([0.0] * 6, [1.0] * 6, [2.0] * 6)

    assert not transport._brainco2_command_guids


def test_matching_brainco2_success_ack_clears_only_its_guid(transport):
    first = transport.set_brainco2_position([0.0] * 6, [1.0] * 6, [2.0] * 6)
    second = transport.set_brainco2_position([2.0] * 6, [3.0] * 6, [4.0] * 6)

    _ws_app().feed_message({
        "title": "response_set_brainco2_hand_cmd",
        "guid": first,
        "data": {"result": "success"},
    })

    assert set(transport._brainco2_command_guids) == {second}
    transport.raise_brainco2_fault()


@pytest.mark.parametrize("response_guid", [None, "unmatched-guid"])
def test_missing_or_unmatched_brainco2_ack_is_ignored(transport, response_guid):
    expected = transport.set_brainco2_position(
        [0.0] * 6,
        [1.0] * 6,
        [2.0] * 6,
    )
    response = {
        "title": "response_set_brainco2_hand_cmd",
        "data": {"result": "failed", "message": "ignored"},
    }
    if response_guid is not None:
        response["guid"] = response_guid

    _ws_app().feed_message(response)

    assert set(transport._brainco2_command_guids) == {expected}
    transport.raise_brainco2_fault()


def test_matching_failed_brainco2_ack_records_one_persistent_fault(transport):
    guid = transport.set_brainco2_position([0.0] * 6, [1.0] * 6, [2.0] * 6)
    _ws_app().feed_message({
        "title": "response_set_brainco2_hand_cmd",
        "guid": guid,
        "data": {"result": "failed", "message": "motor fault"},
    })

    assert not transport._brainco2_command_guids
    for _ in range(2):
        with pytest.raises(CommandError, match="BrainCo2.*motor fault"):
            transport.raise_brainco2_fault()


def test_repeated_brainco2_fault_raises_use_fresh_bounded_exceptions(transport):
    guid = transport.set_brainco2_position([0.0] * 6, [1.0] * 6, [2.0] * 6)
    _ws_app().feed_message({
        "title": "response_set_brainco2_hand_cmd",
        "guid": guid,
        "data": {"result": "failed", "message": "motor fault"},
    })

    faults = []
    traceback_depths = []
    for _ in range(5):
        try:
            transport.raise_brainco2_fault()
        except CommandError as exc:
            faults.append(exc)
            traceback = exc.__traceback__
            depth = 0
            while traceback is not None:
                depth += 1
                traceback = traceback.tb_next
            traceback_depths.append(depth)
        else:
            pytest.fail("persistent BrainCo2 fault was not raised")

    assert len(set(traceback_depths)) == 1
    assert len({id(fault) for fault in faults}) == len(faults)
    assert {str(fault) for fault in faults} == {"BrainCo2 command failed: motor fault"}


def test_brainco2_command_ack_during_send_does_not_leave_dead_guid(
    transport,
    monkeypatch,
):
    send_entered = threading.Event()
    release_send = threading.Event()
    command_guid = []
    original_send = transport._send_request

    def blocking_send(title, data=None, *, guid=None):
        if title != "request_set_brainco2_hand_cmd":
            return original_send(title, data, guid=guid)
        command_guid.append(guid)
        send_entered.set()
        assert release_send.wait(timeout=1.0)
        return True

    monkeypatch.setattr(transport, "_send_request", blocking_send)
    sender = threading.Thread(
        target=lambda: transport.set_brainco2_position(
            [0.0] * 6,
            [1.0] * 6,
            [2.0] * 6,
        )
    )
    sender.start()
    assert send_entered.wait(timeout=1.0)
    acknowledger = threading.Thread(
        target=lambda: transport._on_message(None, json.dumps({
            "title": "response_set_brainco2_hand_cmd",
            "guid": command_guid[0],
            "data": {"result": "success"},
        }))
    )
    acknowledger.start()
    release_send.set()
    sender.join(timeout=1.0)
    acknowledger.join(timeout=1.0)

    assert not sender.is_alive()
    assert not acknowledger.is_alive()
    assert not transport._brainco2_command_guids


def test_brainco2_state_query_uses_exact_title_and_one_in_flight(transport):
    first = transport.request_brainco2_state()
    second = transport.request_brainco2_state()

    sent = _sent_messages("request_get_brainco2_hand_state")
    assert second == first
    assert len(sent) == 1
    assert sent[0]["guid"] == first
    assert sent[0]["data"] == {}


@pytest.mark.parametrize("response_guid", [None, "unmatched-guid"])
def test_missing_or_unmatched_brainco2_state_response_is_ignored(
    transport,
    response_guid,
):
    expected = transport.request_brainco2_state()
    response = {
        "title": "response_get_brainco2_hand_state",
        "data": _brainco_response_data(),
    }
    if response_guid is not None:
        response["guid"] = response_guid

    _ws_app().feed_message(response)

    with pytest.raises(StateError, match="BrainCo2 state timed out"):
        transport.get_brainco2_state(timeout=0.0)
    assert transport.request_brainco2_state() == expected
    assert len(_sent_messages("request_get_brainco2_hand_state")) == 1


def test_brainco2_state_preserves_order_complete_metadata_and_defensive_copies(
    transport,
):
    guid = transport.request_brainco2_state()
    _ws_app().feed_message({
        "title": "response_get_brainco2_hand_state",
        "guid": guid,
        "timestamp": 2000,
        "diagnostics": {"sources": ["controller"]},
        "data": _brainco_response_data(
            details={"temperatures": [30, 31]},
        ),
    })

    first = transport.get_brainco2_state(timeout=0.1)
    assert first["timestamp"] == 2001
    assert first["states"] == list(range(12))
    metadata = first["metadata"]
    assert metadata["robot_timestamp"] == 2001
    assert isinstance(metadata["received_at"], float)
    assert metadata["left_mode"] == 1
    assert metadata["left_names"] == [f"L{index}" for index in range(6)]
    assert metadata["left_vel"] == [10, 11, 12, 13, 14, 15]
    assert metadata["left_current"] == [20, 21, 22, 23, 24, 25]
    assert metadata["left_time"] == [2.0] * 6
    assert metadata["right_mode"] == 1
    assert metadata["right_names"] == [f"R{index}" for index in range(6)]
    assert metadata["right_vel"] == [16, 17, 18, 19, 20, 21]
    assert metadata["right_current"] == [26, 27, 28, 29, 30, 31]
    assert metadata["right_time"] == [2.0] * 6
    assert metadata["diagnostics"] == {"sources": ["controller"]}
    assert metadata["details"] == {"temperatures": [30, 31]}

    first["states"][0] = -999
    first["metadata"]["left_names"][0] = "mutated"
    first["metadata"]["details"]["temperatures"][0] = -999
    second = transport.get_brainco2_state(timeout=0.0)
    assert second["states"] == list(range(12))
    assert second["metadata"]["left_names"][0] == "L0"
    assert second["metadata"]["details"]["temperatures"] == [30, 31]


@pytest.mark.parametrize(
    "invalid_data",
    [
        _brainco_response_data(result="failed"),
        _brainco_response_data(left_pos=[0, 1, 2, 3, 4]),
        _brainco_response_data(left_pos=[0, 1, 2, 3, 4, np.nan]),
        _brainco_response_data(right_pos=[[6, 7, 8, 9, 10, 11]]),
        _brainco_response_data(right_pos=[6, 7, 8, 9, 10, np.inf]),
    ],
)
def test_invalid_brainco2_state_preserves_snapshot_and_releases_query(
    transport,
    invalid_data,
):
    first = transport.request_brainco2_state()
    _feed_brainco_state(first)
    expected = transport.get_brainco2_state(timeout=0.1)

    invalid = transport.request_brainco2_state()
    _ws_app().feed_message({
        "title": "response_get_brainco2_hand_state",
        "guid": invalid,
        "data": invalid_data,
    })

    assert transport.get_brainco2_state(timeout=0.0) == expected
    recovery = transport.request_brainco2_state()
    assert recovery != invalid


def test_brainco2_state_snapshot_honors_max_age(transport, monkeypatch):
    now = [10.0]
    monkeypatch.setattr(_ws_mod.time, "monotonic", lambda: now[0])
    guid = transport.request_brainco2_state()
    _feed_brainco_state(guid)

    now[0] = 10.25
    assert transport.get_brainco2_state(timeout=0.0, max_age=0.5)["timestamp"] == 2001
    now[0] = 10.75
    with pytest.raises(StateError, match="BrainCo2 state is stale"):
        transport.get_brainco2_state(timeout=0.0, max_age=0.5)


def test_late_matching_brainco2_state_is_discarded_before_next_poll(
    transport,
    monkeypatch,
):
    now = [0.0]
    monkeypatch.setattr(transport, "_feedback_clock", lambda: now[0])
    first_guid = transport.request_brainco2_state()
    _feed_brainco_state(first_guid, timestamp=10)
    expected = transport.get_brainco2_state(timeout=0.1)

    late_guid = transport.request_brainco2_state()
    now[0] = 1.001
    _feed_brainco_state(
        late_guid,
        timestamp=20,
        left_pos=[100, 101, 102, 103, 104, 105],
        right_pos=[106, 107, 108, 109, 110, 111],
    )

    assert transport.get_brainco2_state(timeout=0.0) == expected
    assert transport.request_brainco2_state() != late_guid


def test_expired_brainco2_command_ack_faults_and_bounds_pending_on_next_send(
    transport,
    monkeypatch,
):
    now = [0.0]
    monkeypatch.setattr(transport, "_feedback_clock", lambda: now[0])
    transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)
    first_guid = transport.set_brainco2_position(
        [0.0] * 6,
        [1.0] * 6,
        [2.0] * 6,
    )
    assert first_guid in transport._brainco2_command_guids

    now[0] = 0.201
    with pytest.raises(CommandError, match="BrainCo2 acknowledgement timed out"):
        transport.set_brainco2_position([3.0] * 6, [4.0] * 6, [5.0] * 6)

    assert not transport._brainco2_command_guids
    assert len(_sent_messages("request_set_brainco2_hand_cmd")) == 1


def test_poll_tick_expires_brainco2_command_ack_and_late_ack_is_ignored(
    transport,
    monkeypatch,
):
    now = [0.0]
    monkeypatch.setattr(transport, "_feedback_clock", lambda: now[0])
    transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)
    guid = transport.set_brainco2_position(
        [0.0] * 6,
        [1.0] * 6,
        [2.0] * 6,
    )

    now[0] = 0.201
    transport.poll_enabled_state()
    assert not transport._brainco2_command_guids
    with pytest.raises(CommandError, match="BrainCo2 acknowledgement timed out"):
        transport.raise_brainco2_fault()

    _ws_app().feed_message({
        "title": "response_set_brainco2_hand_cmd",
        "guid": guid,
        "data": {"result": "success"},
    })
    with pytest.raises(CommandError, match="BrainCo2 acknowledgement timed out"):
        transport.raise_brainco2_fault()


def test_late_brainco2_ack_itself_records_timeout_fault(transport, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(transport, "_feedback_clock", lambda: now[0])
    transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)
    guid = transport.set_brainco2_position(
        [0.0] * 6,
        [1.0] * 6,
        [2.0] * 6,
    )

    now[0] = 0.201
    _ws_app().feed_message({
        "title": "response_set_brainco2_hand_cmd",
        "guid": guid,
        "data": {"result": "success"},
    })

    assert not transport._brainco2_command_guids
    with pytest.raises(CommandError, match="BrainCo2 acknowledgement timed out"):
        transport.raise_brainco2_fault()


def test_lost_session_with_pending_brainco2_command_requires_fresh_state_recovery(
    transport,
):
    app = _ws_app()
    first_guid = transport.set_brainco2_position(
        [0.0] * 6,
        [1.0] * 6,
        [2.0] * 6,
    )

    transport._on_close(app, 1006, "connection lost")
    transport._on_open(app)
    app.feed_message({
        "title": "notify_robot_info",
        "accid": "RECONNECTED_ROBOT",
        "data": {},
    })

    with pytest.raises(CommandError, match="session lost.*unacknowledged"):
        transport.set_brainco2_position([3.0] * 6, [4.0] * 6, [5.0] * 6)
    app.feed_message({
        "title": "response_set_brainco2_hand_cmd",
        "guid": first_guid,
        "data": {"result": "success"},
    })
    with pytest.raises(StateError, match="fresh BrainCo2 state"):
        transport.clear_brainco2_session_uncertainty(
            transport._session_generation
        )

    state_guid = transport.request_brainco2_state()
    _feed_brainco_state(state_guid)
    transport.clear_brainco2_session_uncertainty(transport._session_generation)
    transport.set_brainco2_position([3.0] * 6, [4.0] * 6, [5.0] * 6)

    commands = _sent_messages("request_set_brainco2_hand_cmd")
    assert len(commands) == 2
    assert commands[0]["guid"] == first_guid


def test_brainco2_session_recovery_does_not_clear_failed_ack_fault(transport):
    guid = transport.set_brainco2_position(
        [0.0] * 6,
        [1.0] * 6,
        [2.0] * 6,
    )
    _ws_app().feed_message({
        "title": "response_set_brainco2_hand_cmd",
        "guid": guid,
        "data": {"result": "failed", "message": "motor rejected"},
    })

    with pytest.raises(CommandError, match="motor rejected"):
        transport.clear_brainco2_session_uncertainty(
            object()
        )

    with pytest.raises(CommandError, match="motor rejected"):
        transport.raise_brainco2_fault()


def test_brainco2_session_recovery_rejects_stale_same_generation_snapshot(
    unpolled_transport,
):
    unpolled_transport.enable_brainco2_polling(
        rate=20.0,
        query_timeout=0.2,
    )
    generation = unpolled_transport._session_generation
    unpolled_transport._brainco2_session_uncertain = True
    with unpolled_transport._brainco2_query_lock:
        unpolled_transport._brainco2_snapshot.update(
            np.arange(12, dtype=np.float64),
            received_at=time.monotonic() - 1.0,
            metadata={"session_generation": generation},
        )
        unpolled_transport._brainco2_state_generation = generation

    with pytest.raises(StateError, match="fresh BrainCo2 state"):
        unpolled_transport.clear_brainco2_session_uncertainty(generation)

    assert unpolled_transport._brainco2_session_uncertain

    with unpolled_transport._brainco2_query_lock:
        unpolled_transport._brainco2_snapshot.update(
            np.arange(12, dtype=np.float64),
            received_at=time.monotonic(),
            metadata={"session_generation": generation},
        )
    unpolled_transport.clear_brainco2_session_uncertainty(generation)
    assert not unpolled_transport._brainco2_session_uncertain


def test_brainco2_session_recovery_checks_snapshot_before_connection_lock(
    unpolled_transport,
):
    generation = unpolled_transport._session_generation
    unpolled_transport._brainco2_session_uncertain = True
    unpolled_transport._brainco2_state_generation = generation
    unpolled_transport._brainco2_snapshot.update(
        np.arange(12, dtype=np.float64),
        received_at=time.monotonic(),
        metadata={"session_generation": generation},
    )
    events = []
    original_command_lock = unpolled_transport._brainco2_command_lock
    original_query_lock = unpolled_transport._brainco2_query_lock
    original_connection_lock = unpolled_transport._connection_state_lock
    original_snapshot = unpolled_transport._brainco2_snapshot

    class RecordingLock:
        def __init__(self, name, lock):
            self._name = name
            self._lock = lock

        def __enter__(self):
            self._lock.acquire()
            events.append(self._name)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._lock.release()

    class RecordingSnapshot:
        def get(self, **kwargs):
            events.append("snapshot")
            return original_snapshot.get(**kwargs)

    unpolled_transport._brainco2_command_lock = RecordingLock(
        "command",
        original_command_lock,
    )
    unpolled_transport._brainco2_query_lock = RecordingLock(
        "query",
        original_query_lock,
    )
    unpolled_transport._connection_state_lock = RecordingLock(
        "connection",
        original_connection_lock,
    )
    unpolled_transport._brainco2_snapshot = RecordingSnapshot()
    try:
        unpolled_transport.clear_brainco2_session_uncertainty(generation)
    finally:
        unpolled_transport._brainco2_command_lock = original_command_lock
        unpolled_transport._brainco2_query_lock = original_query_lock
        unpolled_transport._connection_state_lock = original_connection_lock
        unpolled_transport._brainco2_snapshot = original_snapshot

    assert events == ["command", "query", "snapshot", "connection"]


def test_brainco2_polling_rate_single_flight_and_invalid_response_recovery(
    monkeypatch,
):
    _FakeApp.instances.clear()
    now = [0.0]
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: None,
    )
    config = Tron2Config(robot_ip="127.0.0.1", polling_rate=2.0)
    transport = WebsocketTransport(config, feedback_clock=lambda: now[0])
    deadline = time.time() + 1.0
    while not transport.is_connected() and time.time() < deadline:
        time.sleep(0.01)
    try:
        _ws_app().sent.clear()
        transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)

        transport.poll_enabled_state()
        first = _sent_messages("request_get_brainco2_hand_state")
        assert len(first) == 1

        transport.poll_enabled_state()
        assert _sent_messages("request_get_brainco2_hand_state") == first

        _feed_brainco_state(first[0]["guid"], left_pos=[0, 1, 2, 3, 4, np.nan])
        now[0] = 0.049
        transport.poll_enabled_state()
        assert _sent_messages("request_get_brainco2_hand_state") == first

        now[0] = 0.05
        transport.poll_enabled_state()
        second = _sent_messages("request_get_brainco2_hand_state")
        assert len(second) == 2
        assert second[-1]["guid"] != first[0]["guid"]
    finally:
        transport.disconnect()


def test_brainco2_polling_recovers_after_query_timeout_and_disable_stops_it(
    monkeypatch,
):
    _FakeApp.instances.clear()
    now = [0.0]
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: None,
    )
    config = Tron2Config(robot_ip="127.0.0.1", polling_rate=2.0)
    transport = WebsocketTransport(config, feedback_clock=lambda: now[0])
    deadline = time.time() + 1.0
    while not transport.is_connected() and time.time() < deadline:
        time.sleep(0.01)
    try:
        _ws_app().sent.clear()
        transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)
        transport.poll_enabled_state()
        first = _sent_messages("request_get_brainco2_hand_state")[-1]

        now[0] = 0.201
        transport.poll_enabled_state()
        second = _sent_messages("request_get_brainco2_hand_state")[-1]
        assert second["guid"] != first["guid"]

        transport.disable_brainco2_polling()
        now[0] = 10.0
        transport.poll_enabled_state()
        assert len(_sent_messages("request_get_brainco2_hand_state")) == 2
    finally:
        transport.disconnect()


def test_low_rate_brainco2_poll_requests_before_short_state_wait(
    unpolled_transport,
    monkeypatch,
):
    now = [10.0]
    monkeypatch.setattr(unpolled_transport, "_feedback_clock", lambda: now[0])

    unpolled_transport.enable_brainco2_polling(
        rate=1.0,
        query_timeout=0.5,
    )
    unpolled_transport._poll_feedback_once()

    requests = _sent_messages("request_get_brainco2_hand_state")
    assert len(requests) == 1
    _feed_brainco_state(requests[0]["guid"])
    state = unpolled_transport.get_brainco2_state(
        timeout=0.5,
        max_age=0.5,
    )

    assert state["states"] == pytest.approx(list(range(12)))


def test_feedback_scheduler_keeps_main_and_brainco2_rates_independent(
    monkeypatch,
):
    _FakeApp.instances.clear()
    now = [0.0]
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: None,
    )
    transport = WebsocketTransport(
        Tron2Config(robot_ip="127.0.0.1", polling_rate=2.0),
        feedback_clock=lambda: now[0],
    )
    deadline = time.time() + 1.0
    while not transport.is_connected() and time.time() < deadline:
        time.sleep(0.01)
    try:
        # Prime the main source at t=0, matching the real loop's startup tick,
        # then measure the following one-second interval.
        transport._poll_feedback_once()
        _ws_app().sent.clear()
        transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)
        transport._poll_feedback_once()
        initial = _sent_messages("request_get_brainco2_hand_state")
        assert len(initial) == 1
        _feed_brainco_state(initial[0]["guid"])
        acknowledged = 1

        for millisecond in range(1, 1001):
            now[0] = millisecond / 1000.0
            transport._poll_feedback_once()
            requests = _sent_messages("request_get_brainco2_hand_state")
            for request in requests[acknowledged:]:
                _feed_brainco_state(request["guid"])
            acknowledged = len(requests)

        assert len(_sent_messages("request_get_joint_state")) == 2
        assert len(_sent_messages("request_get_limx_2fclaw_state")) == 2
        assert len(_sent_messages("request_get_brainco2_hand_state")) == 21
    finally:
        transport.disconnect()


@pytest.mark.parametrize(
    ("poll_joint_state", "poll_gripper_state", "expected_titles"),
    [
        (True, True, {"request_get_joint_state", "request_get_limx_2fclaw_state"}),
        (True, False, {"request_get_joint_state"}),
        (False, True, {"request_get_limx_2fclaw_state"}),
        (False, False, set()),
    ],
)
def test_main_feedback_polling_respects_enabled_sources(
    monkeypatch,
    poll_joint_state,
    poll_gripper_state,
    expected_titles,
):
    _FakeApp.instances.clear()
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: None,
    )
    transport = WebsocketTransport(
        Tron2Config(robot_ip="127.0.0.1", polling_rate=2.0),
        poll_joint_state=poll_joint_state,
        poll_gripper_state=poll_gripper_state,
        feedback_clock=lambda: 0.0,
    )
    try:
        _ws_app().sent.clear()
        transport._poll_feedback_once()

        titles = {
            message["title"]
            for message in (
                json.loads(raw_message) for raw_message in _ws_app().sent
            )
        }
        assert titles == expected_titles
        assert (transport._main_feedback_next_at > 0.0) is bool(expected_titles)
    finally:
        transport.disconnect()


def test_legacy_transport_defaults_to_polling_both_main_sources(
    unpolled_transport,
):
    assert unpolled_transport.poll_joint_state is True
    assert unpolled_transport.poll_gripper_state is True

    unpolled_transport._main_feedback_next_at = 0.0
    unpolled_transport._feedback_clock = lambda: 0.0
    unpolled_transport._poll_feedback_once()

    assert len(_sent_messages("request_get_joint_state")) == 1
    assert len(_sent_messages("request_get_limx_2fclaw_state")) == 1


def test_feedback_scheduler_waits_indefinitely_with_every_source_disabled(
    unpolled_transport,
):
    unpolled_transport.poll_joint_state = False
    unpolled_transport.poll_gripper_state = False
    waits = []

    class StopAfterWait:
        def clear(self):
            pass

        def set(self):
            pass

        def wait(self, timeout=None):
            waits.append(timeout)
            unpolled_transport.should_exit = True

    unpolled_transport._feedback_wakeup = StopAfterWait()

    assert unpolled_transport._next_feedback_deadline() is None
    unpolled_transport._poll_feedback()

    assert waits == [None]


def test_feedback_scheduler_caps_finite_wait_at_threading_timeout_max(
    unpolled_transport,
    monkeypatch,
):
    unpolled_transport.poll_joint_state = False
    unpolled_transport.poll_gripper_state = False
    monkeypatch.setattr(unpolled_transport, "_feedback_clock", lambda: 0.0)
    unpolled_transport._pose_query_in_flight = "oversized-internal-deadline"
    unpolled_transport._pose_query_deadline = threading.TIMEOUT_MAX * 2.0
    waits = []

    class StopAfterWait:
        def clear(self):
            pass

        def set(self):
            pass

        def wait(self, timeout=None):
            waits.append(timeout)
            unpolled_transport.should_exit = True

    unpolled_transport._feedback_wakeup = StopAfterWait()

    unpolled_transport._poll_feedback()

    assert waits == [threading.TIMEOUT_MAX]


def test_brainco_enable_disconnect_and_reconnect_wake_idle_scheduler(
    unpolled_transport,
):
    unpolled_transport.poll_joint_state = False
    unpolled_transport.poll_gripper_state = False
    wakeup = unpolled_transport._feedback_wakeup

    wakeup.clear()
    unpolled_transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)
    assert wakeup.is_set()

    wakeup.clear()
    app = _ws_app()
    unpolled_transport._on_close(app, 1006, "connection lost")
    assert wakeup.is_set()

    wakeup.clear()
    unpolled_transport._on_open(app)
    assert wakeup.is_set()

    wakeup.clear()
    app.feed_message({
        "title": "notify_robot_info",
        "accid": "RECONNECTED_ROBOT",
        "data": {},
    })
    assert wakeup.is_set()
    assert unpolled_transport._brainco2_next_poll_at is not None


def test_brainco_enable_during_ready_transition_is_immediately_due(
    unpolled_transport,
    monkeypatch,
):
    now = [5.0]
    monkeypatch.setattr(unpolled_transport, "_feedback_clock", lambda: now[0])
    app = _ws_app()
    unpolled_transport._on_close(app, 1006, "connection lost")
    unpolled_transport._on_open(app)
    assert not unpolled_transport._session_ready.is_set()

    ready_notification_entered = threading.Event()
    allow_ready_publication = threading.Event()
    errors = []

    def gate_ready_notification(ready, _generation):
        if not ready:
            return
        ready_notification_entered.set()
        if not allow_ready_publication.wait(timeout=0.5):
            raise AssertionError("ready publication gate timed out")

    monkeypatch.setattr(
        unpolled_transport,
        "_notify_connection_state",
        gate_ready_notification,
    )

    def announce_robot_identity():
        try:
            unpolled_transport._handle_robot_info(
                app,
                {
                    "title": "notify_robot_info",
                    "accid": "RECONNECTED_ROBOT",
                    "data": {},
                },
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    announcer = threading.Thread(target=announce_robot_identity, daemon=True)
    announcer.start()
    assert ready_notification_entered.wait(timeout=0.5)

    unpolled_transport.enable_brainco2_polling(
        rate=20.0,
        query_timeout=0.2,
    )
    assert unpolled_transport._brainco2_next_poll_at is None
    assert not unpolled_transport._session_ready.is_set()

    allow_ready_publication.set()
    announcer.join(timeout=0.5)

    assert not announcer.is_alive()
    assert errors == []
    assert unpolled_transport._session_ready.is_set()
    assert unpolled_transport._brainco2_next_poll_at == pytest.approx(5.0)

    unpolled_transport._poll_feedback_once()
    requests = _sent_messages("request_get_brainco2_hand_state")
    assert len(requests) == 1
    unpolled_transport._poll_feedback_once()
    assert _sent_messages("request_get_brainco2_hand_state") == requests


def test_pose_query_deadline_wakes_and_is_expired_by_feedback_scheduler(
    unpolled_transport,
    monkeypatch,
):
    now = [0.0]
    monkeypatch.setattr(unpolled_transport, "_feedback_clock", lambda: now[0])
    unpolled_transport.poll_joint_state = False
    unpolled_transport.poll_gripper_state = False

    guid = unpolled_transport.request_pose_state(timeout=0.2)
    assert unpolled_transport._next_feedback_deadline() == pytest.approx(0.2)

    now[0] = 0.2
    unpolled_transport._poll_feedback_once()

    assert unpolled_transport._pose_query_in_flight is None
    assert unpolled_transport._pose_query_deadline is None
    assert unpolled_transport._next_feedback_deadline() is None
    assert [
        message["guid"] for message in _sent_messages("request_get_servo_pose")
    ] == [guid]


def test_pending_brainco2_query_schedules_next_loop_at_recovery_deadline(
    monkeypatch,
):
    _FakeApp.instances.clear()
    now = [0.0]
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: None,
    )
    transport = WebsocketTransport(
        Tron2Config(robot_ip="127.0.0.1", polling_rate=2.0),
        feedback_clock=lambda: now[0],
    )
    deadline = time.time() + 1.0
    while not transport.is_connected() and time.time() < deadline:
        time.sleep(0.01)
    try:
        transport._poll_feedback_once()
        transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)
        transport._poll_feedback_once()
        now[0] = 0.10

        assert transport._next_feedback_deadline() == pytest.approx(0.2)

        now[0] = 0.201
        transport._poll_feedback_once()
        requests = _sent_messages("request_get_brainco2_hand_state")
        assert len(requests) == 2
        assert requests[0]["guid"] != requests[1]["guid"]
    finally:
        transport.disconnect()


def test_disconnected_brainco2_polling_does_not_zero_wait(monkeypatch):
    _FakeApp.instances.clear()
    now = [0.0]
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: None,
    )
    transport = WebsocketTransport(
        Tron2Config(robot_ip="127.0.0.1", polling_rate=2.0),
        feedback_clock=lambda: now[0],
    )
    deadline = time.time() + 1.0
    while not transport.is_connected() and time.time() < deadline:
        time.sleep(0.01)

    waits = []

    class _StopAfterWait:
        def clear(self):
            pass

        def set(self):
            pass

        def wait(self, timeout):
            waits.append(timeout)
            transport.should_exit = True

    try:
        transport._poll_feedback_once()
        transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)
        now[0] = 0.1
        transport._on_close(_ws_app(), 1006, "connection lost")
        transport._feedback_wakeup = _StopAfterWait()

        transport._poll_feedback()

        assert waits == pytest.approx([0.4])
    finally:
        transport.disconnect()


def test_brainco2_polling_resumes_from_reconnect_time(monkeypatch):
    _FakeApp.instances.clear()
    now = [0.0]
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: None,
    )
    transport = WebsocketTransport(
        Tron2Config(robot_ip="127.0.0.1", polling_rate=2.0),
        feedback_clock=lambda: now[0],
    )
    deadline = time.time() + 1.0
    while not transport.is_connected() and time.time() < deadline:
        time.sleep(0.01)
    try:
        transport._poll_feedback_once()
        _ws_app().sent.clear()
        transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)

        now[0] = 0.1
        transport._on_close(_ws_app(), 1006, "connection lost")
        transport._poll_feedback_once()
        assert not _sent_messages("request_get_brainco2_hand_state")

        now[0] = 0.2
        transport._on_open(_ws_app())
        _ws_app().feed_message({
            "title": "notify_robot_info",
            "accid": "RECONNECTED_ROBOT",
            "data": {},
        })
        transport._poll_feedback_once()
        requests = _sent_messages("request_get_brainco2_hand_state")
        assert len(requests) == 1

        transport._poll_feedback_once()
        assert _sent_messages("request_get_brainco2_hand_state") == requests

        now[0] = 0.249
        transport._poll_feedback_once()
        assert _sent_messages("request_get_brainco2_hand_state") == requests
    finally:
        transport.disconnect()


def test_websocket_attempts_reset_state_reconnect_and_stop_during_backoff(
    monkeypatch,
):
    _FakeApp.instances.clear()
    now = [0.0]
    monkeypatch.setattr(_fake_ws_module, "WebSocketApp", _ScriptedAttemptApp)
    monkeypatch.setattr(_ws_mod, "_WEBSOCKET_RECONNECT_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: None,
    )
    transport = WebsocketTransport(
        Tron2Config(robot_ip="127.0.0.1", polling_rate=2.0),
        feedback_clock=lambda: now[0],
    )
    app = _ws_app()
    assert isinstance(app, _ScriptedAttemptApp)
    assert app.attempt_opened[1].wait(timeout=0.5)
    assert transport.is_connected()

    try:
        transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)
        now[0] = 0.1
        app.finish_attempt[1].set()
        assert app.attempt_closed[1].wait(timeout=0.5)

        assert not transport.is_connected()
        assert transport._brainco2_polling_enabled
        assert transport._brainco2_next_poll_at is None

        now[0] = 0.2
        app.allow_attempt_return[1].set()

        assert app.attempt_opened[2].wait(timeout=0.5)
        assert app.attempt_count == 2
        assert app.reconnect_arguments == [None, None]
        assert transport.is_connected()

        transport._poll_feedback_once()
        requests = _sent_messages("request_get_brainco2_hand_state")
        assert len(requests) == 1

        transport._poll_feedback_once()
        assert _sent_messages("request_get_brainco2_hand_state") == requests

        now[0] = 0.249
        transport._poll_feedback_once()
        assert _sent_messages("request_get_brainco2_hand_state") == requests

        backoff = _RecordingStopEvent()
        transport._reconnect_wakeup = backoff
        monkeypatch.setattr(
            _ws_mod,
            "_WEBSOCKET_RECONNECT_DELAY_SECONDS",
            60.0,
        )
        app.finish_attempt[2].set()
        assert app.attempt_closed[2].wait(timeout=0.5)
        app.allow_attempt_return[2].set()
        assert backoff.wait_started.wait(timeout=0.5)

        started = time.monotonic()
        transport.disconnect()
        elapsed = time.monotonic() - started

        assert elapsed < 0.5
        assert backoff.wait_timeouts == [60.0]
        assert app.attempt_count == 2
    finally:
        if not transport.should_exit:
            transport.disconnect()

    assert transport.ws_thread is not None
    assert not transport.ws_thread.is_alive()


def test_websocket_attempt_exception_resets_state_without_close_callback(
    monkeypatch,
):
    _FakeApp.instances.clear()
    monkeypatch.setattr(_fake_ws_module, "WebSocketApp", _ScriptedAttemptApp)
    monkeypatch.setattr(_ws_mod, "_WEBSOCKET_RECONNECT_DELAY_SECONDS", 60.0)
    monkeypatch.setattr(
        WebsocketTransport,
        "_start_polling_thread",
        lambda self: None,
    )
    transport = WebsocketTransport(
        Tron2Config(robot_ip="127.0.0.1", polling_rate=2.0),
        feedback_clock=lambda: 0.0,
    )
    app = _ws_app()
    backoff = _RecordingStopEvent()
    transport._reconnect_wakeup = backoff
    try:
        assert app.attempt_opened[1].wait(timeout=0.5)
        transport.enable_brainco2_polling(rate=20.0, query_timeout=0.2)
        app.raise_without_close.add(1)
        app.finish_attempt[1].set()

        assert app.attempt_returned[1].wait(timeout=0.5)
        assert backoff.wait_started.wait(timeout=0.5)
        assert not transport.is_connected()
        assert transport._brainco2_polling_enabled
        assert transport._brainco2_next_poll_at is None
    finally:
        transport.disconnect()


@pytest.mark.parametrize(
    ("timeout", "expected_error"),
    _INVALID_PUBLIC_WAIT_TIMEOUTS,
)
def test_get_joint_state_rejects_invalid_timeout_before_consuming_queue(
    transport,
    timeout,
    expected_error,
):
    queued = {"sentinel": True}
    with transport._queue_lock:
        transport.joint_state_queue.clear()
        transport.joint_state_queue.append(queued)

    with pytest.raises(expected_error, match="timeout"):
        transport.get_joint_state(timeout=timeout)

    with transport._queue_lock:
        assert list(transport.joint_state_queue) == [queued]


@pytest.mark.parametrize(
    ("timeout", "expected_error"),
    _INVALID_PUBLIC_WAIT_TIMEOUTS,
)
def test_wait_until_reached_rejects_invalid_timeout_before_reading_state(
    transport,
    monkeypatch,
    timeout,
    expected_error,
):
    reads = []

    def record_state_read(*args, **kwargs):
        reads.append((args, kwargs))
        raise AssertionError("state must not be read for an invalid timeout")

    monkeypatch.setattr(transport, "get_joint_state", record_state_read)

    with pytest.raises(expected_error, match="timeout"):
        transport.wait_until_reached(
            np.zeros(JointIndex.MOVEJ_DIM),
            timeout=timeout,
        )

    assert reads == []


def test_get_joint_state_commits_when_both_frames_arrive(transport):
    # Inject one joint state frame — should NOT commit by itself.
    joint_q = list(np.arange(16) * 0.01)
    joint_before_ms = int(time.time() * 1000)
    _ws_app().feed_message({
        "title": "response_get_joint_state",
        "timestamp": 1234567,
        "data": {"q": joint_q},
    })
    joint_after_ms = int(time.time() * 1000)
    with pytest.raises(StateError):
        transport.get_joint_state(timeout=0.05)

    # Inject gripper frame — now commit fires.
    _ws_app().feed_message({
        "title": "response_get_limx_2fclaw_state",
        "timestamp": 1234568,
        "data": {"left_opening": 40, "right_opening": 60},
    })
    state = transport.get_joint_state(timeout=0.5)
    assert joint_before_ms <= state["timestamp"] <= joint_after_ms
    assert state["robot_timestamp"] == 1234567
    assert len(state["states"]) == JointIndex.STATE_DIM
    np.testing.assert_allclose(state["states"][JointIndex.LEFT_ARM], joint_q[:7])
    np.testing.assert_allclose(state["states"][JointIndex.RIGHT_ARM], joint_q[7:14])
    assert state["states"][JointIndex.LEFT_GRIPPER] == pytest.approx(0.4)
    assert state["states"][JointIndex.RIGHT_GRIPPER] == pytest.approx(0.6)
    assert state["states"][JointIndex.HEAD_PITCH] == pytest.approx(joint_q[14])
    assert state["states"][JointIndex.HEAD_YAW] == pytest.approx(joint_q[15])


def test_arm_head_snapshot_does_not_wait_for_gripper(transport):
    _feed_joint_state(list(range(16)), timestamp=2000, guid="joint-guid")

    snapshot = transport.get_arm_head_state(timeout=0.1)

    assert snapshot["timestamp"] == 2000
    assert snapshot["states"] == list(range(16))
    assert snapshot["result"] == "success"
    assert snapshot["guid"] == "joint-guid"
    with pytest.raises(StateError, match="timed out"):
        transport.get_joint_state(timeout=0.0)


@pytest.mark.parametrize("expected_guid", ["", "   ", 1, object()])
def test_get_arm_head_state_rejects_invalid_expected_guid(
    transport,
    expected_guid,
):
    with pytest.raises(ValueError, match="expected_guid must be a non-empty string"):
        transport.get_arm_head_state(
            timeout=0.0,
            expected_guid=expected_guid,
        )


def test_gripper_snapshot_does_not_wait_for_joint(transport):
    _feed_gripper_state(25, 75, timestamp=3000, guid="gripper-guid")

    snapshot = transport.get_gripper_state(timeout=0.1)

    assert snapshot["timestamp"] == 3000
    assert snapshot["states"] == pytest.approx([0.25, 0.75])
    assert snapshot["result"] == "success"
    assert snapshot["guid"] == "gripper-guid"
    with pytest.raises(StateError, match="timed out"):
        transport.get_joint_state(timeout=0.0)


def test_legacy_joint_state_still_combines_gripper_in_canonical_order(transport):
    _feed_joint_state(list(range(16)), timestamp=4000)
    _feed_gripper_state(25, 75, timestamp=4001)

    state = transport.get_joint_state(timeout=0.1)["states"]

    assert state[:8] == [0, 1, 2, 3, 4, 5, 6, 0.25]
    assert state[8:16] == [7, 8, 9, 10, 11, 12, 13, 0.75]
    assert state[16:] == [14, 15]


def test_reconnect_does_not_combine_legacy_frames_from_different_sessions(
    transport,
):
    app = _ws_app()
    _feed_joint_state([1.0] * JointIndex.SERVOJ_DIM, timestamp=4000)

    transport._on_close(app, 1006, "connection lost")
    transport._on_open(app)
    app.feed_message({
        "title": "notify_robot_info",
        "accid": "RECONNECTED_ROBOT",
        "data": {},
    })
    _feed_gripper_state(25, 75, timestamp=4001)

    with pytest.raises(StateError, match="timed out"):
        transport.get_joint_state(timeout=0.0)

    _feed_joint_state([2.0] * JointIndex.SERVOJ_DIM, timestamp=4002)
    state = transport.get_joint_state(timeout=0.1)["states"]
    np.testing.assert_allclose(state[JointIndex.LEFT_ARM], [2.0] * 7)
    np.testing.assert_allclose(state[JointIndex.RIGHT_ARM], [2.0] * 7)
    assert state[JointIndex.LEFT_GRIPPER] == pytest.approx(0.25)
    assert state[JointIndex.RIGHT_GRIPPER] == pytest.approx(0.75)


def test_main_connection_reset_invalidates_all_session_observations(transport):
    app = _ws_app()
    _feed_joint_state([1.0] * JointIndex.SERVOJ_DIM, timestamp=5000)
    _feed_gripper_state(10, 20, timestamp=5001)

    pose_guid = transport.request_pose_state()
    app.feed_message({
        "title": "response_get_servo_pose",
        "guid": pose_guid,
        "data": _pose_response_data(timestamp=5002),
    })
    brainco_guid = transport.request_brainco2_state()
    _feed_brainco_state(brainco_guid, timestamp=5003)
    app.feed_message({
        "title": "response_get_move_pose",
        "guid": "legacy-pose",
        "data": _pose_response_data(timestamp=5004),
    })
    app.feed_message({
        "title": "response_chassis_state",
        "timestamp": 5005,
        "data": {"result": "success", "data": [0.1, -0.2]},
    })
    app.feed_message({
        "title": "response_lifter_state",
        "timestamp": 5006,
        "data": {"result": "success", "q": [0.3]},
    })

    transport._on_close(app, 1006, "connection lost")

    for getter in (
        transport.get_joint_state,
        transport.get_arm_head_state,
        transport.get_gripper_state,
        transport.get_pose_state,
        transport.get_brainco2_state,
    ):
        with pytest.raises(StateError, match="timed out"):
            getter(timeout=0.0)
    assert transport.find_nearest_state(5.0) is None
    assert list(transport.ee_pose_queue) == []
    assert transport.ee_pose_states["timestamp"] == -1
    assert not transport.chassis_states["valid"]
    assert not transport.lifter_states["valid"]
    assert transport._pose_query_in_flight is None
    assert transport._pose_query_deadline is None


def test_legacy_joint_state_uses_oldest_component_receive_time_for_freshness(
    transport,
    monkeypatch,
):
    clock = [10.0]
    monkeypatch.setattr(_ws_mod.time, "monotonic", lambda: clock[0])
    _feed_joint_state([1.0] * 16)
    clock[0] = 10.25
    _feed_gripper_state(10, 20)

    fresh = transport.get_joint_state(timeout=0.1, max_age=0.5)
    assert fresh["received_at"] == pytest.approx(10.0)

    clock[0] = 20.0
    _feed_joint_state([2.0] * 16)
    clock[0] = 20.25
    _feed_gripper_state(30, 40)
    clock[0] = 20.6

    with pytest.raises(StateError, match="joint state is stale"):
        transport.get_joint_state(timeout=0.1, max_age=0.5)


def test_legacy_joint_state_still_needs_one_new_frame_from_each_side(transport):
    _feed_joint_state([1.0] * 16)
    _feed_gripper_state(10, 20)
    transport.get_joint_state(timeout=0.1)

    _feed_joint_state([2.0] * 16)
    _feed_joint_state([3.0] * 16)
    with pytest.raises(StateError, match="timed out"):
        transport.get_joint_state(timeout=0.0)

    _feed_gripper_state(30, 40)
    second = transport.get_joint_state(timeout=0.1)["states"]
    np.testing.assert_allclose(second[JointIndex.LEFT_ARM], [3.0] * 7)
    np.testing.assert_allclose(second[JointIndex.RIGHT_ARM], [3.0] * 7)
    assert second[JointIndex.LEFT_GRIPPER] == pytest.approx(0.3)
    assert second[JointIndex.RIGHT_GRIPPER] == pytest.approx(0.4)

    _feed_gripper_state(50, 60)
    with pytest.raises(StateError, match="timed out"):
        transport.get_joint_state(timeout=0.0)


def test_legacy_joint_state_still_copies_each_queued_state(transport):
    _feed_joint_state([1.0] * 16, timestamp=5000)
    _feed_gripper_state(10, 20, timestamp=5001)
    _feed_joint_state([2.0] * 16, timestamp=5002)
    _feed_gripper_state(30, 40, timestamp=5003)

    first = transport.get_joint_state(timeout=0.1)
    second = transport.get_joint_state(timeout=0.1)

    np.testing.assert_allclose(first["states"][JointIndex.LEFT_ARM], [1.0] * 7)
    assert first["states"][JointIndex.LEFT_GRIPPER] == pytest.approx(0.1)
    np.testing.assert_allclose(second["states"][JointIndex.LEFT_ARM], [2.0] * 7)
    assert second["states"][JointIndex.LEFT_GRIPPER] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "bad_q",
    [
        [1.0] * 15,
        [1.0] * 15 + [np.nan],
    ],
)
def test_invalid_arm_head_snapshot_does_not_replace_or_advance_legacy(
    transport,
    bad_q,
):
    _feed_joint_state([1.0] * 16, timestamp=6000)
    _feed_gripper_state(10, 20, timestamp=6001)
    transport.get_joint_state(timeout=0.1)

    _feed_joint_state(bad_q, timestamp=6002)
    _feed_gripper_state(30, 40, timestamp=6003)

    snapshot = transport.get_arm_head_state(timeout=0.0)
    assert snapshot["timestamp"] == 6000
    assert snapshot["states"] == [1.0] * 16
    with pytest.raises(StateError, match="timed out"):
        transport.get_joint_state(timeout=0.0)

    _feed_joint_state([2.0] * 16, timestamp=6004)
    state = transport.get_joint_state(timeout=0.1)["states"]
    np.testing.assert_allclose(state[JointIndex.LEFT_ARM], [2.0] * 7)
    assert state[JointIndex.LEFT_GRIPPER] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "left,right",
    [
        (np.nan, 50.0),
        (25.0, np.inf),
    ],
)
def test_invalid_gripper_snapshot_does_not_replace_or_advance_legacy(
    transport,
    left,
    right,
):
    _feed_joint_state([1.0] * 16, timestamp=7000)
    _feed_gripper_state(10, 20, timestamp=7001)
    transport.get_joint_state(timeout=0.1)

    _feed_gripper_state(left, right, timestamp=7002)
    _feed_joint_state([2.0] * 16, timestamp=7003)

    snapshot = transport.get_gripper_state(timeout=0.0)
    assert snapshot["timestamp"] == 7001
    assert snapshot["states"] == pytest.approx([0.1, 0.2])
    with pytest.raises(StateError, match="timed out"):
        transport.get_joint_state(timeout=0.0)

    _feed_gripper_state(30, 40, timestamp=7004)
    state = transport.get_joint_state(timeout=0.1)["states"]
    np.testing.assert_allclose(state[JointIndex.LEFT_ARM], [2.0] * 7)
    assert state[JointIndex.LEFT_GRIPPER] == pytest.approx(0.3)


def test_snapshot_times_out_without_a_value():
    snapshot = _new_snapshot(width=2, name="head")

    with pytest.raises(StateError, match="head state timed out"):
        snapshot.get(timeout=0.01)


def test_snapshot_clear_discards_value_and_allows_waiting_for_a_new_sample():
    snapshot = _new_snapshot(width=2, name="head")
    snapshot.update([1.0, 2.0], robot_timestamp=10)

    snapshot.clear()
    with pytest.raises(StateError, match="head state timed out"):
        snapshot.get(timeout=0.0)

    results = []
    reader = threading.Thread(
        target=lambda: results.append(snapshot.get(timeout=0.5)),
        daemon=True,
    )
    reader.start()
    snapshot.update([3.0, 4.0], robot_timestamp=11)
    reader.join(timeout=0.5)

    assert not reader.is_alive()
    assert len(results) == 1
    np.testing.assert_allclose(results[0].values, [3.0, 4.0])
    assert results[0].robot_timestamp == 11


def test_snapshot_predicate_waiter_captures_match_before_latest_is_overwritten():
    snapshot = _new_snapshot(width=2, name="pose")
    snapshot.update([0.0, 0.0], metadata={"guid": "old"})
    cached_value_checked = threading.Event()
    results = []
    errors = []

    def matches_a(item):
        cached_value_checked.set()
        return item.metadata.get("guid") == "A"

    def read_a():
        try:
            results.append(snapshot.get(timeout=0.5, predicate=matches_a))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    reader = threading.Thread(target=read_a, daemon=True)
    reader.start()
    assert cached_value_checked.wait(timeout=0.5)
    with snapshot._condition:
        assert len(snapshot._predicate_waiters) == 1
        snapshot.update([1.0, 1.0], metadata={"guid": "A"})
        snapshot.update([2.0, 2.0], metadata={"guid": "B"})
        assert reader.is_alive()

    reader.join(timeout=0.5)

    assert not reader.is_alive()
    assert errors == []
    assert results[0].metadata["guid"] == "A"
    np.testing.assert_allclose(results[0].values, [1.0, 1.0])
    assert snapshot.get(timeout=0.0).metadata["guid"] == "B"


def test_snapshot_predicate_timeout_removes_waiter():
    snapshot = _new_snapshot(width=2, name="pose")

    with pytest.raises(StateError, match="pose state timed out"):
        snapshot.get(
            timeout=0.0,
            predicate=lambda item: item.metadata.get("guid") == "missing",
        )

    assert snapshot._predicate_waiters == []
    snapshot.update([1.0, 2.0], metadata={"guid": "other"})
    assert snapshot._predicate_waiters == []


def test_snapshot_rejects_oversized_timeout_before_empty_wait():
    snapshot = _new_snapshot(width=2, name="pose")

    with pytest.raises(ValueError, match="threading.TIMEOUT_MAX"):
        snapshot.get(timeout=threading.TIMEOUT_MAX + 1.0)

    assert snapshot._predicate_waiters == []


def test_snapshot_rejects_oversized_timeout_before_predicate_registration():
    snapshot = _new_snapshot(width=2, name="pose")
    snapshot.update([1.0, 2.0], metadata={"guid": "old"})

    with pytest.raises(ValueError, match="threading.TIMEOUT_MAX"):
        snapshot.get(
            timeout=threading.TIMEOUT_MAX + 1.0,
            predicate=lambda item: item.metadata.get("guid") == "new",
        )

    assert snapshot._predicate_waiters == []


def test_snapshot_shared_predicate_waiter_cleanup_uses_waiter_identity(monkeypatch):
    import tron2_env.transport.snapshot as snapshot_module

    snapshot = _new_snapshot(width=2, name="pose")
    snapshot.update([0.0, 0.0], metadata={"guid": "old"})
    registered = {
        "snapshot-waiter-one": threading.Event(),
        "snapshot-waiter-two": threading.Event(),
    }
    clocks = {
        "snapshot-waiter-one": 0.0,
        "snapshot-waiter-two": 0.0,
    }
    results = []
    errors = {}

    def monotonic():
        return clocks.get(threading.current_thread().name, 0.0)

    def shared_predicate(item):
        event = registered.get(threading.current_thread().name)
        if event is not None:
            event.set()
        return item.metadata.get("guid") == "target"

    def read_target():
        try:
            results.append(snapshot.get(timeout=1.0, predicate=shared_predicate))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors[threading.current_thread().name] = exc

    monkeypatch.setattr(snapshot_module.time, "monotonic", monotonic)
    first = threading.Thread(
        target=read_target,
        name="snapshot-waiter-one",
        daemon=True,
    )
    second = threading.Thread(
        target=read_target,
        name="snapshot-waiter-two",
        daemon=True,
    )
    first.start()
    assert registered[first.name].wait(timeout=0.5)
    second.start()
    assert registered[second.name].wait(timeout=0.5)

    with snapshot._condition:
        assert len(snapshot._predicate_waiters) == 2
        clocks[second.name] = 2.0
        snapshot._condition.notify_all()
    second.join(timeout=0.5)

    assert not second.is_alive()
    assert isinstance(errors[second.name], StateError)
    with snapshot._condition:
        assert len(snapshot._predicate_waiters) == 1
        snapshot.update([3.0, 4.0], metadata={"guid": "target"})
    first.join(timeout=0.5)

    assert not first.is_alive()
    assert first.name not in errors
    assert results[0].metadata["guid"] == "target"
    assert snapshot._predicate_waiters == []


def test_snapshot_rejects_stale_data():
    snapshot = _new_snapshot(width=2, name="head")
    snapshot.update([1.0, 2.0], robot_timestamp=10, received_at=1.0)

    with pytest.raises(StateError, match="stale"):
        snapshot.get(max_age=0.5, now=2.0)


def test_snapshot_rejects_invalid_updates_without_overwriting_last_value():
    snapshot = _new_snapshot(width=2, name="gripper")
    snapshot.update([0.25, 0.75], robot_timestamp=10, received_at=1.0)

    for invalid in ([1.0], [[1.0, 2.0]], [1.0, np.nan], [1.0, np.inf]):
        with pytest.raises(ValueError, match="finite shape"):
            snapshot.update(invalid, robot_timestamp=11, received_at=2.0)

    item = snapshot.get(now=2.0)
    np.testing.assert_allclose(item.values, [0.25, 0.75])
    assert item.robot_timestamp == 10


def test_snapshot_defensively_copies_values_and_nested_metadata():
    snapshot = _new_snapshot(width=2, name="head")
    values = np.array([1.0, 2.0])
    metadata = {"diagnostics": {"names": ["pitch", "yaw"]}}

    snapshot.update(
        values,
        robot_timestamp=10,
        received_at=1.0,
        metadata=metadata,
    )
    values[0] = 99.0
    metadata["diagnostics"]["names"][0] = "mutated"

    first = snapshot.get(now=1.0)
    np.testing.assert_allclose(first.values, [1.0, 2.0])
    assert first.metadata == {"diagnostics": {"names": ["pitch", "yaw"]}}
    with pytest.raises(FrozenInstanceError):
        first.received_at = 2.0

    first.values[0] = -99.0
    first.metadata["diagnostics"]["names"][0] = "also-mutated"
    second = snapshot.get(now=1.0)
    np.testing.assert_allclose(second.values, [1.0, 2.0])
    assert second.metadata == {"diagnostics": {"names": ["pitch", "yaw"]}}


@pytest.mark.parametrize(
    "method,kwargs,match",
    [
        ("get", {"timeout": -0.1}, "timeout"),
        ("get", {"timeout": np.nan}, "timeout"),
        ("get", {"max_age": -0.1}, "max_age"),
        ("get", {"max_age": np.nan}, "max_age"),
        ("get", {"now": np.inf}, "now"),
        ("update", {"values": [1.0, 2.0], "received_at": np.nan}, "received_at"),
    ],
)
def test_snapshot_validates_time_arguments(method, kwargs, match):
    snapshot = _new_snapshot(width=2, name="head")

    with pytest.raises(ValueError, match=match):
        getattr(snapshot, method)(**kwargs)


@pytest.mark.parametrize("argument", ["timeout", "max_age"])
def test_snapshot_rejects_infinite_time_bounds(argument):
    snapshot = _new_snapshot(width=2, name="head")
    snapshot.update([1.0, 2.0], received_at=1.0)

    with pytest.raises(ValueError, match=argument):
        snapshot.get(**{argument: np.inf}, now=1.0)


def test_snapshot_waiters_wake_when_first_value_arrives():
    snapshot = _new_snapshot(width=2, name="head")
    barrier = threading.Barrier(3)
    results = []
    errors = []

    def read_snapshot():
        barrier.wait()
        try:
            results.append(snapshot.get(timeout=0.5))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    readers = [threading.Thread(target=read_snapshot) for _ in range(2)]
    for reader in readers:
        reader.start()
    barrier.wait()
    time.sleep(0.02)
    assert all(reader.is_alive() for reader in readers)

    snapshot.update([1.0, 2.0], robot_timestamp=10)
    for reader in readers:
        reader.join(timeout=0.5)

    assert not errors
    assert len(results) == 2
    assert all(not reader.is_alive() for reader in readers)
    for result in results:
        np.testing.assert_allclose(result.values, [1.0, 2.0])


def test_snapshot_capability_protocols_are_optional(transport):
    from tron2_env.transport.base import (
        ArmHeadStateTransport,
        GripperStateTransport,
        RobotTransport,
    )

    assert isinstance(transport, ArmHeadStateTransport)
    assert isinstance(transport, GripperStateTransport)
    assert "get_arm_head_state" not in RobotTransport.__dict__
    assert "get_gripper_state" not in RobotTransport.__dict__


def test_get_head_position_returns_latest(transport):
    joint_q = list(np.zeros(16))
    joint_q[14] = 1.0
    joint_q[15] = -0.5
    _ws_app().feed_message({
        "title": "response_get_joint_state",
        "timestamp": 1,
        "data": {"q": joint_q},
    })
    # No need to commit; get_head_position reads from the snapshot directly.
    np.testing.assert_allclose(transport.get_head_position(), [1.0, -0.5])


def test_set_gripper_emits_message(transport):
    transport.set_gripper(left_opening=33, right_opening=66)
    sent = [json.loads(m) for m in _ws_app().sent if json.loads(m).get("title") == "request_set_limx_2fclaw_cmd"]
    assert sent
    payload = sent[-1]["data"]
    assert payload["left_opening"] == 33
    assert payload["right_opening"] == 66


def test_stream_lifter_position_uses_dedicated_channel_with_default_and_override_velocity(
    chassis_transport,
):
    assert len(_FakeApp.instances) == 2

    chassis_transport.stream_lifter_position(150.04)
    chassis_transport.stream_lifter_position(-12.34, velocity_mm_s=125)
    chassis_transport.stream_lifter_position(612.75)

    sent = _messages_from(_lifter_ws_app(), "request_lifter_stream_position")
    assert [message["data"] for message in sent] == [
        {"position": 150.0, "velocity": 80.0},
        {"position": -12.3, "velocity": 125.0},
        {"position": 612.8, "velocity": 80.0},
    ]
    assert not _messages_from(_main_ws_app(), "request_lifter_stream_position")


def test_stream_lifter_position_sends_consecutive_frames_without_ack(chassis_transport):
    chassis_transport.stream_lifter_position(100.0)
    chassis_transport.stream_lifter_position(120.0)

    sent = _messages_from(_lifter_ws_app(), "request_lifter_stream_position")
    assert [message["data"]["position"] for message in sent] == [100.0, 120.0]


def test_blocked_lifter_stream_send_does_not_block_main_lifter_feedback(
    chassis_transport,
    monkeypatch,
):
    chassis_transport.lifter_state_source = "position_mm"
    lifter = _lifter_ws_app()
    original_send = lifter.send
    send_started = threading.Event()
    release_send = threading.Event()
    feedback_finished = threading.Event()

    def blocking_send(message):
        send_started.set()
        assert release_send.wait(timeout=1.0)
        original_send(message)

    monkeypatch.setattr(lifter, "send", blocking_send)
    streamer = threading.Thread(
        target=chassis_transport.stream_lifter_position,
        args=(120.0,),
        daemon=True,
    )
    feedback = threading.Thread(
        target=lambda: (
            chassis_transport._request_lifter_feedback(),
            feedback_finished.set(),
        ),
        daemon=True,
    )
    try:
        streamer.start()
        assert send_started.wait(timeout=0.5)

        feedback.start()
        assert feedback_finished.wait(timeout=0.2)
        assert _messages_from(_main_ws_app(), "request_get_lifter_position")
    finally:
        release_send.set()
        streamer.join(timeout=1.0)
        feedback.join(timeout=1.0)


def test_lifter_stream_rechecks_generation_after_waiting_for_dedicated_channel(
    chassis_transport,
    monkeypatch,
):
    main = _main_ws_app()
    lifter = _lifter_ws_app()
    original_lock = chassis_transport._lifter_connection_state_lock
    lifter_lock_requested = threading.Event()
    release_lifter_lock = threading.Event()
    stream_errors = []
    old_generation = chassis_transport._session_generation

    class GatedLifterLock:
        def __enter__(self):
            lifter_lock_requested.set()
            assert release_lifter_lock.wait(timeout=1.0)
            original_lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            original_lock.release()

    monkeypatch.setattr(
        chassis_transport,
        "_lifter_connection_state_lock",
        GatedLifterLock(),
    )

    def send_old_generation_frame():
        try:
            with chassis_transport.bind_command_generation(old_generation):
                chassis_transport.stream_lifter_position(120.0)
        except BaseException as exc:  # pragma: no cover - asserted below
            stream_errors.append(exc)

    streamer = threading.Thread(target=send_old_generation_frame, daemon=True)
    try:
        streamer.start()
        assert lifter_lock_requested.wait(timeout=0.5)

        chassis_transport._on_open(main)
        main.feed_message({
            "title": "notify_robot_info",
            "accid": "RECONNECTED_ROBOT",
            "data": {},
        })
        assert chassis_transport._session_generation == old_generation + 1
        assert chassis_transport._session_ready.is_set()

        release_lifter_lock.set()
        streamer.join(timeout=1.0)

        assert not streamer.is_alive()
        assert len(stream_errors) == 1
        assert isinstance(stream_errors[0], CommandError)
        assert not _messages_from(lifter, "request_lifter_stream_position")
    finally:
        release_lifter_lock.set()
        streamer.join(timeout=1.0)


def test_stream_lifter_failure_ack_does_not_block_later_frames(chassis_transport, caplog):
    chassis_transport.stream_lifter_position(100.0)
    first = _messages_from(_lifter_ws_app(), "request_lifter_stream_position")[0]
    _lifter_ws_app().feed_message({
        "title": "response_lifter_stream_position",
        "guid": first["guid"],
        "data": {"result": "failed"},
    })

    chassis_transport.stream_lifter_position(120.0)

    assert len(_messages_from(_lifter_ws_app(), "request_lifter_stream_position")) == 2
    assert first["guid"] in caplog.text
    assert "{'result': 'failed'}" in caplog.text
    assert "lifter stream command failed" in caplog.text


def test_successful_lifter_stream_ack_is_not_logged(chassis_transport, caplog):
    caplog.set_level(logging.DEBUG, logger="tron2_env.transport.websocket")
    chassis_transport.stream_lifter_position(100.0)
    first = _messages_from(_lifter_ws_app(), "request_lifter_stream_position")[0]

    _lifter_ws_app().feed_message({
        "title": "response_lifter_stream_position",
        "guid": first["guid"],
        "data": {"result": "success"},
    })

    assert "lifter stream command acknowledged" not in caplog.text


def test_stop_lifter_stream_sends_zero_velocity(chassis_transport):
    chassis_transport.stop_lifter_stream()

    sent = _messages_from(_lifter_ws_app(), "request_lifter_stream_velocity")
    assert sent[-1]["data"] == {"velocity": 0.0}


@pytest.mark.parametrize(
    ("position", "velocity"),
    [
        (np.nan, 80.0),
        (np.inf, 80.0),
        (100.0, 0.0),
        (100.0, -1.0),
        (100.0, np.nan),
        (100.0, np.inf),
    ],
)
def test_stream_lifter_position_rejects_invalid_position_or_velocity(
    chassis_transport,
    position,
    velocity,
):
    with pytest.raises(CommandError):
        chassis_transport.stream_lifter_position(position, velocity)

    assert not _messages_from(_lifter_ws_app(), "request_lifter_stream_position")


def test_closed_dedicated_lifter_channel_raises_without_main_fallback(chassis_transport):
    _lifter_ws_app().close()

    with pytest.raises(CommandError, match="lifter stream position command send failed"):
        chassis_transport.stream_lifter_position(150.0)

    assert not _messages_from(_main_ws_app(), "request_lifter_stream_position")


def test_stream_lifter_position_requires_dedicated_channel(transport):
    with pytest.raises(CommandError, match="lifter stream position command send failed"):
        transport.stream_lifter_position(150.0)

    assert not _sent_messages("request_lifter_stream_position")


def test_disconnect_stops_chassis_then_closes_main_and_lifter_connections(
    chassis_transport,
    monkeypatch,
):
    main = _main_ws_app()
    lifter = _lifter_ws_app()
    chassis_transport.should_exit = True
    chassis_transport._feedback_wakeup.set()
    polling_thread = chassis_transport._polling_thread
    if polling_thread is not None:
        polling_thread.join(timeout=1.0)
    main.sent.clear()

    stop_sent = threading.Event()
    disconnect_errors = []
    original_send = main.send

    def capture(callable_, errors):
        try:
            callable_()
        except BaseException as exc:
            errors.append(exc)

    def recording_send(message):
        original_send(message)
        payload = json.loads(message)
        if (
            payload.get("title") == "request_chassis_move"
            and payload.get("data") == {"x": 0.0, "yaw": 0.0}
        ):
            stop_sent.set()

    monkeypatch.setattr(main, "send", recording_send)
    disconnector = threading.Thread(
        target=lambda: capture(
            chassis_transport.disconnect,
            disconnect_errors,
        ),
        daemon=True,
    )
    disconnector.start()

    assert stop_sent.wait(timeout=0.5)
    disconnector.join(timeout=1.0)

    stop_messages = _messages_from(main, "request_chassis_move")
    assert [message["data"] for message in stop_messages] == [
        {"x": 0.0, "yaw": 0.0}
    ]
    assert main.closed
    assert lifter.closed
    assert not chassis_transport.connected
    assert not chassis_transport._lifter_command_connected
    assert not disconnector.is_alive()
    assert disconnect_errors == []


def test_disconnect_closes_socket_to_abort_send_holding_connection_lock(
    chassis_transport,
    monkeypatch,
):
    main = _main_ws_app()
    chassis_transport.should_exit = True
    chassis_transport._feedback_wakeup.set()
    polling_thread = chassis_transport._polling_thread
    if polling_thread is not None:
        polling_thread.join(timeout=1.0)

    send_entered = threading.Event()
    release_send = threading.Event()
    abort_entered = threading.Event()
    close_entered = threading.Event()
    send_errors = []
    disconnect_errors = []
    original_close = main.close

    class AbortableSocket:
        def abort(self):
            abort_entered.set()
            release_send.set()

    abort_socket = AbortableSocket()
    with chassis_transport._abort_socket_lock:
        chassis_transport._main_abort_socket = (
            chassis_transport._session_generation,
            abort_socket,
        )
    main.sock = None

    def blocking_send(_message):
        send_entered.set()
        assert release_send.wait(timeout=1.0)
        raise OSError("socket closed during send")

    def recording_close():
        assert abort_entered.is_set()
        close_entered.set()
        original_close()

    def capture(callable_, errors):
        try:
            callable_()
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(main, "send", blocking_send)
    monkeypatch.setattr(main, "close", recording_close)
    sender = threading.Thread(
        target=lambda: capture(
            lambda: chassis_transport.send_joint_cmd(np.zeros(JointIndex.SERVOJ_DIM)),
            send_errors,
        ),
        daemon=True,
    )
    disconnector = threading.Thread(
        target=lambda: capture(chassis_transport.disconnect, disconnect_errors),
        daemon=True,
    )

    sender.start()
    assert send_entered.wait(timeout=0.5)
    disconnector.start()

    try:
        assert close_entered.wait(timeout=0.5)
    finally:
        release_send.set()
    sender.join(timeout=1.0)
    disconnector.join(timeout=1.0)

    assert not sender.is_alive()
    assert not disconnector.is_alive()
    assert len(send_errors) == 1
    assert isinstance(send_errors[0], CommandError)
    assert disconnect_errors == []


def test_abort_websocket_client_selects_no_argument_shutdown_signature(transport):
    calls = []

    class NoArgumentSocket:
        def shutdown(self):
            calls.append(())

    transport._abort_websocket_client(
        types.SimpleNamespace(sock=NoArgumentSocket()),
        None,
    )

    assert calls == [()]


def test_abort_websocket_client_does_not_retry_internal_shutdown_type_error(
    transport,
):
    calls = []

    class FailingSocket:
        def shutdown(self, how):
            calls.append((how,))
            raise TypeError("internal shutdown failure")

    transport._abort_websocket_client(
        types.SimpleNamespace(sock=FailingSocket()),
        None,
    )

    assert calls == [(socket.SHUT_RDWR,)]


def test_disconnect_raises_when_transport_thread_does_not_stop(transport):
    polling_thread = transport._polling_thread
    transport.should_exit = True
    transport._feedback_wakeup.set()
    if polling_thread is not None:
        polling_thread.join(timeout=1.0)

    class StuckThread:
        def __init__(self):
            self.join_timeouts = []

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)

        def is_alive(self):
            return True

    stuck_thread = StuckThread()
    transport._polling_thread = stuck_thread
    try:
        with pytest.raises(RuntimeError, match="polling thread did not stop"):
            transport.disconnect()
        assert stuck_thread.join_timeouts == [2.0]
    finally:
        transport._polling_thread = None


@pytest.mark.parametrize(
    ("timeout", "expected_error"),
    _INVALID_PUBLIC_WAIT_TIMEOUTS,
)
def test_get_chassis_lifter_state_rejects_invalid_timeout(
    chassis_transport,
    timeout,
    expected_error,
):
    with pytest.raises(expected_error, match="timeout"):
        chassis_transport.get_chassis_lifter_state(timeout=timeout)


def test_chassis_lifter_feedback_and_stream_commands_use_current_protocol(chassis_transport):
    _main_ws_app().feed_message({
        "title": "response_chassis_state",
        "timestamp": 100,
        "data": {"result": "success", "timestamp": 101, "data": [0.25, -0.5, 0.0]},
    })
    _main_ws_app().feed_message({
        "title": "response_lifter_state",
        "timestamp": 200,
        "data": {"result": "success", "timestamp": 201, "q": [-254.4], "v": [0.0]},
    })

    state = chassis_transport.get_chassis_lifter_state(timeout=0.1)
    assert {
        key: state[key]
        for key in (
            "linear_velocity",
            "angular_velocity",
            "lifter_state",
            "chassis_timestamp",
            "lifter_timestamp",
        )
    } == {
        "linear_velocity": 0.25,
        "angular_velocity": -0.5,
        "lifter_state": -254.4,
        "chassis_timestamp": 101,
        "lifter_timestamp": 201,
    }
    assert isinstance(state["chassis_received_at"], float)
    assert isinstance(state["lifter_received_at"], float)

    chassis_transport.command_chassis(0.4, -0.6)
    chassis_transport.stream_lifter_position(230.44, velocity_mm_s=120.0)
    chassis = _messages_from(_main_ws_app(), "request_chassis_move")[-1]
    lifter = _messages_from(_lifter_ws_app(), "request_lifter_stream_position")[-1]
    assert chassis["data"] == {"x": 0.4, "yaw": -0.6}
    assert lifter["data"] == {"position": 230.4, "velocity": 120.0}


@pytest.mark.parametrize(
    ("linear_x", "angular_z"),
    [(np.nan, 0.0), (0.0, np.inf), (-np.inf, 0.0)],
)
def test_chassis_command_rejects_nonfinite_values_before_send(
    chassis_transport,
    linear_x,
    angular_z,
):
    with pytest.raises(CommandError, match="finite"):
        chassis_transport.command_chassis(linear_x, angular_z)

    assert _messages_from(_main_ws_app(), "request_chassis_move") == []


def test_chassis_and_lifter_freshness_are_tracked_independently(
    chassis_transport,
    monkeypatch,
):
    clock = [10.0]
    monkeypatch.setattr(_ws_mod.time, "monotonic", lambda: clock[0])

    chassis_transport._handle_chassis_state({
        "timestamp": 100,
        "data": {"result": "success", "timestamp": 101, "data": [0.25, -0.5]},
    })
    clock[0] = 10.25
    chassis_transport._handle_lifter_raw_state({
        "timestamp": 200,
        "data": {"result": "success", "timestamp": 201, "q": [-254.4]},
    })

    state = chassis_transport.get_chassis_lifter_state(timeout=0.1)
    assert state["chassis_received_at"] == pytest.approx(10.0)
    assert state["lifter_received_at"] == pytest.approx(10.25)

    clock[0] = 10.6
    with pytest.raises(StateError, match="chassis.*stale"):
        chassis_transport.get_chassis_lifter_state(timeout=0.1, max_age=0.5)

    clock[0] = 10.7
    chassis_transport._handle_chassis_state({
        "timestamp": 300,
        "data": {"result": "success", "timestamp": 301, "data": [0.3, -0.4]},
    })
    clock[0] = 10.8
    with pytest.raises(StateError, match="lifter.*stale"):
        chassis_transport.get_chassis_lifter_state(timeout=0.1, max_age=0.5)


def test_lifter_position_feedback_above_nominal_range_is_preserved(transport):
    _ws_app().feed_message({
        "title": "response_get_lifter_position",
        "guid": "feedback-guid",
        "data": {"result": "success", "position": 612.75, "timestamp": 123},
    })

    assert transport.lifter_states["value"] == pytest.approx(612.75)


def test_lifter_feedback_allows_only_one_request_in_flight(transport):
    transport.lifter_state_source = "position_mm"

    transport._request_lifter_feedback()
    transport._request_lifter_feedback()

    sent = _sent_messages("request_get_lifter_position")
    assert len(sent) == 1

    _ws_app().feed_message({
        "title": "response_get_lifter_position",
        "guid": sent[0]["guid"],
        "data": {"result": "success", "position": 200.0, "timestamp": 123},
    })
    transport._request_lifter_feedback()

    assert len(_sent_messages("request_get_lifter_position")) == 2


def test_lifter_feedback_timeout_replaces_lost_query_and_ignores_late_response(
    transport,
):
    clock = [10.0]
    transport.lifter_state_source = "position_mm"
    transport._feedback_clock = lambda: clock[0]

    transport._request_lifter_feedback()
    first = _sent_messages("request_get_lifter_position")[0]

    clock[0] = 11.01
    transport._request_lifter_feedback()
    sent = _sent_messages("request_get_lifter_position")
    assert len(sent) == 2
    second = sent[1]
    assert second["guid"] != first["guid"]
    assert transport._lifter_feedback_in_flight == second["guid"]

    _ws_app().feed_message({
        "title": "response_get_lifter_position",
        "guid": first["guid"],
        "data": {"result": "success", "position": 100.0},
    })
    assert transport._lifter_feedback_in_flight == second["guid"]

    _ws_app().feed_message({
        "title": "response_get_lifter_position",
        "guid": second["guid"],
        "data": {"result": "success", "position": 120.0},
    })
    assert transport._lifter_feedback_in_flight is None
    assert transport._lifter_feedback_deadline is None


def test_main_connection_reset_clears_lifter_feedback_deadline(transport):
    with transport._lifter_feedback_lock:
        transport._lifter_feedback_in_flight = "lost-guid"
        transport._lifter_feedback_deadline = 123.0

    transport._reset_main_connection_state()

    assert transport._lifter_feedback_in_flight is None
    assert transport._lifter_feedback_deadline is None


def test_lifter_feedback_and_stream_command_run_in_parallel(chassis_transport):
    chassis_transport.lifter_state_source = "position_mm"
    chassis_transport._request_lifter_feedback()

    feedback = _messages_from(_main_ws_app(), "request_get_lifter_position")
    assert len(feedback) == 1

    chassis_transport.stream_lifter_position(120.0)

    assert _messages_from(_lifter_ws_app(), "request_lifter_stream_position")
    assert chassis_transport._lifter_feedback_in_flight == feedback[0]["guid"]


def test_lifter_feedback_guid_gate_allows_next_query_only_after_matching_response(
    chassis_transport,
):
    chassis_transport.lifter_state_source = "position_mm"
    chassis_transport._request_lifter_feedback()
    first = _messages_from(_main_ws_app(), "request_get_lifter_position")[0]

    chassis_transport._request_lifter_feedback()
    _main_ws_app().feed_message({
        "title": "response_get_lifter_position",
        "guid": "unrelated",
        "data": {"result": "success", "position": 612.75},
    })
    chassis_transport._request_lifter_feedback()
    assert len(_messages_from(_main_ws_app(), "request_get_lifter_position")) == 1

    _main_ws_app().feed_message({
        "title": "response_get_lifter_position",
        "guid": first["guid"],
        "data": {"result": "success", "position": 612.75},
    })
    chassis_transport._request_lifter_feedback()

    assert len(_messages_from(_main_ws_app(), "request_get_lifter_position")) == 2
    assert chassis_transport.lifter_states["value"] == pytest.approx(612.75)


@pytest.mark.parametrize(
    ("timeout", "expected_error"),
    _INVALID_PUBLIC_WAIT_TIMEOUTS,
)
def test_get_ee_poses_rejects_invalid_timeout_before_send_or_queue_clear(
    transport,
    timeout,
    expected_error,
):
    queued = {"sentinel": True}
    with transport._queue_lock:
        transport.ee_pose_queue.clear()
        transport.ee_pose_queue.append(queued)
    sent_before = list(_sent_messages("request_get_move_pose"))

    with pytest.raises(expected_error, match="timeout"):
        transport.get_ee_poses(timeout=timeout)

    assert _sent_messages("request_get_move_pose") == sent_before
    with transport._queue_lock:
        assert list(transport.ee_pose_queue) == [queued]


def test_get_ee_poses_requests_and_returns_latest_pose(transport):
    expected_pose = {
        "timestamp": 1001,
        "left_position": [0.1, 0.2, 0.3],
        "left_quat": [1.0, 0.0, 0.0, 0.0],
        "right_position": [0.4, 0.5, 0.6],
        "right_quat": [1.0, 0.0, 0.0, 0.0],
        "result": "success",
    }

    def feed_pose_after_request():
        deadline = time.time() + 0.5
        while time.time() < deadline:
            sent = [json.loads(m) for m in _ws_app().sent]
            if any(m.get("title") == "request_get_move_pose" for m in sent):
                _ws_app().feed_message({
                    "title": "response_get_move_pose",
                    "timestamp": 999,
                    "data": expected_pose,
                })
                return
            time.sleep(0.001)

    feeder = threading.Thread(target=feed_pose_after_request, daemon=True)
    feeder.start()

    pose = transport.get_ee_poses(timeout=0.5)

    feeder.join(timeout=0.5)
    assert len(_sent_messages("request_get_move_pose")) == 1
    assert not _sent_messages("request_get_servo_pose")
    assert pose["timestamp"] == 1001
    assert pose["left_position"] == expected_pose["left_position"]
    assert pose["right_position"] == expected_pose["right_position"]


def test_get_ee_poses_rejects_fail_not_data_response(transport):
    def feed_failure_after_request():
        deadline = time.time() + 0.5
        while time.time() < deadline:
            sent = [json.loads(m) for m in _ws_app().sent]
            if any(m.get("title") == "request_get_move_pose" for m in sent):
                _ws_app().feed_message({
                    "title": "response_get_move_pose",
                    "timestamp": 999,
                    "data": {
                        "timestamp": 1001,
                        "left_position": [0.0, 0.0, 0.0],
                        "left_quat": [1.0, 0.0, 0.0, 0.0],
                        "right_position": [0.0, 0.0, 0.0],
                        "right_quat": [1.0, 0.0, 0.0, 0.0],
                        "result": "fail_not_data",
                    },
                })
                return
            time.sleep(0.001)

    feeder = threading.Thread(target=feed_failure_after_request, daemon=True)
    feeder.start()

    with pytest.raises(StateError, match="fail_not_data"):
        transport.get_ee_poses(timeout=0.5)

    feeder.join(timeout=0.5)


def _direct_init_transport(
    config: Tron2Config,
    *,
    open_gripper: bool = False,
) -> WebsocketTransport:
    transport = WebsocketTransport.__new__(WebsocketTransport)
    transport.config = config
    transport._second_joint = [0.1] * JointIndex.MOVEJ_DIM
    transport.logger = _ws_mod.logging.getLogger("test-init-pose")
    transport.legacy_init_permissions = LegacyInitPermissions(
        limx_gripper_open=open_gripper,
    )
    transport._connection_state_lock = threading.RLock()
    transport._session_ready = threading.Event()
    transport._session_ready.set()
    transport._session_generation = 1
    transport.connected = True
    return transport


def test_move_to_init_pose_requests_arm_head_then_moves_joints_before_head(
    monkeypatch,
):
    init_joints = [0.2] * JointIndex.MOVEJ_DIM
    init_head = [0.1, -0.1]
    t = _direct_init_transport(
        Tron2Config(
            init_joints=init_joints,
            init_head=init_head,
            init_ee_z_min=None,
        )
    )

    events = []

    def request_joint_state(
        title,
        data=None,
        *,
        guid=None,
        expected_generation=None,
    ):
        events.append((title, guid, expected_generation))
        return True

    def read_arm_head(*, timeout=1.0, expected_guid=None):
        events.append(("read_arm_head", timeout, expected_guid))
        return {"states": [0.5] * JointIndex.SERVOJ_DIM}

    def forbidden_legacy_read(*args, **kwargs):
        raise AssertionError("initialization must not read the legacy 18D queue")

    def forbidden_gripper(*args, **kwargs):
        raise AssertionError("gripper initialization is disabled")

    t._send_request = request_joint_state
    t._new_guid = lambda: "init-joint-guid"
    t.get_arm_head_state = read_arm_head
    t.get_joint_state = forbidden_legacy_read
    t.movej = lambda joints, move_time=2.0: events.append(
        ("movej", list(joints), move_time)
    )
    t.move_head = lambda joints, move_time=1.0: events.append(
        ("moveh", list(joints), move_time)
    )
    t.set_gripper = forbidden_gripper
    monkeypatch.setattr(
        _ws_mod.time,
        "sleep",
        lambda seconds: events.append(("sleep", seconds)),
    )

    t._move_to_init_pose()

    assert events == [
        ("request_get_joint_state", "init-joint-guid", 1),
        ("read_arm_head", 1.0, "init-joint-guid"),
        ("movej", init_joints, 2.0),
        ("sleep", 2.0),
        ("moveh", init_head, 1.0),
        ("sleep", 1.0),
    ]


def test_move_to_init_pose_raises_when_joint_state_request_fails():
    t = _direct_init_transport(
        Tron2Config(
            init_joints=[0.2] * JointIndex.MOVEJ_DIM,
            init_ee_z_min=None,
        )
    )
    t._send_request = lambda *args, **kwargs: False
    t.get_arm_head_state = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("state must not be read after request failure")
    )

    with pytest.raises(StateError, match="joint state request failed"):
        t._move_to_init_pose()


def test_move_to_init_pose_waits_for_snapshot_matching_one_shot_request():
    init_joints = [0.2] * JointIndex.MOVEJ_DIM
    second_joint = [0.1] * JointIndex.MOVEJ_DIM
    t = _direct_init_transport(
        Tron2Config(init_joints=init_joints, init_ee_z_min=None)
    )
    t._second_joint = second_joint
    t._arm_head_snapshot = _new_snapshot(
        JointIndex.SERVOJ_DIM,
        "arm/head",
    )
    t._arm_head_snapshot.update(
        [0.0] * JointIndex.SERVOJ_DIM,
        metadata={"guid": "old-poll-guid"},
    )

    request = {}
    feeder = None

    def send_request(
        title,
        data=None,
        *,
        guid=None,
        expected_generation=None,
    ):
        nonlocal feeder
        request.update(
            title=title,
            guid=guid,
            expected_generation=expected_generation,
        )

        def feed_matching_snapshot():
            time.sleep(0.02)
            t._arm_head_snapshot.update(
                [0.5] * JointIndex.SERVOJ_DIM,
                metadata={"guid": guid},
            )

        feeder = threading.Thread(target=feed_matching_snapshot, daemon=True)
        feeder.start()
        return True

    movej_calls = []
    t._new_guid = lambda: "fresh-init-guid"
    t._send_request = send_request
    t.movej = lambda joints, move_time=2.0: movej_calls.append(list(joints))
    t._sleep_during_init = lambda generation, duration: None

    t._move_to_init_pose()

    assert feeder is not None
    feeder.join(timeout=0.5)
    assert not feeder.is_alive()
    assert request == {
        "title": "request_get_joint_state",
        "guid": "fresh-init-guid",
        "expected_generation": 1,
    }
    assert movej_calls == [init_joints]


def test_move_to_init_pose_uses_servoj_order_for_second_joint_predicate(
    monkeypatch,
):
    init_joints = [0.2] * JointIndex.MOVEJ_DIM
    second_joint = [0.1] * JointIndex.MOVEJ_DIM
    t = _direct_init_transport(
        Tron2Config(init_joints=init_joints, init_ee_z_min=None)
    )
    t._second_joint = second_joint

    arm_head_state = [0.5] * JointIndex.SERVOJ_DIM
    for index in (0, 3, 7, 10):
        arm_head_state[index] = 0.0
    movej_calls = []
    t._send_request = lambda *args, **kwargs: True
    t.get_arm_head_state = lambda timeout=1.0, expected_guid=None: {
        "states": arm_head_state
    }
    t.get_joint_state = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("initialization must not read the legacy 18D queue")
    )
    t.movej = lambda joints, move_time=2.0: movej_calls.append(list(joints))
    monkeypatch.setattr(_ws_mod.time, "sleep", lambda _seconds: None)

    t._move_to_init_pose()

    assert movej_calls == [second_joint, init_joints]


def test_move_to_init_pose_uses_second_joint_when_ee_z_below_threshold(monkeypatch):
    init_joints = [0.2] * JointIndex.MOVEJ_DIM
    second_joint = [0.1] * JointIndex.MOVEJ_DIM
    t = _direct_init_transport(
        Tron2Config(init_joints=init_joints, init_ee_z_min=0.25),
        open_gripper=True,
    )
    t._second_joint = second_joint

    safe_joint_state = [0.5] * JointIndex.SERVOJ_DIM
    movej_calls = []
    gripper_calls = []

    t._send_request = lambda *args, **kwargs: True
    t.get_arm_head_state = lambda timeout=1.0, expected_guid=None: {
        "states": safe_joint_state
    }
    t.get_ee_poses = lambda timeout=1.0, expected_generation=None: {
        "left_position": [0.0, 0.0, 0.2],
        "right_position": [0.0, 0.0, 0.4],
    }
    t.movej = lambda joints, move_time=2.0: movej_calls.append(list(joints))
    t.set_gripper = lambda left_opening=0.0, right_opening=0.0: gripper_calls.append(
        (left_opening, right_opening)
    )
    monkeypatch.setattr(_ws_mod.time, "sleep", lambda _seconds: None)

    t._move_to_init_pose()

    assert movej_calls == [second_joint, init_joints]
    assert gripper_calls == [(100, 100)]


def test_move_to_init_pose_stops_after_session_changes_during_sleep(monkeypatch):
    t = _direct_init_transport(
        Tron2Config(
            init_joints=[0.2] * JointIndex.MOVEJ_DIM,
            init_head=[0.1, -0.1],
            init_ee_z_min=None,
        ),
        open_gripper=True,
    )

    sends = []
    t._send_request = lambda *args, **kwargs: True
    t.get_arm_head_state = lambda timeout=1.0, expected_guid=None: {
        "states": [0.5] * JointIndex.SERVOJ_DIM,
    }
    t.movej = lambda joints, move_time=2.0: sends.append("movej")
    t.move_head = lambda joints, move_time=1.0: sends.append("head")
    t.set_gripper = lambda **kwargs: sends.append("gripper")

    def reconnect_during_sleep(_seconds):
        t._session_generation += 1

    monkeypatch.setattr(_ws_mod.time, "sleep", reconnect_during_sleep)

    with pytest.raises(StateError, match="session changed during initialization"):
        t._move_to_init_pose()

    assert sends == ["movej"]


def test_disconnect_marks_disconnected(transport):
    transport.disconnect()
    assert not transport.is_connected()
