"""Hardware-free tests for Tron2Env policy command ownership."""

from __future__ import annotations

import inspect
import logging
import sys
import threading
import time
import types

import numpy as np
import pytest

import tron2_env.bridge as bridge_module
import tron2_env.env as env_module
import tron2_env.motion.controller as controller_module
from tron2_env.end_effector import GripperProvider
from tron2_env.env import EnvConfig, Tron2Env
from tron2_env.errors import CommandError, StateError
from tron2_env.layout import BrainCo2Config, RobotModules
from tron2_env.motion import MotionController


def test_env_config_preserves_legacy_positional_parameter_prefix():
    legacy_fields = (
        "robot_config",
        "camera_config",
        "control_backend",
        "publish_rate",
        "fps",
        "time_sync_tolerance",
        "time_sync_max_retries",
        "legacy_use_time_sync",
        "init_gripper_opening",
        "raw_config",
        "observation_source",
        "state_dim",
        "state_layout",
        "auxiliary_polling_rate",
        "lifter_state_source",
        "lifter_control_enabled",
        "lifter_command_interval",
        "lifter_command_epsilon_mm",
        "lifter_command_duration_ms",
        "bridge_state_source",
        "bridge_config",
    )
    new_fields = (
        "modules",
        "brainco2_config",
        "state_max_age",
        "head_command_interval",
        "head_move_duration",
        "head_ack_timeout",
        "lifter_stream_velocity_mm_s",
    )
    assert tuple(inspect.signature(EnvConfig).parameters) == legacy_fields + new_fields

    bridge_config = env_module.BridgeConfig()
    config = EnvConfig(
        env_module.Tron2Config(),
        env_module.CameraConfig(),
        "websocket",
        300.0,
        30.0,
        0.01,
        3,
        True,
        0.9,
        {},
        "legacy",
        16,
        None,
        20.0,
        "raw_q",
        False,
        0.1,
        None,
        None,
        "bridge",
        bridge_config,
    )

    assert config.bridge_config is bridge_config
    assert config.modules is None
    assert not hasattr(config, "lifter_command_epsilon_mm")
    assert not hasattr(config, "lifter_command_duration_ms")


@pytest.mark.parametrize("interval", [0.0, 0.299])
def test_lifter_command_interval_accepts_values_shorter_than_watchdog(interval):
    assert EnvConfig(lifter_command_interval=interval).lifter_command_interval == interval


@pytest.mark.parametrize("interval", [-0.1, 0.3, 0.5, np.nan, np.inf])
def test_lifter_command_interval_rejects_values_outside_watchdog_window(interval):
    with pytest.raises(
        ValueError,
        match="lifter_command_interval.*shorter than the default 300 ms watchdog",
    ):
        EnvConfig(lifter_command_interval=interval)


@pytest.mark.parametrize(
    ("legacy_name", "legacy_value"),
    [
        ("lifter_command_epsilon_mm", 0.1),
        ("lifter_command_duration_ms", 0),
    ],
)
def test_removed_lifter_command_options_are_rejected(legacy_name, legacy_value):
    with pytest.raises(
        ValueError,
        match=(
            f"{legacy_name} was removed; lifter streaming uses "
            "lifter_command_interval and lifter_stream_velocity_mm_s"
        ),
    ):
        EnvConfig(**{legacy_name: legacy_value})


class FakeMotionController:
    def __init__(self, close_events=None, disconnect_error=None):
        self.commands = []
        self.grippers = []
        self.arm_commands = []
        self.end_effector_commands = []
        self.command_order = []
        self.fault_checks = 0
        self.fault = None
        self.preflight_calls = []
        self.paused = False
        self.resumed_state = None
        self.gripper_entered = None
        self.gripper_release = None
        self.chassis_commands = []
        self.lifter_commands = []
        self.lifter_stream_stops = 0
        self.disconnected = False
        self.close_events = close_events
        self.disconnect_error = disconnect_error
        self.arm_state = np.arange(14, dtype=np.float64)
        self.arm_metadata = {"timestamp": 101, "guid": "arm-guid"}
        self.end_effector_state = np.array([0.25, 0.75], dtype=np.float64)
        self.end_effector_metadata = {"timestamp": 102, "guid": "effector-guid"}
        self.head_state = np.array([0.1, -0.1], dtype=np.float64)
        self.head_metadata = {"timestamp": 103, "guid": "head-guid"}
        self.arm_state_reads = []
        self.end_effector_state_reads = []
        self.head_state_reads = []
        self.auxiliary_state_reads = []
        self.auxiliary_state = {
            "linear_velocity": 0.2,
            "angular_velocity": -0.3,
            "lifter_state": -254.4,
            "chassis_timestamp": 10,
            "lifter_timestamp": 11,
        }

    def get_head_position(self):
        return np.array([0.1, -0.1])

    def get_joint_states(self, timeout=0.5):
        return {"states": np.zeros(18, dtype=np.float32), "timestamp": 1000}

    def command_joints(self, target):
        self.commands.append(np.asarray(target).copy())

    def command_arm(self, left, right, head=None):
        left = np.asarray(left).copy()
        right = np.asarray(right).copy()
        head_copy = None if head is None else np.asarray(head).copy()
        self.arm_commands.append((left, right, head_copy))
        self.command_order.append("arm")
        held_head = self.get_head_position() if head_copy is None else head_copy
        self.commands.append(np.concatenate((left, right, held_head)))

    def command_end_effector(self, left, right):
        left = np.asarray(left).copy()
        right = np.asarray(right).copy()
        self.end_effector_commands.append((left, right))
        self.command_order.append("end_effector")
        if left.shape == (1,) and right.shape == (1,):
            self.set_gripper(float(left[0]) * 100.0, float(right[0]) * 100.0)

    def preflight_arm(self, left, right, head=None):
        self.preflight_calls.append("arm")
        for name, value, width in (
            ("left arm", left, 7),
            ("right arm", right, 7),
            ("head", head, 2),
        ):
            if value is None:
                continue
            vector = np.asarray(value, dtype=np.float64)
            if vector.shape != (width,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be finite shape ({width},)")

    def preflight_end_effector(self, left, right):
        self.preflight_calls.append("end_effector")
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.shape != right.shape or left.ndim != 1:
            raise ValueError("end-effector targets must have matching vector shapes")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("end-effector targets must be finite")

    def preflight_chassis(self, linear_x, angular_z):
        self.preflight_calls.append("chassis")
        if not np.all(np.isfinite([linear_x, angular_z])):
            raise ValueError("chassis command values must be finite")

    def preflight_lifter_stream_position(self, position_mm, velocity_mm_s=80.0):
        self.preflight_calls.append("lifter")
        if (
            not np.isfinite(float(position_mm))
            or not np.isfinite(float(velocity_mm_s))
            or float(velocity_mm_s) <= 0.0
        ):
            raise ValueError("invalid lifter command")

    def raise_if_faulted(self):
        self.fault_checks += 1
        self.command_order.append("fault")
        if self.fault is not None:
            raise self.fault

    def set_gripper(self, left_opening, right_opening):
        self.grippers.append((left_opening, right_opening))
        if self.gripper_entered is not None:
            self.gripper_entered.set()
        if self.gripper_release is not None:
            self.gripper_release.wait(timeout=1.0)

    def pause(self, timeout=1.0):
        self.paused = True

    def resume_from_state(self, state18=None):
        self.resumed_state = (
            None if state18 is None else np.asarray(state18).copy()
        )
        self.paused = False

    def command_chassis(self, linear_x, angular_z):
        self.command_order.append("chassis")
        self.chassis_commands.append((linear_x, angular_z))

    def stop_chassis(self):
        self.chassis_commands.append((0.0, 0.0))

    def stream_lifter_position(self, position_mm, velocity_mm_s=80.0):
        self.command_order.append("lifter")
        self.lifter_commands.append((position_mm, velocity_mm_s))

    def stop_lifter_stream(self):
        self.lifter_stream_stops += 1

    def disconnect(self):
        self.disconnected = True
        if self.close_events is not None:
            self.close_events.append("robot")
        if self.disconnect_error is not None:
            raise self.disconnect_error

    def get_chassis_lifter_state(self, timeout=1.0, max_age=None):
        self.auxiliary_state_reads.append((timeout, max_age))
        return dict(self.auxiliary_state)

    def get_arm_state(self, timeout=1.0, max_age=None):
        self.arm_state_reads.append((timeout, max_age))
        return self.arm_state.copy(), dict(self.arm_metadata)

    def get_end_effector_state(self, timeout=1.0, max_age=None):
        self.end_effector_state_reads.append((timeout, max_age))
        return self.end_effector_state.copy(), dict(self.end_effector_metadata)

    def get_head_state(self, timeout=1.0, max_age=None):
        self.head_state_reads.append((timeout, max_age))
        return self.head_state.copy(), dict(self.head_metadata)


class FakeBridgeObservationProvider:
    def __init__(
        self,
        obs=None,
        close_events=None,
        stop_error=None,
        start_error=None,
        get_obs_error=None,
    ):
        self.obs = obs
        self.close_events = close_events
        self.stop_error = stop_error
        self.start_error = start_error
        self.get_obs_error = get_obs_error
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        if self.start_error is not None:
            raise self.start_error

    def get_obs(self, timeout=1.0):
        if self.get_obs_error is not None:
            raise self.get_obs_error
        return self.obs

    def stop(self):
        self.stopped = True
        if self.close_events is not None:
            self.close_events.append("bridge")
        if self.stop_error is not None:
            raise self.stop_error


class FakeCameraManager:
    def __init__(self, close_events=None, stop_error=None):
        self.close_events = close_events
        self.stop_error = stop_error
        self.started = False
        self.stopped = False

    def start_capture(self):
        self.started = True

    def stop_capture(self):
        self.stopped = True
        if self.close_events is not None:
            self.close_events.append("camera")
        if self.stop_error is not None:
            raise self.stop_error


class RecordingDebugWriter:
    def __init__(self, debug_dir=None, close_events=None, close_error=None):
        self.debug_dir = debug_dir
        self.close_events = close_events
        self.close_error = close_error
        self.submissions = []
        self.close_timeouts = []
        self.closed = False

    def submit(self, images):
        self.submissions.append(dict(images))

    def close(self, timeout=0.2):
        self.close_timeouts.append(timeout)
        self.closed = True
        if self.close_events is not None:
            self.close_events.append("writer")
        if self.close_error is not None:
            raise self.close_error


class RecordingEndEffectorProvider:
    name = "brainco2"
    width_per_side = 6

    def __init__(self):
        self.commands = []
        self.reads = []
        self.pause_count = 0
        self.resume_count = 0
        self.close_count = 0
        self.fault_checks = 0
        self.state = np.arange(12, dtype=np.float64)
        self.metadata = {"robot_timestamp": 202}

    def read_state(self, timeout=1.0, max_age=None):
        self.reads.append((timeout, max_age))
        return self.state.copy(), dict(self.metadata)

    def command(self, left, right):
        self.commands.append((np.asarray(left).copy(), np.asarray(right).copy()))

    def validate_command(self, left, right):
        return np.asarray(left).copy(), np.asarray(right).copy()

    def pause(self):
        self.pause_count += 1

    def resume(self):
        self.resume_count += 1

    def close(self):
        self.close_count += 1

    def raise_if_faulted(self):
        self.fault_checks += 1


class RecordingControllerStrategy:
    mode = "servoj"
    target_dim = 16

    def __init__(self):
        self.resets = []
        self.destinations = []

    def read_measured_target(self, timeout):
        return np.arange(16, dtype=np.float64)

    def reset(self, measured):
        self.resets.append(np.asarray(measured).copy())

    def set_destination(self, target, eta):
        self.destinations.append((np.asarray(target).copy(), eta))

    def publish_current(self):
        return None


class RecordingControllerTransport:
    def __init__(self):
        self.arm_head_reads = []
        self.gripper_commands = []
        self.disconnected = False

    def get_arm_head_state(self, timeout=1.0, max_age=None):
        self.arm_head_reads.append((timeout, max_age))
        return {
            "states": np.arange(16, dtype=np.float64),
            "timestamp": 101,
            "guid": "arm-guid",
        }

    def get_head_position(self):
        return np.array([14.0, 15.0])

    def set_gripper(self, left, right):
        self.gripper_commands.append((left, right))

    def disconnect(self):
        self.disconnected = True


def make_env(config=None):
    env = Tron2Env.__new__(Tron2Env)
    env.logger = logging.getLogger("test-tron2-env")
    env.config = config or EnvConfig()
    env.layout = env.config.physical_layout
    env.robot = FakeMotionController()
    env.camera_manager = None
    env.bridge_provider = None
    env._debug_image_writer = None
    env._closed = False
    env._pause_complete = False
    env.last_action = None
    env._control_lock = threading.RLock()
    env._control_enabled = True
    env._blocked_step_count = 0
    env._last_lifter_command = None
    env._last_lifter_command_time = float("-inf")
    env.init_joints = env.config.robot_config.init_joints
    return env


def test_explicit_servop_brainco_modules_build_31d_layout():
    modules = RobotModules(
        arm="servop",
        end_effector="brainco2",
        head=True,
        mobile_base=True,
    )

    config = EnvConfig(
        modules=modules,
        brainco2_config=BrainCo2Config(command_time=[2.0] * 6),
        bridge_state_source="legacy",
        lifter_state_source="position_mm",
    )

    assert config.modules is modules
    assert config.state_dim == 31
    assert config.physical_layout.components == (
        "left_arm",
        "left_hand",
        "right_arm",
        "right_hand",
        "head",
        "chassis",
    )


@pytest.mark.parametrize("state_dim", [16.0, True, np.bool_(True)])
def test_modular_state_dim_rejects_non_integer_even_when_value_matches(state_dim):
    with pytest.raises(ValueError, match="state_dim must be an integer"):
        EnvConfig(
            modules=RobotModules(),
            state_dim=state_dim,
        )


def test_explicit_modules_accept_matching_legacy_layout_fields():
    modules = RobotModules(arm="servop", end_effector="brainco2", head=True)
    expected = (
        "left_arm",
        "left_hand",
        "right_arm",
        "right_hand",
        "head",
    )

    config = EnvConfig(
        modules=modules,
        brainco2_config=BrainCo2Config(command_time=2.0),
        state_layout=list(expected),
        state_dim=28,
    )

    assert config.physical_layout.components == expected
    assert config.physical_layout.field_names[3:7] == (
        "left_arm_qw",
        "left_arm_qx",
        "left_arm_qy",
        "left_arm_qz",
    )


@pytest.mark.parametrize(
    "legacy_fields",
    [
        {"state_dim": 26},
        {
            "state_layout": [
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            ]
        },
    ],
)
def test_explicit_modules_reject_mismatched_legacy_layout_fields(legacy_fields):
    with pytest.raises(ValueError, match="modules"):
        EnvConfig(
            modules=RobotModules(
                arm="servop",
                end_effector="brainco2",
                head=True,
            ),
            brainco2_config=BrainCo2Config(command_time=2.0),
            **legacy_fields,
        )


@pytest.mark.parametrize(
    "modules",
    [
        RobotModules(arm="servop"),
        RobotModules(end_effector="brainco2"),
        RobotModules(mobile_base=True),
    ],
)
def test_explicit_modules_reject_unsupported_lowlevel_combinations(modules):
    kwargs = {"modules": modules, "control_backend": "lowlevel"}
    if modules.end_effector == "brainco2":
        kwargs["brainco2_config"] = BrainCo2Config(command_time=2.0)
    with pytest.raises(ValueError, match="lowlevel"):
        EnvConfig(**kwargs)


def test_public_package_rejects_lowlevel_configuration():
    with pytest.raises(ValueError, match="websocket"):
        EnvConfig(modules=RobotModules(), control_backend="lowlevel")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"modules": RobotModules(end_effector="brainco2")},
        {
            "modules": RobotModules(),
            "brainco2_config": BrainCo2Config(command_time=2.0),
        },
    ],
)
def test_brainco_module_and_configuration_must_be_selected_together(kwargs):
    with pytest.raises(ValueError, match="BrainCo2"):
        EnvConfig(**kwargs)


@pytest.mark.parametrize(
    "modules",
    [
        RobotModules(arm="servop"),
        RobotModules(end_effector="brainco2"),
    ],
)
def test_bridge_noncanonical_modular_state_requires_legacy_signaling_source(modules):
    kwargs = {
        "modules": modules,
        "observation_source": "bridge",
        "bridge_state_source": "bridge",
    }
    if modules.end_effector == "brainco2":
        kwargs["brainco2_config"] = BrainCo2Config(command_time=2.0)

    with pytest.raises(ValueError, match="bridge_state_source='legacy'"):
        EnvConfig(**kwargs)


@pytest.mark.parametrize(
    "modules",
    [
        RobotModules(),
        RobotModules(head=True),
        RobotModules(mobile_base=True),
        RobotModules(head=True, mobile_base=True),
    ],
)
def test_bridge_accepts_explicit_servoj_gripper_modules(modules):
    config = EnvConfig(
        modules=modules,
        observation_source="bridge",
        bridge_state_source="bridge",
    )

    assert config.modules is modules


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("state_max_age", 0.0),
        ("state_max_age", np.nan),
        ("head_command_interval", 0.0),
        ("head_move_duration", -1.0),
        ("head_ack_timeout", np.inf),
        ("publish_rate", 0.0),
        ("fps", -1.0),
        ("auxiliary_polling_rate", np.nan),
    ],
)
def test_modular_freshness_and_head_timing_must_be_positive(name, value):
    with pytest.raises(ValueError, match=name):
        EnvConfig(**{name: value})


def test_constructor_forwards_complete_modular_configuration(monkeypatch):
    robot = FakeMotionController()
    calls = []

    def create_controller(robot_config, **kwargs):
        calls.append((robot_config, kwargs))
        return robot

    monkeypatch.setattr(env_module, "create_motion_controller", create_controller)
    monkeypatch.setattr(Tron2Env, "_init_camera", lambda self: FakeCameraManager())
    brainco2_config = BrainCo2Config(
        command_time=[1, 2, 3, 4, 5, 6],
        polling_rate=17.0,
        state_timeout=0.41,
    )
    modules = RobotModules(
        arm="servop",
        end_effector="brainco2",
        head=True,
        mobile_base=True,
    )
    config = EnvConfig(
        modules=modules,
        brainco2_config=brainco2_config,
        state_max_age=0.23,
        head_command_interval=0.07,
        head_move_duration=0.11,
        head_ack_timeout=0.37,
        lifter_state_source="position_mm",
    )
    config.camera_config.save_debug_images = False

    env = Tron2Env(config)
    try:
        assert len(calls) == 1
        assert calls[0][0] is config.robot_config
        assert calls[0][1]["modules"] is modules
        assert calls[0][1]["brainco2_config"] is brainco2_config
        assert calls[0][1]["state_max_age"] == pytest.approx(0.23)
        assert calls[0][1]["head_command_interval"] == pytest.approx(0.07)
        assert calls[0][1]["head_move_duration"] == pytest.approx(0.11)
        assert calls[0][1]["head_ack_timeout"] == pytest.approx(0.37)
    finally:
        env.close()


def test_controller_owns_one_end_effector_provider_and_exposes_modular_state_api():
    transport = RecordingControllerTransport()
    strategy = RecordingControllerStrategy()
    provider = RecordingEndEffectorProvider()
    controller = MotionController(
        transport,
        arm_strategy=strategy,
        end_effector_provider=provider,
        state_max_age=0.4,
        publish_rate=200.0,
    )

    arms, arm_metadata = controller.get_arm_state(timeout=0.3, max_age=0.2)
    head, head_metadata = controller.get_head_state(timeout=0.3, max_age=0.2)
    hands, hand_metadata = controller.get_end_effector_state(
        timeout=0.3,
        max_age=0.2,
    )
    controller.start()
    controller.raise_if_faulted()
    controller.command_end_effector(np.arange(6), np.arange(6, 12))

    np.testing.assert_array_equal(arms, np.arange(14))
    np.testing.assert_array_equal(head, [14.0, 15.0])
    np.testing.assert_array_equal(hands, np.arange(12))
    assert arm_metadata == {"timestamp": 101, "guid": "arm-guid"}
    assert head_metadata == arm_metadata
    assert hand_metadata == {"robot_timestamp": 202}
    assert transport.arm_head_reads == [(0.3, 0.2), (0.3, 0.2)]
    assert provider.reads == [(0.3, 0.2), (2.0, 0.4)]
    assert provider.fault_checks == 1
    assert len(provider.commands) == 1

    controller.pause(timeout=0.5)
    assert provider.pause_count == 1
    controller.resume_from_state(np.arange(18, dtype=np.float64))
    assert provider.resume_count == 1
    controller.disconnect()
    controller.disconnect()
    assert provider.close_count == 1
    assert transport.disconnected


def test_legacy_controller_clips_normalized_gripper_commands():
    transport = RecordingControllerTransport()
    controller = MotionController(
        transport,
        arm_strategy=RecordingControllerStrategy(),
    )

    controller.start()
    try:
        controller.command_end_effector(np.array([-0.25]), np.array([1.25]))
        assert transport.gripper_commands == [(0.0, 100.0)]
    finally:
        controller.disconnect()


def test_factory_preserves_existing_head_timing_positional_arguments(monkeypatch):
    transport = RecordingControllerTransport()
    transport.send_joint_cmd = lambda _target: None
    monkeypatch.setattr(
        controller_module,
        "WebsocketTransport",
        lambda config, **kwargs: transport,
    )

    controller = controller_module.create_motion_controller(
        controller_module.Tron2Config(),
        "websocket",
        200.0,
        0.05,
        None,
        False,
        20.0,
        "raw_q",
        None,
        0.02,
        0.03,
        0.5,
    )
    controller.disconnect()


def test_factory_constructs_selected_provider_with_complete_brainco_config(
    monkeypatch,
):
    transport = RecordingControllerTransport()
    transport.sent = []
    transport.send_joint_cmd = lambda value: transport.sent.append(
        np.asarray(value).copy()
    )
    provider = RecordingEndEffectorProvider()
    provider_calls = []

    monkeypatch.setattr(
        controller_module,
        "WebsocketTransport",
        lambda config, **kwargs: transport,
    )

    def make_provider(name, selected_transport, config):
        provider_calls.append((name, selected_transport, config))
        return provider

    monkeypatch.setattr(
        controller_module,
        "create_end_effector_provider",
        make_provider,
    )
    brainco2_config = BrainCo2Config(
        command_time=[1, 2, 3, 4, 5, 6],
        polling_rate=17.0,
        state_timeout=0.41,
    )

    controller = controller_module.create_motion_controller(
        controller_module.Tron2Config(),
        modules=RobotModules(end_effector="brainco2"),
        brainco2_config=brainco2_config,
        state_max_age=0.23,
        publish_rate=200.0,
    )
    try:
        assert provider_calls == [("brainco2", transport, brainco2_config)]
        assert provider.reads == [(2.0, 0.23)]
    finally:
        controller.disconnect()


def test_gripper_provider_adapts_legacy_combined_state_transport():
    class LegacyCombinedTransport:
        def __init__(self):
            self.commands = []
            self.received_at = time.monotonic()

        def get_joint_state(self, timeout=1.0):
            state = np.arange(18, dtype=np.float64)
            state[7] = 0.25
            state[15] = 0.75
            return {
                "states": state,
                "timestamp": 101,
                "received_at": self.received_at,
            }

        def set_gripper(self, left, right):
            self.commands.append((left, right))

    transport = LegacyCombinedTransport()
    provider = GripperProvider(transport)

    state, metadata = provider.read_state(timeout=0.3, max_age=0.2)
    provider.command([0.1], [0.9])

    np.testing.assert_array_equal(state, [0.25, 0.75])
    assert metadata == {"timestamp": 101, "received_at": transport.received_at}
    assert transport.commands == [(10.0, 90.0)]

    transport.received_at = time.monotonic() - 1.0
    with pytest.raises(StateError, match="legacy joint state is stale"):
        provider.read_state(timeout=0.3, max_age=0.2)


def _make_modular_env(*, mobile_base=True):
    config = EnvConfig(
        modules=RobotModules(
            arm="servop",
            end_effector="brainco2",
            head=True,
            mobile_base=mobile_base,
        ),
        brainco2_config=BrainCo2Config(command_time=2.0),
        state_max_age=0.23,
        lifter_state_source="position_mm",
        lifter_control_enabled=mobile_base,
        lifter_command_interval=0.0,
    )
    env = make_env(config)
    expected = np.arange(config.state_dim, dtype=np.float64)
    env.robot.arm_state = np.concatenate((expected[0:7], expected[13:20]))
    env.robot.end_effector_state = np.concatenate((expected[7:13], expected[20:26]))
    env.robot.end_effector_metadata = {
        "robot_timestamp": 202,
        "received_at": 12.5,
        "left_names": [f"L{index}" for index in range(6)],
        "right_names": [f"R{index}" for index in range(6)],
    }
    env.robot.head_state = expected[26:28]
    if mobile_base:
        env.robot.auxiliary_state = {
            "linear_velocity": expected[28],
            "angular_velocity": expected[29],
            "lifter_state": expected[30],
            "chassis_timestamp": 303,
            "lifter_timestamp": 304,
            "chassis_received_at": 13.0,
            "lifter_received_at": 13.1,
        }
    return env, expected


def test_servop_brainco_head_action_routes_by_component_in_policy_order():
    env, action = _make_modular_env()

    assert env.step(action)

    np.testing.assert_array_equal(env.robot.arm_commands[-1][0], action[0:7])
    np.testing.assert_array_equal(env.robot.end_effector_commands[-1][0], action[7:13])
    np.testing.assert_array_equal(env.robot.arm_commands[-1][1], action[13:20])
    np.testing.assert_array_equal(env.robot.end_effector_commands[-1][1], action[20:26])
    np.testing.assert_array_equal(env.robot.arm_commands[-1][2], action[26:28])
    assert env.robot.chassis_commands[-1] == (28.0, 29.0)
    assert env.robot.lifter_commands[-1] == (30.0, 80.0)
    assert env.robot.command_order == [
        "fault",
        "arm",
        "end_effector",
        "chassis",
        "lifter",
    ]
    action[:] = -1.0
    np.testing.assert_array_equal(env.last_action, np.arange(31, dtype=np.float64))


def test_step_clamps_gripper_actions_without_mutating_model_output():
    env = make_env()
    action = np.zeros(16, dtype=np.float64)
    action[7] = -0.2
    action[15] = 1.2
    original = action.copy()

    assert env.step(action)

    np.testing.assert_array_equal(action, original)
    np.testing.assert_array_equal(env.robot.end_effector_commands[-1][0], [0.0])
    np.testing.assert_array_equal(env.robot.end_effector_commands[-1][1], [1.0])
    assert env.robot.grippers[-1] == (0.0, 100.0)
    expected_action = original.copy()
    expected_action[[7, 15]] = [0.0, 1.0]
    np.testing.assert_array_equal(env.last_action, expected_action)


@pytest.mark.parametrize("shape", [(30,), (32,), (31, 1)])
def test_step_rejects_nonexact_action_shape_before_robot_calls(shape):
    env, _ = _make_modular_env()
    previous_action = np.full(31, -1.0)
    env.last_action = previous_action.copy()

    with pytest.raises(ValueError, match=r"shape \(31,\)"):
        env.step(np.zeros(shape, dtype=np.float64))

    assert env.robot.fault_checks == 0
    assert env.robot.command_order == []
    assert env.robot.arm_commands == []
    assert env.robot.end_effector_commands == []
    assert env.robot.chassis_commands == []
    assert env.robot.lifter_commands == []
    np.testing.assert_array_equal(env.last_action, previous_action)


@pytest.mark.parametrize(
    ("index", "value"),
    [
        (0, np.nan),
        (7, np.inf),
        (26, -np.inf),
        (28, np.nan),
        (30, np.inf),
    ],
)
def test_step_rejects_nonfinite_action_before_any_robot_call(index, value):
    env, action = _make_modular_env()
    previous_action = np.full(31, -1.0)
    env.last_action = previous_action.copy()
    action[index] = value

    with pytest.raises(ValueError, match="finite"):
        env.step(action)

    assert env.robot.fault_checks == 0
    assert env.robot.command_order == []
    assert env.robot.arm_commands == []
    assert env.robot.end_effector_commands == []
    assert env.robot.chassis_commands == []
    assert env.robot.lifter_commands == []
    np.testing.assert_array_equal(env.last_action, previous_action)


def test_28d_servop_brainco_head_action_routes_without_mobile_commands():
    env, action = _make_modular_env(mobile_base=False)

    assert action.shape == (28,)
    assert env.step(action)

    np.testing.assert_array_equal(env.robot.arm_commands[-1][0], action[0:7])
    np.testing.assert_array_equal(env.robot.end_effector_commands[-1][0], action[7:13])
    np.testing.assert_array_equal(env.robot.arm_commands[-1][1], action[13:20])
    np.testing.assert_array_equal(env.robot.end_effector_commands[-1][1], action[20:26])
    np.testing.assert_array_equal(env.robot.arm_commands[-1][2], action[26:28])
    assert env.robot.chassis_commands == []
    assert env.robot.lifter_commands == []
    assert env.robot.command_order == ["fault", "arm", "end_effector"]


def test_async_fault_prevents_every_modular_command():
    env, action = _make_modular_env()
    env.robot.fault = RuntimeError("asynchronous hand fault")

    with pytest.raises(RuntimeError, match="asynchronous hand fault"):
        env.step(action)

    assert env.robot.command_order == ["fault"]
    assert env.robot.arm_commands == []
    assert env.robot.end_effector_commands == []
    assert env.robot.chassis_commands == []
    assert env.robot.lifter_commands == []


def test_future_provider_preflight_rejection_prevents_arm_and_head_dispatch():
    env, action = _make_modular_env(mobile_base=False)

    def reject_future_provider(left, right):
        env.robot.preflight_calls.append("end_effector")
        raise CommandError("future provider rejected command")

    env.robot.preflight_end_effector = reject_future_provider

    with pytest.raises(CommandError, match="future provider rejected command"):
        env.step(action)

    assert env.robot.preflight_calls == ["arm", "end_effector"]
    assert env.robot.command_order == ["fault"]
    assert env.robot.arm_commands == []
    assert env.robot.end_effector_commands == []


def test_modular_state_uses_pose_hand_head_and_mobile_snapshots():
    env, expected = _make_modular_env()

    state, metadata = env._compose_policy_state()

    np.testing.assert_array_equal(state, expected.astype(np.float32))
    assert env.robot.arm_state_reads == [(0.23, 0.23)]
    assert env.robot.end_effector_state_reads == [(0.23, 0.23)]
    assert env.robot.head_state_reads == [(0.23, 0.23)]
    assert env.robot.auxiliary_state_reads == [(0.23, 0.23)]
    assert metadata["arm_control_mode"] == "servop"
    assert metadata["end_effector"] == "brainco2"
    assert metadata["modules"] == {
        "arm": "servop",
        "end_effector": "brainco2",
        "head": True,
        "mobile_base": True,
    }
    assert metadata["field_names"] == env.layout.field_names
    assert metadata["arm_timestamp_ms"] == 101
    assert metadata["end_effector_robot_timestamp"] == 202
    assert metadata["head_timestamp_ms"] == 103
    assert metadata["left_hand_names"] == [f"L{index}" for index in range(6)]
    assert metadata["right_hand_names"] == [f"R{index}" for index in range(6)]
    assert metadata["chassis_timestamp_ms"] == 303
    assert metadata["lifter_timestamp_ms"] == 304
    assert metadata["chassis_received_at"] == 13.0
    assert metadata["lifter_received_at"] == 13.1


def test_bridge_modular_observation_uses_component_snapshots_not_legacy_joint_state():
    env, expected = _make_modular_env()
    images = {"cam_high": np.zeros((2, 2, 3), dtype=np.uint8)}
    env.config.observation_source = "bridge"
    env.config.bridge_state_source = "legacy"
    env.bridge_provider = FakeBridgeObservationProvider(
        {
            "state": np.full(18, -99.0, dtype=np.float32),
            "images": images,
            "metadata": {
                "joint_timestamp_ms": 401,
                "gripper_timestamp_ms": 402,
            },
        }
    )

    def reject_legacy_state(*args, **kwargs):
        raise AssertionError("explicit modules must not read combined legacy state")

    env.robot.get_joint_states = reject_legacy_state

    obs = env.get_obs()

    np.testing.assert_array_equal(obs["state"], expected.astype(np.float32))
    assert obs["images"] is images
    assert obs["metadata"]["state_source"] == "legacy"
    assert obs["metadata"]["bridge_joint_timestamp_ms"] == 401
    assert obs["metadata"]["bridge_gripper_timestamp_ms"] == 402


def test_bridge_servoj_gripper_modules_pack_canonical_state_and_mobile_feedback():
    config = EnvConfig(
        modules=RobotModules(
            arm="servoj",
            end_effector="gripper",
            head=True,
            mobile_base=True,
        ),
        observation_source="bridge",
        bridge_state_source="bridge",
    )
    env = make_env(config)
    canonical = np.arange(18, dtype=np.float32)
    images = {"cam_high": np.zeros((2, 2, 3), dtype=np.uint8)}
    env.robot.auxiliary_state = {
        "linear_velocity": 18.0,
        "angular_velocity": 19.0,
        "lifter_state": 20.0,
        "chassis_timestamp": 303,
        "lifter_timestamp": 304,
        "chassis_received_at": 13.0,
        "lifter_received_at": 13.1,
    }
    env.bridge_provider = FakeBridgeObservationProvider(
        {
            "state": canonical,
            "images": images,
            "metadata": {
                "joint_timestamp_ms": 401,
                "gripper_timestamp_ms": 402,
            },
        }
    )

    obs = env.get_obs()

    np.testing.assert_array_equal(obs["state"], np.arange(21, dtype=np.float32))
    assert env.robot.arm_state_reads == []
    assert env.robot.end_effector_state_reads == []
    assert env.robot.head_state_reads == []
    assert env.robot.auxiliary_state_reads == [(0.5, 0.5)]
    assert obs["metadata"]["state_source"] == "bridge"
    assert obs["metadata"]["joint_timestamp_ms"] == 401
    assert obs["metadata"]["gripper_timestamp_ms"] == 402
    assert obs["metadata"]["arm_control_mode"] == "servoj"
    assert obs["metadata"]["end_effector"] == "gripper"
    assert obs["metadata"]["modules"] == {
        "arm": "servoj",
        "end_effector": "gripper",
        "head": True,
        "mobile_base": True,
    }
    assert obs["metadata"]["field_names"] == env.layout.field_names
    assert obs["metadata"]["chassis_timestamp_ms"] == 303
    assert obs["metadata"]["lifter_timestamp_ms"] == 304
    assert obs["metadata"]["chassis_received_at"] == 13.0
    assert obs["metadata"]["lifter_received_at"] == 13.1
    assert obs["metadata"]["mobile_metadata"] == env.robot.auxiliary_state


def test_brainco_reset_reads_state_without_sending_guessed_hand_command():
    env, _ = _make_modular_env(mobile_base=False)
    env.config.observation_source = "bridge"
    env.config.bridge_state_source = "legacy"
    env.get_obs = lambda: {
        "state": env._compose_policy_state()[0],
        "images": {},
        "metadata": {},
    }

    env.reset()

    assert env.robot.end_effector_state_reads == [(0.23, 0.23)]
    assert env.robot.end_effector_commands == []


def test_pause_and_close_block_every_modular_command():
    env, action = _make_modular_env()

    env.pause_control()
    after_pause = {
        "arm": len(env.robot.arm_commands),
        "hand": len(env.robot.end_effector_commands),
        "head": sum(command[2] is not None for command in env.robot.arm_commands),
        "chassis": len(env.robot.chassis_commands),
        "lifter": len(env.robot.lifter_commands),
    }
    assert not env.step(action)
    assert {
        "arm": len(env.robot.arm_commands),
        "hand": len(env.robot.end_effector_commands),
        "head": sum(command[2] is not None for command in env.robot.arm_commands),
        "chassis": len(env.robot.chassis_commands),
        "lifter": len(env.robot.lifter_commands),
    } == after_pause

    env.close()
    after_close = tuple(
        map(
            len,
            (
                env.robot.arm_commands,
                env.robot.end_effector_commands,
                env.robot.chassis_commands,
                env.robot.lifter_commands,
            ),
        )
    )
    assert not env.step(np.ones(31))
    assert tuple(
        map(
            len,
            (
                env.robot.arm_commands,
                env.robot.end_effector_commands,
                env.robot.chassis_commands,
                env.robot.lifter_commands,
            ),
        )
    ) == after_close


def test_pause_failure_still_gates_commands_and_retries_safety_cleanup():
    env, action = _make_modular_env()
    assert env.step(action)

    def fail_pause(timeout=1.0):
        raise TimeoutError("pause timed out")

    env.robot.pause = fail_pause

    with pytest.raises(TimeoutError, match="pause timed out"):
        env.pause_control(timeout=0.01)

    assert not env.control_enabled
    assert env.robot.lifter_stream_stops == 1
    assert env.robot.chassis_commands[-1] == (0.0, 0.0)
    assert env._last_lifter_command is None
    assert env._last_lifter_command_time == float("-inf")
    after_failure = {
        "arm": len(env.robot.arm_commands),
        "hand": len(env.robot.end_effector_commands),
        "chassis": len(env.robot.chassis_commands),
        "lifter": len(env.robot.lifter_commands),
    }

    assert not env.step(np.ones(31))
    assert len(env.robot.arm_commands) == after_failure["arm"]
    assert len(env.robot.end_effector_commands) == after_failure["hand"]
    assert len(env.robot.chassis_commands) == after_failure["chassis"]
    assert len(env.robot.lifter_commands) == after_failure["lifter"]

    with pytest.raises(TimeoutError, match="pause timed out"):
        env.pause_control(timeout=0.01)
    assert env.robot.lifter_stream_stops == 2
    assert env.robot.chassis_commands[-2:] == [(0.0, 0.0), (0.0, 0.0)]


def test_resume_rejects_incomplete_environment_pause():
    env, _ = _make_modular_env()
    resume_calls = []

    def fail_lifter_stop():
        raise RuntimeError("lifter stop failed")

    env.robot.stop_lifter_stream = fail_lifter_stop
    env.robot.resume_from_state = lambda state=None: resume_calls.append(state)

    with pytest.raises(RuntimeError, match="lifter stop failed"):
        env.pause_control()

    with pytest.raises(RuntimeError, match="pause.*complete"):
        env.resume_control()

    assert not env.control_enabled
    assert not env._pause_complete
    assert resume_calls == []


def test_failed_environment_resume_keeps_gate_closed_for_retry():
    env, _ = _make_modular_env()
    env.pause_control()

    def fail_resume(state=None):
        raise RuntimeError("resume failed")

    env.robot.resume_from_state = fail_resume

    with pytest.raises(RuntimeError, match="resume failed"):
        env.resume_control()

    assert not env.control_enabled
    assert env._pause_complete

    env.robot.resume_from_state = lambda state=None: None
    env.resume_control()

    assert env.control_enabled
    assert not env._pause_complete


def test_pause_completes_mobile_safety_before_blocking_robot_pause():
    env, _ = _make_modular_env()
    pause_entered = threading.Event()
    release_pause = threading.Event()
    events = []
    errors = []

    env.robot.stop_lifter_stream = lambda: events.append("lifter")
    env.robot.stop_chassis = lambda: events.append("chassis")

    def blocking_pause(timeout=1.0):
        events.append("pause")
        pause_entered.set()
        release_pause.wait(timeout=0.5)

    def pause_control():
        try:
            env.pause_control()
        except BaseException as exc:
            errors.append(exc)

    env.robot.pause = blocking_pause
    pause_thread = threading.Thread(target=pause_control)
    pause_thread.start()

    assert pause_entered.wait(timeout=0.5)
    assert events == ["lifter", "chassis", "pause"]

    release_pause.set()
    pause_thread.join(timeout=0.5)
    assert not pause_thread.is_alive()
    assert errors == []


def test_close_completes_mobile_safety_before_blocking_robot_pause():
    env, _ = _make_modular_env()
    pause_entered = threading.Event()
    release_pause = threading.Event()
    events = []

    env.robot.stop_lifter_stream = lambda: events.append("lifter")
    env.robot.stop_chassis = lambda: events.append("chassis")
    env.robot.disconnect = lambda: events.append("disconnect")

    def blocking_pause(timeout=1.0):
        events.append("pause")
        pause_entered.set()
        release_pause.wait(timeout=0.5)

    env.robot.pause = blocking_pause
    close_thread = threading.Thread(target=env.close)
    close_thread.start()

    assert pause_entered.wait(timeout=0.5)
    assert events == ["lifter", "chassis", "pause"]

    release_pause.set()
    close_thread.join(timeout=0.5)
    assert not close_thread.is_alive()
    assert events == ["lifter", "chassis", "pause", "disconnect"]


def test_close_disconnects_when_lifter_stop_fails_and_records_cleanup_error(caplog):
    env, action = _make_modular_env()
    calls = {"lifter": 0, "chassis": 0, "pause": 0, "disconnect": 0}

    def stop_lifter():
        calls["lifter"] += 1
        raise RuntimeError("lifter cleanup failed")

    def stop_chassis():
        calls["chassis"] += 1

    def pause_robot(timeout=1.0):
        calls["pause"] += 1

    def disconnect_robot():
        calls["disconnect"] += 1
        env.robot.disconnected = True

    env.robot.stop_lifter_stream = stop_lifter
    env.robot.stop_chassis = stop_chassis
    env.robot.pause = pause_robot
    env.robot.disconnect = disconnect_robot

    env.close()

    assert not env.control_enabled
    assert not env.step(action)
    assert env._close_complete
    assert env.robot.disconnected
    assert calls == {
        "lifter": 1,
        "chassis": 1,
        "pause": 1,
        "disconnect": 1,
    }
    assert "Failed to stop the lifter stream" in caplog.text

    env.close()

    assert env._close_complete
    assert env.robot.disconnected
    assert calls == {"lifter": 1, "chassis": 1, "pause": 1, "disconnect": 1}


def test_close_disconnect_fallback_completes_chassis_stop_failure_idempotently():
    env, _ = _make_modular_env()
    calls = {"chassis": 0, "disconnect": 0}

    def fail_chassis_stop():
        calls["chassis"] += 1
        raise RuntimeError("chassis cleanup failed")

    def disconnect_robot():
        calls["disconnect"] += 1
        env.robot.disconnected = True

    env.robot.stop_chassis = fail_chassis_stop
    env.robot.disconnect = disconnect_robot

    env.close()

    assert env._close_complete
    assert env.robot.disconnected
    assert calls == {"chassis": 1, "disconnect": 1}

    env.close()

    assert calls == {"chassis": 1, "disconnect": 1}


def test_close_reuses_unreleased_lifter_stop_worker_until_it_exits():
    env, _ = _make_modular_env()
    release_stop = threading.Event()
    stop_entered = threading.Event()
    calls = {"lifter": 0, "disconnect": 0}

    def block_lifter_stop():
        calls["lifter"] += 1
        stop_entered.set()
        release_stop.wait(timeout=1.0)

    def disconnect_robot():
        calls["disconnect"] += 1
        env.robot.disconnected = True

    env.robot.stop_lifter_stream = block_lifter_stop
    env.robot.disconnect = disconnect_robot

    env.close()

    worker = env._lifter_stop_thread
    assert stop_entered.is_set()
    assert worker is not None and worker.is_alive()
    assert not env._close_complete
    assert calls == {"lifter": 1, "disconnect": 1}

    env.close()

    assert env._lifter_stop_thread is worker
    assert calls == {"lifter": 1, "disconnect": 1}
    assert not env._close_complete

    release_stop.set()
    worker.join(timeout=0.5)
    assert not worker.is_alive()

    env.close()

    assert env._close_complete
    assert env._lifter_stop_thread is None


def test_close_retry_does_not_repeat_successful_lifter_stop_after_other_failure():
    env, _ = _make_modular_env()

    class FailingCamera:
        def __init__(self):
            self.calls = 0

        def stop_capture(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("camera cleanup failed")

    camera = FailingCamera()
    env.camera_manager = camera

    env.close()

    assert not env._close_complete
    assert env.robot.lifter_stream_stops == 1
    assert camera.calls == 1

    env.close()

    assert env._close_complete
    assert env.robot.lifter_stream_stops == 1
    assert camera.calls == 2


def test_pause_bounds_blocked_lifter_stop_and_keeps_resume_closed():
    env, _ = _make_modular_env()
    stop_entered = threading.Event()
    release_stop = threading.Event()
    stop_finished = threading.Event()
    pause_calls = []

    def block_lifter_stop():
        stop_entered.set()
        try:
            release_stop.wait(timeout=1.0)
        finally:
            stop_finished.set()

    env.robot.stop_lifter_stream = block_lifter_stop
    env.robot.pause = lambda timeout=1.0: pause_calls.append(timeout)

    started_at = time.monotonic()
    with pytest.raises(TimeoutError, match="lifter stream stop timed out"):
        env.pause_control(timeout=0.01)

    assert time.monotonic() - started_at < 0.2
    assert stop_entered.is_set()
    assert pause_calls == [0.01]
    assert not env._pause_complete
    with pytest.raises(RuntimeError, match="pause.*complete"):
        env.resume_control()

    release_stop.set()
    assert stop_finished.wait(timeout=0.5)


def test_close_disconnects_to_release_blocked_lifter_stop_without_thread_leak():
    env, _ = _make_modular_env()
    stop_entered = threading.Event()
    release_stop = threading.Event()
    stop_finished = threading.Event()
    events = []

    def block_lifter_stop():
        events.append("lifter")
        stop_entered.set()
        try:
            release_stop.wait(timeout=1.0)
        finally:
            stop_finished.set()

    def disconnect_robot():
        events.append("disconnect")
        env.robot.disconnected = True
        release_stop.set()

    env.robot.stop_lifter_stream = block_lifter_stop
    env.robot.stop_chassis = lambda: events.append("chassis")
    env.robot.pause = lambda timeout=1.0: events.append("pause")
    env.robot.disconnect = disconnect_robot

    started_at = time.monotonic()
    env.close()

    assert time.monotonic() - started_at < 0.2
    assert stop_entered.is_set()
    assert stop_finished.wait(timeout=0.5)
    assert events == ["lifter", "chassis", "pause", "disconnect"]
    assert env.robot.disconnected
    assert env._close_complete
    assert env._lifter_stop_thread is None


def test_successful_disconnect_supersedes_failed_close_pause():
    env, _ = _make_modular_env()
    pause_attempts = []

    def fail_pause(timeout=1.0):
        pause_attempts.append(timeout)
        raise TimeoutError("pause timed out")

    env.robot.pause = fail_pause

    with pytest.raises(TimeoutError, match="pause timed out"):
        env.pause_control(timeout=0.01)

    env.close()
    assert env._close_complete
    env.close()

    assert pause_attempts == [0.01, 1.0]
    assert env.robot.lifter_stream_stops == 2
    assert env.robot.chassis_commands == [(0.0, 0.0), (0.0, 0.0)]
    assert env.robot.disconnected


def test_failed_disconnect_does_not_supersede_failed_close_pause():
    env, _ = _make_modular_env()
    pause_attempts = []

    def fail_pause(timeout=1.0):
        pause_attempts.append(timeout)
        raise TimeoutError("pause timed out")

    env.robot.pause = fail_pause
    env.robot.disconnect_error = RuntimeError("disconnect failed")

    env.close()

    assert not env._close_complete
    assert not getattr(env, "_robot_pause_complete", False)
    assert pause_attempts == [1.0]

    env.robot.disconnect_error = None
    env.close()

    assert env._close_complete
    assert pause_attempts == [1.0, 1.0]


def test_pause_resume_allows_same_lifter_target_to_be_sent_again():
    env, action = _make_modular_env()

    assert env.step(action)
    assert env.robot.lifter_commands == [(30.0, 80.0)]

    env.pause_control()
    env.resume_control()
    assert env.step(action)

    assert env.robot.lifter_commands == [(30.0, 80.0), (30.0, 80.0)]


def test_close_resets_lifter_dedupe_cache():
    env, action = _make_modular_env()
    assert env.step(action)
    assert env._last_lifter_command == 30.0

    env.close()

    assert env._last_lifter_command is None
    assert env._last_lifter_command_time == float("-inf")


def test_close_directly_closes_step_gate_before_cleanup():
    env, action = _make_modular_env()

    env.close()
    after_close = (
        len(env.robot.arm_commands),
        len(env.robot.end_effector_commands),
        len(env.robot.chassis_commands),
        len(env.robot.lifter_commands),
    )

    assert not env.step(action)
    assert (
        len(env.robot.arm_commands),
        len(env.robot.end_effector_commands),
        len(env.robot.chassis_commands),
        len(env.robot.lifter_commands),
    ) == after_close


def test_servop_resume_uses_controller_fresh_state_path_not_vector_shape():
    env, _ = _make_modular_env(mobile_base=False)
    env.pause_control()

    env.resume_control(np.arange(18, dtype=np.float64))

    assert env.control_enabled
    assert env.robot.resumed_state is None


def _patch_constructor_resources(
    monkeypatch,
    *,
    close_events=None,
    robot_error=None,
    camera_error=None,
    bridge_error=None,
):
    robot = FakeMotionController(close_events, robot_error)
    camera = FakeCameraManager(close_events, camera_error)
    bridge = FakeBridgeObservationProvider(
        close_events=close_events,
        stop_error=bridge_error,
    )
    monkeypatch.setattr(env_module, "create_motion_controller", lambda *args, **kwargs: robot)
    monkeypatch.setattr(Tron2Env, "_init_camera", lambda self: camera)
    monkeypatch.setattr(Tron2Env, "_init_bridge", lambda self: bridge)
    return robot, camera, bridge


@pytest.mark.parametrize(
    (
        "observation_source",
        "legacy_debug_enabled",
        "bridge_debug_enabled",
        "expected_debug_dir",
    ),
    [
        ("legacy", True, False, "legacy-debug"),
        ("legacy", False, True, None),
        ("bridge", False, True, "bridge-debug"),
        ("bridge", True, False, None),
    ],
)
def test_constructor_selects_debug_writer_for_active_observation_source(
    monkeypatch,
    observation_source,
    legacy_debug_enabled,
    bridge_debug_enabled,
    expected_debug_dir,
):
    robot, camera, bridge = _patch_constructor_resources(monkeypatch)
    writers = []

    def create_writer(debug_dir):
        writer = RecordingDebugWriter(debug_dir)
        writers.append(writer)
        return writer

    monkeypatch.setattr(env_module, "AsyncDebugImageWriter", create_writer)
    config = EnvConfig(observation_source=observation_source)
    config.camera_config.save_debug_images = legacy_debug_enabled
    config.camera_config.debug_image_dir = "legacy-debug"
    config.bridge_config.save_debug_images = bridge_debug_enabled
    config.bridge_config.debug_image_dir = "bridge-debug"

    env = Tron2Env(config)
    try:
        assert env.robot is robot
        if observation_source == "legacy":
            assert env.camera_manager is camera
            assert env.bridge_provider is None
        else:
            assert env.camera_manager is None
            assert env.bridge_provider is bridge

        if expected_debug_dir is None:
            assert writers == []
            assert env._debug_image_writer is None
        else:
            assert len(writers) == 1
            assert writers[0].debug_dir == expected_debug_dir
            assert env._debug_image_writer is writers[0]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("observation_source", "expected_close_events"),
    [
        ("legacy", ["robot", "camera"]),
        ("bridge", ["robot", "bridge"]),
    ],
)
def test_writer_construction_failure_rolls_back_started_resources(
    monkeypatch,
    observation_source,
    expected_close_events,
):
    close_events = []
    _patch_constructor_resources(monkeypatch, close_events=close_events)

    def fail_writer_construction(debug_dir):
        raise RuntimeError("writer construction failed")

    monkeypatch.setattr(env_module, "AsyncDebugImageWriter", fail_writer_construction)
    config = EnvConfig(observation_source=observation_source)

    with pytest.raises(RuntimeError, match="writer construction failed"):
        Tron2Env(config)

    assert close_events == expected_close_events


def test_constructor_cleans_camera_and_robot_on_warmup_keyboard_interrupt(monkeypatch):
    close_events = []
    robot = FakeMotionController(close_events)
    camera = FakeCameraManager(close_events)
    fake_camera_module = types.ModuleType("tron2_env.camera")

    class FakeMultiCameraManager:
        @classmethod
        def from_config(cls, config):
            return camera

    fake_camera_module.MultiCameraManager = FakeMultiCameraManager
    monkeypatch.setitem(sys.modules, "tron2_env.camera", fake_camera_module)
    monkeypatch.setattr(env_module, "create_motion_controller", lambda *args, **kwargs: robot)

    def interrupt_warmup(seconds):
        raise KeyboardInterrupt("camera warmup interrupted")

    monkeypatch.setattr(env_module.time, "sleep", interrupt_warmup)
    config = EnvConfig(observation_source="legacy")
    config.camera_config.save_debug_images = False

    with pytest.raises(KeyboardInterrupt, match="camera warmup interrupted"):
        Tron2Env(config)

    assert camera.started
    assert camera.stopped
    assert robot.disconnected
    assert close_events == ["camera", "robot"]


@pytest.mark.parametrize(
    ("failure_stage", "failure"),
    [
        ("start", RuntimeError("bridge start failed")),
        ("get_obs", KeyboardInterrupt("bridge readiness interrupted")),
    ],
)
def test_constructor_cleans_bridge_provider_and_robot_when_bridge_init_fails(
    monkeypatch,
    failure_stage,
    failure,
):
    close_events = []
    robot = FakeMotionController(close_events)
    provider = FakeBridgeObservationProvider(
        close_events=close_events,
        start_error=failure if failure_stage == "start" else None,
        get_obs_error=failure if failure_stage == "get_obs" else None,
    )
    monkeypatch.setattr(env_module, "create_motion_controller", lambda *args, **kwargs: robot)

    monkeypatch.setattr(
        bridge_module,
        "BridgeObservationProvider",
        lambda config: provider,
    )
    config = EnvConfig(observation_source="bridge")
    config.bridge_config.save_debug_images = False

    with pytest.raises(type(failure), match=str(failure)):
        Tron2Env(config)

    assert provider.started
    assert provider.stopped
    assert robot.disconnected
    assert close_events == ["bridge", "robot"]


@pytest.mark.parametrize(
    ("observation_source", "expected_close_events"),
    [
        ("legacy", ["robot", "camera", "writer"]),
        ("bridge", ["robot", "bridge", "writer"]),
    ],
)
def test_close_stops_active_resources_before_debug_writer(
    monkeypatch,
    observation_source,
    expected_close_events,
):
    close_events = []
    _patch_constructor_resources(monkeypatch, close_events=close_events)
    monkeypatch.setattr(
        env_module,
        "AsyncDebugImageWriter",
        lambda debug_dir: RecordingDebugWriter(debug_dir, close_events),
    )
    env = Tron2Env(EnvConfig(observation_source=observation_source))

    env.close()
    env.close()

    assert close_events == expected_close_events


@pytest.mark.parametrize(
    ("observation_source", "expected_close_events"),
    [
        ("legacy", ["robot", "camera", "writer"]),
        ("bridge", ["robot", "bridge", "writer"]),
    ],
)
def test_close_attempts_all_resources_when_cleanup_raises(
    monkeypatch,
    observation_source,
    expected_close_events,
):
    close_events = []
    _patch_constructor_resources(
        monkeypatch,
        close_events=close_events,
        robot_error=RuntimeError("robot cleanup failed"),
        camera_error=RuntimeError("camera cleanup failed"),
        bridge_error=RuntimeError("bridge cleanup failed"),
    )
    monkeypatch.setattr(
        env_module,
        "AsyncDebugImageWriter",
        lambda debug_dir: RecordingDebugWriter(
            debug_dir,
            close_events,
            RuntimeError("writer cleanup failed"),
        ),
    )
    env = Tron2Env(EnvConfig(observation_source=observation_source))

    env.close()

    assert close_events == expected_close_events


def test_bridge_get_obs_submits_images_to_debug_writer_once(tmp_path):
    env = make_env()
    env.config = EnvConfig(observation_source="bridge")
    env.config.bridge_config.debug_image_dir = str(tmp_path)
    env.layout = env.config.physical_layout
    images = {
        "cam_high": np.full((2, 2, 3), 1, dtype=np.uint8),
        "cam_left_wrist": np.full((2, 2, 3), 2, dtype=np.uint8),
        "cam_right_wrist": np.full((2, 2, 3), 3, dtype=np.uint8),
    }
    env.bridge_provider = FakeBridgeObservationProvider(
        {"state": np.zeros(18, dtype=np.float32), "images": images, "metadata": {}}
    )
    writer = RecordingDebugWriter()
    env._debug_image_writer = writer

    env.get_obs()

    assert len(writer.submissions) == 1
    assert set(writer.submissions[0]) == set(images)
    for name, image in images.items():
        assert writer.submissions[0][name] is image


def test_legacy_get_obs_submits_only_camera_images_to_debug_writer(tmp_path):
    env = make_env()
    env.config.legacy_use_time_sync = False
    env.config.camera_config.debug_image_dir = str(tmp_path)
    images = {
        "cam_high": np.full((2, 2, 3), 1, dtype=np.uint8),
        "cam_left_wrist": np.full((2, 2, 3), 2, dtype=np.uint8),
        "cam_right_wrist": np.full((2, 2, 3), 3, dtype=np.uint8),
    }
    rgb_images = {
        **images,
        "cam_high_timestamp": 1.0,
        "cam_left_wrist_timestamp": 1.1,
        "cam_right_wrist_timestamp": 1.2,
    }
    env._get_images = lambda: rgb_images
    writer = RecordingDebugWriter()
    env._debug_image_writer = writer

    env.get_obs()

    assert len(writer.submissions) == 1
    assert set(writer.submissions[0]) == set(images)
    for name, image in images.items():
        assert writer.submissions[0][name] is image


def test_close_closes_debug_writer_once_with_bounded_timeout():
    env = make_env()
    writer = RecordingDebugWriter()
    env._debug_image_writer = writer

    env.close()
    env.close()

    assert writer.closed
    assert writer.close_timeouts == [0.2]


def test_pause_blocks_arm_and_gripper_commands():
    env = make_env()

    env.pause_control()

    assert env.step(np.zeros(16)) is False
    assert env.robot.commands == []
    assert env.robot.grippers == []
    assert env.blocked_step_count == 1


def test_resume_rebases_before_reopening_gate():
    env = make_env()
    env.pause_control()
    state18 = np.arange(18, dtype=np.float64)

    env.resume_control(state18)

    assert env.control_enabled
    np.testing.assert_array_equal(env.robot.resumed_state, state18)
    assert env.step(np.zeros(16)) is True
    assert len(env.robot.commands) == 1
    assert len(env.robot.grippers) == 1


def test_pause_waits_for_in_flight_step_and_blocks_following_steps():
    env = make_env()
    env.robot.gripper_entered = threading.Event()
    env.robot.gripper_release = threading.Event()
    pause_returned = threading.Event()

    step_thread = threading.Thread(target=env.step, args=(np.zeros(16),))
    step_thread.start()
    assert env.robot.gripper_entered.wait(timeout=0.5)

    def pause():
        env.pause_control()
        pause_returned.set()

    pause_thread = threading.Thread(target=pause)
    pause_thread.start()
    time.sleep(0.03)
    assert not pause_returned.is_set()

    env.robot.gripper_release.set()
    step_thread.join(timeout=0.5)
    pause_thread.join(timeout=0.5)
    assert pause_returned.is_set()

    command_count = len(env.robot.commands)
    gripper_count = len(env.robot.grippers)
    assert env.step(np.ones(16)) is False
    assert len(env.robot.commands) == command_count
    assert len(env.robot.grippers) == gripper_count


def test_resume_rejects_wrong_state_shape():
    env = make_env()
    env.pause_control()

    try:
        env.resume_control(np.zeros(16))
    except ValueError as exc:
        assert "18" in str(exc)
    else:
        raise AssertionError("resume_control should reject non-18D measured state")


def test_chassis_layout_routes_velocity_but_does_not_send_lifter_by_default():
    env = make_env()
    env.config = EnvConfig(
        state_layout=["left_arm", "left_gripper", "right_arm", "right_gripper", "chassis"],
        state_dim=19,
    )
    env.layout = env.config.physical_layout
    action = np.arange(19, dtype=np.float64)
    action[16:] = [0.4, -0.5, 230.2]

    assert env.step(action)

    assert env.robot.chassis_commands == [(0.4, -0.5)]
    assert env.robot.lifter_commands == []
    assert env.robot.commands[0].shape == (16,)

    env.step(action)
    assert env.robot.lifter_commands == []


def test_lifter_control_requires_mm_feedback_and_can_be_enabled_explicitly():
    env = make_env()
    env.config = EnvConfig(
        state_layout=["left_arm", "left_gripper", "right_arm", "right_gripper", "chassis"],
        state_dim=19,
        lifter_state_source="position_mm",
        lifter_control_enabled=True,
    )
    env.layout = env.config.physical_layout
    action = np.zeros(19, dtype=np.float64)
    action[-1] = 230.2

    env.step(action)
    env.step(action)

    assert env.robot.lifter_commands == [(230.2, 80.0)]


def test_lifter_stream_velocity_defaults_to_80_and_allows_override():
    assert EnvConfig().lifter_stream_velocity_mm_s == 80.0
    assert (
        EnvConfig(lifter_stream_velocity_mm_s=125.5).lifter_stream_velocity_mm_s
        == 125.5
    )


@pytest.mark.parametrize("velocity", [0.0, -1.0, np.nan, np.inf, "fast"])
def test_lifter_stream_velocity_must_be_finite_and_positive(velocity):
    with pytest.raises(
        ValueError, match="lifter_stream_velocity_mm_s must be finite and positive"
    ):
        EnvConfig(lifter_stream_velocity_mm_s=velocity)


def test_lifter_stream_refreshes_same_target_only_after_configured_interval(
    monkeypatch,
):
    env, action = _make_modular_env()
    env.config.lifter_command_interval = 0.1
    now = [10.0]
    monkeypatch.setattr(env_module.time, "monotonic", lambda: now[0])

    assert env.step(action)
    now[0] = 10.05
    assert env.step(action)
    now[0] = 10.1
    assert env.step(action)

    assert env.robot.lifter_commands == [(30.0, 80.0), (30.0, 80.0)]


def test_lifter_stream_uses_configured_velocity_for_preflight_and_dispatch():
    env, action = _make_modular_env()
    env.config.lifter_stream_velocity_mm_s = 125.5

    assert env.step(action)

    assert env.robot.preflight_calls[-1] == "lifter"
    assert env.robot.lifter_commands == [(30.0, 125.5)]


def test_raw_lifter_state_cannot_be_enabled_for_control():
    with np.testing.assert_raises_regex(ValueError, "raw q actions"):
        EnvConfig(
            state_layout=["left_arm", "left_gripper", "right_arm", "right_gripper", "chassis"],
            state_dim=19,
            lifter_state_source="raw_q",
            lifter_control_enabled=True,
        )


def test_lifter_control_requires_chassis_layout():
    with np.testing.assert_raises_regex(ValueError, "chassis.*state_layout"):
        EnvConfig(lifter_state_source="position_mm", lifter_control_enabled=True)


def test_chassis_layout_composes_auxiliary_feedback_in_policy_order():
    env = make_env()
    env.config = EnvConfig(
        state_layout=["left_arm", "left_gripper", "right_arm", "right_gripper", "chassis"],
        state_dim=19,
    )
    env.layout = env.config.physical_layout

    state, metadata = env._compose_policy_state(np.arange(18, dtype=np.float32))

    np.testing.assert_allclose(state[-3:], [0.2, -0.3, -254.4])
    assert metadata == {
        "chassis_timestamp_ms": 10,
        "lifter_timestamp_ms": 11,
        "lifter_state_source": "raw_q",
    }


def test_pause_stops_chassis_for_auxiliary_layout():
    env = make_env()
    env.config = EnvConfig(
        state_layout=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
            "chassis",
        ],
        state_dim=19,
    )
    env.layout = env.config.physical_layout

    env.pause_control()

    assert env.robot.paused
    assert env.robot.chassis_commands == [(0.0, 0.0)]


def test_pause_does_not_stop_lifter_stream_when_lifter_control_is_disabled():
    env = make_env()
    env.config = EnvConfig(
        state_layout=["left_arm", "left_gripper", "right_arm", "right_gripper", "chassis"],
        state_dim=19,
    )
    env.layout = env.config.physical_layout

    env.pause_control()

    assert env.robot.lifter_stream_stops == 0


def test_close_does_not_stop_lifter_stream_when_lifter_control_is_disabled():
    env = make_env()
    env.config = EnvConfig(
        state_layout=["left_arm", "left_gripper", "right_arm", "right_gripper", "chassis"],
        state_dim=19,
    )
    env.layout = env.config.physical_layout

    env.close()

    assert env.robot.lifter_stream_stops == 0
    assert env.robot.disconnected
