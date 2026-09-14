"""Tests for MotionController — uses a FakeTransport, no robot needed."""

from __future__ import annotations

import inspect
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest

from tron2_env.end_effector import GripperProvider
from tron2_env.errors import CommandError, StateError
from tron2_env.interpolation import LinearInterpolator, PoseInterpolator
from tron2_env.joints import JointIndex
from tron2_env.layout import RobotModules
from tron2_env.motion import MotionController, create_motion_controller
import tron2_env.motion.arm_strategy as arm_strategy_module
import tron2_env.motion.controller as controller_module


class FakeTransport:
    """Minimal transport that records every send_joint_cmd call."""

    def __init__(self, initial_state=None, fail_state: bool = False):
        if initial_state is None:
            initial_state = list(np.zeros(JointIndex.STATE_DIM, dtype=np.float64))
        self._state = list(initial_state)
        self._fail_state = fail_state
        self._sends: list = []  # list[(timestamp, q)]
        self._lock = threading.Lock()
        self.connected = True
        self.gripper_calls: list = []
        self.disconnected = False
        self.lifter_stream_commands: list = []
        self.lifter_stream_stops = 0

    # RobotTransport API

    def send_joint_cmd(self, q):
        with self._lock:
            self._sends.append((time.perf_counter(), np.asarray(q, dtype=np.float64).copy()))

    def get_joint_state(self, timeout: float = 1.0):
        if self._fail_state:
            raise StateError("fake transport configured to fail")
        return {
            "timestamp": int(time.time() * 1000),
            "received_at": time.monotonic(),
            "states": list(self._state),
        }

    def get_head_position(self):
        return np.asarray(self._state[JointIndex.HEAD], dtype=np.float64)

    def set_gripper(self, left, right):
        self.gripper_calls.append((left, right))

    def wait_until_reached(self, target, tolerance=0.05, timeout=10.0):
        return True

    def disconnect(self):
        self.disconnected = True

    def stream_lifter_position(self, position_mm, velocity_mm_s=80.0):
        self.lifter_stream_commands.append((position_mm, velocity_mm_s))

    def stop_lifter_stream(self):
        self.lifter_stream_stops += 1

    def is_connected(self):
        return self.connected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # introspection

    def sends(self):
        with self._lock:
            return list(self._sends)


def valid_pose(offset: float = 0.0) -> np.ndarray:
    return np.array(
        [
            offset + 0.1,
            offset + 0.2,
            offset + 0.3,
            1.0,
            0.0,
            0.0,
            0.0,
            offset + 0.4,
            offset + 0.5,
            offset + 0.6,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def wait_for(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    assert predicate(), "condition was not met before timeout"


class _ProviderContractMixin:
    @staticmethod
    def validate_command(left, right):
        return np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)


class FakeDualModeTransport(FakeTransport):
    """Complete fake for both optional arm capabilities and head callbacks."""

    def __init__(self, initial_state=None, measured_head=None):
        super().__init__(initial_state=initial_state)
        self.arm_head_state = MotionController._state_to_servoj(
            np.asarray(self._state, dtype=np.float64)
        )
        if measured_head is not None:
            self.arm_head_state[14:16] = measured_head
            self._state[JointIndex.HEAD] = list(measured_head)
        self.pose_state = valid_pose()
        self.pose_guid = None
        self.requested_pose_guids = []
        self.requested_pose_timeouts = []
        self.joint_state_reads = 0
        self.arm_head_reads = 0
        self.pose_commands = []
        self.head_commands = []
        self.head_callback = None
        self.head_callback_registrations = 0
        self.head_callback_unregistrations = 0

    @property
    def joint_commands(self):
        return [q for _timestamp, q in self.sends()]

    def get_joint_state(self, timeout: float = 1.0):
        self.joint_state_reads += 1
        return super().get_joint_state(timeout)

    def get_arm_head_state(self, timeout: float = 1.0, max_age=None):
        self.arm_head_reads += 1
        return {"timestamp": 1, "states": self.arm_head_state.copy()}

    def request_pose_state(self, timeout: float = 1.0):
        self.requested_pose_timeouts.append(timeout)
        guid = f"pose-{len(self.requested_pose_guids) + 1}"
        self.requested_pose_guids.append(guid)
        self.pose_guid = guid
        return guid

    def get_pose_state(self, timeout: float = 1.0, max_age=None):
        return {
            "timestamp": len(self.requested_pose_guids),
            "states": self.pose_state.copy(),
            "guid": self.pose_guid,
        }

    def send_pose_cmd(self, left, right):
        self.pose_commands.append(
            np.concatenate(
                (
                    np.asarray(left, dtype=np.float64),
                    np.asarray(right, dtype=np.float64),
                )
            )
        )

    def send_head_move(self, target, duration, guid):
        self.head_commands.append(
            (np.asarray(target, dtype=np.float64).copy(), duration, guid)
        )
        return True

    def register_head_response_callback(self, callback):
        self.head_callback = callback
        self.head_callback_registrations += 1

    def unregister_head_response_callback(self):
        self.head_callback = None
        self.head_callback_unregistrations += 1


class ReconnectingFakeTransport(FakeDualModeTransport):
    """Reconnect-capable fake implementing the optional session callback API."""

    def __init__(self, initial_state=None):
        super().__init__(initial_state=initial_state)
        self.session_generation = 1
        self.connection_callback = None
        self.connection_callback_unregistrations = 0
        self.chassis_commands = []
        self.lifter_commands = []

    def register_connection_state_callback(self, callback):
        self.connection_callback = callback
        callback(True, self.session_generation)

    def unregister_connection_state_callback(self):
        self.connection_callback = None
        self.connection_callback_unregistrations += 1

    def lose_connection(self):
        self.connected = False
        self.connection_callback(False, self.session_generation)

    def reconnect(self):
        self.session_generation += 1
        self.connected = True
        self.connection_callback(True, self.session_generation)

    def command_chassis(self, linear_x, angular_z):
        self.chassis_commands.append((linear_x, angular_z))

    def stop_chassis(self):
        self.chassis_commands.append((0.0, 0.0))

    def stream_lifter_position(self, position_mm, velocity_mm_s=80.0):
        self.lifter_commands.append((position_mm, velocity_mm_s))


class RecordingArmStrategy:
    def __init__(self, mode="servoj", measured=None):
        self.mode = mode
        self.target_dim = 16 if mode == "servoj" else 14
        if measured is None:
            measured = (
                np.zeros(self.target_dim)
                if mode == "servoj"
                else valid_pose()
            )
        self.measured = np.asarray(measured, dtype=np.float64)
        self.reads = 0
        self.resets = []
        self.destinations = []
        self.publishes = 0
        self.publish_entered = threading.Event()

    def read_measured_target(self, timeout: float = 2.0):
        self.reads += 1
        return self.measured.copy()

    def reset(self, measured):
        self.resets.append(np.asarray(measured, dtype=np.float64).copy())

    def set_destination(self, target, eta):
        self.destinations.append(
            (np.asarray(target, dtype=np.float64).copy(), float(eta))
        )

    def publish_current(self):
        self.publishes += 1
        self.publish_entered.set()


class RecordingHeadScheduler:
    def __init__(self, block_tick: bool = False):
        self.submissions = []
        self.tick_count = 0
        self.pause_count = 0
        self.resume_count = 0
        self.close_count = 0
        self.reconnect_pause_count = 0
        self.tick_entered = threading.Event()
        self.release_tick = threading.Event()
        if not block_tick:
            self.release_tick.set()

    def submit(self, target):
        self.submissions.append(np.asarray(target, dtype=np.float64).copy())

    def tick(self):
        self.tick_count += 1
        self.tick_entered.set()
        assert self.release_tick.wait(timeout=1.0)

    def handle_response(self, guid, result):
        pass

    def raise_if_faulted(self):
        pass

    def pause(self):
        self.pause_count += 1

    def pause_for_reconnect(self):
        self.reconnect_pause_count += 1

    def resume(self):
        self.resume_count += 1

    def close(self):
        self.close_count += 1


def test_start_seeds_interpolator_from_transport_state():
    """First publish should match the measured q, not zeros."""
    state18 = np.zeros(JointIndex.STATE_DIM)
    state18[JointIndex.LEFT_ARM] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    state18[JointIndex.RIGHT_ARM] = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]
    state18[JointIndex.HEAD] = [0.8, -0.8]
    fake = FakeTransport(initial_state=state18)

    mc = MotionController(transport=fake, interpolator=LinearInterpolator(), publish_rate=200.0)
    mc.start()
    time.sleep(0.05)  # let a few publish cycles run

    try:
        sends = fake.sends()
        assert sends, "publish loop should have sent at least one frame"
        first_q = sends[0][1]
        expected = np.concatenate([state18[JointIndex.LEFT_ARM], state18[JointIndex.RIGHT_ARM], state18[JointIndex.HEAD]])
        np.testing.assert_allclose(first_q, expected, atol=1e-9)
    finally:
        mc.disconnect()


def test_start_and_normal_resume_bound_measured_seed_freshness():
    class FreshnessRecordingStrategy(RecordingArmStrategy):
        def __init__(self):
            super().__init__(mode="servoj")
            self.read_calls = []

        def read_measured_target(self, timeout=2.0, max_age=None):
            self.read_calls.append((timeout, max_age))
            return self.measured.copy()

    strategy = FreshnessRecordingStrategy()
    controller = MotionController(
        FakeTransport(),
        arm_strategy=strategy,
        state_max_age=0.23,
        publish_rate=200.0,
    )

    controller.start()
    try:
        controller.pause(timeout=0.5)
        controller.resume_from_state()
        assert strategy.read_calls == [(2.0, 0.23), (2.0, 0.23)]
    finally:
        controller.disconnect()


def test_command_apis_fail_closed_while_start_is_installing_measured_seed():
    reset_entered = threading.Event()
    release_reset = threading.Event()
    start_errors = []

    class GenerationTrackingTransport(ReconnectingFakeTransport):
        def __init__(self):
            super().__init__()
            self.bound_generations = []

        @contextmanager
        def bind_command_generation(self, generation):
            self.bound_generations.append(generation)
            yield

    class BlockingResetStrategy(RecordingArmStrategy):
        def reset(self, measured):
            super().reset(measured)
            reset_entered.set()
            assert release_reset.wait(timeout=1.0)

    transport = GenerationTrackingTransport()
    strategy = BlockingResetStrategy(mode="servop")
    scheduler = RecordingHeadScheduler()
    controller = MotionController(
        transport,
        arm_mode="servop",
        arm_strategy=strategy,
        head_scheduler=scheduler,
        publish_rate=200.0,
    )

    def start_controller():
        try:
            controller.start()
        except BaseException as exc:  # pragma: no cover - asserted below
            start_errors.append(exc)

    starter = threading.Thread(target=start_controller, daemon=True)
    starter.start()
    assert reset_entered.wait(timeout=0.5)

    try:
        ordinary_apis = (
            lambda: controller.command_arm(
                valid_pose()[:7],
                valid_pose()[7:],
                head=np.zeros(2),
            ),
            lambda: controller.preflight_arm(
                valid_pose()[:7],
                valid_pose()[7:],
                head=np.zeros(2),
            ),
            lambda: controller.command_head(np.zeros(2)),
            lambda: controller.preflight_head(np.zeros(2)),
            lambda: controller.command_end_effector(np.zeros(1), np.zeros(1)),
            lambda: controller.preflight_end_effector(np.zeros(1), np.zeros(1)),
            controller.raise_if_faulted,
            lambda: controller.set_gripper(25.0, 75.0),
            lambda: controller.command_chassis(0.1, -0.2),
            lambda: controller.preflight_chassis(0.1, -0.2),
            lambda: controller.stream_lifter_position(100.0),
            lambda: controller.preflight_lifter_stream_position(100.0),
        )
        for invoke in ordinary_apis:
            with pytest.raises(RuntimeError, match="has not started"):
                invoke()

        assert strategy.destinations == []
        assert scheduler.submissions == []
        assert transport.gripper_calls == []
        assert transport.chassis_commands == []
        assert transport.lifter_commands == []
        assert transport.bound_generations == []
    finally:
        release_reset.set()
        starter.join(timeout=1.0)
        controller.disconnect()

    assert not starter.is_alive()
    assert start_errors == []


def test_publish_rate_close_to_requested():
    fake = FakeTransport()
    mc = MotionController(transport=fake, publish_rate=200.0)
    mc.start()
    try:
        time.sleep(0.6)
        sends = fake.sends()
    finally:
        mc.disconnect()

    assert len(sends) >= 50, f"expected ~120 sends in 0.6s @ 200Hz, got {len(sends)}"
    # rough rate sanity — within 25% of requested
    duration = sends[-1][0] - sends[0][0]
    measured_rate = (len(sends) - 1) / max(duration, 1e-6)
    assert 150.0 <= measured_rate <= 260.0, f"measured rate {measured_rate:.1f} far from 200"


def test_command_joints_non_blocking_and_retargets():
    fake = FakeTransport()
    mc = MotionController(transport=fake, publish_rate=200.0, eta_default=0.05)
    mc.start()
    try:
        # Initially publishes zeros. After command_joints, the published q
        # should converge toward the new target within eta.
        target = np.ones(JointIndex.SERVOJ_DIM) * 0.5

        t_before = time.perf_counter()
        mc.command_joints(target)
        elapsed = time.perf_counter() - t_before
        assert elapsed < 0.01, f"command_joints should be non-blocking, took {elapsed*1000:.1f}ms"

        time.sleep(0.15)  # well past eta
        sends = fake.sends()
    finally:
        mc.disconnect()

    last_q = sends[-1][1]
    np.testing.assert_allclose(last_q, target, atol=1e-6)


def test_pause_acknowledgement_stops_joint_publication():
    fake = FakeTransport()
    mc = MotionController(transport=fake, publish_rate=200.0)
    mc.start()
    try:
        time.sleep(0.03)
        mc.pause(timeout=0.5)
        count_at_pause = len(fake.sends())

        time.sleep(0.05)

        assert len(fake.sends()) == count_at_pause
        assert mc.is_paused
    finally:
        mc.disconnect()


@pytest.mark.parametrize("failing_hook", ["head", "provider"])
def test_pause_failure_still_stops_arm_and_attempts_every_hook(failing_hook):
    class FailingPauseScheduler(RecordingHeadScheduler):
        def __init__(self):
            super().__init__()
            self.fail_pause = failing_hook == "head"

        def pause(self):
            super().pause()
            if self.fail_pause:
                raise RuntimeError("head pause failed")

    class FailingPauseProvider(_ProviderContractMixin):
        name = "gripper"
        width_per_side = 1

        def __init__(self):
            self.pause_count = 0
            self.fail_pause = failing_hook == "provider"

        def read_state(self, timeout=1.0, max_age=None):
            return np.zeros(2), {}

        def command(self, left, right):
            return None

        def pause(self):
            self.pause_count += 1
            if self.fail_pause:
                raise RuntimeError("provider pause failed")

        def resume(self):
            return None

        def close(self):
            return None

        def raise_if_faulted(self):
            return None

    strategy = RecordingArmStrategy(mode="servop")
    scheduler = FailingPauseScheduler()
    provider = FailingPauseProvider()
    controller = MotionController(
        FakeTransport(),
        arm_mode="servop",
        arm_strategy=strategy,
        head_scheduler=scheduler,
        end_effector_provider=provider,
        publish_rate=200.0,
    )
    controller.start()
    wait_for(lambda: strategy.publishes >= 2)

    try:
        with pytest.raises(RuntimeError, match=f"{failing_hook} pause failed"):
            controller.pause(timeout=0.5)

        count_at_failure = strategy.publishes
        time.sleep(0.03)
        assert strategy.publishes == count_at_failure
        assert controller.is_paused
        assert scheduler.pause_count == 1
        assert provider.pause_count == 1

        scheduler.fail_pause = False
        provider.fail_pause = False
        controller.pause(timeout=0.5)

        assert scheduler.pause_count == (2 if failing_hook == "head" else 1)
        assert provider.pause_count == (2 if failing_hook == "provider" else 1)
        controller.resume_from_state()
        assert not controller.is_paused
    finally:
        controller.disconnect()


@pytest.mark.parametrize("failing_hook", ["head", "provider"])
def test_resume_rejects_incomplete_optional_pause_hooks(failing_hook):
    class FailingPauseScheduler(RecordingHeadScheduler):
        def pause(self):
            super().pause()
            if failing_hook == "head":
                raise RuntimeError("head pause failed")

    class FailingPauseProvider(_ProviderContractMixin):
        name = "gripper"
        width_per_side = 1

        def __init__(self):
            self.pause_count = 0
            self.resume_count = 0

        def read_state(self, timeout=1.0, max_age=None):
            return np.zeros(2), {}

        def command(self, left, right):
            return None

        def pause(self):
            self.pause_count += 1
            if failing_hook == "provider":
                raise RuntimeError("provider pause failed")

        def resume(self):
            self.resume_count += 1

        def close(self):
            return None

        def raise_if_faulted(self):
            return None

    strategy = RecordingArmStrategy(mode="servop")
    scheduler = FailingPauseScheduler()
    provider = FailingPauseProvider()
    controller = MotionController(
        FakeTransport(),
        arm_mode="servop",
        arm_strategy=strategy,
        head_scheduler=scheduler,
        end_effector_provider=provider,
        publish_rate=200.0,
    )
    controller.start()

    try:
        with pytest.raises(RuntimeError, match=f"{failing_hook} pause failed"):
            controller.pause(timeout=0.5)
        reset_count = len(strategy.resets)

        with pytest.raises(RuntimeError, match="paused"):
            controller.resume_from_state()

        assert controller.is_paused
        assert len(strategy.resets) == reset_count
        assert scheduler.resume_count == 0
        assert provider.resume_count == 0
    finally:
        controller.disconnect()


def test_resume_from_state_discards_old_interpolator_destination():
    fake = FakeTransport()
    mc = MotionController(transport=fake, publish_rate=200.0, eta_default=0.5)
    mc.start()
    try:
        mc.command_joints(np.ones(JointIndex.SERVOJ_DIM))
        time.sleep(0.02)
        mc.pause(timeout=0.5)

        state18 = np.zeros(JointIndex.STATE_DIM)
        state18[JointIndex.LEFT_ARM] = 0.2
        state18[JointIndex.RIGHT_ARM] = -0.2
        state18[JointIndex.HEAD] = [0.3, -0.3]
        expected = MotionController._state_to_servoj(state18)

        mc.resume_from_state(state18)
        time.sleep(0.03)

        np.testing.assert_allclose(fake.sends()[-1][1], expected, atol=1e-9)
        assert not mc.is_paused
    finally:
        mc.disconnect()


def test_resume_from_state_requires_paused_controller():
    fake = FakeTransport()
    mc = MotionController(transport=fake, publish_rate=200.0)
    mc.start()
    try:
        with pytest.raises(RuntimeError, match="paused"):
            mc.resume_from_state(np.zeros(JointIndex.STATE_DIM))
    finally:
        mc.disconnect()


def test_disconnect_stops_publish_thread():
    fake = FakeTransport()
    mc = MotionController(transport=fake, publish_rate=200.0)
    mc.start()
    time.sleep(0.05)
    mc.disconnect()

    assert fake.disconnected
    # no new sends after disconnect
    snapshot = len(fake.sends())
    time.sleep(0.05)
    assert len(fake.sends()) == snapshot


def test_disconnect_interrupts_low_rate_publish_sleep():
    strategy = RecordingArmStrategy()
    controller = MotionController(
        FakeTransport(),
        arm_strategy=strategy,
        publish_rate=0.1,
    )
    controller.start()
    wait_for(lambda: strategy.publishes >= 2, timeout=0.5)
    thread = controller._publish_thread

    started_at = time.monotonic()
    controller.disconnect()
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert thread is not None and not thread.is_alive()


def test_pause_interrupts_low_rate_publish_sleep_and_resume_continues():
    strategy = RecordingArmStrategy()
    controller = MotionController(
        FakeTransport(),
        arm_strategy=strategy,
        publish_rate=0.1,
    )
    controller.start()
    thread = controller._publish_thread

    try:
        wait_for(lambda: strategy.publishes >= 2, timeout=0.5)
        pause_started_at = time.monotonic()

        controller.pause(timeout=0.2)

        assert time.monotonic() - pause_started_at < 0.2
        assert controller.is_paused
        publishes_at_pause = strategy.publishes
        time.sleep(0.03)
        assert strategy.publishes == publishes_at_pause

        controller.resume_from_state()
        wait_for(lambda: strategy.publishes > publishes_at_pause, timeout=0.5)

        disconnect_started_at = time.monotonic()
        controller.disconnect()
        assert time.monotonic() - disconnect_started_at < 0.5
        assert thread is not None and not thread.is_alive()
    finally:
        controller.disconnect()


def test_stop_lifter_stream_delegates_while_controller_is_paused():
    fake = FakeTransport()
    mc = MotionController(transport=fake)
    mc.start()
    mc.pause()

    mc.stop_lifter_stream()
    mc.disconnect()

    assert fake.lifter_stream_stops == 1


def test_lifter_stream_preflight_is_side_effect_free_and_dispatches_default_velocity():
    fake = FakeTransport()
    controller = MotionController(transport=fake)
    controller.start()
    try:
        controller.preflight_lifter_stream_position(123.4)
        assert fake.lifter_stream_commands == []

        controller.stream_lifter_position(123.4)
        assert fake.lifter_stream_commands == [(123.4, 80.0)]
    finally:
        controller.disconnect()


@pytest.mark.parametrize(
    "position_mm, velocity_mm_s",
    [(np.nan, 80.0), (1.0, np.inf), (1.0, 0.0), (1.0, -1.0)],
)
def test_lifter_stream_validation_happens_before_transport_dispatch(
    position_mm,
    velocity_mm_s,
):
    fake = FakeTransport()
    controller = MotionController(transport=fake)
    controller.start()
    try:
        with pytest.raises(CommandError, match="lifter stream position/velocity"):
            controller.stream_lifter_position(position_mm, velocity_mm_s)
        assert fake.lifter_stream_commands == []
    finally:
        controller.disconnect()


def test_start_fails_when_transport_has_no_state():
    fake = FakeTransport(fail_state=True)
    mc = MotionController(transport=fake, publish_rate=200.0)
    with pytest.raises(RuntimeError, match="initial joint state"):
        mc.start()
    with pytest.raises(RuntimeError, match="has not started"):
        mc.set_gripper(75.0, 25.0)
    assert fake.gripper_calls == []
    mc.disconnect()


def test_set_gripper_delegates():
    fake = FakeTransport()
    mc = MotionController(transport=fake, publish_rate=200.0)
    mc.start()
    try:
        mc.set_gripper(75.0, 25.0)
        assert fake.gripper_calls == [(75.0, 25.0)]
    finally:
        mc.disconnect()


def test_set_gripper_routes_percentages_through_selected_gripper_provider():
    class RecordingGripperProvider(_ProviderContractMixin):
        name = "gripper"
        width_per_side = 1

        def __init__(self):
            self.commands = []

        def read_state(self, timeout=1.0, max_age=None):
            return np.zeros(2), {}

        def command(self, left, right):
            self.commands.append((np.asarray(left).copy(), np.asarray(right).copy()))

        def pause(self):
            return None

        def resume(self):
            return None

        def close(self):
            return None

        def raise_if_faulted(self):
            return None

    fake = FakeTransport()
    provider = RecordingGripperProvider()
    mc = MotionController(fake, end_effector_provider=provider)
    mc.start()
    try:
        mc.set_gripper(75.0, 25.0)

        assert fake.gripper_calls == []
        np.testing.assert_allclose(provider.commands[0][0], [0.75])
        np.testing.assert_allclose(provider.commands[0][1], [0.25])
    finally:
        mc.disconnect()


def test_set_gripper_rejects_non_gripper_provider_before_transport_send():
    class BrainCoProvider(_ProviderContractMixin):
        name = "brainco2"
        width_per_side = 6

        def read_state(self, timeout=1.0, max_age=None):
            return np.zeros(12), {}

        def command(self, left, right):
            raise AssertionError("set_gripper must not route through BrainCo2")

        def pause(self):
            return None

        def resume(self):
            return None

        def close(self):
            return None

        def raise_if_faulted(self):
            return None

    fake = FakeTransport()
    mc = MotionController(fake, end_effector_provider=BrainCoProvider())
    mc.start()
    try:
        with pytest.raises(RuntimeError, match="selected end-effector provider"):
            mc.set_gripper(75.0, 25.0)

        assert fake.gripper_calls == []
    finally:
        mc.disconnect()


def test_design_protocol_provider_without_resume_or_max_age_can_be_used():
    class MinimalProvider(_ProviderContractMixin):
        name = "gripper"
        width_per_side = 1

        def __init__(self):
            self.read_timeouts = []
            self.validations = []
            self.commands = []
            self.pause_count = 0
            self.close_count = 0

        def read_state(self, timeout=1.0):
            self.read_timeouts.append(timeout)
            return np.array([0.25, 0.75]), {"source": "minimal"}

        def validate_command(self, left, right):
            left = np.asarray(left, dtype=np.float64) + 0.1
            right = np.asarray(right, dtype=np.float64) + 0.2
            self.validations.append((left.copy(), right.copy()))
            return left, right

        def command(self, left, right):
            self.commands.append((np.asarray(left).copy(), np.asarray(right).copy()))

        def pause(self):
            self.pause_count += 1

        def close(self):
            self.close_count += 1

        def raise_if_faulted(self):
            return None

    provider = MinimalProvider()
    controller = MotionController(
        FakeTransport(),
        arm_strategy=RecordingArmStrategy(),
        end_effector_provider=provider,
        state_max_age=0.23,
        publish_rate=200.0,
    )
    controller.start()
    try:
        state, metadata = controller.get_end_effector_state(
            timeout=0.4,
            max_age=0.2,
        )
        controller.command_end_effector(np.array([0.1]), np.array([0.2]))
        controller.pause(timeout=0.5)
        controller.resume_from_state(np.zeros(JointIndex.STATE_DIM))

        np.testing.assert_array_equal(state, [0.25, 0.75])
        assert metadata == {"source": "minimal"}
        assert provider.read_timeouts == [2.0, 0.4]
        np.testing.assert_allclose(provider.commands[-1][0], [0.2])
        np.testing.assert_allclose(provider.commands[-1][1], [0.4])
        assert provider.pause_count == 1
        assert not hasattr(provider, "resume")
    finally:
        controller.disconnect()


def test_get_head_position_delegates():
    state18 = np.zeros(JointIndex.STATE_DIM)
    state18[JointIndex.HEAD] = [1.2, -0.3]
    fake = FakeTransport(initial_state=state18)
    mc = MotionController(transport=fake, publish_rate=200.0)
    mc.start()
    try:
        np.testing.assert_allclose(mc.get_head_position(), [1.2, -0.3])
    finally:
        mc.disconnect()


def test_motion_package_exports_arm_strategy_types():
    from tron2_env.motion import ArmStrategy, ServoJStrategy, ServoPStrategy

    assert ArmStrategy is not None
    assert ServoJStrategy.mode == "servoj"
    assert ServoPStrategy.mode == "servop"


def test_servoj_prefers_independent_arm_head_state_before_publish():
    fake = FakeDualModeTransport(measured_head=[0.2, -0.3])
    fake.arm_head_state[:14] = np.arange(14, dtype=np.float64) / 10.0
    fake._fail_state = True
    mc = MotionController(fake, publish_rate=200.0)

    mc.start()
    try:
        wait_for(lambda: bool(fake.joint_commands))
        np.testing.assert_allclose(fake.joint_commands[0], fake.arm_head_state)
        assert fake.arm_head_reads == 1
        assert fake.joint_state_reads == 0
        assert fake.pose_commands == []
    finally:
        mc.disconnect()


def test_servoj_falls_back_to_legacy_18d_state_in_canonical_order():
    state18 = np.arange(JointIndex.STATE_DIM, dtype=np.float64)
    fake = FakeTransport(initial_state=state18)
    mc = MotionController(fake, publish_rate=200.0)

    mc.start()
    try:
        wait_for(lambda: bool(fake.sends()))
        np.testing.assert_allclose(
            fake.sends()[0][1],
            np.concatenate(
                (
                    state18[JointIndex.LEFT_ARM],
                    state18[JointIndex.RIGHT_ARM],
                    state18[JointIndex.HEAD],
                )
            ),
        )
    finally:
        mc.disconnect()


@pytest.mark.parametrize(
    ("interpolator", "error_type", "message"),
    [
        (object(), TypeError, "JointInterpolator"),
        (PoseInterpolator(), ValueError, "PoseInterpolator"),
    ],
)
def test_servoj_rejects_invalid_interpolators_at_construction(
    interpolator, error_type, message
):
    with pytest.raises(error_type, match=message):
        MotionController(FakeTransport(), interpolator=interpolator)


def test_servop_controller_seeds_from_requested_measured_pose_before_publish():
    fake = FakeDualModeTransport()
    fake.pose_state = valid_pose(offset=0.25)
    mc = MotionController(fake, arm_mode="servop", publish_rate=200.0)

    mc.start()
    try:
        wait_for(lambda: bool(fake.pose_commands))
        np.testing.assert_allclose(fake.pose_commands[0], fake.pose_state)
        assert fake.requested_pose_guids == ["pose-1"]
        assert fake.requested_pose_timeouts == [2.0]
        assert fake.joint_commands == []
    finally:
        mc.disconnect()


def test_servop_start_seeds_from_atomic_pose_when_latest_is_already_overwritten():
    class AtomicSeedTransport(FakeDualModeTransport):
        def __init__(self):
            super().__init__()
            self.captured_pose = valid_pose(offset=1.0)
            self.pose_state = valid_pose(offset=2.0)
            self.pose_get_calls = []

        def get_pose_state(
            self,
            timeout: float = 1.0,
            max_age=None,
            expected_guid=None,
        ):
            self.pose_get_calls.append((max_age, expected_guid))
            assert expected_guid == self.pose_guid
            return {
                "timestamp": 1,
                "states": self.captured_pose.copy(),
                "guid": expected_guid,
            }

    transport = AtomicSeedTransport()
    controller = MotionController(transport, arm_mode="servop", publish_rate=200.0)

    controller.start()
    try:
        wait_for(lambda: bool(transport.pose_commands))
        np.testing.assert_allclose(
            transport.pose_commands[0],
            transport.captured_pose,
        )
        assert transport.pose_get_calls == [(0.5, "pose-1")]
    finally:
        controller.disconnect()


def test_servop_strategy_propagates_expected_guid_getter_error_once():
    from tron2_env.motion import ServoPStrategy

    sentinel = StateError("atomic pose read failed")

    class FailingAtomicTransport(FakeDualModeTransport):
        def __init__(self):
            super().__init__()
            self.pose_reads = 0

        def get_pose_state(
            self,
            timeout: float = 1.0,
            max_age=None,
            expected_guid=None,
        ):
            self.pose_reads += 1
            raise sentinel

    transport = FailingAtomicTransport()
    strategy = ServoPStrategy(transport)

    with pytest.raises(StateError) as exc_info:
        strategy.read_measured_target(timeout=0.1, max_age=0.02)

    assert exc_info.value is sentinel
    assert transport.pose_reads == 1


@pytest.mark.parametrize("cached_guid", ["old-pose", None])
def test_servop_waits_for_pose_response_matching_new_request(cached_guid):
    class DelayedPoseTransport(FakeDualModeTransport):
        def __init__(self):
            super().__init__()
            self._pose_snapshot_lock = threading.Lock()
            self.set_pose_snapshot(valid_pose(offset=-1.0), cached_guid)

        def set_pose_snapshot(self, states, guid):
            with self._pose_snapshot_lock:
                self.pose_state = np.asarray(states, dtype=np.float64).copy()
                self.pose_guid = guid

        def request_pose_state(self):
            guid = f"pose-{len(self.requested_pose_guids) + 1}"
            self.requested_pose_guids.append(guid)
            return guid

        def get_pose_state(self, timeout: float = 1.0, max_age=None):
            with self._pose_snapshot_lock:
                return {
                    "timestamp": len(self.requested_pose_guids),
                    "states": self.pose_state.copy(),
                    "guid": self.pose_guid,
                }

    fake = DelayedPoseTransport()
    expected = valid_pose(offset=2.0)
    mc = MotionController(fake, arm_mode="servop", publish_rate=200.0)
    start_errors = []
    starter = threading.Thread(
        target=lambda: _capture_exception(mc.start, start_errors),
        daemon=True,
    )

    starter.start()
    wait_for(lambda: fake.requested_pose_guids == ["pose-1"])
    time.sleep(0.02)
    assert fake.pose_commands == []

    fake.set_pose_snapshot(expected, "pose-1")
    starter.join(timeout=1.0)
    try:
        assert not starter.is_alive()
        assert start_errors == []
        wait_for(lambda: bool(fake.pose_commands))
        np.testing.assert_allclose(fake.pose_commands[0], expected)
    finally:
        mc.disconnect()


def test_servop_uses_one_getter_result_for_pose_guid_and_freshness():
    class AtomicPoseTransport(FakeDualModeTransport):
        def __init__(self):
            super().__init__()
            self.pose_state = valid_pose(offset=2.0)
            self.get_calls = []

        def request_pose_state(self):
            self.requested_pose_guids.append("pose-new")
            return "pose-new"

        def get_pose_state(
            self,
            timeout: float = 1.0,
            max_age=None,
            expected_guid=None,
        ):
            self.get_calls.append((max_age, expected_guid))
            if len(self.get_calls) > 1:
                raise AssertionError("pose getter must return one atomic snapshot")
            return {
                "timestamp": 2,
                "states": self.pose_state.copy(),
                "guid": "pose-new",
            }

    transport = AtomicPoseTransport()
    controller = MotionController(transport, arm_mode="servop")

    state, metadata = controller.get_arm_state(timeout=0.1, max_age=0.02)

    np.testing.assert_allclose(state, transport.pose_state)
    assert metadata["guid"] == "pose-new"
    assert transport.get_calls == [(0.02, "pose-new")]


def test_servop_legacy_getter_retries_stale_cached_pose_until_matching_pose():
    class LegacyStalePoseTransport(FakeDualModeTransport):
        def __init__(self):
            super().__init__()
            self.pose_reads = 0

        def request_pose_state(self):
            return "pose-new"

        def get_pose_state(self, timeout: float = 1.0, max_age=None):
            self.pose_reads += 1
            if self.pose_reads == 1:
                self.pose_guid = "pose-old"
                raise StateError("pose state is stale")
            self.pose_guid = "pose-new"
            return super().get_pose_state(timeout=timeout, max_age=max_age)

    transport = LegacyStalePoseTransport()
    controller = MotionController(transport, arm_mode="servop")

    state, metadata = controller.get_arm_state(timeout=0.1, max_age=0.02)

    np.testing.assert_allclose(state, transport.pose_state)
    assert metadata["guid"] == "pose-new"
    assert transport.pose_reads == 2


def test_servop_legacy_getter_that_stays_stale_fails_at_total_deadline(
    monkeypatch,
):
    class AlwaysStalePoseTransport(FakeDualModeTransport):
        def __init__(self):
            super().__init__()
            self.pose_reads = 0

        def request_pose_state(self):
            return "pose-new"

        def get_pose_state(self, timeout: float = 1.0, max_age=None):
            self.pose_reads += 1
            raise StateError("pose state is stale")

    now = [0.0]
    monkeypatch.setattr(arm_strategy_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        arm_strategy_module.time,
        "sleep",
        lambda duration: now.__setitem__(0, now[0] + duration),
    )
    transport = AlwaysStalePoseTransport()
    controller = MotionController(transport, arm_mode="servop")

    with pytest.raises(StateError, match="timed out waiting"):
        controller.get_arm_state(timeout=0.002, max_age=0.02)

    assert transport.pose_reads == 2


def test_servop_public_state_read_passes_timeout_to_pose_request():
    transport = FakeDualModeTransport()
    controller = MotionController(transport, arm_mode="servop")

    state, _metadata = controller.get_arm_state(timeout=0.25)

    np.testing.assert_allclose(state, transport.pose_state)
    assert transport.requested_pose_timeouts == [0.25]


def test_servop_rejects_invalid_pose_request_guid_without_publishing():
    class InvalidGuidTransport(FakeDualModeTransport):
        def request_pose_state(self):
            return ""

    fake = InvalidGuidTransport()
    mc = MotionController(fake, arm_mode="servop", publish_rate=200.0)

    with pytest.raises(RuntimeError, match="initial pose state"):
        mc.start()

    assert fake.pose_commands == []
    assert fake.joint_commands == []


def test_servop_rejects_legacy_linear_interpolator_override():
    fake = FakeDualModeTransport()

    with pytest.raises(ValueError, match="PoseInterpolator"):
        MotionController(
            fake,
            interpolator=LinearInterpolator(),
            arm_mode="servop",
        )


def _capture_exception(callable_, errors):
    try:
        callable_()
    except BaseException as exc:  # pragma: no cover - only populated on failure
        errors.append(exc)


@pytest.mark.parametrize(
    ("arm_mode", "method", "target"),
    [
        ("servop", "command_joints", np.zeros(JointIndex.SERVOJ_DIM)),
        ("servoj", "command_poses", (valid_pose()[:7], valid_pose()[7:])),
    ],
)
def test_mode_specific_command_apis_reject_the_wrong_arm_mode(
    arm_mode, method, target
):
    strategy = RecordingArmStrategy(mode=arm_mode)
    mc = MotionController(
        FakeTransport(),
        arm_mode=arm_mode,
        arm_strategy=strategy,
    )

    with pytest.raises(RuntimeError, match=arm_mode):
        if method == "command_joints":
            mc.command_joints(target)
        else:
            mc.command_poses(*target)

    assert strategy.destinations == []


def test_controller_arm_mode_cannot_change_after_construction():
    mc = MotionController(FakeTransport(), arm_mode="servoj")

    with pytest.raises(AttributeError):
        mc.arm_mode = "servop"


def test_servoj_combined_state_fallbacks_forward_freshness_and_slice_head():
    class CombinedOnlyTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.state_reads = []

        def get_joint_state(self, timeout: float = 1.0, max_age=None):
            self.state_reads.append((timeout, max_age))
            return {
                "states": np.arange(JointIndex.STATE_DIM, dtype=np.float64),
                "timestamp": 101,
                "received_at": time.monotonic(),
            }

        def get_head_position(self):
            raise AssertionError("head fallback must use the freshness-aware state")

    transport = CombinedOnlyTransport()
    controller = MotionController(
        transport,
        arm_strategy=RecordingArmStrategy(mode="servoj"),
    )
    gripper = GripperProvider(transport)

    arms, arm_metadata = controller.get_arm_state(timeout=0.3, max_age=0.2)
    head, head_metadata = controller.get_head_state(timeout=0.3, max_age=0.2)
    grippers, gripper_metadata = gripper.read_state(timeout=0.3, max_age=0.2)

    state = np.arange(JointIndex.STATE_DIM, dtype=np.float64)
    np.testing.assert_array_equal(
        arms,
        np.concatenate((state[JointIndex.LEFT_ARM], state[JointIndex.RIGHT_ARM])),
    )
    np.testing.assert_array_equal(head, state[JointIndex.HEAD])
    np.testing.assert_array_equal(
        grippers,
        state[[JointIndex.LEFT_GRIPPER, JointIndex.RIGHT_GRIPPER]],
    )
    assert arm_metadata["timestamp"] == 101
    assert head_metadata["timestamp"] == 101
    assert gripper_metadata["timestamp"] == 101
    assert transport.state_reads == [(0.3, 0.2)] * 3


@pytest.mark.parametrize(
    ("received_at", "message"),
    [
        (None, "invalid received_at"),
        ("invalid", "invalid received_at"),
        (np.nan, "invalid received_at"),
        ("stale", "is stale"),
    ],
)
def test_legacy_combined_state_requires_valid_received_at_when_max_age_set(
    received_at,
    message,
):
    class LegacyCombinedTransport(FakeTransport):
        def get_joint_state(self, timeout=1.0):
            payload = {
                "states": np.arange(JointIndex.STATE_DIM, dtype=np.float64),
                "timestamp": 101,
            }
            if received_at is not None:
                payload["received_at"] = (
                    time.monotonic() - 1.0
                    if received_at == "stale"
                    else received_at
                )
            return payload

    controller = MotionController(
        LegacyCombinedTransport(),
        arm_strategy=RecordingArmStrategy(mode="servoj"),
    )

    state = controller.get_joint_states(timeout=0.3, max_age=None)
    assert state["timestamp"] == 101

    with pytest.raises(StateError, match=message):
        controller.get_joint_states(timeout=0.3, max_age=0.2)


def test_legacy_mobile_getter_signature_still_enforces_component_freshness():
    class LegacyMobileTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.mobile_reads = []
            self.chassis_received_at = time.monotonic()
            self.lifter_received_at = time.monotonic()

        def get_chassis_lifter_state(self, timeout=1.0):
            self.mobile_reads.append(timeout)
            return {
                "linear_velocity": 0.1,
                "angular_velocity": -0.2,
                "lifter_state": 30.0,
                "chassis_received_at": self.chassis_received_at,
                "lifter_received_at": self.lifter_received_at,
            }

    transport = LegacyMobileTransport()
    controller = MotionController(
        transport,
        arm_strategy=RecordingArmStrategy(mode="servoj"),
    )

    state = controller.get_chassis_lifter_state(timeout=0.3, max_age=0.2)
    assert state["lifter_state"] == 30.0
    assert transport.mobile_reads == [0.3]

    transport.chassis_received_at = time.monotonic() - 1.0
    with pytest.raises(StateError, match="chassis state is stale"):
        controller.get_chassis_lifter_state(timeout=0.3, max_age=0.2)


@pytest.mark.parametrize(
    ("invalid_component", "message"),
    [
        ("missing_chassis", "chassis state has invalid received_at"),
        ("missing_lifter", "lifter state has invalid received_at"),
        ("invalid_chassis", "chassis state has invalid received_at"),
        ("invalid_lifter", "lifter state has invalid received_at"),
        ("stale_chassis", "chassis state is stale"),
        ("stale_lifter", "lifter state is stale"),
    ],
)
def test_legacy_mobile_requires_each_component_received_at_when_max_age_set(
    invalid_component,
    message,
):
    class LegacyMobileTransport(FakeTransport):
        def get_chassis_lifter_state(self, timeout=1.0):
            now = time.monotonic()
            payload = {
                "linear_velocity": 0.1,
                "angular_velocity": -0.2,
                "lifter_state": 30.0,
                "chassis_received_at": now,
                "lifter_received_at": now,
            }
            component = invalid_component.rsplit("_", 1)[-1]
            key = f"{component}_received_at"
            if invalid_component.startswith("missing"):
                payload.pop(key)
            elif invalid_component.startswith("invalid"):
                payload[key] = np.nan
            else:
                payload[key] = now - 1.0
            return payload

    controller = MotionController(
        LegacyMobileTransport(),
        arm_strategy=RecordingArmStrategy(mode="servoj"),
    )

    state = controller.get_chassis_lifter_state(timeout=0.3, max_age=None)
    assert state["lifter_state"] == 30.0

    with pytest.raises(StateError, match=message):
        controller.get_chassis_lifter_state(timeout=0.3, max_age=0.2)


@pytest.mark.parametrize(
    "target",
    [
        np.zeros(15),
        np.zeros(17),
        np.full(16, np.nan),
        np.full(16, np.inf),
    ],
)
def test_command_joints_rejects_wrong_shape_and_nonfinite_values(target):
    strategy = RecordingArmStrategy(mode="servoj")
    mc = MotionController(FakeTransport(), arm_strategy=strategy)
    mc.start()
    try:
        with pytest.raises(ValueError, match="joint target"):
            mc.command_joints(target)

        assert strategy.destinations == []
    finally:
        mc.disconnect()


@pytest.mark.parametrize(
    ("left", "right", "head"),
    [
        (np.zeros(6), np.zeros(7), np.zeros(2)),
        (np.zeros(7), np.full(7, np.nan), np.zeros(2)),
        (np.zeros(7), np.zeros(7), np.zeros(1)),
        (np.zeros(7), np.zeros(7), np.array([0.0, np.inf])),
    ],
)
def test_command_arm_validates_all_components_before_changing_targets(
    left, right, head
):
    strategy = RecordingArmStrategy(mode="servop")
    scheduler = RecordingHeadScheduler()
    mc = MotionController(
        FakeTransport(),
        arm_mode="servop",
        arm_strategy=strategy,
        head_scheduler=scheduler,
    )
    mc.start()
    try:
        with pytest.raises(ValueError):
            mc.command_arm(left, right, head=head)

        assert strategy.destinations == []
        assert scheduler.submissions == []
    finally:
        mc.disconnect()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("publish_rate", 0.0),
        ("publish_rate", -1.0),
        ("publish_rate", np.nan),
        ("publish_rate", np.inf),
        ("eta_default", 0.0),
        ("eta_default", -1.0),
        ("eta_default", np.nan),
        ("eta_default", np.inf),
    ],
)
def test_controller_rejects_non_positive_or_nonfinite_timing(name, value):
    kwargs = {name: value}
    with pytest.raises(ValueError, match=name):
        MotionController(FakeTransport(), **kwargs)


@pytest.mark.parametrize("eta", [0.0, -1.0, np.nan, np.inf])
def test_command_rejects_non_positive_or_nonfinite_eta_without_retargeting(eta):
    strategy = RecordingArmStrategy(mode="servoj")
    mc = MotionController(FakeTransport(), arm_strategy=strategy)
    mc.start()
    try:
        with pytest.raises(ValueError, match="eta"):
            mc.command_joints(np.zeros(JointIndex.SERVOJ_DIM), eta=eta)

        assert strategy.destinations == []
    finally:
        mc.disconnect()


def test_servoj_component_command_holds_latest_measured_head_when_omitted():
    fake = FakeDualModeTransport(measured_head=[0.2, -0.3])
    strategy = RecordingArmStrategy(mode="servoj", measured=fake.arm_head_state)
    mc = MotionController(fake, arm_strategy=strategy, eta_default=0.05)
    mc.start()
    try:
        fake._state[JointIndex.HEAD] = [0.8, -0.9]
        left = np.arange(7, dtype=np.float64)
        right = np.arange(7, dtype=np.float64) + 10.0

        mc.command_arm(left, right)

        target, eta = strategy.destinations[-1]
        np.testing.assert_allclose(target, np.concatenate((left, right, [0.8, -0.9])))
        assert eta == pytest.approx(0.05)
    finally:
        mc.disconnect()


def test_servop_component_command_copies_poses_and_submits_optional_head():
    strategy = RecordingArmStrategy(mode="servop")
    scheduler = RecordingHeadScheduler()
    mc = MotionController(
        FakeTransport(),
        arm_mode="servop",
        arm_strategy=strategy,
        head_scheduler=scheduler,
        eta_default=0.05,
    )
    left = valid_pose()[:7]
    right = valid_pose()[7:]
    head = np.array([0.2, -0.4])
    mc.start()
    try:
        mc.command_poses(left, right)
        mc.command_head(head)
        left[:] = 9.0
        right[:] = 8.0
        head[:] = 7.0

        np.testing.assert_allclose(strategy.destinations[0][0], valid_pose())
        np.testing.assert_allclose(scheduler.submissions[0], [0.2, -0.4])
    finally:
        mc.disconnect()


def test_servop_head_command_requires_configured_scheduler():
    mc = MotionController(FakeDualModeTransport(), arm_mode="servop")
    mc.start()
    try:
        with pytest.raises(RuntimeError, match="head scheduler"):
            mc.command_head(np.zeros(2))

        with pytest.raises(RuntimeError, match="head scheduler"):
            mc.command_arm(valid_pose()[:7], valid_pose()[7:], head=np.zeros(2))
    finally:
        mc.disconnect()


def test_pause_acknowledges_only_after_arm_and_head_tick_are_quiescent():
    strategy = RecordingArmStrategy(mode="servop")
    scheduler = RecordingHeadScheduler(block_tick=True)
    mc = MotionController(
        FakeTransport(),
        arm_mode="servop",
        arm_strategy=strategy,
        head_scheduler=scheduler,
        publish_rate=200.0,
    )
    mc.start()
    assert scheduler.tick_entered.wait(timeout=1.0)
    pause_errors = []
    pauser = threading.Thread(
        target=lambda: _capture_exception(
            lambda: mc.pause(timeout=0.5), pause_errors
        ),
        daemon=True,
    )

    pauser.start()
    time.sleep(0.02)
    assert pauser.is_alive()
    scheduler.release_tick.set()
    pauser.join(timeout=1.0)
    try:
        assert pause_errors == []
        assert not pauser.is_alive()
        publish_count = strategy.publishes
        tick_count = scheduler.tick_count
        time.sleep(0.03)
        assert strategy.publishes == publish_count
        assert scheduler.tick_count == tick_count
        assert scheduler.pause_count == 1
    finally:
        mc.disconnect()


def test_pause_rejects_oversized_timeout_without_changing_pause_state():
    strategy = RecordingArmStrategy(mode="servop")
    scheduler = RecordingHeadScheduler()
    controller = MotionController(
        FakeTransport(),
        arm_mode="servop",
        arm_strategy=strategy,
        head_scheduler=scheduler,
        publish_rate=200.0,
    )
    controller.start()
    try:
        wait_for(lambda: strategy.publishes > 0)
        pause_generation = controller._pause_generation
        paused_generation = controller._paused_generation
        scheduler_pauses = scheduler.pause_count

        with pytest.raises(ValueError, match="threading.TIMEOUT_MAX"):
            controller.pause(timeout=threading.TIMEOUT_MAX + 1.0)

        assert controller._publish_allowed.is_set()
        assert controller._pause_generation == pause_generation
        assert controller._paused_generation == paused_generation
        assert scheduler.pause_count == scheduler_pauses
    finally:
        controller.disconnect()


def test_rapid_resume_then_pause_rejects_stale_loop_acknowledgement():
    publisher_trace_installed = threading.Event()
    paused_branch_armed = threading.Event()
    paused_branch_entered = threading.Event()
    release_paused_branch = threading.Event()
    active_branch_armed = threading.Event()
    active_branch_entered = threading.Event()
    active_branch_released = threading.Event()
    second_pause_gate_cleared = threading.Event()
    second_pause_condition_waiting = threading.Event()
    release_active_branch = threading.Event()
    second_pause_returned = threading.Event()
    post_gate_publish_completed = threading.Event()
    abort_trace_waits = threading.Event()
    trace_errors = []

    class OrderedArmStrategy(RecordingArmStrategy):
        def publish_current(self):
            super().publish_current()
            if active_branch_released.is_set():
                post_gate_publish_completed.set()

    loop_code = MotionController._publish_loop.__code__
    loop_source, first_line = inspect.getsourcelines(MotionController._publish_loop)
    pause_code = MotionController.pause.__code__
    pause_source, pause_first_line = inspect.getsourcelines(MotionController.pause)
    second_pauser_name = "MotionController-second-pause"

    def line_number(source, source_first_line, fragment):
        matches = [
            source_first_line + offset
            for offset, line in enumerate(source)
            if fragment in line
        ]
        assert len(matches) == 1, f"expected one line containing {fragment!r}"
        return matches[0]

    paused_branch_line = line_number(
        loop_source,
        first_line,
        "if not publish_allowed:",
    )
    active_branch_line = line_number(
        loop_source,
        first_line,
        "self._arm_strategy.publish_current()",
    )
    second_pause_gate_line = line_number(
        pause_source,
        pause_first_line,
        "self._loop_wakeup.set()",
    )

    def wait_for_trace_gate(gate, message):
        if gate.wait(timeout=1.0):
            return True
        trace_errors.append(message)
        while not abort_trace_waits.wait(timeout=0.1):
            pass
        return False

    def trace_publish_loop(frame, event, _arg):
        if frame.f_code is loop_code:
            publisher_trace_installed.set()
            if event != "line":
                return trace_publish_loop
            if (
                frame.f_lineno == paused_branch_line
                and paused_branch_armed.is_set()
                and not paused_branch_entered.is_set()
            ):
                paused_branch_entered.set()
                wait_for_trace_gate(
                    release_paused_branch,
                    "timed out releasing stale paused branch",
                )
            elif (
                frame.f_lineno == active_branch_line
                and active_branch_armed.is_set()
                and not active_branch_entered.is_set()
            ):
                active_branch_entered.set()
                if wait_for_trace_gate(
                    second_pause_gate_cleared,
                    "second pause did not clear the publish gate",
                ) and wait_for_trace_gate(
                    release_active_branch,
                    "second pause neither returned nor entered its condition wait",
                ):
                    active_branch_released.set()
        elif (
            threading.current_thread().name == second_pauser_name
            and frame.f_code is pause_code
            and event == "line"
            and frame.f_lineno == second_pause_gate_line
        ):
            second_pause_gate_cleared.set()
        elif (
            threading.current_thread().name == second_pauser_name
            and frame.f_code is threading.Condition.wait.__code__
            and event == "call"
            and second_pause_gate_cleared.is_set()
        ):
            second_pause_condition_waiting.set()
            release_active_branch.set()
        return trace_publish_loop

    strategy = OrderedArmStrategy()
    controller = MotionController(
        FakeTransport(),
        arm_strategy=strategy,
        publish_rate=200.0,
    )
    second_pause_errors = []
    publishes_when_second_pause_returned = []
    pauser = None
    started = False
    get_thread_trace = getattr(threading, "gettrace", None)
    previous_trace = (
        get_thread_trace()
        if get_thread_trace is not None
        else getattr(threading, "_trace_hook", None)
    )

    threading.settrace(trace_publish_loop)
    try:
        controller.start()
        started = True
        assert publisher_trace_installed.wait(timeout=1.0), (
            "publish thread did not install the trace hook"
        )
        controller.pause(timeout=0.5)
        paused_branch_armed.set()
        assert paused_branch_entered.wait(timeout=1.0), (
            "publish loop did not enter the armed paused branch"
        )

        controller.resume_from_state()
        active_branch_armed.set()
        release_paused_branch.set()
        assert active_branch_entered.wait(timeout=1.0), (
            "publish loop did not enter the armed active branch"
        )

        def pause_again():
            try:
                controller.pause(timeout=0.5)
            except BaseException as exc:
                second_pause_errors.append(exc)
            publishes_when_second_pause_returned.append(strategy.publishes)
            second_pause_returned.set()
            if not second_pause_condition_waiting.is_set():
                release_active_branch.set()

        pauser = threading.Thread(
            target=pause_again,
            daemon=True,
            name=second_pauser_name,
        )
        pauser.start()

        assert second_pause_gate_cleared.wait(timeout=1.0), (
            "second pause did not clear the publish gate"
        )
        assert release_active_branch.wait(timeout=1.0), (
            "second pause neither returned nor entered its condition wait"
        )
        assert post_gate_publish_completed.wait(timeout=1.0), (
            "publish loop did not finish the gated active publication"
        )
        assert second_pause_returned.wait(timeout=1.0), (
            "second pause did not return after the loop acknowledged it"
        )
        pauser.join(timeout=1.0)

        assert trace_errors == []
        assert second_pause_errors == []
        assert not pauser.is_alive()
        assert strategy.publishes == publishes_when_second_pause_returned[0]
    finally:
        abort_trace_waits.set()
        release_paused_branch.set()
        release_active_branch.set()
        try:
            if pauser is not None:
                pauser.join(timeout=1.0)
            if started:
                controller.disconnect()
        finally:
            threading.settrace(previous_trace)


def test_servop_resume_ignores_legacy_state_and_reseeds_from_fresh_pose():
    fake = FakeDualModeTransport()
    mc = MotionController(fake, arm_mode="servop", publish_rate=200.0)
    mc.start()
    try:
        wait_for(lambda: bool(fake.pose_commands))
        mc.pause(timeout=0.5)
        fake.pose_state = valid_pose(offset=1.0)
        before = len(fake.pose_commands)

        mc.resume_from_state(np.full(JointIndex.STATE_DIM, 99.0))

        wait_for(lambda: len(fake.pose_commands) > before)
        np.testing.assert_allclose(fake.pose_commands[before], fake.pose_state)
        assert fake.requested_pose_guids == ["pose-1", "pose-2"]
    finally:
        mc.disconnect()


def test_servop_reconnect_seed_passes_state_max_age_to_pose_getter():
    class ReconnectingPoseTransport(ReconnectingFakeTransport):
        def __init__(self):
            super().__init__()
            self.pose_get_calls = []

        def get_pose_state(
            self,
            timeout: float = 1.0,
            max_age=None,
            expected_guid=None,
        ):
            self.pose_get_calls.append((max_age, expected_guid))
            return {
                "timestamp": len(self.requested_pose_guids),
                "states": self.pose_state.copy(),
                "guid": expected_guid,
            }

    transport = ReconnectingPoseTransport()
    controller = MotionController(
        transport,
        arm_mode="servop",
        publish_rate=200.0,
        state_max_age=0.23,
    )
    controller.start()
    try:
        transport.lose_connection()
        controller.pause(timeout=0.5)
        transport.reconnect()
        controller.resume_from_state()

        assert transport.pose_get_calls == [
            (0.23, "pose-1"),
            (0.23, "pose-2"),
        ]
    finally:
        controller.disconnect()


def test_servoj_resume_without_argument_reads_fresh_measured_state():
    fake = FakeDualModeTransport()
    mc = MotionController(fake, publish_rate=200.0)
    mc.start()
    try:
        wait_for(lambda: bool(fake.joint_commands))
        mc.pause(timeout=0.5)
        fake.arm_head_state = np.arange(JointIndex.SERVOJ_DIM, dtype=np.float64)
        before = len(fake.joint_commands)

        mc.resume_from_state()

        wait_for(lambda: len(fake.joint_commands) > before)
        np.testing.assert_allclose(fake.joint_commands[before], fake.arm_head_state)
        assert fake.arm_head_reads == 2
    finally:
        mc.disconnect()


def test_reconnect_latches_pause_rejects_commands_and_reseeds_before_publish():
    transport = ReconnectingFakeTransport()
    provider = GripperProvider(transport)
    controller = MotionController(
        transport,
        end_effector_provider=provider,
        publish_rate=200.0,
        eta_default=0.01,
    )
    controller.start()
    try:
        stale_target = np.full(JointIndex.SERVOJ_DIM, 5.0)
        controller.command_joints(stale_target)
        wait_for(
            lambda: bool(transport.joint_commands)
            and np.allclose(transport.joint_commands[-1], stale_target)
        )

        transport.lose_connection()
        controller.pause(timeout=0.5)
        publishes_after_loss = len(transport.joint_commands)
        transport.reconnect()

        assert controller.is_paused
        assert len(transport.joint_commands) == publishes_after_loss
        blocked_commands = (
            lambda: controller.raise_if_faulted(),
            lambda: controller.command_arm(np.zeros(7), np.zeros(7)),
            lambda: controller.command_end_effector(np.zeros(1), np.zeros(1)),
            lambda: controller.set_gripper(0.0, 0.0),
            lambda: controller.command_chassis(0.1, -0.2),
            lambda: controller.stream_lifter_position(100.0),
        )
        for command in blocked_commands:
            with pytest.raises(RuntimeError, match="resume_from_state"):
                command()

        controller.stop_chassis()
        controller.stop_lifter_stream()
        assert transport.chassis_commands == [(0.0, 0.0)]
        assert transport.lifter_stream_stops == 1

        fresh = np.arange(JointIndex.SERVOJ_DIM, dtype=np.float64) + 10.0
        transport.arm_head_state = fresh.copy()
        transport._state = list(np.arange(JointIndex.STATE_DIM, dtype=np.float64))
        controller.resume_from_state(np.full(JointIndex.STATE_DIM, 99.0))

        wait_for(lambda: len(transport.joint_commands) > publishes_after_loss)
        np.testing.assert_allclose(
            transport.joint_commands[publishes_after_loss],
            fresh,
        )
    finally:
        controller.disconnect()

    assert transport.connection_callback is None
    assert transport.connection_callback_unregistrations == 1


def test_reconnect_resume_requires_fresh_arm_head_and_end_effector_state():
    class RecordingProvider(_ProviderContractMixin):
        name = "gripper"
        width_per_side = 1

        def __init__(self):
            self.reads = 0
            self.pause_count = 0
            self.commands = []

        def read_state(self, timeout=1.0):
            self.reads += 1
            return np.array([0.25, 0.75]), {"received_at": time.monotonic()}

        def command(self, left, right):
            self.commands.append((np.asarray(left), np.asarray(right)))

        def pause(self):
            self.pause_count += 1

        def close(self):
            return None

        def raise_if_faulted(self):
            return None

    transport = ReconnectingFakeTransport()
    strategy = RecordingArmStrategy(mode="servop")
    scheduler = RecordingHeadScheduler()
    provider = RecordingProvider()
    controller = MotionController(
        transport,
        arm_mode="servop",
        arm_strategy=strategy,
        head_scheduler=scheduler,
        end_effector_provider=provider,
        publish_rate=200.0,
    )
    controller.start()
    try:
        transport.lose_connection()
        controller.pause(timeout=0.5)
        transport.reconnect()
        strategy.measured = valid_pose(offset=3.0)
        transport.arm_head_state = np.arange(
            JointIndex.SERVOJ_DIM,
            dtype=np.float64,
        )

        controller.resume_from_state()

        np.testing.assert_allclose(strategy.resets[-1], strategy.measured)
        assert strategy.reads == 2
        assert transport.arm_head_reads == 1
        assert provider.reads == 2
        assert provider.pause_count >= 1
        assert not hasattr(provider, "resume")
        assert scheduler.reconnect_pause_count >= 1
        assert scheduler.resume_count == 1
    finally:
        controller.disconnect()


def test_reconnect_callback_does_not_wait_for_in_flight_command():
    command_entered = threading.Event()
    release_command = threading.Event()
    reconnect_returned = threading.Event()

    class BlockingReconnectTransport(ReconnectingFakeTransport):
        def __init__(self):
            super().__init__()
            self._command_generation_local = threading.local()
            self.rejected_gripper_commands = 0

        @contextmanager
        def bind_command_generation(self, generation):
            previous = getattr(
                self._command_generation_local,
                "expected_generation",
                None,
            )
            self._command_generation_local.expected_generation = generation
            try:
                yield
            finally:
                self._command_generation_local.expected_generation = previous

        def set_gripper(self, left, right):
            expected_generation = getattr(
                self._command_generation_local,
                "expected_generation",
                None,
            )
            if expected_generation != self.session_generation:
                self.rejected_gripper_commands += 1
                raise CommandError("stale command generation")
            super().set_gripper(left, right)

    class BlockingGripperProvider(GripperProvider):
        def command(self, left, right):
            with self._lifecycle_lock:
                self._raise_if_command_blocked()
                left_target, right_target = self.validate_command(left, right)
                command_entered.set()
                assert release_command.wait(timeout=1.0)
                self._transport.set_gripper(
                    float(left_target[0]) * 100.0,
                    float(right_target[0]) * 100.0,
                )

    transport = BlockingReconnectTransport()
    provider = BlockingGripperProvider(transport)
    controller = MotionController(
        transport,
        end_effector_provider=provider,
        publish_rate=200.0,
    )
    command_errors = []
    reconnect_errors = []
    command_thread = threading.Thread(
        target=lambda: _capture_exception(
            lambda: controller.command_end_effector(
                np.array([0.1]),
                np.array([0.9]),
            ),
            command_errors,
        ),
        daemon=True,
    )
    reconnect_thread = threading.Thread(
        target=lambda: (
            _capture_exception(transport.reconnect, reconnect_errors),
            reconnect_returned.set(),
        ),
        daemon=True,
    )

    controller.start()
    command_thread.start()
    assert command_entered.wait(timeout=0.5)
    transport.lose_connection()
    reconnect_thread.start()

    try:
        assert reconnect_returned.wait(timeout=0.5), (
            "connection callback waited for an admitted command"
        )
        with pytest.raises(RuntimeError, match="commands are still in flight"):
            controller.resume_from_state()

        release_command.set()
        command_thread.join(timeout=1.0)
        controller.pause(timeout=0.5)
        controller.resume_from_state()
        controller.command_end_effector(
            np.array([0.25]),
            np.array([0.75]),
        )

        assert not controller.is_paused
        assert len(command_errors) == 1
        assert isinstance(command_errors[0], CommandError)
        assert transport.rejected_gripper_commands == 1
        assert transport.gripper_calls == [(25.0, 75.0)]
        assert reconnect_errors == []
    finally:
        release_command.set()
        command_thread.join(timeout=1.0)
        reconnect_thread.join(timeout=1.0)
        controller.disconnect()


def test_disconnect_commits_shutdown_while_resume_hook_is_blocked():
    resume_entered = threading.Event()
    transport_disconnected = threading.Event()

    class RecordingDisconnectTransport(FakeTransport):
        def disconnect(self):
            transport_disconnected.set()
            super().disconnect()

    class BlockingResumeProvider(_ProviderContractMixin):
        name = "gripper"
        width_per_side = 1

        def __init__(self):
            self.pause_count = 0

        def read_state(self, timeout=1.0, max_age=None):
            return np.zeros(2), {}

        def command(self, left, right):
            return None

        def pause(self):
            self.pause_count += 1

        def resume(self):
            resume_entered.set()
            assert transport_disconnected.wait(timeout=1.0)

        def close(self):
            return None

        def raise_if_faulted(self):
            return None

    transport = RecordingDisconnectTransport()
    provider = BlockingResumeProvider()
    controller = MotionController(
        transport,
        arm_strategy=RecordingArmStrategy(),
        end_effector_provider=provider,
        publish_rate=200.0,
    )
    resume_errors = []
    disconnect_errors = []
    resume_thread = threading.Thread(
        target=lambda: _capture_exception(
            controller.resume_from_state,
            resume_errors,
        ),
        daemon=True,
    )
    disconnect_thread = threading.Thread(
        target=lambda: _capture_exception(controller.disconnect, disconnect_errors),
        daemon=True,
    )

    controller.start()
    controller.pause(timeout=0.5)
    resume_thread.start()
    assert resume_entered.wait(timeout=0.5)
    disconnect_thread.start()

    assert transport_disconnected.wait(timeout=0.5), (
        "disconnect could not commit shutdown while resume hook was blocked"
    )
    resume_thread.join(timeout=1.0)
    disconnect_thread.join(timeout=1.0)

    assert not resume_thread.is_alive()
    assert not disconnect_thread.is_alive()
    assert len(resume_errors) == 1
    assert isinstance(resume_errors[0], RuntimeError)
    assert "disconnected while resuming" in str(resume_errors[0])
    assert disconnect_errors == []
    assert provider.pause_count == 2


def test_failed_resume_keeps_controller_and_scheduler_paused():
    strategy = RecordingArmStrategy(mode="servop")
    scheduler = RecordingHeadScheduler()
    mc = MotionController(
        FakeTransport(),
        arm_mode="servop",
        arm_strategy=strategy,
        head_scheduler=scheduler,
        publish_rate=200.0,
    )
    mc.start()
    mc.pause(timeout=0.5)

    def fail_read(timeout=2.0):
        raise StateError("fresh pose unavailable")

    strategy.read_measured_target = fail_read
    try:
        with pytest.raises(StateError, match="fresh pose unavailable"):
            mc.resume_from_state()
        assert mc.is_paused
        assert scheduler.resume_count == 0
    finally:
        mc.disconnect()


def test_scheduler_resume_failure_keeps_pause_acknowledged_for_retry():
    class RetryableScheduler(RecordingHeadScheduler):
        fail_resume = True

        def resume(self):
            self.resume_count += 1
            if self.fail_resume:
                raise RuntimeError("resume failed")

    strategy = RecordingArmStrategy(mode="servop")
    scheduler = RetryableScheduler()
    mc = MotionController(
        FakeTransport(),
        arm_mode="servop",
        arm_strategy=strategy,
        head_scheduler=scheduler,
        publish_rate=200.0,
    )
    mc.start()
    mc.pause(timeout=0.5)
    try:
        with pytest.raises(RuntimeError, match="resume failed"):
            mc.resume_from_state()
        assert mc.is_paused

        scheduler.fail_resume = False
        mc.resume_from_state()
        assert not mc.is_paused
    finally:
        mc.disconnect()


def test_resume_compensation_preserves_original_scheduler_error():
    scheduler_error = RuntimeError("scheduler resume failed")
    compensation_error = RuntimeError("provider re-pause failed")

    class FailingResumeScheduler(RecordingHeadScheduler):
        def resume(self):
            self.resume_count += 1
            raise scheduler_error

    class FailingCompensationProvider(_ProviderContractMixin):
        name = "gripper"
        width_per_side = 1

        def __init__(self):
            self.pause_count = 0
            self.resume_count = 0

        def read_state(self, timeout=1.0, max_age=None):
            return np.zeros(2), {}

        def command(self, left, right):
            return None

        def pause(self):
            self.pause_count += 1
            if self.pause_count == 2:
                raise compensation_error

        def resume(self):
            self.resume_count += 1

        def close(self):
            return None

        def raise_if_faulted(self):
            return None

    scheduler = FailingResumeScheduler()
    provider = FailingCompensationProvider()
    controller = MotionController(
        FakeTransport(),
        arm_mode="servop",
        arm_strategy=RecordingArmStrategy(mode="servop"),
        head_scheduler=scheduler,
        end_effector_provider=provider,
        publish_rate=200.0,
    )
    controller.start()
    controller.pause(timeout=0.5)

    try:
        with pytest.raises(RuntimeError) as raised:
            controller.resume_from_state()

        assert raised.value is scheduler_error
        assert controller.is_paused
        assert provider.resume_count == 1
        assert provider.pause_count == 2
        assert not controller._end_effector_pause_complete
    finally:
        controller.disconnect()


def test_disconnect_closes_scheduler_and_unregisters_callback_once():
    fake = FakeDualModeTransport()
    scheduler = RecordingHeadScheduler()
    fake.register_head_response_callback(scheduler.handle_response)
    mc = MotionController(
        fake,
        arm_mode="servop",
        head_scheduler=scheduler,
        publish_rate=200.0,
    )
    mc.start()
    mc.disconnect()
    mc.disconnect()

    assert scheduler.close_count == 1
    assert fake.head_callback_unregistrations == 1


def test_disconnect_aborts_blocked_provider_command_before_other_cleanup():
    transport_disconnected = threading.Event()
    provider_command_entered = threading.Event()
    provider_close_entered = threading.Event()
    release_provider_close = threading.Event()

    class AbortableTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.disconnect_count = 0

        def disconnect(self):
            self.disconnect_count += 1
            transport_disconnected.set()
            super().disconnect()

    class BlockingProvider(_ProviderContractMixin):
        name = "gripper"
        width_per_side = 1

        def __init__(self):
            self.command_count = 0

        def read_state(self, timeout=1.0, max_age=None):
            return np.zeros(2), {}

        def command(self, left, right):
            self.command_count += 1
            provider_command_entered.set()
            assert transport_disconnected.wait(timeout=1.0)
            raise RuntimeError("provider send aborted")

        def pause(self):
            return None

        def resume(self):
            return None

        def close(self):
            assert transport_disconnected.is_set()
            provider_close_entered.set()
            assert release_provider_close.wait(timeout=1.0)

        def raise_if_faulted(self):
            return None

    transport = AbortableTransport()
    provider = BlockingProvider()
    controller = MotionController(
        transport,
        arm_strategy=RecordingArmStrategy(),
        end_effector_provider=provider,
    )
    command_errors = []
    disconnect_errors = []
    command_thread = threading.Thread(
        target=lambda: _capture_exception(
            lambda: controller.command_end_effector(np.zeros(1), np.zeros(1)),
            command_errors,
        ),
        daemon=True,
    )
    disconnect_thread = threading.Thread(
        target=lambda: _capture_exception(controller.disconnect, disconnect_errors),
        daemon=True,
    )

    controller.start()
    command_thread.start()
    assert provider_command_entered.wait(timeout=0.5)
    disconnect_thread.start()
    assert transport_disconnected.wait(timeout=0.5)
    assert provider_close_entered.wait(timeout=0.5)

    with pytest.raises(RuntimeError, match="disconnect"):
        controller.command_end_effector(np.zeros(1), np.zeros(1))
    assert provider.command_count == 1

    release_provider_close.set()
    command_thread.join(timeout=1.0)
    disconnect_thread.join(timeout=1.0)

    assert not command_thread.is_alive()
    assert not disconnect_thread.is_alive()
    assert len(command_errors) == 1
    assert str(command_errors[0]) == "provider send aborted"
    assert disconnect_errors == []
    assert transport.disconnect_count == 1


def test_pause_timeout_is_bounded_by_in_flight_ordinary_command():
    command_entered = threading.Event()
    release_command = threading.Event()

    class BlockingCommandTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.command_count = 0

        def command_chassis(self, linear_x, angular_z):
            self.command_count += 1
            command_entered.set()
            assert release_command.wait(timeout=1.0)

        def disconnect(self):
            release_command.set()
            super().disconnect()

    transport = BlockingCommandTransport()
    controller = MotionController(
        transport,
        arm_strategy=RecordingArmStrategy(),
        publish_rate=200.0,
    )
    command_errors = []
    command_thread = threading.Thread(
        target=lambda: _capture_exception(
            lambda: controller.command_chassis(0.1, -0.2),
            command_errors,
        ),
        daemon=True,
    )
    controller.start()
    command_thread.start()
    assert command_entered.wait(timeout=0.5)

    try:
        started_at = time.monotonic()
        with pytest.raises(TimeoutError, match="did not pause in time"):
            controller.pause(timeout=0.05)
        elapsed = time.monotonic() - started_at

        assert elapsed < 0.25
        assert controller.is_paused
        with pytest.raises(RuntimeError, match="paused"):
            controller.command_chassis(0.2, 0.3)
        assert transport.command_count == 1

        release_command.set()
        command_thread.join(timeout=1.0)
        controller.pause(timeout=0.5)
        assert command_errors == []
    finally:
        release_command.set()
        controller.disconnect()
        command_thread.join(timeout=1.0)


def test_disconnect_aborts_blocked_publish_before_joining_publish_thread():
    publish_entered = threading.Event()
    transport_disconnected = threading.Event()

    class BlockingPublishTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.disconnect_count = 0

        def send_joint_cmd(self, q):
            publish_entered.set()
            assert transport_disconnected.wait(timeout=1.0)
            raise RuntimeError("publish send aborted")

        def disconnect(self):
            self.disconnect_count += 1
            transport_disconnected.set()
            super().disconnect()

    transport = BlockingPublishTransport()
    controller = MotionController(transport, publish_rate=200.0)
    controller.start()
    assert publish_entered.wait(timeout=0.5)
    publish_thread = controller._publish_thread

    started_at = time.monotonic()
    controller.disconnect()
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert transport.disconnect_count == 1
    assert publish_thread is not None and not publish_thread.is_alive()
    with pytest.raises(RuntimeError, match="disconnect"):
        controller.command_joints(np.zeros(JointIndex.SERVOJ_DIM))


def test_disconnect_best_effort_retries_failed_cleanup_without_leaking_thread():
    class FailingScheduler(RecordingHeadScheduler):
        def __init__(self):
            super().__init__()
            self.fail_close = True

        def close(self):
            self.close_count += 1
            if self.fail_close:
                self.fail_close = False
                raise RuntimeError("scheduler cleanup failed")

    class FailingProvider(_ProviderContractMixin):
        name = "gripper"
        width_per_side = 1

        def __init__(self):
            self.close_count = 0
            self.fail_close = True

        def read_state(self, timeout=1.0, max_age=None):
            return np.zeros(2), {}

        def command(self, left, right):
            return None

        def pause(self):
            return None

        def resume(self):
            return None

        def close(self):
            self.close_count += 1
            if self.fail_close:
                self.fail_close = False
                raise RuntimeError("provider cleanup failed")

        def raise_if_faulted(self):
            return None

    class FailingTransport(FakeDualModeTransport):
        def __init__(self):
            super().__init__()
            self.unregister_attempts = 0
            self.disconnect_attempts = 0
            self.fail_cleanup = True

        def unregister_head_response_callback(self):
            self.unregister_attempts += 1
            if self.fail_cleanup:
                raise RuntimeError("callback cleanup failed")
            super().unregister_head_response_callback()

        def disconnect(self):
            self.disconnect_attempts += 1
            if self.fail_cleanup:
                raise RuntimeError("transport cleanup failed")
            super().disconnect()

    transport = FailingTransport()
    scheduler = FailingScheduler()
    provider = FailingProvider()
    controller = MotionController(
        transport,
        arm_mode="servop",
        arm_strategy=RecordingArmStrategy(mode="servop"),
        head_scheduler=scheduler,
        end_effector_provider=provider,
        publish_rate=200.0,
    )
    controller.start()
    thread = controller._publish_thread

    try:
        with pytest.raises(RuntimeError, match="scheduler cleanup failed"):
            controller.disconnect()

        assert scheduler.close_count == 1
        assert transport.unregister_attempts == 1
        assert provider.close_count == 1
        assert transport.disconnect_attempts == 1
        assert thread is not None and not thread.is_alive()

        transport.fail_cleanup = False
        controller.disconnect()
        controller.disconnect()

        assert scheduler.close_count == 2
        assert transport.unregister_attempts == 2
        assert provider.close_count == 2
        assert transport.disconnect_attempts == 2
        assert transport.disconnected
    finally:
        controller._shutdown.set()
        controller._publish_allowed.set()
        if thread is not None:
            thread.join(timeout=1.0)


def test_disconnect_runs_external_cleanup_without_lifecycle_lock_held():
    controller = None

    def assert_lifecycle_unlocked():
        assert controller is not None
        assert not controller._lifecycle_lock.locked()

    class LockCheckingScheduler(RecordingHeadScheduler):
        def close(self):
            assert_lifecycle_unlocked()
            super().close()

    class LockCheckingProvider(_ProviderContractMixin):
        name = "gripper"
        width_per_side = 1

        def read_state(self, timeout=1.0, max_age=None):
            return np.zeros(2), {}

        def command(self, left, right):
            return None

        def pause(self):
            return None

        def resume(self):
            return None

        def close(self):
            assert_lifecycle_unlocked()

        def raise_if_faulted(self):
            return None

    class LockCheckingTransport(FakeDualModeTransport):
        def unregister_head_response_callback(self):
            assert_lifecycle_unlocked()
            super().unregister_head_response_callback()

        def disconnect(self):
            assert_lifecycle_unlocked()
            super().disconnect()

    controller = MotionController(
        LockCheckingTransport(),
        arm_mode="servop",
        arm_strategy=RecordingArmStrategy(mode="servop"),
        head_scheduler=LockCheckingScheduler(),
        end_effector_provider=LockCheckingProvider(),
    )

    controller.disconnect()


def test_pause_rejects_once_disconnect_transition_is_committed():
    class BlockingCloseScheduler(RecordingHeadScheduler):
        def __init__(self):
            super().__init__()
            self.close_entered = threading.Event()
            self.release_close = threading.Event()

        def close(self):
            self.close_entered.set()
            assert self.release_close.wait(timeout=1.0)
            super().close()

    scheduler = BlockingCloseScheduler()
    controller = MotionController(
        FakeTransport(),
        arm_mode="servop",
        arm_strategy=RecordingArmStrategy(mode="servop"),
        head_scheduler=scheduler,
        publish_rate=200.0,
    )
    controller.start()
    disconnect_errors = []
    disconnect_thread = threading.Thread(
        target=lambda: _capture_exception(controller.disconnect, disconnect_errors),
        daemon=True,
    )
    disconnect_thread.start()
    assert scheduler.close_entered.wait(timeout=0.5)

    try:
        with pytest.raises(RuntimeError, match="disconnect"):
            controller.pause(timeout=0.1)
        assert scheduler.pause_count == 0
    finally:
        scheduler.release_close.set()
        disconnect_thread.join(timeout=1.0)

    assert not disconnect_thread.is_alive()
    assert disconnect_errors == []


def test_factory_rejects_lowlevel_backend_in_public_package():
    with pytest.raises(ValueError, match="websocket"):
        create_motion_controller(
            controller_module.Tron2Config(),
            backend="lowlevel",
            modules=RobotModules(),
        )


@pytest.mark.parametrize(
    ("config", "modules"),
    [
        (
            controller_module.Tron2Config(init_joints=[0.0] * JointIndex.MOVEJ_DIM),
            RobotModules(arm="servop"),
        ),
        (
            controller_module.Tron2Config(init_head=[0.0, 0.0]),
            RobotModules(head=False),
        ),
    ],
)
def test_factory_allows_init_independent_of_inference_modules(
    monkeypatch,
    config,
    modules,
):
    calls = []

    def capture_transport(received_config, **kwargs):
        calls.append((received_config, kwargs))
        raise RuntimeError("captured")

    monkeypatch.setattr(controller_module, "WebsocketTransport", capture_transport)

    with pytest.raises(RuntimeError, match="captured"):
        create_motion_controller(config, modules=modules)

    assert calls[0][0] is config


def test_factory_maps_explicit_modules_to_legacy_init_permissions(monkeypatch):
    calls = []

    def capture_transport(config, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(controller_module, "WebsocketTransport", capture_transport)

    with pytest.raises(RuntimeError, match="captured"):
        create_motion_controller(
            controller_module.Tron2Config(
                init_joints=[0.0] * JointIndex.MOVEJ_DIM,
            ),
            modules=RobotModules(
                arm="servop",
                end_effector="brainco2",
                head=False,
            ),
            brainco2_config=controller_module.BrainCo2Config(command_time=2.0),
        )

    permissions = calls[0]["legacy_init_permissions"]
    assert permissions == controller_module.LegacyInitPermissions(
        limx_gripper_open=False,
    )
    assert vars(permissions) == {"limx_gripper_open": False}
    assert calls[0]["poll_joint_state"] is False
    assert calls[0]["poll_gripper_state"] is False


def test_factory_init_independent_completes_transport_before_servo_start(monkeypatch):
    events = []
    transport = object()

    def initialize_transport(config, **kwargs):
        events.extend(("init-start", "movej", "moveh", "init-complete"))
        return transport

    class RecordingController:
        def __init__(self, **kwargs):
            assert kwargs["transport"] is transport

        def start(self):
            events.append("servo-start")

    monkeypatch.setattr(
        controller_module,
        "WebsocketTransport",
        initialize_transport,
    )
    monkeypatch.setattr(controller_module, "MotionController", RecordingController)

    create_motion_controller(
        controller_module.Tron2Config(
            init_joints=[0.0] * JointIndex.MOVEJ_DIM,
            init_head=[0.0] * JointIndex.HEAD_DIM,
        )
    )

    assert events == [
        "init-start",
        "movej",
        "moveh",
        "init-complete",
        "servo-start",
    ]


@pytest.mark.parametrize(
    ("modules", "expected_joint", "expected_gripper"),
    [
        (RobotModules(arm="servoj", end_effector="gripper"), True, True),
        (RobotModules(arm="servoj", end_effector="brainco2"), True, False),
        (
            RobotModules(arm="servop", end_effector="brainco2", head=True),
            True,
            False,
        ),
        (RobotModules(arm="servop", end_effector="gripper"), False, True),
        (RobotModules(arm="servop", end_effector="brainco2"), False, False),
    ],
)
def test_factory_maps_explicit_modules_to_feedback_polling_sources(
    monkeypatch,
    modules,
    expected_joint,
    expected_gripper,
):
    calls = []

    def capture_transport(config, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(controller_module, "WebsocketTransport", capture_transport)
    brainco2_config = (
        controller_module.BrainCo2Config(command_time=2.0)
        if modules.end_effector == "brainco2"
        else None
    )

    with pytest.raises(RuntimeError, match="captured"):
        create_motion_controller(
            controller_module.Tron2Config(),
            modules=modules,
            brainco2_config=brainco2_config,
        )

    assert calls[0]["poll_joint_state"] is expected_joint
    assert calls[0]["poll_gripper_state"] is expected_gripper


@pytest.mark.parametrize(
    ("modules", "legacy_enable", "expected_enable"),
    [
        (RobotModules(mobile_base=True), False, True),
        (RobotModules(mobile_base=False), True, False),
    ],
)
def test_factory_derives_chassis_feedback_from_explicit_modules(
    monkeypatch,
    modules,
    legacy_enable,
    expected_enable,
):
    calls = []

    def capture_transport(config, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(controller_module, "WebsocketTransport", capture_transport)

    with pytest.raises(RuntimeError, match="captured"):
        create_motion_controller(
            controller_module.Tron2Config(),
            modules=modules,
            enable_chassis_lifter=legacy_enable,
        )

    assert calls[0]["enable_chassis_lifter"] is expected_enable


def test_factory_rollback_attempts_all_cleanup_and_preserves_original_error(
    monkeypatch,
):
    class FailingProvider(_ProviderContractMixin):
        width_per_side = 6

        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1
            raise RuntimeError("provider rollback failed")

    class FailingScheduler:
        def __init__(self, *args, **kwargs):
            self.close_count = 0

        def handle_response(self, guid, result):
            return None

        def close(self):
            self.close_count += 1
            raise RuntimeError("scheduler rollback failed")

    class FailingTransport(FakeDualModeTransport):
        def __init__(self):
            super().__init__()
            self.unregister_attempts = 0
            self.disconnect_attempts = 0

        def unregister_head_response_callback(self):
            self.unregister_attempts += 1
            raise RuntimeError("callback rollback failed")

        def disconnect(self):
            self.disconnect_attempts += 1
            raise RuntimeError("transport rollback failed")

    transport = FailingTransport()
    provider = FailingProvider()
    scheduler = FailingScheduler()
    monkeypatch.setattr(
        controller_module,
        "WebsocketTransport",
        lambda config, **kwargs: transport,
    )
    monkeypatch.setattr(
        controller_module,
        "create_end_effector_provider",
        lambda name, selected_transport, config: provider,
    )
    monkeypatch.setattr(
        controller_module,
        "HeadCommandScheduler",
        lambda **kwargs: scheduler,
    )

    def fail_controller_construction(*args, **kwargs):
        raise ValueError("controller construction failed")

    monkeypatch.setattr(
        controller_module,
        "MotionController",
        fail_controller_construction,
    )

    with pytest.raises(ValueError, match="controller construction failed"):
        create_motion_controller(
            controller_module.Tron2Config(),
            modules=RobotModules(
                arm="servop",
                end_effector="brainco2",
                head=True,
            ),
            brainco2_config=controller_module.BrainCo2Config(command_time=2.0),
        )

    assert provider.close_count == 1
    assert scheduler.close_count == 1
    assert transport.unregister_attempts == 1
    assert transport.disconnect_attempts == 1


def test_factory_preserves_start_error_when_controller_disconnect_also_fails(
    monkeypatch,
):
    class FailingController:
        def __init__(self, **kwargs):
            self.disconnect_count = 0

        def start(self):
            raise ValueError("controller start failed")

        def disconnect(self):
            self.disconnect_count += 1
            raise RuntimeError("controller rollback failed")

    transport = FakeTransport()
    created = []

    def make_controller(**kwargs):
        controller = FailingController(**kwargs)
        created.append(controller)
        return controller

    monkeypatch.setattr(
        controller_module,
        "WebsocketTransport",
        lambda config, **kwargs: transport,
    )
    monkeypatch.setattr(controller_module, "MotionController", make_controller)

    with pytest.raises(ValueError, match="controller start failed"):
        create_motion_controller(controller_module.Tron2Config())

    assert len(created) == 1
    assert created[0].disconnect_count == 1


def test_factory_preserves_legacy_positional_arguments(monkeypatch):
    fake = FakeTransport()
    constructor_calls = []

    def make_websocket(config, **kwargs):
        constructor_calls.append((config, kwargs))
        return fake

    monkeypatch.setattr(controller_module, "WebsocketTransport", make_websocket)
    config = controller_module.Tron2Config()
    interpolator = LinearInterpolator()

    mc = create_motion_controller(
        config,
        "websocket",
        200.0,
        0.05,
        interpolator,
        False,
        20.0,
        "raw_q",
    )
    try:
        wait_for(lambda: bool(fake.sends()))
        assert constructor_calls == [
            (
                config,
                {
                    "enable_chassis_lifter": False,
                    "auxiliary_polling_rate": 20.0,
                    "lifter_state_source": "raw_q",
                },
            )
        ]
    finally:
        mc.disconnect()


def test_factory_installs_servop_head_scheduler_and_unregisters_on_disconnect(
    monkeypatch,
):
    fake = FakeDualModeTransport()
    monkeypatch.setattr(
        controller_module,
        "WebsocketTransport",
        lambda config, **kwargs: fake,
    )

    mc = create_motion_controller(
        controller_module.Tron2Config(),
        backend="websocket",
        publish_rate=200.0,
        eta_default=0.04,
        modules=RobotModules(arm="servop", head=True),
        head_command_interval=0.02,
        head_move_duration=0.03,
        head_ack_timeout=0.5,
    )
    try:
        assert fake.head_callback_registrations == 1
        mc.command_head([0.1, -0.2])
        assert len(fake.head_commands) == 1
        np.testing.assert_allclose(fake.head_commands[0][0], [0.1, -0.2])
        assert fake.head_commands[0][1] == pytest.approx(0.03)
    finally:
        mc.disconnect()

    assert fake.head_callback is None
    assert fake.head_callback_unregistrations == 1


def test_factory_only_installs_head_scheduler_for_servop_with_head(monkeypatch):
    created = []

    def make_websocket(config, **kwargs):
        fake = FakeDualModeTransport()
        created.append(fake)
        return fake

    monkeypatch.setattr(controller_module, "WebsocketTransport", make_websocket)

    for modules in (
        RobotModules(arm="servoj", head=True),
        RobotModules(arm="servop", head=False),
    ):
        mc = create_motion_controller(
            controller_module.Tron2Config(),
            backend="websocket",
            publish_rate=200.0,
            modules=modules,
        )
        mc.disconnect()

    assert [fake.head_callback_registrations for fake in created] == [0, 0]
