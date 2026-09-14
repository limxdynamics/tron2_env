"""Robot-wide end-effector provider tests without robot hardware."""

from __future__ import annotations

import copy
import time

import numpy as np
import pytest

from tron2_env.end_effector import (
    BrainCo2Provider,
    EndEffectorProvider,
    GripperProvider,
    create_end_effector_provider,
    register_end_effector_provider,
)
from tron2_env.errors import CommandError, StateError
from tron2_env.layout import BrainCo2Config


class FakeEndEffectorTransport:
    """Behavioral fake for the complete provider-facing transport surface."""

    def __init__(self) -> None:
        self.gripper_commands = []
        self.brainco_commands = []
        self.gripper_reads = []
        self.brainco_reads = []
        self.polling_enables = []
        self.polling_disable_count = 0
        self.brainco_recovery_generations = []
        self.brainco_fault = None
        self.gripper_state = {
            "timestamp": 101,
            "states": [0.25, 0.75],
            "diagnostics": {"source": ["gripper"]},
        }
        self.brainco_state = {
            "timestamp": 202,
            "states": list(range(12)),
            "metadata": {
                "robot_timestamp": 202,
                "received_at": 10.0,
                "diagnostics": {"source": ["brainco2"]},
            },
        }

    def set_gripper(self, left_opening, right_opening) -> None:
        self.gripper_commands.append((left_opening, right_opening))

    def get_gripper_state(self, timeout=1.0, max_age=None) -> dict:
        self.gripper_reads.append((timeout, max_age))
        return copy.deepcopy(self.gripper_state)

    def set_brainco2_position(self, left, right, command_time) -> str:
        self.brainco_commands.append(
            (list(left), list(right), list(command_time))
        )
        return f"command-{len(self.brainco_commands)}"

    def request_brainco2_state(self) -> str:
        return "state-guid"

    def get_brainco2_state(self, timeout=1.0, max_age=None) -> dict:
        self.brainco_reads.append((timeout, max_age))
        return copy.deepcopy(self.brainco_state)

    def raise_brainco2_fault(self) -> None:
        if self.brainco_fault is not None:
            raise self.brainco_fault

    def enable_brainco2_polling(self, rate, query_timeout) -> None:
        self.polling_enables.append((rate, query_timeout))

    def disable_brainco2_polling(self) -> None:
        self.polling_disable_count += 1

    def clear_brainco2_session_uncertainty(self, generation: int) -> None:
        self.brainco_recovery_generations.append(generation)


@pytest.fixture
def fake_transport():
    return FakeEndEffectorTransport()


def test_builtin_registry_builds_one_robot_wide_provider(fake_transport):
    gripper = create_end_effector_provider("gripper", fake_transport)
    brainco = create_end_effector_provider(
        "brainco2",
        fake_transport,
        BrainCo2Config(command_time=[1, 2, 3, 4, 5, 6]),
    )

    assert isinstance(gripper, GripperProvider)
    assert isinstance(brainco, BrainCo2Provider)
    assert isinstance(gripper, EndEffectorProvider)
    assert isinstance(brainco, EndEffectorProvider)
    assert gripper.width_per_side == 1
    assert brainco.width_per_side == 6
    brainco.close()


def test_registry_rejects_duplicate_and_unknown_names(fake_transport):
    with pytest.raises(ValueError, match="already registered"):
        register_end_effector_provider("gripper", lambda transport, config: None)

    with pytest.raises(ValueError, match="unknown end effector provider"):
        create_end_effector_provider("mixed-left-right", fake_transport)


def test_brainco_factory_requires_config(fake_transport):
    with pytest.raises(ValueError, match="BrainCo2Config"):
        create_end_effector_provider("brainco2", fake_transport)


def test_gripper_provider_maps_normalized_scalars_to_percent(fake_transport):
    provider = GripperProvider(fake_transport)

    provider.command(np.array([0.125]), np.array([0.875]))

    assert fake_transport.gripper_commands == [(12.5, 87.5)]


@pytest.mark.parametrize(
    "left,right",
    [
        ([0.0, 1.0], [0.5]),
        ([[0.5]], [0.5]),
        ([np.nan], [0.5]),
        ([0.5], [np.inf]),
        ([-0.01], [0.5]),
        ([0.5], [1.01]),
    ],
)
def test_gripper_provider_rejects_bad_shape_nonfinite_and_out_of_range(
    fake_transport,
    left,
    right,
):
    provider = GripperProvider(fake_transport)

    with pytest.raises(CommandError, match="gripper"):
        provider.command(left, right)

    assert not fake_transport.gripper_commands


def test_gripper_provider_reads_two_values_and_defensive_metadata(fake_transport):
    provider = GripperProvider(fake_transport)

    values, metadata = provider.read_state(timeout=0.3, max_age=0.2)

    np.testing.assert_array_equal(values, [0.25, 0.75])
    assert metadata == {
        "timestamp": 101,
        "diagnostics": {"source": ["gripper"]},
    }
    assert fake_transport.gripper_reads == [(0.3, 0.2)]
    metadata["diagnostics"]["source"][0] = "mutated"
    assert fake_transport.gripper_state["diagnostics"] == {"source": ["gripper"]}


@pytest.mark.parametrize(
    ("received_at", "message"),
    [
        (None, "invalid freshness"),
        ("invalid", "invalid freshness"),
        (np.nan, "invalid freshness"),
        ("stale", "is stale"),
    ],
)
def test_legacy_gripper_requires_valid_received_at_when_max_age_set(
    received_at,
    message,
):
    class LegacyGripperTransport:
        def get_joint_state(self, timeout=1.0):
            payload = {
                "states": np.arange(18, dtype=np.float64),
                "timestamp": 101,
            }
            if received_at is not None:
                payload["received_at"] = (
                    time.monotonic() - 1.0
                    if received_at == "stale"
                    else received_at
                )
            return payload

        def set_gripper(self, left, right):
            return None

    provider = GripperProvider(LegacyGripperTransport())

    values, _metadata = provider.read_state(timeout=0.3, max_age=None)
    np.testing.assert_array_equal(values, [7.0, 15.0])

    with pytest.raises(StateError, match=message):
        provider.read_state(timeout=0.3, max_age=0.2)


def test_gripper_pause_resume_close_gate_and_fault_noop(fake_transport):
    provider = GripperProvider(fake_transport)
    provider.raise_if_faulted()

    provider.pause()
    with pytest.raises(RuntimeError, match="paused"):
        provider.command([0.25], [0.75])
    provider.resume()
    provider.command([0.25], [0.75])

    provider.close()
    with pytest.raises(RuntimeError, match="closed"):
        provider.command([0.25], [0.75])
    with pytest.raises(RuntimeError, match="closed"):
        provider.resume()


def test_brainco_provider_enables_polling_and_expands_scalar_time(fake_transport):
    provider = BrainCo2Provider(
        fake_transport,
        BrainCo2Config(command_time=2.0, polling_rate=25.0, state_timeout=0.4),
    )

    provider.command(np.arange(6), np.arange(6, 12))

    assert fake_transport.polling_enables == [(25.0, 0.4)]
    assert fake_transport.brainco_commands == [
        (list(range(6)), list(range(6, 12)), [2.0] * 6)
    ]
    provider.close()


def test_brainco_provider_preserves_vector_time_and_raw_positions(fake_transport):
    provider = BrainCo2Provider(
        fake_transport,
        BrainCo2Config(command_time=[1, 2, 3, 4, 5, 6]),
    )
    left = [-20.0, 0.5, 100.5, 3.0, 2.0, 1.0]
    right = [60.0, 50.0, 40.0, 30.0, 20.0, 10.0]

    provider.command(left, right)

    assert fake_transport.brainco_commands == [
        (left, right, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ]
    provider.close()


@pytest.mark.parametrize(
    "left,right",
    [
        ([0.0] * 5, [0.0] * 6),
        ([[0.0] * 6], [0.0] * 6),
        ([0.0] * 6, [0.0] * 7),
        ([0.0, 0.0, 0.0, 0.0, 0.0, np.nan], [0.0] * 6),
        ([0.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.0, np.inf]),
    ],
)
def test_brainco_provider_rejects_bad_shape_and_nonfinite_values(
    fake_transport,
    left,
    right,
):
    provider = BrainCo2Provider(fake_transport, BrainCo2Config(command_time=2.0))

    with pytest.raises(CommandError, match="finite shape \\(6,\\)"):
        provider.command(left, right)

    assert not fake_transport.brainco_commands
    provider.close()


def test_brainco_provider_reads_12d_state_with_stricter_freshness(fake_transport):
    provider = BrainCo2Provider(
        fake_transport,
        BrainCo2Config(command_time=2.0, state_timeout=0.5),
    )

    values, metadata = provider.read_state(timeout=0.25)
    strict_values, strict_metadata = provider.read_state(timeout=0.1, max_age=0.2)

    np.testing.assert_array_equal(values, np.arange(12))
    np.testing.assert_array_equal(strict_values, np.arange(12))
    assert metadata == fake_transport.brainco_state["metadata"]
    assert strict_metadata == metadata
    assert fake_transport.brainco_reads == [(0.25, 0.5), (0.1, 0.2)]
    metadata["diagnostics"]["source"][0] = "mutated"
    assert fake_transport.brainco_state["metadata"]["diagnostics"] == {
        "source": ["brainco2"]
    }
    provider.close()


def test_brainco_provider_command_propagates_recorded_transport_fault(fake_transport):
    provider = BrainCo2Provider(fake_transport, BrainCo2Config(command_time=2.0))
    fake_transport.brainco_fault = CommandError("BrainCo2 asynchronous failure")

    with pytest.raises(CommandError, match="asynchronous failure"):
        provider.command([0.0] * 6, [0.0] * 6)

    assert not fake_transport.brainco_commands
    provider.close()


def test_brainco_pause_resume_and_close_do_not_guess_physical_commands(
    fake_transport,
):
    provider = BrainCo2Provider(fake_transport, BrainCo2Config(command_time=2.0))

    provider.pause()
    with pytest.raises(RuntimeError, match="paused"):
        provider.command([0.0] * 6, [0.0] * 6)
    assert not fake_transport.brainco_commands

    provider.resume()
    provider.command([1.0] * 6, [2.0] * 6)
    assert len(fake_transport.brainco_commands) == 1

    provider.close()
    assert fake_transport.polling_disable_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        provider.command([0.0] * 6, [0.0] * 6)
    with pytest.raises(RuntimeError, match="closed"):
        provider.resume()
    assert len(fake_transport.brainco_commands) == 1

    provider.close()
    assert fake_transport.polling_disable_count == 1


def test_brainco_reconnect_recovery_is_allowed_after_provider_resume(
    fake_transport,
):
    provider = BrainCo2Provider(fake_transport, BrainCo2Config(command_time=2.0))
    provider.pause()
    provider.resume()

    provider.recover_after_reconnect(7)

    assert fake_transport.brainco_recovery_generations == [7]
    provider.close()


def test_brainco_close_retries_polling_cleanup_while_commands_stay_blocked(
    fake_transport,
):
    attempts = []

    def fail_once():
        attempts.append("disable")
        if len(attempts) == 1:
            raise RuntimeError("disable polling failed")
        fake_transport.polling_disable_count += 1

    fake_transport.disable_brainco2_polling = fail_once
    provider = BrainCo2Provider(fake_transport, BrainCo2Config(command_time=2.0))

    with pytest.raises(RuntimeError, match="disable polling failed"):
        provider.close()
    with pytest.raises(RuntimeError, match="closed"):
        provider.command([0.0] * 6, [0.0] * 6)
    with pytest.raises(RuntimeError, match="closed"):
        provider.resume()

    provider.close()
    provider.close()

    assert attempts == ["disable", "disable"]
    assert fake_transport.polling_disable_count == 1
