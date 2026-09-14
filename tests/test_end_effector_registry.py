"""Atomic end-effector registration contracts without robot hardware."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

import tron2_env._end_effector_registry as _registry
from tron2_env.end_effector import (
    END_EFFECTOR_PROVIDERS,
    create_end_effector_provider,
    register_end_effector_provider,
)
from tron2_env.env import EnvConfig
from tron2_env.layout import (
    END_EFFECTOR_LAYOUTS,
    ComponentSpec,
    EndEffectorLayoutSpec,
    PhysicalLayout,
    RobotModules,
)


@pytest.fixture(autouse=True)
def _restore_end_effector_registrations():
    with _registry._REGISTRY_LOCK:
        registrations = _registry._REGISTRATIONS.copy()
    try:
        yield
    finally:
        with _registry._REGISTRY_LOCK:
            _registry._REGISTRATIONS.clear()
            _registry._REGISTRATIONS.update(registrations)


class _CompleteProvider:
    def __init__(
        self,
        name: str,
        width_per_side: int,
        *,
        close_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.width_per_side = width_per_side
        self.close_calls = 0
        self._close_error = close_error

    def read_state(self, timeout=1.0, max_age=None):
        return [], {}

    def command(self, left, right):
        return None

    def validate_command(self, left, right):
        return left, right

    def pause(self):
        return None

    def resume(self):
        return None

    def close(self):
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error

    def raise_if_faulted(self):
        return None


def _layout(
    name: str,
    left_component: str,
    right_component: str,
    *,
    width: int = 2,
) -> EndEffectorLayoutSpec:
    return EndEffectorLayoutSpec(
        name=name,
        left=ComponentSpec(
            left_component,
            tuple(f"{left_component}_{index}" for index in range(width)),
        ),
        right=ComponentSpec(
            right_component,
            tuple(f"{right_component}_{index}" for index in range(width)),
        ),
    )


def test_registration_atomically_updates_stable_read_only_projections():
    layouts = END_EFFECTOR_LAYOUTS
    providers = END_EFFECTOR_PROVIDERS
    layout = _layout(
        "registry_atomic_hand",
        "left_registry_atomic_hand",
        "right_registry_atomic_hand",
    )

    def factory(transport, config):
        return _CompleteProvider("registry_atomic_hand", 2)

    register_end_effector_provider(
        "registry_atomic_hand",
        factory,
        layout=layout,
    )

    assert END_EFFECTOR_LAYOUTS is layouts
    assert END_EFFECTOR_PROVIDERS is providers
    assert layouts["registry_atomic_hand"] is layout
    assert providers["registry_atomic_hand"] is factory
    assert isinstance(layouts, Mapping)
    assert isinstance(providers, Mapping)
    with pytest.raises(TypeError):
        layouts["replacement"] = layout
    with pytest.raises(TypeError):
        providers["replacement"] = factory

    modules = RobotModules(end_effector="registry_atomic_hand")
    assert PhysicalLayout.from_modules(modules).components == (
        "left_arm",
        "left_registry_atomic_hand",
        "right_arm",
        "right_registry_atomic_hand",
    )
    assert create_end_effector_provider(
        "registry_atomic_hand",
        object(),
    ).width_per_side == 2


@pytest.mark.parametrize("width", [1, 2])
def test_legacy_env_config_rejects_every_registered_non_gripper_layout(width):
    name = f"registry_explicit_only_{width}"
    layout = _layout(
        name,
        f"left_{name}",
        f"right_{name}",
        width=width,
    )
    register_end_effector_provider(
        name,
        lambda transport, config: _CompleteProvider(name, width),
        layout=layout,
    )
    components = (
        "left_arm",
        layout.left.name,
        "right_arm",
        layout.right.name,
    )

    with pytest.raises(ValueError, match="explicit RobotModules"):
        EnvConfig(state_layout=list(components))

    explicit = EnvConfig(modules=RobotModules(end_effector=name))
    assert explicit.physical_layout.components == components
    assert explicit.physical_layout.end_effector_name == name


def test_unknown_two_argument_registration_fails_without_partial_entry():
    name = "registry_missing_layout"

    with pytest.raises(ValueError, match="layout.*required"):
        register_end_effector_provider(name, lambda transport, config: None)

    assert name not in END_EFFECTOR_LAYOUTS
    assert name not in END_EFFECTOR_PROVIDERS


def test_duplicate_registration_cannot_replace_either_projection():
    original_layout = END_EFFECTOR_LAYOUTS["gripper"]
    original_factory = END_EFFECTOR_PROVIDERS["gripper"]

    with pytest.raises(ValueError, match="already registered"):
        register_end_effector_provider(
            "gripper",
            lambda transport, config: None,
            layout=original_layout,
        )

    assert END_EFFECTOR_LAYOUTS["gripper"] is original_layout
    assert END_EFFECTOR_PROVIDERS["gripper"] is original_factory


@pytest.mark.parametrize("name", ["", "   ", None])
def test_registration_rejects_empty_or_non_string_names_before_mutation(name):
    layout = _layout(
        "registry_valid_name",
        "left_registry_valid_name",
        "right_registry_valid_name",
    )

    with pytest.raises(ValueError, match="non-empty string"):
        register_end_effector_provider(
            name,
            lambda transport, config: None,
            layout=layout,
        )

    assert "registry_valid_name" not in END_EFFECTOR_PROVIDERS


def test_registration_rejects_non_callable_factory_and_layout_name_mismatch():
    layout = _layout(
        "registry_layout_name",
        "left_registry_layout_name",
        "right_registry_layout_name",
    )

    with pytest.raises(TypeError, match="factory.*callable"):
        register_end_effector_provider(
            "registry_layout_name",
            object(),
            layout=layout,
        )
    with pytest.raises(ValueError, match="must match"):
        register_end_effector_provider(
            "registry_other_name",
            lambda transport, config: None,
            layout=layout,
        )

    assert "registry_layout_name" not in END_EFFECTOR_PROVIDERS
    assert "registry_other_name" not in END_EFFECTOR_PROVIDERS


@pytest.mark.parametrize("component_name", ["left_arm", "head", "left_gripper"])
def test_registration_rejects_reserved_or_existing_component_names(component_name):
    name = f"registry_component_conflict_{component_name}"
    layout = _layout(name, component_name, f"right_{name}")

    with pytest.raises(ValueError, match="component.*conflict"):
        register_end_effector_provider(
            name,
            lambda transport, config: None,
            layout=layout,
        )

    assert name not in END_EFFECTOR_LAYOUTS
    assert name not in END_EFFECTOR_PROVIDERS


def test_component_and_layout_specs_reject_invalid_fields_and_widths():
    with pytest.raises(ValueError, match="field.*non-empty string"):
        ComponentSpec("left_invalid_fields", ("",))
    with pytest.raises(ValueError, match="distinct component"):
        _layout("registry_same_component", "same_component", "same_component")
    with pytest.raises(ValueError, match="equal left/right widths"):
        EndEffectorLayoutSpec(
            name="registry_unequal_width",
            left=ComponentSpec("left_registry_unequal_width", ("left_value",)),
            right=ComponentSpec(
                "right_registry_unequal_width",
                ("right_value_0", "right_value_1"),
            ),
        )


def test_create_closes_provider_that_does_not_satisfy_protocol():
    name = "registry_invalid_protocol"
    layout = _layout(name, f"left_{name}", f"right_{name}")

    class IncompleteProvider:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    provider = IncompleteProvider()
    register_end_effector_provider(
        name,
        lambda transport, config: provider,
        layout=layout,
    )

    with pytest.raises(TypeError, match="EndEffectorProvider"):
        create_end_effector_provider(name, object())

    assert provider.close_calls == 1


@pytest.mark.parametrize(
    ("method_name", "expected_close_calls"),
    [
        ("read_state", 1),
        ("validate_command", 1),
        ("command", 1),
        ("pause", 1),
        ("close", 0),
        ("raise_if_faulted", 1),
    ],
)
def test_create_rejects_non_callable_required_provider_method(
    method_name,
    expected_close_calls,
):
    name = f"registry_non_callable_{method_name}"
    layout = _layout(name, f"left_{name}", f"right_{name}")
    provider = _CompleteProvider(name, 2)
    setattr(provider, method_name, object())
    register_end_effector_provider(
        name,
        lambda transport, config: provider,
        layout=layout,
    )

    with pytest.raises(
        TypeError,
        match=rf"EndEffectorProvider.*callable {method_name}\(\)",
    ):
        create_end_effector_provider(name, object())

    assert provider.close_calls == expected_close_calls


def test_create_accepts_design_protocol_without_resume_or_max_age():
    name = "registry_minimal_protocol"
    layout = _layout(name, f"left_{name}", f"right_{name}")

    class MinimalProvider:
        width_per_side = 2

        def __init__(self):
            self.name = name

        def read_state(self, timeout=1.0):
            return np.zeros(4), {}

        def validate_command(self, left, right):
            return np.asarray(left), np.asarray(right)

        def command(self, left, right):
            return None

        def pause(self):
            return None

        def close(self):
            return None

        def raise_if_faulted(self):
            return None

    provider = MinimalProvider()
    register_end_effector_provider(
        name,
        lambda transport, config: provider,
        layout=layout,
    )

    assert create_end_effector_provider(name, object()) is provider
    assert not hasattr(provider, "resume")


def test_create_validates_provider_name_and_preserves_error_when_close_fails():
    name = "registry_wrong_provider_name"
    layout = _layout(name, f"left_{name}", f"right_{name}")
    provider = _CompleteProvider(
        "wrong_name",
        2,
        close_error=RuntimeError("close must not replace validation error"),
    )
    register_end_effector_provider(
        name,
        lambda transport, config: provider,
        layout=layout,
    )

    with pytest.raises(ValueError, match="provider name.*wrong_name"):
        create_end_effector_provider(name, object())

    assert provider.close_calls == 1


def test_create_validates_provider_width_against_registered_layout_and_closes():
    name = "registry_wrong_provider_width"
    layout = _layout(name, f"left_{name}", f"right_{name}", width=3)
    provider = _CompleteProvider(name, 2)
    register_end_effector_provider(
        name,
        lambda transport, config: provider,
        layout=layout,
    )

    with pytest.raises(ValueError, match="width_per_side.*3.*2"):
        create_end_effector_provider(name, object())

    assert provider.close_calls == 1
