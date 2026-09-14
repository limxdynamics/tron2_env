from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import tron2_env
from tron2_env.layout import (
    ARM_LAYOUTS,
    END_EFFECTOR_LAYOUTS,
    BrainCo2Config,
    ComponentSpec,
    PhysicalLayout,
    RobotModules,
)


@pytest.mark.parametrize(
    ("end_effector", "head", "mobile_base", "dimension"),
    [
        ("gripper", False, False, 16),
        ("gripper", True, False, 18),
        ("gripper", False, True, 19),
        ("gripper", True, True, 21),
        ("brainco2", False, False, 26),
        ("brainco2", True, False, 28),
        ("brainco2", False, True, 29),
        ("brainco2", True, True, 31),
    ],
)
def test_layout_from_modules_has_expected_dimension(
    end_effector, head, mobile_base, dimension
):
    modules = RobotModules(
        arm="servoj",
        end_effector=end_effector,
        head=head,
        mobile_base=mobile_base,
    )

    assert PhysicalLayout.from_modules(modules).dim == dimension


def test_layout_from_modules_uses_canonical_component_order():
    layout = PhysicalLayout.from_modules(
        RobotModules(end_effector="brainco2", head=True, mobile_base=True)
    )

    assert layout.components == (
        "left_arm",
        "left_hand",
        "right_arm",
        "right_hand",
        "head",
        "chassis",
    )


def test_servop_layout_uses_wxyz_pose_field_names():
    layout = PhysicalLayout.from_modules(
        RobotModules(arm="servop", end_effector="gripper")
    )

    assert layout.field_names[:7] == (
        "left_arm_x",
        "left_arm_y",
        "left_arm_z",
        "left_arm_qw",
        "left_arm_qx",
        "left_arm_qy",
        "left_arm_qz",
    )
    assert layout.field_names[8:15] == (
        "right_arm_x",
        "right_arm_y",
        "right_arm_z",
        "right_arm_qw",
        "right_arm_qx",
        "right_arm_qy",
        "right_arm_qz",
    )


def test_brainco_layout_preserves_index_order_and_exposes_effector_metadata():
    layout = PhysicalLayout.from_modules(
        RobotModules(arm="servoj", end_effector="brainco2")
    )

    assert layout.field_names[7:13] == tuple(
        f"left_hand_{index}" for index in range(6)
    )
    assert layout.field_names[20:26] == tuple(
        f"right_hand_{index}" for index in range(6)
    )
    assert layout.left_end_effector_component == "left_hand"
    assert layout.right_end_effector_component == "right_hand"
    assert layout.end_effector_width == 6


def test_default_modules_are_frozen_servoj_grippers():
    modules = RobotModules()

    assert modules == RobotModules(
        arm="servoj", end_effector="gripper", head=False, mobile_base=False
    )
    with pytest.raises(FrozenInstanceError):
        modules.arm = "servop"


@pytest.mark.parametrize(
    ("registry", "key", "replacement"),
    [
        (ARM_LAYOUTS, "servoj", ARM_LAYOUTS["servop"]),
        (END_EFFECTOR_LAYOUTS, "gripper", END_EFFECTOR_LAYOUTS["brainco2"]),
    ],
)
def test_builtin_layout_registries_reject_replacement_and_deletion(
    registry, key, replacement
):
    modules = RobotModules()
    expected = PhysicalLayout.from_modules(modules)
    original = registry[key]

    try:
        with pytest.raises(TypeError):
            registry[key] = replacement
    finally:
        if registry.get(key) is not original:
            registry[key] = original

    try:
        with pytest.raises(TypeError):
            del registry[key]
    finally:
        if key not in registry:
            registry[key] = original

    assert PhysicalLayout.from_modules(modules) == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"arm": "movej"}, "arm"),
        ({"arm": []}, "arm"),
        ({"end_effector": "unknown"}, "end effector"),
        ({"head": 1}, "head"),
        ({"mobile_base": "yes"}, "mobile_base"),
    ],
)
def test_robot_modules_validate_registered_options_and_flags(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RobotModules(**kwargs)


def test_new_config_types_are_exported_from_package():
    assert tron2_env.RobotModules is RobotModules
    assert tron2_env.BrainCo2Config is BrainCo2Config


def test_brainco_command_time_expands_scalar_and_preserves_vector_order():
    config = BrainCo2Config(command_time=2.0)

    assert config.expanded_command_time == (2.0,) * 6
    assert config.polling_rate == 20.0
    assert config.state_timeout == 0.5
    with pytest.raises(FrozenInstanceError):
        config.polling_rate = 10.0
    assert BrainCo2Config(
        command_time=[1, 2, 3, 4, 5, 6]
    ).expanded_command_time == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_brainco_config_copies_mutable_command_time_input():
    command_time = [1, 2, 3, 4, 5, 6]
    config = BrainCo2Config(command_time=command_time)

    command_time[0] = 99

    assert config.command_time == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert config.expanded_command_time == config.command_time


@pytest.mark.parametrize(
    "command_time",
    [None, [1, 2, 3, 4, 5], [[1, 2, 3], [4, 5, 6]]],
)
def test_brainco_command_time_rejects_missing_or_wrong_shape(command_time):
    with pytest.raises(ValueError, match="command_time"):
        BrainCo2Config(command_time=command_time)


@pytest.mark.parametrize(
    "command_time",
    [np.nan, np.inf, [1, 2, 3, 4, 5, np.nan], [1, 2, 3, 4, 5, -np.inf]],
)
def test_brainco_command_time_must_be_finite(command_time):
    with pytest.raises(ValueError, match="finite"):
        BrainCo2Config(command_time=command_time)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polling_rate": 0},
        {"polling_rate": -1},
        {"state_timeout": 0},
        {"state_timeout": -1},
    ],
)
def test_brainco_polling_rate_and_state_timeout_must_be_positive(kwargs):
    with pytest.raises(ValueError, match="positive"):
        BrainCo2Config(command_time=2.0, **kwargs)


@pytest.mark.parametrize(
    ("state_dim", "components"),
    [
        (16, ("left_arm", "left_gripper", "right_arm", "right_gripper")),
        (np.int64(16), ("left_arm", "left_gripper", "right_arm", "right_gripper")),
        (
            18,
            ("left_arm", "left_gripper", "right_arm", "right_gripper", "head"),
        ),
    ],
)
def test_legacy_dimensions_resolve_to_servoj_gripper_layouts(state_dim, components):
    layout = PhysicalLayout.resolve(state_dim=state_dim)

    assert layout.components == components
    assert layout.field_names[:7] == tuple(f"left_arm_{index}" for index in range(7))


@pytest.mark.parametrize(
    ("components", "dimension"),
    [
        (["left_arm", "left_gripper", "right_arm", "right_gripper", "head"], 18),
        (["left_arm", "left_gripper", "right_arm", "right_gripper", "chassis"], 19),
        (
            [
                "left_arm",
                "left_gripper",
                "right_arm",
                "right_gripper",
                "head",
                "chassis",
            ],
            21,
        ),
        (["left_arm", "left_hand", "right_arm", "right_hand"], 26),
        (["left_arm", "left_hand", "right_arm", "right_hand", "head"], 28),
        (["left_arm", "left_hand", "right_arm", "right_hand", "chassis"], 29),
        (["left_arm", "left_hand", "right_arm", "right_hand", "head", "chassis"], 31),
    ],
)
def test_explicit_legacy_layouts_retain_dimensions(components, dimension):
    assert PhysicalLayout.resolve(components).dim == dimension


def test_gripper_layout_exposes_effector_metadata():
    layout = PhysicalLayout.resolve()

    assert layout.left_end_effector_component == "left_gripper"
    assert layout.right_end_effector_component == "right_gripper"
    assert layout.end_effector_width == 1


def test_layout_repr_includes_components_and_dimension():
    layout = PhysicalLayout.resolve()

    rendered = repr(layout)

    assert f"components={layout.components!r}" in rendered
    assert f"dim={layout.dim}" in rendered


def test_nonlegacy_dimension_requires_explicit_layout():
    with pytest.raises(ValueError, match="ambiguous"):
        PhysicalLayout.resolve(state_dim=19)


@pytest.mark.parametrize(
    ("state_dim", "components"),
    [
        (16.9, None),
        (18.2, None),
        (16.0, None),
        (True, None),
        ("16", None),
        (np.bool_(True), None),
        (
            16.5,
            ["left_arm", "left_gripper", "right_arm", "right_gripper"],
        ),
    ],
)
def test_resolve_rejects_non_integer_state_dim(state_dim, components):
    with pytest.raises(ValueError, match="state_dim must be an integer"):
        PhysicalLayout.resolve(components, state_dim=state_dim)


def test_explicit_layout_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        PhysicalLayout.resolve(
            ["left_arm", "left_gripper", "right_arm", "right_gripper", "chassis"],
            state_dim=18,
        )


def test_layout_pack_unpack_roundtrip():
    layout = PhysicalLayout.resolve(
        ["left_arm", "left_gripper", "right_arm", "right_gripper", "head", "chassis"]
    )
    vector = np.arange(layout.dim, dtype=np.float32)
    parts = layout.unpack(vector)

    np.testing.assert_array_equal(layout.pack(parts), vector)
    assert layout.field_names[-3:] == (
        "chassis_linear_x",
        "chassis_angular_z",
        "lifter_state",
    )

    float64_vector = vector.astype(np.float64)
    float64_parts = layout.unpack(float64_vector)
    assert float64_parts["left_arm"].dtype == np.float64

    overflow = float64_vector.copy()
    overflow[0] = float(np.finfo(np.float32).max) * 2.0
    with pytest.raises(ValueError, match="float32"):
        layout.unpack(overflow)


def test_dynamic_layout_pack_unpack_and_component_slice_use_instance_specs():
    layout = PhysicalLayout.from_modules(
        RobotModules(arm="servop", end_effector="brainco2", head=True)
    )
    vector = np.arange(layout.dim, dtype=np.float32)
    parts = layout.unpack(vector)

    np.testing.assert_array_equal(layout.pack(parts), vector)
    assert layout.component_slice("left_hand") == slice(7, 13)
    assert layout.component_slice("right_arm") == slice(13, 20)
    assert layout.has("head")
    assert not layout.has("chassis")


def test_direct_layout_rejects_registered_end_effector_schema_override():
    with pytest.raises(ValueError, match="registered schema"):
        PhysicalLayout(
            (
                ARM_LAYOUTS["servoj"][0],
                ComponentSpec("left_gripper", ("left_finger_0", "left_finger_1")),
                ARM_LAYOUTS["servoj"][1],
                ComponentSpec("right_gripper", ("right_finger_0", "right_finger_1")),
            )
        )


def test_direct_layout_rejects_mixed_servoj_and_servop_arm_specs():
    with pytest.raises(ValueError, match="same supported mode"):
        PhysicalLayout(
            (
                ARM_LAYOUTS["servoj"][0],
                END_EFFECTOR_LAYOUTS["gripper"].left,
                ARM_LAYOUTS["servop"][1],
                END_EFFECTOR_LAYOUTS["gripper"].right,
            )
        )


@pytest.mark.parametrize(
    "components",
    [
        ["left_arm", "left_hand", "right_arm", "right_gripper"],
        ["left_arm", "left_gripper", "right_arm", "right_hand"],
    ],
)
def test_explicit_layout_rejects_mixed_robot_wide_end_effectors(components):
    with pytest.raises(ValueError, match="robot-wide end effector"):
        PhysicalLayout.resolve(components)


def test_layout_rejects_gripper_and_hand_together():
    with pytest.raises(ValueError, match="exactly one"):
        PhysicalLayout.resolve(
            ["left_arm", "left_gripper", "left_hand", "right_arm", "right_gripper"]
        )


@pytest.mark.parametrize(
    ("components", "message"),
    [
        (
            ["left_arm", "left_gripper", "right_arm", "right_gripper", "camera"],
            "Unknown",
        ),
        (
            ["left_arm", "left_gripper", "right_arm", "right_gripper", "head", "head"],
            "duplicate",
        ),
        (
            ["right_arm", "right_gripper", "left_arm", "left_gripper"],
            "canonical order",
        ),
        (["left_gripper", "right_arm", "right_gripper"], "left_arm"),
        (["left_arm", "left_gripper", "right_gripper"], "right_arm"),
    ],
)
def test_explicit_layout_retains_structural_validation(components, message):
    with pytest.raises(ValueError, match=message):
        PhysicalLayout.resolve(components)
