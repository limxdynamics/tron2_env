"""Explicit physical state/action layouts used by TRON2 policies.

The data platform stores every physical signal in one flat vector. This
module keeps the flat representation configurable without guessing semantics
from the vector length.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np

from tron2_env._end_effector_registry import (
    END_EFFECTOR_LAYOUTS,
    ComponentSpec,
    EndEffectorLayoutSpec,
)


def _validated_float32_vector(
    value,
    *,
    width: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the source vector and its finite float32 policy representation."""
    try:
        vector = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric finite shape ({width},)") from exc
    if vector.shape != (width,):
        raise ValueError(
            f"{name} must have shape ({width},), got {vector.shape}"
        )
    if (
        not (
            np.issubdtype(vector.dtype, np.integer)
            or np.issubdtype(vector.dtype, np.floating)
        )
        or not np.all(np.isfinite(vector))
    ):
        raise ValueError(f"{name} must be a numeric finite shape ({width},)")
    with np.errstate(over="ignore", invalid="ignore"):
        float32_vector = vector.astype(np.float32, copy=False)
    if not np.all(np.isfinite(float32_vector)):
        raise ValueError(f"{name} must remain finite after float32 conversion")
    return vector, float32_vector


def _arm_component(side: str, mode: str) -> ComponentSpec:
    if mode == "servoj":
        fields = tuple(f"{side}_arm_{index}" for index in range(7))
    else:
        fields = tuple(
            f"{side}_arm_{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")
        )
    return ComponentSpec(f"{side}_arm", fields)


_ARM_LAYOUTS: dict[str, tuple[ComponentSpec, ComponentSpec]] = {
    mode: (_arm_component("left", mode), _arm_component("right", mode))
    for mode in ("servoj", "servop")
}

ARM_LAYOUTS: Mapping[str, tuple[ComponentSpec, ComponentSpec]] = MappingProxyType(
    _ARM_LAYOUTS
)

HEAD_COMPONENT = ComponentSpec("head", ("head_pitch", "head_yaw"))
CHASSIS_COMPONENT = ComponentSpec(
    "chassis", ("chassis_linear_x", "chassis_angular_z", "lifter_state")
)

# These names remain public for callers that imported the legacy constants.
# PhysicalLayout itself uses its instance-owned ComponentSpec objects.
COMPONENT_FIELDS: dict[str, tuple[str, ...]] = {
    spec.name: spec.fields
    for spec in (
        _ARM_LAYOUTS["servoj"][0],
        END_EFFECTOR_LAYOUTS["gripper"].left,
        END_EFFECTOR_LAYOUTS["brainco2"].left,
        _ARM_LAYOUTS["servoj"][1],
        END_EFFECTOR_LAYOUTS["gripper"].right,
        END_EFFECTOR_LAYOUTS["brainco2"].right,
        HEAD_COMPONENT,
        CHASSIS_COMPONENT,
    )
}
CANONICAL_COMPONENT_ORDER = tuple(COMPONENT_FIELDS)
DEFAULT_LAYOUT = ("left_arm", "left_gripper", "right_arm", "right_gripper")
DEFAULT_LAYOUT_WITH_HEAD = (*DEFAULT_LAYOUT, "head")


@dataclass(frozen=True)
class RobotModules:
    """Immutable robot-wide module selection used to derive a layout."""

    arm: Literal["servoj", "servop"] = "servoj"
    end_effector: str = "gripper"
    head: bool = False
    mobile_base: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.arm, str) or self.arm not in _ARM_LAYOUTS:
            raise ValueError("arm must be 'servoj' or 'servop'")
        if (
            not isinstance(self.end_effector, str)
            or self.end_effector not in END_EFFECTOR_LAYOUTS
        ):
            raise ValueError(f"unknown end effector: {self.end_effector!r}")
        if not isinstance(self.head, bool):
            raise ValueError("head must be a bool")
        if not isinstance(self.mobile_base, bool):
            raise ValueError("mobile_base must be a bool")


@dataclass(frozen=True)
class BrainCo2Config:
    """Configuration shared by both BrainCo2 hands."""

    command_time: float | Sequence[float] | None
    polling_rate: float = 20.0
    state_timeout: float = 0.5

    def __post_init__(self) -> None:
        # Resolve eagerly so invalid command timing fails at configuration time.
        command_time = self.expanded_command_time
        object.__setattr__(self, "command_time", command_time)
        rates: list[tuple[str, float]] = []
        for name in ("polling_rate", "state_timeout"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "BrainCo2 polling_rate/state_timeout must be positive"
                ) from exc
            rates.append((name, value))
        if any(not np.isfinite(value) or value <= 0 for _, value in rates):
            raise ValueError("BrainCo2 polling_rate/state_timeout must be positive")
        for name, value in rates:
            object.__setattr__(self, name, value)

    @property
    def expanded_command_time(self) -> tuple[float, ...]:
        if self.command_time is None:
            raise ValueError("BrainCo2 command_time is required")
        try:
            values = np.asarray(self.command_time, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "BrainCo2 command_time must be a scalar or exactly 6 values"
            ) from exc
        if values.ndim == 0:
            values = np.repeat(values, 6)
        if values.shape != (6,):
            raise ValueError(
                "BrainCo2 command_time must be a scalar or exactly 6 values"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("BrainCo2 command_time must be finite")
        return tuple(float(value) for value in values)


def _registered_component_specs() -> dict[str, ComponentSpec]:
    """Return legacy ServoJ specs plus the current end-effector registry."""

    left_arm, right_arm = _ARM_LAYOUTS["servoj"]
    specs = {
        left_arm.name: left_arm,
        right_arm.name: right_arm,
        HEAD_COMPONENT.name: HEAD_COMPONENT,
        CHASSIS_COMPONENT.name: CHASSIS_COMPONENT,
    }
    for end_effector in END_EFFECTOR_LAYOUTS.values():
        specs[end_effector.left.name] = end_effector.left
        specs[end_effector.right.name] = end_effector.right
    return specs


def _canonical_component_ranks() -> dict[str, int]:
    ranks = {"left_arm": 0, "right_arm": 2, "head": 4, "chassis": 5}
    for end_effector in END_EFFECTOR_LAYOUTS.values():
        ranks[end_effector.left.name] = 1
        ranks[end_effector.right.name] = 3
    return ranks


@dataclass(frozen=True, init=False)
class PhysicalLayout:
    """Validated ordered mapping between named hardware groups and a flat vector."""

    _component_specs: tuple[ComponentSpec, ...] = field(repr=False)
    _end_effector_spec: EndEffectorLayoutSpec = field(repr=False)

    def __init__(self, components: Sequence[str] | Sequence[ComponentSpec]) -> None:
        if isinstance(components, (str, bytes)):
            raise ValueError("state_layout must be a YAML list, not a string")
        items = tuple(components)
        if all(isinstance(item, ComponentSpec) for item in items):
            specs = items
        elif any(isinstance(item, ComponentSpec) for item in items):
            raise ValueError("state_layout must not mix component names and specifications")
        else:
            registered = _registered_component_specs()
            names = tuple(str(item) for item in items)
            unknown = [name for name in names if name not in registered]
            if unknown:
                raise ValueError(
                    f"Unknown state_layout components: {', '.join(unknown)}"
                )
            specs = tuple(registered[name] for name in names)

        object.__setattr__(self, "_component_specs", specs)
        end_effector_spec = self._validate()
        object.__setattr__(self, "_end_effector_spec", end_effector_spec)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(components={self.components!r}, dim={self.dim})"
        )

    def _validate(self) -> EndEffectorLayoutSpec:
        if not self._component_specs:
            raise ValueError("state_layout must contain at least one component")

        names = self.components
        registered = _registered_component_specs()
        unknown = [name for name in names if name not in registered]
        if unknown:
            raise ValueError(f"Unknown state_layout components: {', '.join(unknown)}")
        if len(set(names)) != len(names):
            raise ValueError("state_layout must not contain duplicate components")

        ranks = _canonical_component_ranks()
        positions = [ranks[name] for name in names]
        if positions != sorted(positions):
            raise ValueError(
                "state_layout must follow canonical order: left_arm, "
                "left_end_effector, right_arm, right_end_effector, head, chassis"
            )

        for side in ("left", "right"):
            arm = f"{side}_arm"
            if arm not in names:
                raise ValueError(f"state_layout is missing required component {arm!r}")

        specs_by_name = {spec.name: spec for spec in self._component_specs}
        if not any(
            specs_by_name["left_arm"] == left_arm
            and specs_by_name["right_arm"] == right_arm
            for left_arm, right_arm in _ARM_LAYOUTS.values()
        ):
            raise ValueError(
                "state_layout arm components must be a complete pair from the "
                "same supported mode"
            )

        for component in (HEAD_COMPONENT, CHASSIS_COMPONENT):
            selected_component = specs_by_name.get(component.name)
            if selected_component is not None and selected_component != component:
                raise ValueError(
                    f"Component {component.name!r} must match its registered schema"
                )

        left_matches = [
            spec
            for spec in END_EFFECTOR_LAYOUTS.values()
            if spec.left.name in names
        ]
        right_matches = [
            spec
            for spec in END_EFFECTOR_LAYOUTS.values()
            if spec.right.name in names
        ]
        if len(left_matches) != 1 or len(right_matches) != 1:
            raise ValueError(
                "state_layout must contain exactly one end effector for each side"
            )
        if left_matches[0].name != right_matches[0].name:
            raise ValueError(
                "state_layout must use one robot-wide end effector on both sides"
            )

        selected = left_matches[0]
        for registered_component in (selected.left, selected.right):
            if specs_by_name[registered_component.name] != registered_component:
                raise ValueError(
                    f"Component {registered_component.name!r} must match its "
                    "registered schema"
                )

        fields = tuple(field for spec in self._component_specs for field in spec.fields)
        if len(set(fields)) != len(fields):
            raise ValueError("state_layout component fields must not overlap")

        return selected

    @classmethod
    def from_modules(cls, modules: RobotModules) -> "PhysicalLayout":
        """Build the canonical physical layout for an explicit module selection."""

        if not isinstance(modules, RobotModules):
            raise TypeError("modules must be a RobotModules instance")
        left_arm, right_arm = _ARM_LAYOUTS[modules.arm]
        end_effector = END_EFFECTOR_LAYOUTS[modules.end_effector]
        specs = [left_arm, end_effector.left, right_arm, end_effector.right]
        if modules.head:
            specs.append(HEAD_COMPONENT)
        if modules.mobile_base:
            specs.append(CHASSIS_COMPONENT)
        return cls(specs)

    @classmethod
    def resolve(
        cls,
        components: Sequence[str] | None = None,
        state_dim: int | None = None,
    ) -> "PhysicalLayout":
        """Resolve a layout, retaining 16/18D legacy config compatibility.

        New dimensions require ``state_layout`` so a value such as 19 cannot be
        silently assigned the wrong physical meaning. Explicit component names
        retain the legacy ServoJ arm interpretation.
        """

        if state_dim is not None:
            if isinstance(state_dim, (bool, np.bool_)) or not isinstance(
                state_dim,
                (int, np.integer),
            ):
                raise ValueError("state_dim must be an integer")
            state_dim = int(state_dim)

        if components is None:
            dim = 16 if state_dim is None else state_dim
            if dim == 16:
                return cls.from_modules(RobotModules())
            if dim == 18:
                return cls.from_modules(RobotModules(head=True))
            raise ValueError(
                f"state_dim={dim} is ambiguous; set state_layout explicitly "
                "for non-16/18D policies"
            )

        layout = cls(components)
        if state_dim is not None and state_dim != layout.dim:
            raise ValueError(
                f"state_dim={state_dim} does not match state_layout "
                f"dimension {layout.dim}"
            )
        return layout

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._component_specs)

    @property
    def dim(self) -> int:
        return sum(spec.width for spec in self._component_specs)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field for spec in self._component_specs for field in spec.fields)

    @property
    def left_end_effector_component(self) -> str:
        return self._end_effector_spec.left.name

    @property
    def right_end_effector_component(self) -> str:
        return self._end_effector_spec.right.name

    @property
    def end_effector_name(self) -> str:
        return self._end_effector_spec.name

    @property
    def end_effector_width(self) -> int:
        return self._end_effector_spec.width_per_side

    def has(self, component: str) -> bool:
        return component in self.components

    def component_slice(self, component: str) -> slice:
        start = 0
        for spec in self._component_specs:
            if spec.name == component:
                return slice(start, start + spec.width)
            start += spec.width
        raise KeyError(f"Component {component!r} is not present in state_layout")

    def pack(self, values: Mapping[str, Sequence[float] | np.ndarray]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for spec in self._component_specs:
            if spec.name not in values:
                raise ValueError(
                    f"Missing value for state_layout component {spec.name!r}"
                )
            _source, part = _validated_float32_vector(
                values[spec.name],
                width=spec.width,
                name=f"Component {spec.name!r}",
            )
            parts.append(part)
        return np.concatenate(parts)

    def unpack(self, vector: Sequence[float] | np.ndarray) -> dict[str, np.ndarray]:
        """Validate float32 policy safety and split without changing source dtype."""
        array, _float32_vector = _validated_float32_vector(
            vector,
            width=self.dim,
            name="Physical vector",
        )
        return {name: array[self.component_slice(name)] for name in self.components}
