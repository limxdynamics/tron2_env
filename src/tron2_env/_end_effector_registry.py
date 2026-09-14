"""Atomic registry shared by end-effector layouts and provider factories."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Mapping, TypeVar


@dataclass(frozen=True)
class ComponentSpec:
    """One named, contiguous component of a physical vector."""

    name: str
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("component name must be a non-empty string")
        if isinstance(self.fields, (str, bytes)):
            raise ValueError(
                f"Component {self.name!r} fields must be a sequence of names"
            )
        try:
            fields = tuple(self.fields)
        except TypeError as exc:
            raise ValueError(
                f"Component {self.name!r} fields must be a sequence of names"
            ) from exc
        object.__setattr__(self, "fields", fields)
        if not fields:
            raise ValueError(f"Component {self.name!r} must contain at least one field")
        if any(not isinstance(field, str) or not field.strip() for field in fields):
            raise ValueError(
                f"Component {self.name!r} fields must be non-empty strings"
            )
        if len(set(fields)) != len(fields):
            raise ValueError(f"Component {self.name!r} must not contain duplicate fields")

    @property
    def width(self) -> int:
        return len(self.fields)


@dataclass(frozen=True)
class EndEffectorLayoutSpec:
    """The paired components contributed by one robot-wide end effector."""

    name: str
    left: ComponentSpec
    right: ComponentSpec

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("end effector layout name must be a non-empty string")
        if not isinstance(self.left, ComponentSpec) or not isinstance(
            self.right, ComponentSpec
        ):
            raise TypeError("end effector layout sides must be ComponentSpec instances")
        if self.left.name == self.right.name:
            raise ValueError(
                f"End effector {self.name!r} must use distinct component names"
            )
        if self.left.width != self.right.width:
            raise ValueError(
                f"End effector {self.name!r} must have equal left/right widths"
            )
        if set(self.left.fields) & set(self.right.fields):
            raise ValueError(
                f"End effector {self.name!r} left/right fields must not overlap"
            )

    @property
    def width_per_side(self) -> int:
        return self.left.width


ProviderFactory = Callable[[object, object], object]


@dataclass(frozen=True)
class _EndEffectorRegistration:
    name: str
    layout: EndEffectorLayoutSpec
    factory: ProviderFactory


_REGISTRY_LOCK = threading.RLock()
_REGISTRATIONS: dict[str, _EndEffectorRegistration] = {}

_RESERVED_COMPONENT_NAMES = frozenset(
    {"left_arm", "right_arm", "head", "chassis"}
)
_RESERVED_COMPONENT_FIELDS = frozenset(
    {
        *(f"{side}_arm_{index}" for side in ("left", "right") for index in range(7)),
        *(
            f"{side}_arm_{axis}"
            for side in ("left", "right")
            for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")
        ),
        "head_pitch",
        "head_yaw",
        "chassis_linear_x",
        "chassis_angular_z",
        "lifter_state",
    }
)


_ProjectedValue = TypeVar("_ProjectedValue")


class _RegistryProjection(Mapping[str, _ProjectedValue], Generic[_ProjectedValue]):
    """Read-only live view of one field in each immutable registration."""

    def __init__(
        self,
        project: Callable[[_EndEffectorRegistration], _ProjectedValue],
    ) -> None:
        self._project = project

    def __getitem__(self, name: str) -> _ProjectedValue:
        with _REGISTRY_LOCK:
            registration = _REGISTRATIONS[name]
        return self._project(registration)

    def __iter__(self) -> Iterator[str]:
        with _REGISTRY_LOCK:
            names = tuple(_REGISTRATIONS)
        return iter(names)

    def __len__(self) -> int:
        with _REGISTRY_LOCK:
            return len(_REGISTRATIONS)

    def __repr__(self) -> str:
        return repr(self.copy())

    def copy(self) -> dict[str, _ProjectedValue]:
        with _REGISTRY_LOCK:
            return {
                name: self._project(registration)
                for name, registration in _REGISTRATIONS.items()
            }


END_EFFECTOR_LAYOUTS: Mapping[str, EndEffectorLayoutSpec] = _RegistryProjection(
    lambda registration: registration.layout
)
END_EFFECTOR_PROVIDERS: Mapping[str, ProviderFactory] = _RegistryProjection(
    lambda registration: registration.factory
)


def _validate_layout(layout: EndEffectorLayoutSpec, expected_name: str) -> None:
    if not isinstance(layout, EndEffectorLayoutSpec):
        raise TypeError("end effector layout must be an EndEffectorLayoutSpec")
    if layout.name != expected_name:
        raise ValueError(
            f"end effector provider name {expected_name!r} must match "
            f"layout name {layout.name!r}"
        )

    # Recheck frozen-spec invariants defensively in case a caller bypassed the
    # dataclass constructors with object.__setattr__.
    for side, component in (("left", layout.left), ("right", layout.right)):
        if not isinstance(component, ComponentSpec):
            raise TypeError(f"end effector {side} component must be a ComponentSpec")
        if not isinstance(component.name, str) or not component.name.strip():
            raise ValueError(f"end effector {side} component name must be non-empty")
        if not component.fields or any(
            not isinstance(field, str) or not field.strip()
            for field in component.fields
        ):
            raise ValueError(
                f"end effector {side} component fields must be non-empty strings"
            )
        if len(set(component.fields)) != len(component.fields):
            raise ValueError(
                f"end effector {side} component fields must not contain duplicates"
            )

    if layout.left.name == layout.right.name:
        raise ValueError("end effector layout must use distinct component names")
    if layout.left.width != layout.right.width:
        raise ValueError("end effector layout must have equal left/right widths")
    if set(layout.left.fields) & set(layout.right.fields):
        raise ValueError("end effector layout left/right fields must not overlap")


def register_end_effector(
    name: str,
    factory: ProviderFactory,
    *,
    layout: EndEffectorLayoutSpec | None = None,
) -> None:
    """Validate and atomically add one immutable end-effector registration."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("end effector provider name must be a non-empty string")
    if not callable(factory):
        raise TypeError("end effector provider factory must be callable")

    with _REGISTRY_LOCK:
        if name in _REGISTRATIONS:
            raise ValueError(f"end effector provider {name!r} is already registered")
        if layout is None:
            raise ValueError(
                f"end effector layout is required for new provider {name!r}"
            )
        _validate_layout(layout, name)

        component_names = {layout.left.name, layout.right.name}
        registered_component_names = {
            component.name
            for registration in _REGISTRATIONS.values()
            for component in (registration.layout.left, registration.layout.right)
        }
        component_conflicts = component_names & (
            _RESERVED_COMPONENT_NAMES | registered_component_names
        )
        if component_conflicts:
            rendered = ", ".join(sorted(component_conflicts))
            raise ValueError(
                f"end effector component names conflict with reserved or "
                f"registered components: {rendered}"
            )

        fields = set(layout.left.fields) | set(layout.right.fields)
        registered_fields = {
            field
            for registration in _REGISTRATIONS.values()
            for component in (registration.layout.left, registration.layout.right)
            for field in component.fields
        }
        field_conflicts = fields & (_RESERVED_COMPONENT_FIELDS | registered_fields)
        if field_conflicts:
            rendered = ", ".join(sorted(field_conflicts))
            raise ValueError(
                "end effector fields conflict with reserved or registered "
                f"component fields: {rendered}"
            )

        registration = _EndEffectorRegistration(name, layout, factory)
        _REGISTRATIONS[name] = registration


def get_end_effector_registration(name: str) -> _EndEffectorRegistration:
    """Return one immutable registration under the registry lock."""

    with _REGISTRY_LOCK:
        return _REGISTRATIONS[name]


def _create_gripper(transport, config=None):
    if config is not None:
        raise ValueError("gripper provider does not accept BrainCo2Config")
    from tron2_env.end_effector.gripper import GripperProvider

    return GripperProvider(transport)


def _create_brainco2(transport, config=None):
    from tron2_env.end_effector.brainco2 import BrainCo2Provider
    from tron2_env.layout import BrainCo2Config

    if not isinstance(config, BrainCo2Config):
        raise ValueError("brainco2 provider requires BrainCo2Config")
    return BrainCo2Provider(transport, config)


register_end_effector(
    "gripper",
    _create_gripper,
    layout=EndEffectorLayoutSpec(
        name="gripper",
        left=ComponentSpec("left_gripper", ("left_gripper",)),
        right=ComponentSpec("right_gripper", ("right_gripper",)),
    ),
)
register_end_effector(
    "brainco2",
    _create_brainco2,
    layout=EndEffectorLayoutSpec(
        name="brainco2",
        left=ComponentSpec(
            "left_hand", tuple(f"left_hand_{index}" for index in range(6))
        ),
        right=ComponentSpec(
            "right_hand", tuple(f"right_hand_{index}" for index in range(6))
        ),
    ),
)
