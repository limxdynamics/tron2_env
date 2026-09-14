"""Robot-wide end-effector provider protocol and controlled registry."""

from __future__ import annotations

from typing import Callable, Mapping, Protocol, cast, runtime_checkable

import numpy as np

from tron2_env._end_effector_registry import (
    END_EFFECTOR_PROVIDERS as _END_EFFECTOR_PROVIDERS,
    EndEffectorLayoutSpec,
    get_end_effector_registration,
    register_end_effector,
)


@runtime_checkable
class EndEffectorProvider(Protocol):
    """One provider owns both left and right end effectors."""

    name: str
    width_per_side: int

    def read_state(self, timeout: float = 1.0) -> tuple[np.ndarray, dict]: ...

    def validate_command(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def command(self, left: np.ndarray, right: np.ndarray) -> None: ...

    def pause(self) -> None: ...

    def close(self) -> None: ...

    def raise_if_faulted(self) -> None: ...


ProviderFactory = Callable[[object, object], EndEffectorProvider]
END_EFFECTOR_PROVIDERS = cast(
    Mapping[str, ProviderFactory],
    _END_EFFECTOR_PROVIDERS,
)


def register_end_effector_provider(
    name: str,
    factory: ProviderFactory,
    *,
    layout: EndEffectorLayoutSpec | None = None,
) -> None:
    """Atomically register one matching layout and provider factory."""

    register_end_effector(name, factory, layout=layout)


def _best_effort_close(provider: object) -> None:
    try:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    except BaseException:
        # Validation errors describe the broken provider contract. Cleanup is
        # best-effort and must never replace that original error.
        pass


def create_end_effector_provider(
    name: str,
    transport,
    config=None,
) -> EndEffectorProvider:
    """Build the named robot-wide provider from the immutable registry."""
    try:
        registration = get_end_effector_registration(name)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown end effector provider: {name!r}") from exc

    provider = registration.factory(transport, config)
    try:
        for method_name in (
            "read_state",
            "validate_command",
            "command",
            "pause",
            "close",
            "raise_if_faulted",
        ):
            if not callable(getattr(provider, method_name, None)):
                raise TypeError(
                    f"factory for {name!r} did not return an EndEffectorProvider "
                    f"with callable {method_name}()"
                )
        if not isinstance(provider, EndEffectorProvider):
            raise TypeError(
                f"factory for {name!r} did not return an EndEffectorProvider"
            )
        if provider.name != registration.name:
            raise ValueError(
                f"factory for {name!r} returned provider name {provider.name!r}; "
                f"expected {registration.name!r}"
            )
        expected_width = registration.layout.width_per_side
        if (
            isinstance(provider.width_per_side, bool)
            or not isinstance(provider.width_per_side, int)
            or provider.width_per_side != expected_width
        ):
            raise ValueError(
                f"factory for {name!r} expected width_per_side {expected_width}, "
                f"got {provider.width_per_side!r}"
            )
    except BaseException:
        _best_effort_close(provider)
        raise
    return provider
