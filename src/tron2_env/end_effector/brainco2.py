"""BrainCo2 paired-hand provider."""

from __future__ import annotations

import copy
import threading
from typing import Optional

import numpy as np

from tron2_env.errors import CommandError, StateError
from tron2_env.layout import BrainCo2Config
from tron2_env.transport.base import (
    BrainCo2SessionRecoveryTransport,
    BrainCo2Transport,
)


class BrainCo2Provider:
    """Own both BrainCo2 hands and their polling lifecycle."""

    name = "brainco2"
    width_per_side = 6

    def __init__(self, transport, config: BrainCo2Config) -> None:
        if not isinstance(config, BrainCo2Config):
            raise ValueError("brainco2 provider requires BrainCo2Config")
        if not isinstance(transport, BrainCo2Transport):
            raise TypeError("brainco2 provider requires BrainCo2Transport")
        enable_polling = getattr(transport, "enable_brainco2_polling", None)
        disable_polling = getattr(transport, "disable_brainco2_polling", None)
        if not callable(enable_polling) or not callable(disable_polling):
            raise TypeError("brainco2 transport requires polling lifecycle methods")
        self._transport = transport
        self._config = config
        self._command_time = config.expanded_command_time
        self._disable_polling = disable_polling
        self._lifecycle_lock = threading.RLock()
        self._paused = False
        self._closed = False
        self._polling_disabled = False
        enable_polling(config.polling_rate, config.state_timeout)

    @staticmethod
    def _position(value, side: str) -> np.ndarray:
        try:
            position = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CommandError(f"{side} position must be finite shape (6,)") from exc
        if position.shape != (6,) or not np.all(np.isfinite(position)):
            raise CommandError(f"{side} position must be finite shape (6,)")
        return position

    def read_state(
        self,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
    ) -> tuple[np.ndarray, dict]:
        effective_max_age = self._config.state_timeout
        if max_age is not None:
            try:
                requested_max_age = float(max_age)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_age must be finite and non-negative") from exc
            if not np.isfinite(requested_max_age) or requested_max_age < 0.0:
                raise ValueError("max_age must be finite and non-negative")
            effective_max_age = min(effective_max_age, requested_max_age)
        state = self._transport.get_brainco2_state(
            timeout=timeout,
            max_age=effective_max_age,
        )
        try:
            values = np.asarray(state["states"], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise StateError("BrainCo2 provider received invalid state") from exc
        if values.shape != (12,) or not np.all(np.isfinite(values)):
            raise StateError("BrainCo2 provider received invalid state")
        metadata = copy.deepcopy(state.get("metadata", {}))
        return values.copy(), metadata

    def command(self, left: np.ndarray, right: np.ndarray) -> None:
        with self._lifecycle_lock:
            self._raise_if_command_blocked()
            self.raise_if_faulted()
            left_position, right_position = self.validate_command(left, right)
            self._transport.set_brainco2_position(
                left_position,
                right_position,
                self._command_time,
            )

    def validate_command(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate paired raw hand positions without touching provider state."""
        return self._position(left, "left"), self._position(right, "right")

    def pause(self) -> None:
        with self._lifecycle_lock:
            if not self._closed:
                self._paused = True

    def resume(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("brainco2 provider is closed")
            self._paused = False

    def recover_after_reconnect(self, generation: int) -> None:
        """Clear only a lost-session ACK fault after current-session feedback."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("brainco2 provider is closed")
            if not isinstance(self._transport, BrainCo2SessionRecoveryTransport):
                raise RuntimeError(
                    "brainco2 transport does not support session recovery"
                )
            self._transport.clear_brainco2_session_uncertainty(generation)

    def close(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
            self._paused = False
            if self._polling_disabled:
                return
            self._disable_polling()
            self._polling_disabled = True

    def raise_if_faulted(self) -> None:
        self._transport.raise_brainco2_fault()

    def _raise_if_command_blocked(self) -> None:
        if self._closed:
            raise RuntimeError("brainco2 provider is closed")
        if self._paused:
            raise RuntimeError("brainco2 provider is paused")
