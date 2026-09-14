"""Latest-only scheduling for ServoP head commands."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from tron2_env.errors import CommandError


@dataclass(frozen=True)
class _InFlightCommand:
    guid: str
    deadline: float


@dataclass(frozen=True)
class _FaultInfo:
    message: str
    clear_on_reconnect: bool = False


class HeadCommandScheduler:
    """Serialize head moves while retaining only the newest waiting target."""

    def __init__(
        self,
        sender: Callable[[np.ndarray, float, str], bool],
        min_interval: float,
        move_duration: float,
        ack_timeout: float,
        clock: Callable[[], float] = time.monotonic,
        guid_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        if not callable(sender):
            raise TypeError("sender must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(guid_factory):
            raise TypeError("guid_factory must be callable")

        self._min_interval = self._validate_time(
            min_interval,
            name="min_interval",
            allow_zero=True,
        )
        self._move_duration = self._validate_time(
            move_duration,
            name="move_duration",
            allow_zero=False,
        )
        self._ack_timeout = self._validate_time(
            ack_timeout,
            name="ack_timeout",
            allow_zero=False,
        )
        self._sender = sender
        self._clock = clock
        self._guid_factory = guid_factory
        self._lock = threading.RLock()
        self._active_send: Optional[str] = None
        self._in_flight: Optional[_InFlightCommand] = None
        self._pending: Optional[np.ndarray] = None
        self._next_send_at: Optional[float] = None
        self._fault: Optional[_FaultInfo] = None
        self._paused = False
        self._closed = False

    @staticmethod
    def _validate_time(value, *, name: str, allow_zero: bool) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite and positive") from exc
        invalid_bound = number < 0.0 if allow_zero else number <= 0.0
        if not np.isfinite(number) or invalid_bound:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")
        return number

    @staticmethod
    def _target(value) -> np.ndarray:
        try:
            target = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CommandError("head target must be finite shape (2,)") from exc
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise CommandError("head target must be finite shape (2,)")
        return target.copy()

    @staticmethod
    def _exception_detail(exc: Exception) -> str:
        try:
            detail = str(exc)
        except Exception:
            detail = ""
        name = type(exc).__name__
        return f"{name}: {detail}" if detail else name

    def _set_fault_locked(
        self,
        message: str,
        *,
        clear_on_reconnect: bool = False,
    ) -> None:
        if self._fault is None:
            self._fault = _FaultInfo(
                message=str(message),
                clear_on_reconnect=clear_on_reconnect,
            )
        self._in_flight = None
        self._pending = None

    def _read_clock_locked(self) -> Optional[float]:
        try:
            now = float(self._clock())
        except Exception as exc:
            self._set_fault_locked(
                f"head scheduler clock failed: {self._exception_detail(exc)}"
            )
            return None
        if not np.isfinite(now):
            self._set_fault_locked("head scheduler clock failed: non-finite value")
            return None
        return now

    def _latch_timeout_locked(self) -> None:
        if self._fault is not None or self._in_flight is None:
            return

        now = self._read_clock_locked()
        if now is None:
            return

        in_flight = self._in_flight
        if in_flight is not None and now >= in_flight.deadline:
            self._set_fault_locked("head command acknowledgement timed out")

    def _new_guid_locked(self) -> Optional[str]:
        try:
            guid = self._guid_factory()
        except Exception as exc:
            self._set_fault_locked(
                f"head command GUID creation failed: {self._exception_detail(exc)}"
            )
            return None
        if not isinstance(guid, str) or not guid.strip():
            self._set_fault_locked("head command GUID creation failed: invalid GUID")
            return None
        return guid

    def _reserve_send_locked(
        self,
        target: np.ndarray,
        now: float,
    ) -> Optional[tuple[np.ndarray, float, str]]:
        self._pending = None
        guid = self._new_guid_locked()
        if guid is None:
            return None

        self._active_send = guid
        self._in_flight = _InFlightCommand(
            guid=guid,
            deadline=now + self._ack_timeout,
        )
        self._next_send_at = now + self._min_interval
        return target.copy(), self._move_duration, guid

    def _send_reserved(
        self,
        reserved: tuple[np.ndarray, float, str],
        *,
        raise_on_failure: bool = False,
    ) -> None:
        target, duration, guid = reserved
        failure = None
        try:
            sent = self._sender(target, duration, guid)
        except Exception as exc:
            failure = (
                f"head command send failed: {self._exception_detail(exc)}"
            )
        else:
            if not sent:
                failure = "head command send failed"

        fault_message = None
        with self._lock:
            if self._active_send != guid:
                return
            self._latch_timeout_locked()
            self._active_send = None
            if failure is not None and not self._closed:
                # A synchronous success ACK may already have cleared in_flight,
                # but a False/exception return from the sender is still a send
                # failure and must remain visible to the submitter.
                self._set_fault_locked(failure, clear_on_reconnect=True)
            if raise_on_failure and self._fault is not None:
                # This includes a failed ACK delivered reentrantly by sender().
                fault_message = self._fault.message
        if fault_message is not None:
            raise CommandError(fault_message)

    def submit(self, target: np.ndarray) -> None:
        """Submit a target, raising failures completed within this call."""
        value = self._target(target)
        reserved = None
        fault_message = None
        with self._lock:
            if self._closed:
                return
            self._latch_timeout_locked()
            if self._fault is not None:
                fault_message = self._fault.message
            elif self._paused:
                return
            elif self._active_send is not None or self._in_flight is not None:
                self._pending = value
                return
            else:
                now = self._read_clock_locked()
                if now is None:
                    fault_message = self._fault.message
                elif self._next_send_at is None or now >= self._next_send_at:
                    reserved = self._reserve_send_locked(value, now)
                    if reserved is None and self._fault is not None:
                        fault_message = self._fault.message
                else:
                    self._pending = value
        if fault_message is not None:
            raise CommandError(fault_message)
        if reserved is not None:
            self._send_reserved(reserved, raise_on_failure=True)

    def tick(self) -> None:
        """Advance timeout and minimum-interval gates without replaying failures."""
        reserved = None
        with self._lock:
            if self._closed:
                return
            self._latch_timeout_locked()
            if self._fault is not None:
                return
            if self._in_flight is not None:
                return
            if (
                self._active_send is None
                and self._in_flight is None
                and (self._paused or self._pending is None)
            ):
                return
            now = self._read_clock_locked()
            if now is None:
                return
            if self._active_send is not None:
                return
            if self._paused or self._pending is None:
                return
            if self._next_send_at is not None and now < self._next_send_at:
                return
            target = self._pending
            self._pending = None
            reserved = self._reserve_send_locked(target, now)
        if reserved is not None:
            self._send_reserved(reserved)

    def handle_response(self, guid: Optional[str], result: object) -> None:
        """Consume a matching move-head acknowledgement."""
        with self._lock:
            self._latch_timeout_locked()
            if self._in_flight is None or guid != self._in_flight.guid:
                return
            self._in_flight = None
            if result != "success":
                self._set_fault_locked(f"head command failed: {result}")

    def raise_if_faulted(self) -> None:
        """Raise a fresh exception for the retained asynchronous fault."""
        with self._lock:
            self._latch_timeout_locked()
            message = None if self._fault is None else self._fault.message
        if message is not None:
            raise CommandError(message)

    def pause(self) -> None:
        """Close the send gate and discard work that has not been sent."""
        with self._lock:
            if self._closed:
                return
            self._paused = True
            self._pending = None

    def pause_for_reconnect(self) -> None:
        """Invalidate all command state owned by the lost transport session."""
        with self._lock:
            if self._closed:
                return
            self._latch_timeout_locked()
            self._paused = True
            self._pending = None
            self._in_flight = None
            self._active_send = None
            self._next_send_at = None
            if self._fault is not None and self._fault.clear_on_reconnect:
                self._fault = None

    def resume(self) -> None:
        """Re-open a paused scheduler without dispatching a command."""
        fault_message = None
        with self._lock:
            if self._closed:
                return
            self._latch_timeout_locked()
            if self._fault is not None:
                self._paused = True
                fault_message = self._fault.message
            else:
                self._paused = False
        if fault_message is not None:
            raise CommandError(fault_message)

    def close(self) -> None:
        """Permanently close the scheduler and discard all command state."""
        with self._lock:
            self._closed = True
            self._paused = True
            self._pending = None
            self._in_flight = None
            self._active_send = None
