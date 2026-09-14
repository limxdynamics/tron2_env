"""Deterministic tests for the ServoP head command scheduler."""

from __future__ import annotations

import itertools
import threading

import numpy as np
import pytest

from tron2_env.errors import CommandError
from tron2_env.motion.head_scheduler import HeadCommandScheduler


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def make_scheduler(
    *,
    min_interval: float = 0.1,
    move_duration: float = 0.05,
    ack_timeout: float = 1.0,
    sender=None,
    clock=None,
):
    sent = []
    fake_clock = clock or FakeClock()
    guids = (f"head-{index}" for index in itertools.count(1))

    def recording_sender(target, duration, guid):
        sent.append((target.copy(), duration, guid))
        return True

    scheduler = HeadCommandScheduler(
        sender=sender or recording_sender,
        min_interval=min_interval,
        move_duration=move_duration,
        ack_timeout=ack_timeout,
        clock=fake_clock,
        guid_factory=lambda: next(guids),
    )
    return scheduler, sent, fake_clock


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_interval": -0.01},
        {"min_interval": np.nan},
        {"min_interval": np.inf},
        {"move_duration": 0.0},
        {"move_duration": -0.01},
        {"move_duration": np.nan},
        {"ack_timeout": 0.0},
        {"ack_timeout": -0.01},
        {"ack_timeout": np.inf},
    ],
)
def test_head_scheduler_rejects_invalid_time_configuration(kwargs):
    with pytest.raises(ValueError):
        HeadCommandScheduler(
            sender=lambda *_args: True,
            min_interval=kwargs.get("min_interval", 0.0),
            move_duration=kwargs.get("move_duration", 0.1),
            ack_timeout=kwargs.get("ack_timeout", 1.0),
        )


@pytest.mark.parametrize(
    "target",
    [
        [0.0],
        [0.0, 0.0, 0.0],
        [[0.0, 0.0]],
        [np.nan, 0.0],
        [0.0, np.inf],
        ["bad", 0.0],
    ],
)
def test_head_scheduler_rejects_non_finite_or_non_2d_targets(target):
    scheduler, sent, _clock = make_scheduler()

    with pytest.raises(CommandError, match="finite shape \\(2,\\)"):
        scheduler.submit(target)

    assert sent == []


def test_head_scheduler_sends_first_target_with_duration_and_reserved_guid():
    scheduler, sent, _clock = make_scheduler(move_duration=0.25)

    scheduler.submit(np.array([0.1, -0.2]))

    assert len(sent) == 1
    np.testing.assert_allclose(sent[0][0], [0.1, -0.2])
    assert sent[0][1:] == (0.25, "head-1")


def test_head_scheduler_keeps_only_latest_pending_and_honors_min_interval():
    scheduler, sent, clock = make_scheduler(min_interval=0.1)
    scheduler.submit([0.1, 0.2])
    scheduler.submit([0.3, 0.4])
    scheduler.submit([0.5, 0.6])

    scheduler.handle_response("head-1", "success")
    scheduler.tick()
    assert len(sent) == 1

    clock.advance(0.1)
    scheduler.tick()

    assert len(sent) == 2
    np.testing.assert_allclose(sent[1][0], [0.5, 0.6])
    assert sent[1][2] == "head-2"


def test_due_submit_replaces_stale_pending_instead_of_replaying_it():
    scheduler, sent, clock = make_scheduler(min_interval=0.1)
    scheduler.submit([0.1, 0.2])
    scheduler.submit([0.3, 0.4])
    scheduler.handle_response("head-1", "success")
    clock.advance(0.1)

    scheduler.submit([0.5, 0.6])
    scheduler.handle_response("head-2", "success")
    clock.advance(0.1)
    scheduler.tick()

    assert len(sent) == 2
    np.testing.assert_allclose(sent[1][0], [0.5, 0.6])


def test_head_scheduler_copies_immediate_and_pending_targets():
    scheduler, sent, clock = make_scheduler(min_interval=0.1)
    immediate = np.array([0.1, 0.2])
    pending = np.array([0.3, 0.4])

    scheduler.submit(immediate)
    scheduler.submit(pending)
    immediate[:] = 9.0
    pending[:] = 8.0
    scheduler.handle_response("head-1", "success")
    clock.advance(0.1)
    scheduler.tick()

    np.testing.assert_allclose(sent[0][0], [0.1, 0.2])
    np.testing.assert_allclose(sent[1][0], [0.3, 0.4])


def test_unmatched_ack_does_not_release_inflight_or_dispatch_pending():
    scheduler, sent, clock = make_scheduler(min_interval=0.1)
    scheduler.submit([0.1, 0.2])
    scheduler.submit([0.3, 0.4])

    scheduler.handle_response("unmatched", "success")
    clock.advance(0.1)
    scheduler.tick()
    assert len(sent) == 1

    scheduler.handle_response("head-1", "success")
    scheduler.tick()
    assert len(sent) == 2


def test_failed_ack_faults_and_discards_pending_without_replay():
    scheduler, sent, clock = make_scheduler()
    scheduler.submit([0.1, 0.2])
    scheduler.submit([0.3, 0.4])

    scheduler.handle_response("head-1", "failed")
    clock.advance(2.0)
    scheduler.tick()
    with pytest.raises(CommandError, match="head command failed.*failed"):
        scheduler.submit([0.5, 0.6])

    assert len(sent) == 1
    with pytest.raises(CommandError, match="head command failed.*failed"):
        scheduler.raise_if_faulted()


def test_ack_timeout_faults_and_never_replays_pending_or_late_ack():
    scheduler, sent, clock = make_scheduler(ack_timeout=0.5)
    scheduler.submit([0.1, 0.2])
    scheduler.submit([0.3, 0.4])

    clock.advance(0.5)
    scheduler.tick()
    scheduler.handle_response("head-1", "success")
    clock.advance(1.0)
    scheduler.tick()

    assert len(sent) == 1
    with pytest.raises(CommandError, match="head command acknowledgement timed out"):
        scheduler.raise_if_faulted()


@pytest.mark.parametrize("failure_mode", ["false", "exception"])
@pytest.mark.parametrize("dispatch_origin", ["submit", "tick"])
def test_send_failure_or_exception_faults_without_retry(
    failure_mode,
    dispatch_origin,
):
    attempts = []

    def sender(target, duration, guid):
        attempts.append((target.copy(), duration, guid))
        if dispatch_origin == "tick" and len(attempts) == 1:
            return True
        if failure_mode == "exception":
            raise RuntimeError("socket exploded")
        return False

    scheduler, _sent, clock = make_scheduler(sender=sender)
    if dispatch_origin == "submit":
        with pytest.raises(CommandError, match="head command send failed") as raised:
            scheduler.submit([0.1, 0.2])
        assert raised.value.__cause__ is None
    else:
        scheduler.submit([0.1, 0.2])
        scheduler.submit([0.3, 0.4])
        scheduler.handle_response("head-1", "success")
        clock.advance(0.1)
        # Publish-loop dispatch failures remain latched instead of escaping tick().
        scheduler.tick()

    clock.advance(5.0)
    scheduler.tick()
    with pytest.raises(CommandError, match="head command send failed"):
        scheduler.submit([0.5, 0.6])

    assert len(attempts) == (1 if dispatch_origin == "submit" else 2)
    with pytest.raises(CommandError, match="head command send failed") as raised:
        scheduler.raise_if_faulted()
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("failure_mode", ["false", "exception"])
def test_reconnect_clears_transport_send_fault_and_allows_new_send(
    failure_mode,
):
    attempts = []

    def sender(target, duration, guid):
        attempts.append((target.copy(), duration, guid))
        if len(attempts) == 1:
            if failure_mode == "exception":
                raise RuntimeError("socket exploded")
            return False
        return True

    scheduler, _sent, _clock = make_scheduler(sender=sender)
    with pytest.raises(CommandError, match="head command send failed"):
        scheduler.submit([0.1, 0.2])

    scheduler.pause_for_reconnect()
    scheduler.resume()
    scheduler.submit([0.3, 0.4])

    assert [attempt[2] for attempt in attempts] == ["head-1", "head-2"]
    np.testing.assert_allclose(attempts[1][0], [0.3, 0.4])
    scheduler.handle_response("head-2", "success")
    scheduler.raise_if_faulted()


def test_synchronous_success_ack_does_not_deadlock_or_resurrect_guid():
    holder = {}
    sent = []

    def sender(target, duration, guid):
        sent.append((target.copy(), duration, guid))
        holder["scheduler"].handle_response(guid, "success")
        return True

    scheduler, _unused, _clock = make_scheduler(
        min_interval=0.0,
        sender=sender,
    )
    holder["scheduler"] = scheduler

    worker = threading.Thread(
        target=lambda: (
            scheduler.submit([0.1, 0.2]),
            scheduler.submit([0.3, 0.4]),
        ),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert [item[2] for item in sent] == ["head-1", "head-2"]


def test_synchronous_failed_ack_fault_is_not_erased_after_sender_returns():
    holder = {}

    def sender(_target, _duration, guid):
        holder["scheduler"].handle_response(guid, "controller rejected")
        return True

    scheduler, _sent, _clock = make_scheduler(sender=sender)
    holder["scheduler"] = scheduler

    with pytest.raises(CommandError, match="controller rejected"):
        scheduler.submit([0.1, 0.2])

    with pytest.raises(CommandError, match="controller rejected"):
        scheduler.raise_if_faulted()


def test_cross_thread_synchronous_ack_completes_before_sender_returns():
    holder = {}
    ack_completed = threading.Event()
    ack_completed_inside_sender = []
    ack_threads = []

    def sender(_target, _duration, guid):
        def acknowledge():
            holder["scheduler"].handle_response(guid, "success")
            ack_completed.set()

        ack_thread = threading.Thread(target=acknowledge, daemon=True)
        ack_threads.append(ack_thread)
        ack_thread.start()
        ack_completed_inside_sender.append(ack_completed.wait(timeout=0.2))
        return True

    scheduler, _sent, _clock = make_scheduler(sender=sender)
    holder["scheduler"] = scheduler

    scheduler.submit([0.1, 0.2])
    for ack_thread in ack_threads:
        ack_thread.join(timeout=0.5)

    assert ack_completed_inside_sender == [True]
    assert all(not ack_thread.is_alive() for ack_thread in ack_threads)


@pytest.mark.parametrize(
    ("lifecycle_action", "send_result"),
    [("pause", True), ("close", False)],
)
def test_lifecycle_action_does_not_wait_for_blocked_sender(
    lifecycle_action,
    send_result,
):
    sender_entered = threading.Event()
    release_sender = threading.Event()
    action_completed = threading.Event()
    send_calls = []

    def sender(target, duration, guid):
        send_calls.append((target.copy(), duration, guid))
        sender_entered.set()
        assert release_sender.wait(timeout=1.0)
        return send_result

    scheduler, _sent, clock = make_scheduler(
        min_interval=0.0,
        sender=sender,
    )
    submit_thread = threading.Thread(
        target=lambda: scheduler.submit([0.1, 0.2]),
        daemon=True,
    )
    submit_thread.start()
    assert sender_entered.wait(timeout=0.5)

    def run_lifecycle_action():
        getattr(scheduler, lifecycle_action)()
        action_completed.set()

    action_thread = threading.Thread(target=run_lifecycle_action, daemon=True)
    action_thread.start()
    try:
        assert action_completed.wait(timeout=0.2)
    finally:
        release_sender.set()
        submit_thread.join(timeout=0.5)
        action_thread.join(timeout=0.5)

    assert not submit_thread.is_alive()
    assert not action_thread.is_alive()
    scheduler.handle_response("head-1", "success")
    scheduler.submit([0.3, 0.4])
    clock.advance(1.0)
    scheduler.tick()
    assert len(send_calls) == 1
    if lifecycle_action == "close":
        scheduler.raise_if_faulted()


def test_reconnect_pause_invalidates_blocked_send_without_resurrection():
    sender_entered = threading.Event()
    release_sender = threading.Event()
    reconnect_completed = threading.Event()
    send_calls = []

    def sender(target, duration, guid):
        send_calls.append((target.copy(), duration, guid))
        if guid == "head-1":
            sender_entered.set()
            release_sender.wait(timeout=1.0)
        return True

    scheduler, _sent, _clock = make_scheduler(
        min_interval=0.0,
        sender=sender,
    )
    submit_thread = threading.Thread(
        target=lambda: scheduler.submit([0.1, 0.2]),
        daemon=True,
    )

    def pause_for_reconnect():
        scheduler.pause_for_reconnect()
        reconnect_completed.set()

    reconnect_thread = threading.Thread(
        target=pause_for_reconnect,
        daemon=True,
    )
    submit_started = False
    reconnect_started = False
    try:
        submit_thread.start()
        submit_started = True
        assert sender_entered.wait(timeout=0.5)

        reconnect_thread.start()
        reconnect_started = True
        assert reconnect_completed.wait(timeout=0.2)
        assert scheduler._active_send is None
        assert scheduler._in_flight is None

        scheduler.resume()
        scheduler.submit([0.3, 0.4])
    finally:
        release_sender.set()
        if submit_started:
            submit_thread.join(timeout=0.5)
        if reconnect_started:
            reconnect_thread.join(timeout=0.5)

    assert not submit_thread.is_alive()
    assert not reconnect_thread.is_alive()
    assert [call[2] for call in send_calls] == ["head-1", "head-2"]
    assert scheduler._in_flight is not None
    assert scheduler._in_flight.guid == "head-2"
    scheduler.handle_response("head-2", "success")
    scheduler.raise_if_faulted()


@pytest.mark.parametrize("send_outcome", ["true", "false", "exception"])
@pytest.mark.parametrize("dispatch_origin", ["submit", "tick"])
def test_blocked_send_crossing_deadline_latches_non_reconnectable_timeout(
    send_outcome,
    dispatch_origin,
):
    sender_entered = threading.Event()
    release_sender = threading.Event()
    send_calls = []
    dispatch_errors = []
    clock = FakeClock()

    def sender(target, duration, guid):
        send_calls.append((target.copy(), duration, guid))
        blocked_call = dispatch_origin == "submit" or len(send_calls) == 2
        if not blocked_call:
            return True
        sender_entered.set()
        release_sender.wait(timeout=1.0)
        if send_outcome == "exception":
            raise RuntimeError("socket exploded")
        return send_outcome == "true"

    scheduler, _sent, _clock = make_scheduler(
        ack_timeout=0.5,
        sender=sender,
        clock=clock,
    )
    if dispatch_origin == "tick":
        scheduler.submit([0.1, 0.2])
        scheduler.submit([0.3, 0.4])
        scheduler.handle_response("head-1", "success")
        clock.advance(0.1)

    def dispatch():
        try:
            if dispatch_origin == "submit":
                scheduler.submit([0.1, 0.2])
            else:
                scheduler.tick()
        except Exception as exc:
            dispatch_errors.append(exc)

    dispatch_thread = threading.Thread(target=dispatch, daemon=True)
    dispatch_started = False
    try:
        dispatch_thread.start()
        dispatch_started = True
        assert sender_entered.wait(timeout=0.5)
        clock.advance(0.5)
    finally:
        release_sender.set()
        if dispatch_started:
            dispatch_thread.join(timeout=0.5)

    assert not dispatch_thread.is_alive()
    if dispatch_origin == "submit":
        assert len(dispatch_errors) == 1
        submitted_error = dispatch_errors[0]
        assert isinstance(submitted_error, CommandError)
        assert str(submitted_error) == "head command acknowledgement timed out"
        assert submitted_error.__cause__ is None
    else:
        assert dispatch_errors == []

    with pytest.raises(CommandError, match="head command acknowledgement timed out"):
        scheduler.raise_if_faulted()
    scheduler.pause_for_reconnect()
    with pytest.raises(
        CommandError,
        match="head command acknowledgement timed out",
    ) as resumed:
        scheduler.resume()

    assert scheduler._paused is True
    if dispatch_origin == "submit":
        assert resumed.value is not submitted_error
        assert resumed.value.__cause__ is None


def test_pause_clears_pending_and_resume_does_not_replay_it():
    scheduler, sent, clock = make_scheduler()
    scheduler.submit([0.1, 0.2])
    scheduler.submit([0.3, 0.4])
    scheduler.pause()
    scheduler.submit([0.5, 0.6])
    scheduler.handle_response("head-1", "success")
    clock.advance(1.0)
    scheduler.tick()

    scheduler.resume()
    scheduler.tick()
    assert len(sent) == 1

    scheduler.submit([0.7, 0.8])
    assert len(sent) == 2
    np.testing.assert_allclose(sent[1][0], [0.7, 0.8])


def test_pause_retains_ack_deadline_for_raise_if_faulted():
    scheduler, sent, clock = make_scheduler(ack_timeout=0.5)
    scheduler.submit([0.1, 0.2])
    scheduler.pause()

    clock.advance(0.5)

    with pytest.raises(CommandError, match="head command acknowledgement timed out"):
        scheduler.raise_if_faulted()
    assert len(sent) == 1


def test_late_success_ack_during_pause_does_not_erase_timeout():
    scheduler, sent, clock = make_scheduler(ack_timeout=0.5)
    scheduler.submit([0.1, 0.2])
    scheduler.pause()

    clock.advance(0.5)
    scheduler.handle_response("head-1", "success")

    with pytest.raises(CommandError, match="head command acknowledgement timed out"):
        scheduler.raise_if_faulted()
    assert len(sent) == 1


def test_resume_latches_expired_ack_and_keeps_scheduler_paused():
    scheduler, sent, clock = make_scheduler(ack_timeout=0.5)
    scheduler.submit([0.1, 0.2])
    scheduler.pause()

    clock.advance(0.5)

    with pytest.raises(CommandError, match="head command acknowledgement timed out"):
        scheduler.resume()
    assert scheduler._paused is True

    scheduler.tick()
    assert len(sent) == 1


def test_reconnect_pause_discards_old_ack_deadline():
    scheduler, sent, clock = make_scheduler(ack_timeout=0.5)
    scheduler.submit([0.1, 0.2])
    scheduler.pause_for_reconnect()

    clock.advance(0.5)
    scheduler.resume()
    scheduler.submit([0.3, 0.4])

    assert len(sent) == 2
    np.testing.assert_allclose(sent[1][0], [0.3, 0.4])
    assert sent[1][2] == "head-2"


def test_reconnect_pause_latches_ack_timeout_before_discarding_old_session():
    scheduler, sent, clock = make_scheduler(ack_timeout=0.5)
    scheduler.submit([0.1, 0.2])

    clock.advance(0.5)
    scheduler.pause_for_reconnect()

    with pytest.raises(CommandError, match="head command acknowledgement timed out"):
        scheduler.resume()
    assert scheduler._paused is True
    assert len(sent) == 1


def test_submit_while_paused_latches_expired_ack():
    scheduler, sent, clock = make_scheduler(ack_timeout=0.5)
    scheduler.submit([0.1, 0.2])
    scheduler.pause()

    clock.advance(0.5)

    with pytest.raises(CommandError, match="head command acknowledgement timed out"):
        scheduler.submit([0.3, 0.4])
    assert len(sent) == 1


def test_reconnect_preserves_failed_ack_and_resume_raises_fresh_errors():
    scheduler, sent, _clock = make_scheduler()
    scheduler.submit([0.1, 0.2])
    scheduler.handle_response("head-1", "controller rejected")
    scheduler.pause_for_reconnect()

    with pytest.raises(CommandError, match="controller rejected") as first:
        scheduler.resume()
    with pytest.raises(CommandError, match="controller rejected") as second:
        scheduler.resume()

    assert scheduler._paused is True
    assert first.value is not second.value
    assert first.value.__cause__ is None
    assert second.value.__cause__ is None
    assert first.value.__context__ is None
    assert second.value.__context__ is None
    assert first.value.__traceback__ is not second.value.__traceback__
    scheduler.tick()
    assert len(sent) == 1


def test_reconnect_preserves_ack_timeout_fault():
    scheduler, sent, clock = make_scheduler(ack_timeout=0.5)
    scheduler.submit([0.1, 0.2])
    clock.advance(0.5)
    scheduler.tick()

    scheduler.pause_for_reconnect()

    with pytest.raises(CommandError, match="head command acknowledgement timed out"):
        scheduler.resume()
    assert scheduler._paused is True
    assert len(sent) == 1


def test_reconnect_preserves_clock_fault():
    class FailingClock(FakeClock):
        fail = False

        def __call__(self):
            if self.fail:
                raise RuntimeError("clock broke")
            return super().__call__()

    clock = FailingClock()
    scheduler, sent, _clock = make_scheduler(clock=clock)
    scheduler.submit([0.1, 0.2])
    clock.fail = True

    scheduler.pause_for_reconnect()

    with pytest.raises(CommandError, match="head scheduler clock failed.*clock broke"):
        scheduler.resume()
    assert scheduler._paused is True
    assert len(sent) == 1


def test_reconnect_preserves_guid_factory_fault():
    def broken_guid_factory():
        raise RuntimeError("GUID source broke")

    sent = []
    scheduler = HeadCommandScheduler(
        sender=lambda target, duration, guid: sent.append(
            (target.copy(), duration, guid)
        ),
        min_interval=0.1,
        move_duration=0.05,
        ack_timeout=1.0,
        clock=FakeClock(),
        guid_factory=broken_guid_factory,
    )
    with pytest.raises(CommandError, match="GUID creation failed.*GUID source broke"):
        scheduler.submit([0.1, 0.2])

    scheduler.pause_for_reconnect()

    with pytest.raises(CommandError, match="GUID creation failed.*GUID source broke"):
        scheduler.resume()
    assert scheduler._paused is True
    assert sent == []


def test_close_is_permanent_and_prevents_all_later_sends():
    scheduler, sent, clock = make_scheduler()
    scheduler.submit([0.1, 0.2])
    scheduler.submit([0.3, 0.4])
    scheduler.close()

    scheduler.handle_response("head-1", "success")
    scheduler.resume()
    scheduler.submit([0.5, 0.6])
    clock.advance(2.0)
    scheduler.tick()

    assert len(sent) == 1


@pytest.mark.parametrize("bad_clock", [lambda: np.nan, lambda: np.inf])
def test_non_finite_clock_faults_safely_without_sending(bad_clock):
    scheduler, sent, _clock = make_scheduler(clock=bad_clock)

    with pytest.raises(CommandError, match="head scheduler clock failed"):
        scheduler.submit([0.1, 0.2])

    assert sent == []
    with pytest.raises(CommandError, match="head scheduler clock failed"):
        scheduler.raise_if_faulted()


def test_clock_exception_during_tick_faults_and_clears_pending_safely():
    class FailingClock(FakeClock):
        fail = False

        def __call__(self):
            if self.fail:
                raise RuntimeError("clock broke")
            return super().__call__()

    clock = FailingClock()
    scheduler, sent, _clock = make_scheduler(clock=clock)
    scheduler.submit([0.1, 0.2])
    scheduler.submit([0.3, 0.4])
    clock.fail = True

    scheduler.tick()
    scheduler.handle_response("head-1", "success")

    assert len(sent) == 1
    with pytest.raises(CommandError, match="head scheduler clock failed.*clock broke"):
        scheduler.raise_if_faulted()


def test_raise_if_faulted_creates_fresh_errors_without_retaining_traceback():
    def sender(*_args):
        raise RuntimeError("boom")

    scheduler, _sent, _clock = make_scheduler(sender=sender)
    with pytest.raises(CommandError, match="head command send failed") as submitted:
        scheduler.submit([0.1, 0.2])

    with pytest.raises(CommandError) as first:
        scheduler.raise_if_faulted()
    with pytest.raises(CommandError) as second:
        scheduler.raise_if_faulted()

    assert submitted.value is not first.value
    assert first.value is not second.value
    assert submitted.value.__cause__ is None
    assert first.value.__cause__ is None
    assert second.value.__cause__ is None
