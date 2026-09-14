"""Tests for Cartesian pose interpolation — pure math, no robot needed."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from tron2_env.interpolation import PoseInterpolator


def _pose(
    left_xyz=(0.0, 0.0, 0.0),
    left_quaternion=(1.0, 0.0, 0.0, 0.0),
    right_xyz=(0.0, 0.0, 0.0),
    right_quaternion=(1.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build the fixed [left XYZ/WXYZ, right XYZ/WXYZ] representation."""

    return np.asarray(
        (*left_xyz, *left_quaternion, *right_xyz, *right_quaternion),
        dtype=np.float64,
    )


class _BlockingSequenceClock:
    """Return increasing values while allowing the first sample to be gated."""

    def __init__(self, values: tuple[float, ...]) -> None:
        if any(first >= second for first, second in zip(values, values[1:])):
            raise ValueError("clock values must be strictly increasing")
        self._values = values
        self._index = 0
        self._lock = threading.Lock()
        self.first_call_entered = threading.Event()
        self.release_first_call = threading.Event()
        self.returned_values: list[float] = []

    def __call__(self) -> float:
        with self._lock:
            if self._index >= len(self._values):
                raise AssertionError("controlled clock exhausted")
            index = self._index
            value = self._values[index]
            self._index += 1
            self.returned_values.append(value)

        if index == 0:
            self.first_call_entered.set()
            if not self.release_first_call.wait(timeout=2.0):
                raise TimeoutError("controlled clock was not released")
        return value


class _IncrementingClock:
    """Thread-safe monotonic fake clock for concurrency smoke tests."""

    def __init__(self, *, step: float = 1e-4) -> None:
        self._value = 0.0
        self._step = step
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            value = self._value
            self._value += self._step
            return value


@pytest.fixture
def interpolator(monkeypatch: pytest.MonkeyPatch) -> PoseInterpolator:
    monkeypatch.setattr("tron2_env.interpolation.pose.time.perf_counter", lambda: 0.0)
    result = PoseInterpolator()
    result.reset(_pose())
    return result


def test_xyz_interpolation_clamps_to_endpoints_and_hits_midpoint(
    interpolator: PoseInterpolator,
) -> None:
    start = _pose(left_xyz=(0.0, 2.0, 4.0), right_xyz=(10.0, 20.0, 30.0))
    target = _pose(left_xyz=(2.0, 6.0, 10.0), right_xyz=(-10.0, 0.0, 50.0))
    interpolator.reset(start)
    interpolator.set_destination(target, eta=4.0, now=10.0)

    np.testing.assert_allclose(interpolator.current(t=9.0), start)
    np.testing.assert_allclose(interpolator.current(t=10.0), start)
    np.testing.assert_allclose(
        interpolator.current(t=12.0),
        _pose(left_xyz=(1.0, 4.0, 7.0), right_xyz=(0.0, 10.0, 40.0)),
    )
    np.testing.assert_allclose(interpolator.current(t=14.0), target)
    np.testing.assert_allclose(interpolator.current(t=20.0), target)
    assert not interpolator.at_destination(t=13.999)
    assert interpolator.at_destination(t=14.0)


def test_reset_normalizes_wxyz_quaternions_and_copies_input(
    interpolator: PoseInterpolator,
) -> None:
    initial = _pose(
        left_xyz=(1.0, 2.0, 3.0),
        left_quaternion=(2.0, 0.0, 0.0, 0.0),
        right_xyz=(4.0, 5.0, 6.0),
        right_quaternion=(0.0, 0.0, 0.0, -4.0),
    )
    expected = _pose(
        left_xyz=(1.0, 2.0, 3.0),
        right_xyz=(4.0, 5.0, 6.0),
        right_quaternion=(0.0, 0.0, 0.0, -1.0),
    )

    interpolator.reset(initial)
    initial[:] = 123.0

    output = interpolator.current()
    np.testing.assert_allclose(output, expected)
    np.testing.assert_allclose(np.linalg.norm(output[3:7]), 1.0)
    np.testing.assert_allclose(np.linalg.norm(output[10:14]), 1.0)


def test_set_destination_normalizes_quaternions_and_copies_input(
    interpolator: PoseInterpolator,
) -> None:
    target = _pose(
        left_xyz=(1.0, 2.0, 3.0),
        left_quaternion=(0.0, 2.0, 0.0, 0.0),
        right_xyz=(4.0, 5.0, 6.0),
        right_quaternion=(0.0, 0.0, 3.0, 0.0),
    )
    expected = _pose(
        left_xyz=(1.0, 2.0, 3.0),
        left_quaternion=(0.0, 1.0, 0.0, 0.0),
        right_xyz=(4.0, 5.0, 6.0),
        right_quaternion=(0.0, 0.0, 1.0, 0.0),
    )

    interpolator.set_destination(target, eta=1.0, now=0.0)
    target[:] = 123.0

    np.testing.assert_allclose(interpolator.current(t=1.0), expected)


def test_slerp_uses_wxyz_order_and_shortest_path_for_negative_dot(
    interpolator: PoseInterpolator,
) -> None:
    half_angle = np.pi / 4.0
    positive_end = np.array(
        [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]
    )
    negative_end = -positive_end
    target = _pose(left_quaternion=negative_end)

    interpolator.set_destination(target, eta=1.0, now=0.0)
    midpoint = interpolator.current(t=0.5)

    expected = np.array(
        [np.cos(np.pi / 8.0), 0.0, 0.0, np.sin(np.pi / 8.0)]
    )
    np.testing.assert_allclose(midpoint[3:7], expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(midpoint[3:7]), 1.0, atol=1e-12)
    assert np.all(np.isfinite(midpoint))


def test_near_identical_quaternions_use_normalized_linear_interpolation(
    interpolator: PoseInterpolator,
) -> None:
    start_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    end_quaternion = np.array([np.cos(0.02), np.sin(0.02), 0.0, 0.0])
    assert np.dot(start_quaternion, end_quaternion) > 0.9995
    interpolator.set_destination(
        _pose(left_quaternion=end_quaternion), eta=1.0, now=0.0
    )

    output = interpolator.current(t=0.25)
    expected = 0.75 * start_quaternion + 0.25 * end_quaternion
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(output[3:7], expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(output[3:7]), 1.0, atol=1e-12)


def test_tiny_positive_eta_still_interpolates(
    interpolator: PoseInterpolator,
) -> None:
    eta = 5e-10
    target = _pose(
        left_xyz=(2.0, 4.0, 6.0),
        left_quaternion=(np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)),
    )
    interpolator.set_destination(target, eta=eta, now=0.0)

    midpoint = interpolator.current(t=eta / 2.0)

    np.testing.assert_allclose(midpoint[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        midpoint[3:7],
        [np.cos(np.pi / 8.0), 0.0, 0.0, np.sin(np.pi / 8.0)],
        atol=1e-12,
    )
    assert not interpolator.at_destination(t=eta / 2.0)


def test_negative_dot_endpoint_keeps_shortest_path_quaternion_sign(
    interpolator: PoseInterpolator,
) -> None:
    aligned_left = np.array(
        [np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)]
    )
    aligned_right = np.array(
        [np.cos(np.pi / 3.0), np.sin(np.pi / 3.0), 0.0, 0.0]
    )
    interpolator.set_destination(
        _pose(
            left_quaternion=-aligned_left,
            right_quaternion=-aligned_right,
        ),
        eta=1.0,
        now=0.0,
    )

    just_before_end = interpolator.current(t=np.nextafter(1.0, 0.0))
    at_end = interpolator.current(t=1.0)
    after_end = interpolator.current(t=2.0)

    np.testing.assert_allclose(at_end[3:7], aligned_left, atol=1e-12)
    np.testing.assert_allclose(at_end[10:14], aligned_right, atol=1e-12)
    np.testing.assert_allclose(after_end, at_end)
    assert np.dot(just_before_end[3:7], at_end[3:7]) > 0.999999999999
    assert np.dot(just_before_end[10:14], at_end[10:14]) > 0.999999999999


def test_left_and_right_quaternions_are_interpolated_independently(
    interpolator: PoseInterpolator,
) -> None:
    target = _pose(
        left_quaternion=(np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0, 0.0),
        right_quaternion=(0.0, 0.0, 1.0, 0.0),
    )
    interpolator.set_destination(target, eta=2.0, now=0.0)

    midpoint = interpolator.current(t=1.0)
    np.testing.assert_allclose(
        midpoint[3:7],
        [np.cos(np.pi / 8.0), np.sin(np.pi / 8.0), 0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        midpoint[10:14],
        [np.cos(np.pi / 4.0), 0.0, np.sin(np.pi / 4.0), 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.norm(midpoint[3:7]), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(midpoint[10:14]), 1.0, atol=1e-12)


@pytest.mark.parametrize("bad_shape", [(13,), (15,), (14, 1), (2, 7)])
def test_reset_and_set_destination_require_exact_shape(bad_shape: tuple[int, ...]) -> None:
    bad_pose = np.zeros(bad_shape)

    with pytest.raises(ValueError, match="shape"):
        PoseInterpolator().reset(bad_pose)
    with pytest.raises(ValueError, match="shape"):
        PoseInterpolator().set_destination(bad_pose, eta=1.0, now=0.0)


@pytest.mark.parametrize(
    ("index", "value"),
    [(0, np.nan), (8, np.inf), (4, -np.inf)],
)
def test_reset_and_set_destination_reject_non_finite_values(
    index: int, value: float
) -> None:
    bad_pose = _pose()
    bad_pose[index] = value

    with pytest.raises(ValueError, match="finite"):
        PoseInterpolator().reset(bad_pose)
    with pytest.raises(ValueError, match="finite"):
        PoseInterpolator().set_destination(bad_pose, eta=1.0, now=0.0)


@pytest.mark.parametrize("quaternion", [(0.0, 0.0, 0.0, 0.0), (9e-9, 0.0, 0.0, 0.0)])
@pytest.mark.parametrize("side", ["left", "right"])
def test_reset_and_set_destination_reject_too_small_quaternions(
    side: str, quaternion: tuple[float, float, float, float]
) -> None:
    kwargs = {f"{side}_quaternion": quaternion}
    bad_pose = _pose(**kwargs)

    with pytest.raises(ValueError, match="quaternion"):
        PoseInterpolator().reset(bad_pose)
    with pytest.raises(ValueError, match="quaternion"):
        PoseInterpolator().set_destination(bad_pose, eta=1.0, now=0.0)


def test_quaternion_at_minimum_allowed_norm_is_accepted() -> None:
    interpolator = PoseInterpolator()
    interpolator.reset(
        _pose(
            left_quaternion=(1e-8, 0.0, 0.0, 0.0),
            right_quaternion=(0.0, -1e-8, 0.0, 0.0),
        )
    )

    output = interpolator.current()
    np.testing.assert_allclose(output[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(output[10:14], [0.0, -1.0, 0.0, 0.0])


@pytest.mark.parametrize("eta", [0.0, -1.0])
def test_non_positive_eta_jumps_immediately(
    interpolator: PoseInterpolator, eta: float
) -> None:
    target = _pose(
        left_xyz=(1.0, 2.0, 3.0),
        left_quaternion=(0.0, 0.0, 0.0, 2.0),
        right_xyz=(4.0, 5.0, 6.0),
        right_quaternion=(0.0, 2.0, 0.0, 0.0),
    )
    expected = _pose(
        left_xyz=(1.0, 2.0, 3.0),
        left_quaternion=(0.0, 0.0, 0.0, 1.0),
        right_xyz=(4.0, 5.0, 6.0),
        right_quaternion=(0.0, 1.0, 0.0, 0.0),
    )

    interpolator.set_destination(target, eta=eta, now=5.0)

    np.testing.assert_allclose(interpolator.current(t=5.0), expected)
    assert interpolator.at_destination(t=5.0)


def test_midflight_retarget_starts_continuously_from_current_pose(
    interpolator: PoseInterpolator,
) -> None:
    first_target = _pose(
        left_xyz=(10.0, 0.0, 0.0),
        left_quaternion=(0.0, 0.0, 0.0, 1.0),
        right_xyz=(0.0, 20.0, 0.0),
        right_quaternion=(0.0, 1.0, 0.0, 0.0),
    )
    second_target = _pose(
        left_xyz=(4.0, 8.0, 0.0),
        left_quaternion=(0.0, 0.0, 1.0, 0.0),
        right_xyz=(3.0, 4.0, 5.0),
        right_quaternion=(0.0, 0.0, 0.0, 1.0),
    )
    interpolator.set_destination(first_target, eta=1.0, now=0.0)
    before_retarget = interpolator.current(t=0.4)

    interpolator.set_destination(second_target, eta=2.0, now=0.4)

    np.testing.assert_allclose(interpolator.current(t=0.4), before_retarget)
    np.testing.assert_allclose(interpolator.current(t=2.4), second_target)


def test_default_set_destination_linearizes_clock_sampling_with_state(
    interpolator: PoseInterpolator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpolator.set_destination(
        _pose(left_xyz=(10.0, 0.0, 0.0)), eta=10.0, now=0.0
    )
    clock = _BlockingSequenceClock((5.0, 6.0))
    monkeypatch.setattr("tron2_env.interpolation.pose.time.perf_counter", clock)
    reader_started = threading.Event()
    reader_done = threading.Event()
    observations: list[np.ndarray] = []
    errors: list[BaseException] = []

    def retarget() -> None:
        try:
            interpolator.set_destination(
                _pose(left_xyz=(5.0, 0.0, 0.0)), eta=5.0
            )
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    def read_later_old_timeline() -> None:
        reader_started.set()
        try:
            observations.append(interpolator.current(t=6.0))
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)
        finally:
            reader_done.set()

    writer = threading.Thread(target=retarget)
    reader = threading.Thread(target=read_later_old_timeline)
    reader_finished_before_clock_release = False
    writer.start()
    try:
        assert clock.first_call_entered.wait(timeout=1.0)
        reader.start()
        assert reader_started.wait(timeout=1.0)
        reader_finished_before_clock_release = reader_done.wait(timeout=0.5)
    finally:
        clock.release_first_call.set()
        writer.join(timeout=2.0)
        if reader.ident is not None:
            reader.join(timeout=2.0)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert not errors

    after_retarget = interpolator.current()
    np.testing.assert_allclose(observations[0], after_retarget)
    np.testing.assert_allclose(after_retarget[:3], [5.0, 0.0, 0.0])
    assert not reader_finished_before_clock_release
    assert clock.returned_values == [5.0, 6.0]


def test_default_at_destination_linearizes_clock_sampling_with_state(
    interpolator: PoseInterpolator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpolator.set_destination(
        _pose(left_xyz=(5.0, 0.0, 0.0)), eta=5.0, now=0.0
    )
    clock = _BlockingSequenceClock((6.0, 7.0))
    monkeypatch.setattr("tron2_env.interpolation.pose.time.perf_counter", clock)
    writer_started = threading.Event()
    writer_done = threading.Event()
    destination_results: list[bool] = []
    errors: list[BaseException] = []

    def check_destination() -> None:
        try:
            destination_results.append(interpolator.at_destination())
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    def extend_trajectory() -> None:
        writer_started.set()
        try:
            interpolator.set_destination(
                _pose(left_xyz=(15.0, 0.0, 0.0)), eta=10.0, now=6.0
            )
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)
        finally:
            writer_done.set()

    checker = threading.Thread(target=check_destination)
    writer = threading.Thread(target=extend_trajectory)
    writer_finished_before_clock_release = False
    checker.start()
    try:
        assert clock.first_call_entered.wait(timeout=1.0)
        writer.start()
        assert writer_started.wait(timeout=1.0)
        writer_finished_before_clock_release = writer_done.wait(timeout=0.5)
    finally:
        clock.release_first_call.set()
        checker.join(timeout=2.0)
        if writer.ident is not None:
            writer.join(timeout=2.0)

    assert not checker.is_alive()
    assert not writer.is_alive()
    assert not errors
    assert destination_results == [True]
    assert not writer_finished_before_clock_release

    calls_before_explicit_time = len(clock.returned_values)
    assert interpolator.at_destination(t=16.0)
    assert len(clock.returned_values) == calls_before_explicit_time
    assert not interpolator.at_destination()
    assert clock.returned_values == [6.0, 7.0]


def test_first_set_destination_initializes_at_normalized_target() -> None:
    target = _pose(
        left_xyz=(1.0, 2.0, 3.0),
        left_quaternion=(2.0, 0.0, 0.0, 0.0),
        right_xyz=(4.0, 5.0, 6.0),
        right_quaternion=(0.0, 0.0, -2.0, 0.0),
    )
    expected = _pose(
        left_xyz=(1.0, 2.0, 3.0),
        right_xyz=(4.0, 5.0, 6.0),
        right_quaternion=(0.0, 0.0, -1.0, 0.0),
    )
    interpolator = PoseInterpolator()

    interpolator.set_destination(target, eta=10.0, now=5.0)

    np.testing.assert_allclose(interpolator.current(t=5.0), expected)
    assert interpolator.at_destination(t=5.0)


def test_current_requires_initialization() -> None:
    with pytest.raises(RuntimeError, match="reset"):
        PoseInterpolator().current()


def test_thread_safety_smoke(
    interpolator: PoseInterpolator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tron2_env.interpolation.pose.time.perf_counter", _IncrementingClock()
    )
    errors: list[BaseException] = []
    start = threading.Barrier(3)

    def writer() -> None:
        try:
            start.wait()
            for index in range(250):
                sign = -1.0 if index % 2 else 1.0
                interpolator.set_destination(
                    _pose(
                        left_xyz=(sign, 0.0, 0.0),
                        right_xyz=(0.0, sign, 0.0),
                    ),
                    eta=0.01,
                )
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    def reader() -> None:
        try:
            start.wait()
            for _ in range(250):
                output = interpolator.current()
                assert output.shape == (14,)
                assert np.all(np.isfinite(output))
                np.testing.assert_allclose(np.linalg.norm(output[3:7]), 1.0)
                np.testing.assert_allclose(np.linalg.norm(output[10:14]), 1.0)
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2.0)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
