"""Two things that are easy to get quietly wrong.

`&mut [f64]` is an output written through an argument. A binding that
copied the values in and dropped the results would compile, run, and give
the wrong answer with no error anywhere -- so it is checked directly.

Releasing the GIL is the other. It is done for calls that take or return
arrays, which is where it pays; the test is that the results are still
right when several threads do it at once, and that it actually overlaps.
"""

import math
import threading

import pytest

import numeria as nm


def test_an_in_place_argument_is_written_back():
    """`velocity_verlet` advances position and velocity through `&mut [f64]`.

    It returns nothing: the whole result is the mutation. A binding that
    copied the lists in and threw the results away would run cleanly and
    leave both lists untouched.
    """
    dt = 0.01
    x = [1.0]
    v = [0.0]
    nm.numerical.ode.symplectic.velocity_verlet(lambda pos: [-p for p in pos], x, v, dt)

    # v½ = -dt/2; x₁ = x + v½·dt; v₁ = v½ - (dt/2)·x₁.
    half = -0.5 * dt
    expected_x = 1.0 + half * dt
    expected_v = half - 0.5 * dt * expected_x
    assert x == [pytest.approx(expected_x)]
    assert v == [pytest.approx(expected_v)]


def test_stepping_repeatedly_traces_the_oscillator():
    """Successive calls have to see the previous call's writes."""
    x, v = [1.0], [0.0]
    dt = 0.001
    for _ in range(1000):
        nm.numerical.ode.symplectic.velocity_verlet(
            lambda pos: [-p for p in pos], x, v, dt
        )
    # One radian of a unit-frequency oscillator: x = cos(1), v = -sin(1).
    assert x[0] == pytest.approx(math.cos(1.0), abs=1e-6)
    assert v[0] == pytest.approx(-math.sin(1.0), abs=1e-6)


def test_an_in_place_argument_must_be_something_that_can_receive_the_result():
    with pytest.raises(TypeError) as excinfo:
        nm.numerical.ode.symplectic.velocity_verlet(
            lambda pos: [-p for p in pos], (1.0,), [0.0], 0.01
        )
    assert "in place" in str(excinfo.value)


def test_results_are_right_under_concurrency():
    """Array calls release the GIL. Doing that wrong corrupts results."""
    data = [math.sin(i / 7.0) for i in range(20_000)]
    expected = nm.statistics.descriptive.mean(data)
    results = []
    errors = []

    def work():
        try:
            for _ in range(20):
                results.append(nm.statistics.descriptive.mean(data))
        except Exception as exc:  # pragma: no cover - a failure is the point
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 160
    assert all(r == expected for r in results)


def test_a_callback_from_several_threads_stays_correct():
    """Callbacks hold the GIL; each call still has to see its own state."""
    outcomes = {}
    lock = threading.Lock()

    def work(k):
        got = nm.numerical.integrate.simpson(lambda x: x**k, 0.0, 1.0, 2000)
        with lock:
            outcomes[k] = got

    threads = [threading.Thread(target=work, args=(k,)) for k in range(1, 7)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # ∫₀¹ xᵏ dx = 1/(k+1)
    for k, got in outcomes.items():
        assert got == pytest.approx(1.0 / (k + 1), abs=1e-9)


def test_an_exception_on_one_thread_does_not_disturb_another():
    """The panic guard's state is thread-local; it had better be."""
    ok = []
    caught = []

    def good():
        for _ in range(200):
            ok.append(nm.classical.acceleration(10.0, 2.0))

    def bad():
        for _ in range(200):
            try:
                nm.classical.acceleration(10.0, -1.0)
            except nm.InvalidArgumentError:
                caught.append(1)

    threads = [threading.Thread(target=good), threading.Thread(target=bad)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(caught) == 200
    assert ok == [5.0] * 200
