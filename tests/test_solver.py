"""Exactness tests: the DP solver is compared against brute force enumeration."""

import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.solver import solve  # noqa: E402


def convolve(a, h, n):
    pred = [0] * n
    for p, v in enumerate(a):
        for t, hv in enumerate(h):
            if p + t < n:
                pred[p + t] += v * hv
    return pred


def brute(samples, kernel, residual_limit, amplitude_limit, max_events):
    """Return (feasible, best_key_or_None, first_bad)."""
    n = len(samples)
    m = n - len(kernel) + 1
    best = None
    for a in itertools.product(range(amplitude_limit + 1), repeat=m):
        events = sum(1 for v in a if v)
        if events > max_events:
            continue
        pred = convolve(a, kernel, n)
        res = [samples[j] - pred[j] for j in range(n)]
        mx = max(abs(r) for r in res)
        if mx > residual_limit:
            continue
        key = (mx, sum(abs(r) for r in res), events, tuple(a))
        if best is None or key < best:
            best = key

    first_bad = None
    if best is None:
        # Earliest row for which no prefix assignment satisfies rows 0..j.
        for j in range(n):
            exists = False
            for a in itertools.product(range(amplitude_limit + 1), repeat=min(j + 1, m)):
                pred = convolve(list(a), kernel, n)
                if all(abs(samples[t] - pred[t]) <= residual_limit for t in range(j + 1)):
                    exists = True
                    break
            if not exists:
                first_bad = j
                break
    return best is not None, best, first_bad


def make_case(rng, n, k, amax, noise, tail_only=False):
    kernel = [rng.randint(1, 4) for _ in range(k)]
    m = n - k + 1
    amps = [rng.randint(0, amax) for _ in range(m)]
    if tail_only:
        amps = [0] * m
    pred = convolve(amps, kernel, n)
    samples = [v + rng.randint(-noise, noise) for v in pred]
    return samples, kernel, amps


class SolverTests(unittest.TestCase):
    def test_exact_known_overlapping_pulses(self):
        # Two events one sample apart, triangular-ish kernel; exact recovery.
        kernel = [3, 2, 1]
        amps = [0, 4, 0, 0, 5, 0, 0, 0, 0, 0]
        n = 12
        samples = convolve(amps, kernel, n)
        r = solve(samples, kernel, 0, 9, 6)
        self.assertTrue(r.feasible)
        self.assertEqual(r.amplitudes, tuple(amps))
        self.assertEqual(r.num_events, 2)
        self.assertEqual(r.max_abs_residual, 0)
        self.assertEqual(r.sum_abs_residual, 0)
        self.assertEqual(r.predicted, tuple(samples))
        self.assertEqual(r.residuals, tuple([0] * n))

    def test_no_peak_by_peak_subtraction_trap(self):
        # A naive greedy fitter would put an event at the local maximum j=1;
        # the joint optimum instead places events dictated by the full tail.
        kernel = [2, 1, 1]
        amps = [3, 0, 2] + [0] * 7
        n = 12
        samples = convolve(amps, kernel, n)
        r = solve(samples, kernel, 0, 5, 6)
        self.assertTrue(r.feasible)
        self.assertEqual(r.amplitudes[1], 0)
        self.assertEqual(r.amplitudes, tuple(amps))

    def test_random_vs_brute(self):
        rng = random.Random(20261001)
        trials = 0
        for n in (8, 9, 10):
            for k in (3, 4, 5):
                for _ in range(25):
                    samples, kernel, _ = make_case(rng, n, k, amax=2, noise=rng.choice([0, 1, 2]))
                    R = rng.choice([0, 1, 2, 3])
                    A = 2
                    E = rng.choice([n, n // 2, 2])
                    with self.subTest(n=n, k=k, R=R, E=E, s=samples, h=kernel):
                        r = solve(samples, kernel, R, A, E)
                        feasible, best, first_bad = brute(samples, kernel, R, A, E)
                        self.assertEqual(r.feasible, feasible)
                        if feasible:
                            self.assertEqual(
                                (r.max_abs_residual, r.sum_abs_residual, r.num_events, r.amplitudes),
                                best,
                            )
                            self.assertLessEqual(r.max_abs_residual, R)
                            self.assertLessEqual(r.num_events, E)
                            self.assertTrue(all(0 <= v <= A for v in r.amplitudes))
                        else:
                            if r.reason == "residual_limit":
                                self.assertEqual(r.first_unexplainable_position, first_bad)
                            else:
                                self.assertEqual(r.reason, "max_events")
                        trials += 1
        self.assertGreater(trials, 100)

    def test_infeasible_residual_points_to_first_unexplainable(self):
        kernel = [3, 2, 1]
        n = 12
        samples = [1] * n  # row 0 alone cannot be within R=0 of 3*v for any v
        r = solve(samples, kernel, 0, 5, 6)
        self.assertFalse(r.feasible)
        self.assertEqual(r.reason, "residual_limit")
        self.assertEqual(r.first_unexplainable_position, 0)

    def test_first_unexplainable_late_row_regression(self):
        # Earlier rows are individually/prefix satisfiable; the joint prefix
        # only becomes impossible at index 5. (Regression: a horizon-pruned
        # diagnostic once reported the later index 6.)
        samples = [4, 7, 11, 15, 19, 23, 20, 15, 11, 7]
        kernel = [2, 2, 3, 2, 3]
        r = solve(samples, kernel, 1, 2, len(samples) - len(kernel) + 1)
        self.assertFalse(r.feasible)
        self.assertEqual(r.reason, "residual_limit")
        self.assertEqual(r.first_unexplainable_position, 5)

    def test_infeasible_event_count(self):
        kernel = [2, 1, 1]
        amps = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
        n = 12
        samples = convolve(amps, kernel, n)
        r = solve(samples, kernel, 0, 5, 2)  # 3 events required
        self.assertFalse(r.feasible)
        self.assertEqual(r.reason, "max_events")
        # Relaxing the event cap makes it recoverable.
        r2 = solve(samples, kernel, 0, 5, 3)
        self.assertTrue(r2.feasible)
        self.assertEqual(r2.amplitudes, tuple(amps))

    def test_lexicographic_tie_breaking(self):
        # With R large, all-zero samples prefer zero amplitudes; force a tie
        # where different placements give identical (max, sum, events).
        kernel = [1, 1, 1]
        n = 9
        samples = [1, 1, 1, 0, 0, 0, 0, 0, 0]
        r = solve(samples, kernel, 0, 1, 3)
        self.assertTrue(r.feasible)
        # event at position 0 covers the first three samples; lexicographically
        # earliest representation of the unique one-event solution.
        self.assertEqual(r.amplitudes, (1, 0, 0, 0, 0, 0, 0))

    def test_larger_instance_performance(self):
        rng = random.Random(42)
        n, k = 36, 6
        kernel = [rng.randint(1, 5) for _ in range(k)]
        amps = [rng.randint(0, 20) if rng.random() < 0.3 else 0 for _ in range(n - k + 1)]
        samples = convolve(amps, kernel, n)
        samples = [v + rng.randint(-2, 2) for v in samples]
        r = solve(samples, kernel, 2, 25, 15)
        self.assertTrue(r.feasible)
        self.assertLessEqual(r.max_abs_residual, 2)
        self.assertLessEqual(r.num_events, 15)
        self.assertEqual(r.predicted, tuple(convolve(list(r.amplitudes), kernel, n)))


if __name__ == "__main__":
    unittest.main()
