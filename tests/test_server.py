"""Tests for the HTTP layer's validation and response shaping (no sockets)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.server import deconvolve  # noqa: E402


def convolve(a, h, n):
    pred = [0] * n
    for p, v in enumerate(a):
        for t, hv in enumerate(h):
            if p + t < n:
                pred[p + t] += v * hv
    return pred


class ServerTests(unittest.TestCase):
    def test_ok_response_shape(self):
        kernel = [3, 2, 1]
        amps = [2, 0, 0, 3, 0, 0, 0, 0, 0, 0]
        samples = convolve(amps, kernel, 12)
        status, body = deconvolve(
            {
                "samples": samples,
                "kernel": kernel,
                "residual_limit": 0,
                "amplitude_limit": 5,
                "max_events": 6,
            }
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["deconvolvable"])
        self.assertEqual(body["amplitudes"], amps)
        self.assertEqual(len(body["predicted_waveform"]), 12)
        self.assertEqual(len(body["residuals"]), 12)
        self.assertEqual(body["num_events"], 2)
        self.assertEqual(body["max_abs_residual"], 0)
        self.assertEqual(
            body["objective"]["order"],
            ["max_abs_residual", "sum_abs_residual", "num_events", "amplitudes"],
        )

    def test_full_size_36_overlapping_recovery(self):
        kernel = [5, 3, 2, 1, 1]
        amps = [0] * 32
        for p in (1, 4, 9, 15, 22, 28):
            amps[p] = (p % 7) + 2
        samples = convolve(amps, kernel, 36)
        status, body = deconvolve(
            {
                "samples": samples,
                "kernel": kernel,
                "residual_limit": 0,
                "amplitude_limit": 20,
                "max_events": 12,
            }
        )
        self.assertEqual(status, 200, body)
        self.assertTrue(body["deconvolvable"])
        self.assertEqual(body["amplitudes"], amps)
        self.assertEqual(body["predicted_waveform"], samples)
        self.assertEqual(body["num_events"], 6)

    def test_non_invertible_carries_position(self):
        status, body = deconvolve(
            {
                "samples": [1] * 12,
                "kernel": [4, 2, 1],
                "residual_limit": 0,
                "amplitude_limit": 10,
                "max_events": 10,
            }
        )
        self.assertEqual(status, 422)
        self.assertFalse(body["deconvolvable"])
        self.assertEqual(body["reason"], "residual_limit")
        self.assertEqual(body["first_unexplainable_position"], 0)

    def test_event_cap_failure_distinguished(self):
        amps = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
        status, body = deconvolve(
            {
                "samples": convolve(amps, [2, 1, 1], 12),
                "kernel": [2, 1, 1],
                "residual_limit": 0,
                "amplitude_limit": 5,
                "max_events": 2,
            }
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["reason"], "max_events")
        self.assertIsNone(body["first_unexplainable_position"])

    def test_validation_errors(self):
        good = {
            "samples": [0] * 12,
            "kernel": [3, 2, 1],
            "residual_limit": 1,
            "amplitude_limit": 5,
            "max_events": 5,
        }
        for field, bad in [
            ("samples", [1] * 11),
            ("samples", [1] * 37),
            ("samples", [1.0] * 12),
            ("kernel", [1, 2]),
            ("kernel", [1, 2, 0]),
            ("kernel", [1, 2, -3]),
            ("residual_limit", -1),
            ("amplitude_limit", 0),
            ("max_events", 0),
        ]:
            payload = dict(good)
            payload[field] = bad
            status, body = deconvolve(payload)
            self.assertEqual(status, 400, (field, bad, body))
            self.assertEqual(body["error"], "validation_error")
            self.assertEqual(body["field"], field)

        status, body = deconvolve({"samples": [0] * 12})
        self.assertEqual(status, 400)
        status, body = deconvolve("not-an-object")
        self.assertEqual(status, 400)
        status, body = deconvolve({**good, "max_events": 99})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
