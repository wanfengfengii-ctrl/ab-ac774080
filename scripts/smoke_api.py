#!/usr/bin/env python3
"""End-to-end smoke tests for POST /api/pulses/deconvolve.

Exits non-zero on the first failed check; prints one line per assertion.
Standard library only.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
failures = 0
checks = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global checks, failures
    checks += 1
    if cond:
        print(f"  ok   {name}")
    else:
        failures += 1
        print(f"  FAIL {name} {detail}")


def post(path: str, body: dict):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


def convolve(amplitudes, kernel, n):
    out = [0] * n
    for p, v in enumerate(amplitudes):
        for t, hv in enumerate(kernel):
            if p + t < n:
                out[p + t] += v * hv
    return out


def main() -> int:
    print("[1] health")
    status, body = get("/healthz")
    check("GET /healthz -> 200", status == 200, str(status))
    check("health payload reports ok", body.get("status") == "ok", str(body))

    print("[2] overlapping pulses recovered jointly")
    # Three events spaced closer than the kernel length: tails overlap, so a
    # peak-by-peak subtraction cannot recover them, but the joint solver does.
    kernel = [4, 3, 2, 1]
    true_amps = [0, 6, 0, 5, 0, 0, 0, 7, 0, 0, 0]
    n = 14
    samples = convolve(true_amps, kernel, n)
    status, body = post(
        "/api/pulses/deconvolve",
        {
            "samples": samples,
            "kernel": kernel,
            "residual_limit": 0,
            "amplitude_limit": 12,
            "max_events": 8,
        },
    )
    check("status 200", status == 200, f"{status} {body}")
    check("deconvolvable true", body.get("deconvolvable") is True, str(body))
    check("exact amplitude recovery", body.get("amplitudes") == true_amps,
          f"{body.get('amplitudes')} != {true_amps}")
    check("zero max residual", body.get("max_abs_residual") == 0)
    check("zero residual sum", body.get("sum_abs_residual") == 0)
    check("three events", body.get("num_events") == 3, str(body.get("num_events")))
    check("predicted equals input waveform", body.get("predicted_waveform") == samples)
    check("residuals all zero", set(body.get("residuals", [])) == {0})
    check(
        "events list matches nonzero amplitudes",
        [(e["position"], e["amplitude"]) for e in body.get("events", [])]
        == [(p, a) for p, a in enumerate(true_amps) if a],
    )
    # Internal consistency: predicted must be the zero-padded convolution.
    check(
        "predicted is the zero-padded convolution",
        body.get("predicted_waveform")
        == convolve(body["amplitudes"], kernel, n),
    )

    print("[3] noisy overlapping pulses within residual limit")
    noisy = [v + d for v, d in zip(samples, [0, 0, 1, -1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0])]
    status, body = post(
        "/api/pulses/deconvolve",
        {
            "samples": noisy,
            "kernel": kernel,
            "residual_limit": 1,
            "amplitude_limit": 12,
            "max_events": 8,
        },
    )
    check("status 200", status == 200, f"{status} {body}")
    check("deconvolvable true", body.get("deconvolvable") is True)
    check("max residual within limit", body.get("max_abs_residual") <= 1)
    check(
        "reported residuals reproduce samples-predicted",
        all(
            noisy[i] - body["predicted_waveform"][i] == body["residuals"][i]
            for i in range(n)
        ),
    )
    check("residual vector length equals sample length",
          len(body["residuals"]) == n)

    print("[4] residual limit impossible -> identifiable non-invertibility")
    bad = [1] * 12  # kernel starts at 4: no multiple of 4 is within 0 of 1
    status, body = post(
        "/api/pulses/deconvolve",
        {
            "samples": bad,
            "kernel": [4, 2, 1],
            "residual_limit": 0,
            "amplitude_limit": 10,
            "max_events": 10,
        },
    )
    check("status 422", status == 422, f"{status} {body}")
    check("deconvolvable false", body.get("deconvolvable") is False)
    check("reason identifies residual limit", body.get("reason") == "residual_limit")
    check("first unexplainable position is 0",
          body.get("first_unexplainable_position") == 0, str(body))

    print("[5] event cap impossible")
    status, body = post(
        "/api/pulses/deconvolve",
        {
            "samples": convolve([1, 0, 1, 0, 1, 0, 0, 0, 0, 0], [2, 1, 1], 12),
            "kernel": [2, 1, 1],
            "residual_limit": 0,
            "amplitude_limit": 5,
            "max_events": 2,
        },
    )
    check("status 422", status == 422, f"{status} {body}")
    check("deconvolvable false", body.get("deconvolvable") is False)
    check("reason identifies event cap", body.get("reason") == "max_events")

    print("[6] validation errors rejected")
    status, body = post(
        "/api/pulses/deconvolve",
        {"samples": [1, 2], "kernel": [4, 2, 1], "residual_limit": 0,
         "amplitude_limit": 10, "max_events": 5},
    )
    check("too few samples -> 400", status == 400, f"{status}")
    status, body = post(
        "/api/pulses/deconvolve",
        {"samples": [0] * 12, "kernel": [4, 0, 1], "residual_limit": 0,
         "amplitude_limit": 10, "max_events": 5},
    )
    check("non-positive kernel entry -> 400", status == 400, f"{status} {body}")
    status, body = post(
        "/api/pulses/deconvolve",
        {"samples": [0] * 12, "kernel": [4, 2, 1], "residual_limit": -1,
         "amplitude_limit": 10, "max_events": 5},
    )
    check("negative residual limit -> 400", status == 400, f"{status}")
    status, body = post("/api/pulses/deconvolve", {"samples": [0] * 12})
    check("missing fields -> 400", status == 400, f"{status}")

    print(f"\n{checks - failures}/{checks} checks passed")
    if failures:
        print(f"{failures} FAILURES", file=sys.stderr)
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
