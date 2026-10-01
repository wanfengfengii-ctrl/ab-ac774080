# Scintillator Pulse Deconvolution Service

High count rates make the tails of adjacent scintillator events overlap.
This service recovers the underlying event sequence from 12–36 discrete
samples by **jointly** choosing non-negative integer event amplitudes — it
never subtracts response tails peak by peak (which double-counts shared tail
energy).

The measured waveform is modelled as the zero-padded discrete convolution

```
y_hat[j] = Σ_p a[p] · kernel[j - p]
```

where `a[p] ≥ 0` is the integer amplitude of an event starting at position
`p` (`a[p] = 0` means no event there). The solver is an exact
branch-and-bound-style dynamic program over the banded convolution matrix
(kernel length 3–6), implemented with the Python standard library only.

## Optimization objective

Among all amplitude sequences satisfying the supplied limits, the service
minimizes lexicographically:

1. maximum absolute residual `max |y[j] − y_hat[j]|`
2. sum of absolute residuals `Σ |y[j] − y_hat[j]|`
3. number of events (`a[p] > 0`)
4. the amplitude sequence lexicographically (earlier positions dominate)

## API

`POST /api/pulses/deconvolve`

```json
{
  "samples": [/* 12–36 integers */],
  "kernel":  [/* 3–6 positive integers, the detector response */],
  "residual_limit": 2,
  "amplitude_limit": 20,
  "max_events": 12
}
```

- `residual_limit` — uniform inclusive cap on every per-point absolute residual
- `amplitude_limit` — inclusive per-event amplitude cap (positive integer)
- `max_events` — inclusive cap on the number of non-zero events

### Success — `200 OK`

```json
{
  "deconvolvable": true,
  "events": [{"position": 1, "amplitude": 6}],
  "amplitudes": [0, 6, 0],
  "predicted_waveform": [/* zero-padded convolution, full length */],
  "residuals": [/* samples - predicted, full length */],
  "max_abs_residual": 0,
  "sum_abs_residual": 0,
  "num_events": 1,
  "objective": {"order": ["max_abs_residual", "sum_abs_residual",
                          "num_events", "amplitudes"], "...": "..."},
  "limits": {"residual_limit": 2, "amplitude_limit": 20, "max_events": 12}
}
```

### Not invertible — `422 Unprocessable Entity`

- `"reason": "residual_limit"` — no candidate keeps every sample within the
  residual cap. `first_unexplainable_position` is the index of the earliest
  sample that cannot be explained by **any** complete candidate (i.e. no
  assignment of all positions can satisfy rows `0..j` simultaneously).
- `"reason": "max_events"` — the residual cap is attainable, but only with
  more than `max_events` events.

### Other responses

- `400` — malformed request / parameters outside their allowed ranges
- `503` (`error: computation_limit_reached`) — an unusual parameter
  combination exhausted the exact-search transition budget; loosen the
  limits or shorten the window. The search is bounded so requests cannot
  hang indefinitely.
- `GET /healthz` — liveness/readiness probe (`200 {"status":"ok"}`)

## Running with Docker

```bash
# host port configurable via HOST_PORT
HOST_PORT=9000 docker compose up --build api
```

The API container publishes `${HOST_PORT:-8000}:8000` and defines a
container health check on `/healthz`.

## One-shot verification

```bash
docker compose up --build verify
```

The `verify` service waits until the API is **healthy**, then runs and
aggregates:

1. code tests — the exact solver vs exhaustive brute force
   (`python -m unittest discover -s tests`)
2. application build check — byte-compilation of every module
3. overlapping-pulse API smoke tests against the live container
   (`scripts/smoke_api.py`)

It exits `0` only when all three stages pass, and shuts itself down.

## Local development

```bash
python3 -m unittest discover -s tests    # standard library only
PORT=8000 python3 -m app.server
API_BASE=http://127.0.0.1:8000 python3 scripts/smoke_api.py
```

## Layout

```
app/solver.py      exact joint deconvolution (window DP + interval propagation)
app/server.py      HTTP API, validation, identifiable status codes
tests/             brute-force cross-checks and HTTP-layer tests
scripts/           verify.sh aggregator and API smoke test
Dockerfile         python:3.11-slim, no third-party runtime dependencies
docker-compose.yml api service (configurable port + healthcheck) and verify
```
