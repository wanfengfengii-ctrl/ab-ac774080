"""HTTP API for the pulse deconvolution solver (standard library only)."""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .solver import ComputationLimitReached, solve

SAMPLE_MIN, SAMPLE_MAX = 12, 36
KERNEL_MIN, KERNEL_MAX = 3, 6


class ValidationError(ValueError):
    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _validate_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValidationError("request body must be a JSON object")

    required = ("samples", "kernel", "residual_limit", "amplitude_limit", "max_events")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValidationError(f"missing required field(s): {', '.join(missing)}")

    samples = payload["samples"]
    kernel = payload["kernel"]
    residual_limit = payload["residual_limit"]
    amplitude_limit = payload["amplitude_limit"]
    max_events = payload["max_events"]

    if not isinstance(samples, list) or not samples or not all(_is_int(v) for v in samples):
        raise ValidationError("samples must be a non-empty list of integers", "samples")
    if not (SAMPLE_MIN <= len(samples) <= SAMPLE_MAX):
        raise ValidationError(
            f"samples length must be between {SAMPLE_MIN} and {SAMPLE_MAX}", "samples"
        )

    if not isinstance(kernel, list) or not kernel or not all(_is_int(v) for v in kernel):
        raise ValidationError("kernel must be a non-empty list of integers", "kernel")
    if not (KERNEL_MIN <= len(kernel) <= KERNEL_MAX):
        raise ValidationError(
            f"kernel length must be between {KERNEL_MIN} and {KERNEL_MAX}", "kernel"
        )
    if any(v <= 0 for v in kernel):
        raise ValidationError("every kernel coefficient must be a positive integer", "kernel")

    if not _is_int(residual_limit) or residual_limit < 0:
        raise ValidationError("residual_limit must be a non-negative integer", "residual_limit")
    if not _is_int(amplitude_limit) or amplitude_limit < 1:
        raise ValidationError("amplitude_limit must be a positive integer", "amplitude_limit")
    if not _is_int(max_events) or max_events < 1:
        raise ValidationError("max_events must be a positive integer", "max_events")
    if max_events > len(samples) - len(kernel) + 1:
        raise ValidationError(
            "max_events cannot exceed the number of possible starting positions",
            "max_events",
        )

    return {
        "samples": samples,
        "kernel": kernel,
        "residual_limit": residual_limit,
        "amplitude_limit": amplitude_limit,
        "max_events": max_events,
    }


def deconvolve(payload: object) -> tuple[int, dict]:
    """Validate and solve; return (http_status, response_body)."""
    try:
        params = _validate_payload(payload)
    except ValidationError as exc:
        body = {
            "deconvolvable": None,
            "error": "validation_error",
            "field": exc.field,
            "message": str(exc),
        }
        return HTTPStatus.BAD_REQUEST, body

    try:
        result = solve(
            params["samples"],
            params["kernel"],
            params["residual_limit"],
            params["amplitude_limit"],
            params["max_events"],
        )
    except ComputationLimitReached:
        body = {
            "deconvolvable": None,
            "error": "computation_limit_reached",
            "message": (
                "the exact joint search exceeded its transition budget for this "
                "parameter combination; loosen the limits or reduce the window"
            ),
        }
        return HTTPStatus.SERVICE_UNAVAILABLE, body

    limits_echo = {
        "residual_limit": params["residual_limit"],
        "amplitude_limit": params["amplitude_limit"],
        "max_events": params["max_events"],
    }

    if not result.feasible:
        if result.reason == "residual_limit":
            message = (
                "no non-negative integer event sequence keeps every sample within "
                "the residual limit"
            )
        else:
            message = "fitting the residual limit requires more events than max_events"
        return HTTPStatus.UNPROCESSABLE_ENTITY, {
            "deconvolvable": False,
            "reason": result.reason,
            "message": message,
            "first_unexplainable_position": result.first_unexplainable_position,
            "limits": limits_echo,
        }

    return HTTPStatus.OK, {
        "deconvolvable": True,
        "events": [
            {"position": p, "amplitude": a} for p, a in result.events
        ],
        "amplitudes": list(result.amplitudes),
        "predicted_waveform": list(result.predicted),
        "residuals": list(result.residuals),
        "max_abs_residual": result.max_abs_residual,
        "sum_abs_residual": result.sum_abs_residual,
        "num_events": result.num_events,
        "objective": {
            "order": ["max_abs_residual", "sum_abs_residual", "num_events", "amplitudes"],
            "max_abs_residual": result.max_abs_residual,
            "sum_abs_residual": result.sum_abs_residual,
            "num_events": result.num_events,
        },
        "limits": limits_echo,
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "PulseDeconv/1.0"

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/health", "/healthz", "/ready"):
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/":
            self._send_json(
                HTTPStatus.OK,
                {
                    "service": "scintillator-pulse-deconvolver",
                    "endpoint": "POST /api/pulses/deconvolve",
                    "health": "/healthz",
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/pulses/deconvolve":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "validation_error",
                                                     "message": "invalid Content-Length"})
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "validation_error", "message": "request body must be valid JSON"},
            )
            return
        status, body = deconvolve(payload)
        self._send_json(status, body)

    def log_message(self, fmt, *args) -> None:  # quieter, structured logs
        if os.environ.get("QUIET_LOGS"):
            return
        super().log_message(fmt, *args)


def create_server(host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _Handler)


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    httpd = create_server(host, port)
    print(f"pulse deconvolution API listening on {host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
