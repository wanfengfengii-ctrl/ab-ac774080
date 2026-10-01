#!/usr/bin/env bash
# One-shot verification: unit tests, application build check, API smoke.
# Aggregates every stage into a single exit code (0 only if all pass).
set -u

cd "$(dirname "$0")/.."
rc=0

if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

echo "== [1/3] code tests (exact solver vs brute force) =="
if "$PY" -m unittest discover -s tests -v; then
  echo "tests: PASS"
else
  echo "tests: FAIL"
  rc=1
fi

echo
echo "== [2/3] application build check (byte-compile all modules) =="
if "$PY" -m compileall -q app tests scripts; then
  echo "build check: PASS"
else
  echo "build check: FAIL"
  rc=1
fi

echo
echo "== [3/3] overlapping-pulse API smoke against ${API_BASE:-http://api:8000} =="
if "$PY" scripts/smoke_api.py; then
  echo "smoke: PASS"
else
  echo "smoke: FAIL"
  rc=1
fi

echo
if [ "$rc" -eq 0 ]; then
  echo "VERIFY OK"
else
  echo "VERIFY FAILED"
fi
exit "$rc"
