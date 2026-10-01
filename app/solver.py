"""Discrete deconvolution of a sampled scintillator waveform.

Model (zero-padded discrete convolution)::

    y_hat[j] = sum_p a[p] * h[j - p]

with non-negative integer amplitudes ``a[p]`` (0 means no event starts at
position ``p``), a positive integer response kernel ``h`` and implicit zero
padding outside the sample window.

At high count rates response tails overlap, so amplitudes MUST be chosen
jointly: subtracting a fitted pulse peak-by-peak silently attributes shared
tail energy twice.

Within the caller supplied limits (uniform per-point absolute residual cap
``R``, per-event amplitude cap ``A`` and event-count cap ``E``) the solution
minimizes, in lexicographic order:

    1. maximum absolute residual
    2. sum of absolute residuals
    3. number of events (a[p] > 0)
    4. the amplitude sequence lexicographically (earlier positions dominate)

Exact algorithm, standard library only.  The kernel is short (k <= 6), so the
convolution matrix is banded: scanning samples left to right, sample ``j`` is
the *first* sample touched by amplitude ``a[j]``.  The residual cap pins
``a[j]`` to a small integer interval, and once a row is passed its residual
never changes again.  A window of the last k-1 amplitudes is therefore a
sufficient DP state, pruned by exact forward box-propagation over the next
k-1 rows (each row introduces exactly one new variable, so the propagation is
complete for that horizon).
"""

from __future__ import annotations

from dataclasses import dataclass


class ComputationLimitReached(RuntimeError):
    """Raised when the exact search exceeds its internal transition budget."""


@dataclass(frozen=True)
class DeconvResult:
    feasible: bool
    reason: str | None  # "residual_limit" | "max_events" when infeasible
    amplitudes: tuple[int, ...]
    predicted: tuple[int, ...]
    residuals: tuple[int, ...]
    events: tuple[tuple[int, int], ...]  # (position, amplitude)
    max_abs_residual: int
    sum_abs_residual: int
    num_events: int
    first_unexplainable_position: int | None
    explored_states: int


def _ceil_div(a: int, b: int) -> int:
    """ceil(a / b) for b > 0 (a may be negative)."""
    return -((-a) // b)


def _convolve(amplitudes, kernel: list[int], n: int) -> list[int]:
    k = len(kernel)
    pred = [0] * n
    for p, v in enumerate(amplitudes):
        if v:
            upper = min(k, n - p)
            for t in range(upper):
                pred[p + t] += v * kernel[t]
    return pred


def _candidate_interval(need: int, h0: int, cap: int, amax: int) -> tuple[int, int]:
    """Integer v in [0, amax] satisfying |need - v*h0| <= cap."""
    lo = max(0, _ceil_div(need - cap, h0))
    hi = min(amax, (need + cap) // h0)
    return lo, hi


def _past_tail(win: tuple[int, ...], kernel: list[int], k: int) -> int:
    """Contribution at the current row from amplitudes already in the window."""
    L = len(win)
    acc = 0
    for t in range(1, min(k, L + 1)):
        acc += win[L - t] * kernel[t]
    return acc


def _horizon_check(
    win: tuple[int, ...],
    jnext: int,
    samples: list[int],
    kernel: list[int],
    k: int,
    m: int,
    cap: int,
    amax: int,
    lookahead: int | None = None,
) -> int | None:
    """Exact local feasibility via forward interval propagation.

    The window holds the newest past amplitudes: ``win[-u] == a[jnext-u]``.
    Processing rows from ``jnext`` onward, every row introduces at most one
    new variable (``a[q]``).  Given the propagated [lo, hi] intervals of the
    earlier variables, the row's single absolute-value inequality pins ``a[q]``
    to an exact integer interval (positive coefficients make interval
    arithmetic exact).  An empty intersection at row q means no completion of
    this window can survive row q; that q is returned.  Information only flows
    forward (later variables never enter earlier rows), so a single forward
    sweep over ``lookahead`` rows is complete for that horizon.
    """
    L = len(win)
    n = len(samples)
    if lookahead is None:
        lookahead = k - 1
    last = min(jnext + lookahead - 1, n - 1)
    intervals: dict[int, tuple[int, int]] = {}
    h0 = kernel[0]
    for q in range(jnext, last + 1):
        fixed = 0
        lo_other = 0
        hi_other = 0
        for t in range(1, k):
            p = q - t
            if p < 0 or p >= m:
                continue
            if p >= jnext:
                lo_v, hi_v = intervals[p]
                lo_other += kernel[t] * lo_v
                hi_other += kernel[t] * hi_v
            else:
                idx = L - (jnext - p)
                if idx >= 0:
                    fixed += win[idx] * kernel[t]
        y = samples[q]
        if q < m:
            # Existential range: values a_q can take across SOME choice of
            # the earlier horizon variables within their propagated ranges.
            # Sound (a joint assignment implies these ranges), exact per row
            # for a banded matrix where each row introduces one new variable.
            lo_q = max(0, _ceil_div(y - fixed - hi_other - cap, h0))
            hi_q = min(amax, (y - fixed - lo_other + cap) // h0)
            if lo_q > hi_q:
                return q
            intervals[q] = (lo_q, hi_q)
        else:
            if y < fixed + lo_other - cap or y > fixed + hi_other + cap:
                return q
    return None


def _slide(win: tuple[int, ...], v: int, w: int) -> tuple[int, ...]:
    return (win + (v,))[-w:] if w > 0 else ()


def _greedy_d(
    samples: list[int],
    kernel: list[int],
    cap: int,
    amax: int,
    max_events: int | None = None,
) -> int:
    """Single-path greedy: per row the amplitude zeroing the local residual.

    Returns a feasible max residual (an upper bound), or ``cap`` (the trivial
    bound) when the greedy path dead-ends or exceeds the event budget.
    """
    n = len(samples)
    k = len(kernel)
    m = n - k + 1
    w = k - 1
    h0 = kernel[0]
    win: tuple[int, ...] = ()
    d = 0
    events = 0
    for j in range(n):
        need = samples[j] - _past_tail(win, kernel, k)
        if j < m:
            lo, hi = _candidate_interval(need, h0, cap, amax)
            if lo > hi:
                return cap
            cand = sorted(range(lo, hi + 1), key=lambda v: (abs(need - v * h0), v))
            chosen = None
            for v in cand:
                if max_events is not None and events + (1 if v else 0) > max_events:
                    continue
                nw = _slide(win, v, w)
                if _horizon_check(nw, j + 1, samples, kernel, k, m, cap, amax) is None:
                    chosen = v
                    win = nw
                    break
            if chosen is None:
                return cap
            events += 1 if chosen else 0
            r = abs(need - chosen * h0)
        else:
            r = abs(need)
            if r > cap:
                return cap
            win = _slide(win, 0, w)
        d = max(d, r)
    return d


def _greedy_sum(
    samples: list[int],
    kernel: list[int],
    cap: int,
    amax: int,
    max_events: int,
) -> int | None:
    """Single-path O(n) upper bound on the residual sum under both caps.

    Mirrors :func:`_greedy_d` (locally smallest residual, horizon-verified),
    accumulating the absolute residuals; returns None if the path dead-ends
    or needs more than ``max_events`` events.
    """
    n = len(samples)
    k = len(kernel)
    m = n - k + 1
    w = k - 1
    h0 = kernel[0]
    win: tuple[int, ...] = ()
    events = 0
    total = 0
    for j in range(n):
        need = samples[j] - _past_tail(win, kernel, k)
        if j < m:
            lo, hi = _candidate_interval(need, h0, cap, amax)
            if lo > hi:
                return None
            chosen = None
            for v in sorted(range(lo, hi + 1), key=lambda v: (abs(need - v * h0), v)):
                if events + (1 if v else 0) > max_events:
                    continue
                nw = _slide(win, v, w)
                if _horizon_check(nw, j + 1, samples, kernel, k, m, cap, amax) is None:
                    chosen = v
                    win = nw
                    break
            if chosen is None:
                return None
            events += 1 if chosen else 0
            total += abs(need - chosen * h0)
        else:
            if abs(need) > cap:
                return None
            win = _slide(win, 0, w)
            total += abs(need)
    return total


def _min_d_dp(
    samples: list[int],
    kernel: list[int],
    cap: int,
    amax: int,
    max_events: int | None,
    budget: int = 4_000_000,
) -> tuple[int | None, int]:
    """Minimum attainable max |r| with d <= cap and (optionally) event cap.

    Returns ``(d_star, expanded)``; ``d_star`` is None when infeasible.

    Two label regimes, each with its own valid incumbent:

    * ``max_events is None`` -- one label per window (the smallest running
      maximum residual); the greedy incumbent ignores the event count.
    * ``max_events`` given  -- a Pareto frontier of (events, max residual)
      per window; the incumbent comes from an event-aware greedy path, so
      pruning d > incumbent can never discard the event-capped optimum.
    """
    n = len(samples)
    k = len(kernel)
    m = n - k + 1
    w = k - 1
    h0 = kernel[0]
    incumbent = _greedy_d(samples, kernel, cap, amax, max_events)

    def add_label(front: list[tuple[int, int]], e: int, d: int) -> None:
        for qe, qd in front:
            if qe <= e and qd <= d:
                return
        kept = [(e, d)]
        for qe, qd in front:
            if e <= qe and d <= qd:
                continue
            kept.append((qe, qd))
        front[:] = kept

    if max_events is None:
        states: dict = {(): 0}
    else:
        states = {(): [(0, 0)]}
    expanded = 0

    def tick() -> None:
        if expanded > budget:
            raise ComputationLimitReached("min-max DP transition budget exceeded")

    for j in range(n):
        nxt: dict = {}
        horizon_cache: dict[tuple[int, ...], int | None] = {}
        last_row = j == n - 1

        def horizon_ok(nw: tuple[int, ...]) -> bool:
            hf = horizon_cache.get(nw, -1)
            if hf == -1:
                hf = _horizon_check(nw, j + 1, samples, kernel, k, m, cap, amax)
                horizon_cache[nw] = hf
            return hf is None

        if j < m:
            for win, lab in states.items():
                need = samples[j] - _past_tail(win, kernel, k)
                lo, hi = _candidate_interval(need, h0, cap, amax)
                pairs = [(0, lab)] if max_events is None else [(e0, d0) for e0, d0 in lab]
                for e0, d0 in pairs:
                    order = sorted(range(lo, hi + 1), key=lambda v: (abs(need - v * h0), v))
                    for v in order:
                        r = abs(need - v * h0)
                        d1 = d0 if d0 >= r else r
                        if d1 > incumbent:
                            continue
                        e1 = e0 + (1 if v else 0)
                        if max_events is not None and e1 > max_events:
                            continue
                        nw = _slide(win, v, w)
                        expanded += 1
                        tick()
                        if not horizon_ok(nw):
                            continue
                        if last_row and d1 < incumbent:
                            incumbent = d1
                        if max_events is None:
                            old = nxt.get(nw)
                            if old is None or d1 < old:
                                nxt[nw] = d1
                        else:
                            add_label(nxt.setdefault(nw, []), e1, d1)
        else:
            for win, lab in states.items():
                r = abs(samples[j] - _past_tail(win, kernel, k))
                if r > cap:
                    expanded += 1
                    tick()
                    continue
                nw = _slide(win, 0, w)
                expanded += 1
                tick()
                if not horizon_ok(nw):
                    continue
                pairs = [(0, lab)] if max_events is None else list(lab)
                for e0, d0 in pairs:
                    d1 = d0 if d0 >= r else r
                    if d1 > incumbent:
                        continue
                    if last_row and d1 < incumbent:
                        incumbent = d1
                    if max_events is None:
                        old = nxt.get(nw)
                        if old is None or d1 < old:
                            nxt[nw] = d1
                    else:
                        add_label(nxt.setdefault(nw, []), e0, d1)

        states = nxt

    if not states:
        return None, expanded
    if max_events is None:
        return min(states.values()), expanded
    return min(d for front in states.values() for _e, d in front), expanded


def _first_blocking_row(
    samples: list[int],
    kernel: list[int],
    cap: int,
    amax: int,
    budget: int = 4_000_000,
) -> tuple[int, int]:
    """Earliest sample no complete candidate can explain (+ transition count).

    Exact existence DP with no look-ahead pruning: a window survives layer j
    exactly when some fixed prefix keeps rows 0..j within the cap (the event
    count is irrelevant here -- this runs only after residual infeasibility
    is established for any number of events).  Consequently the first empty
    layer is, definitionally, the earliest sample no complete candidate can
    explain.  Windows are merged in a set because their past is then fixed.
    """
    n = len(samples)
    k = len(kernel)
    m = n - k + 1
    w = k - 1
    h0 = kernel[0]

    states: set[tuple[int, ...]] = {()}
    expanded = 0

    for j in range(n):
        nxt: set[tuple[int, ...]] = set()

        if j < m:
            for win in states:
                need = samples[j] - _past_tail(win, kernel, k)
                lo = max(0, _ceil_div(need - cap, h0))
                hi = min(amax, (need + cap) // h0)
                for v in range(lo, hi + 1):
                    expanded += 1
                    if expanded > budget:
                        raise ComputationLimitReached("diagnostic DP budget exceeded")
                    nxt.add(_slide(win, v, w))
        else:
            for win in states:
                expanded += 1
                if expanded > budget:
                    raise ComputationLimitReached("diagnostic DP budget exceeded")
                if abs(samples[j] - _past_tail(win, kernel, k)) <= cap:
                    nxt.add(_slide(win, 0, w))

        if not nxt:
            return j, expanded
        states = nxt

    return -1, expanded  # actually feasible


def _offer(bucket: list[list], s: int, e: int, pref: tuple[int, ...]) -> None:
    """Insert [sum, events, prefix] into a per-window Pareto bucket."""
    for q in bucket:
        if q[0] <= s and q[1] <= e and (q[0] < s or q[1] < e or q[2] <= pref):
            return
    kept = [[s, e, pref]]
    for q in bucket:
        if s <= q[0] and e <= q[1] and (s < q[0] or e < q[1] or pref <= q[2]):
            continue
        kept.append(q)
    bucket[:] = kept


def _optimal_pareto(
    samples: list[int],
    kernel: list[int],
    cap: int,
    amax: int,
    max_events: int,
    budget: int = 6_000_000,
) -> tuple[tuple[int, ...], int, int, int]:
    """Lexicographically optimize (sum |r|, events, amplitude tuple) at cap."""
    n = len(samples)
    k = len(kernel)
    m = n - k + 1
    w = k - 1
    h0 = kernel[0]

    incumbent_sum = _greedy_sum(samples, kernel, cap, amax, max_events)
    if incumbent_sum is None:
        # The cheap path dead-ended even though a solution exists; disable the
        # sum pruning and let the exact DP (bounded) run.
        incumbent_sum = 10**30

    states: dict[tuple[int, ...], list[list]] = {(): [[0, 0, ()]]}
    expanded = 0

    for j in range(n):
        nxt: dict[tuple[int, ...], list[list]] = {}
        horizon_cache: dict[tuple[int, ...], int | None] = {}

        if j < m:
            for win, labels in states.items():
                need = samples[j] - _past_tail(win, kernel, k)
                lo, hi = _candidate_interval(need, h0, cap, amax)
                if lo > hi:
                    continue
                for s0, e0, pref in labels:
                    if s0 > incumbent_sum:
                        continue
                    order = sorted(
                        range(lo, hi + 1), key=lambda v: (abs(need - v * h0), v)
                    )
                    for v in order:
                        e1 = e0 + (1 if v > 0 else 0)
                        if e1 > max_events:
                            continue
                        r = need - v * h0
                        s1 = s0 + abs(r)
                        if s1 > incumbent_sum:
                            continue
                        nw = _slide(win, v, w)
                        expanded += 1
                        if expanded > budget:
                            raise ComputationLimitReached("optimization state budget exceeded")
                        hf = horizon_cache.get(nw)
                        if hf is None:
                            hf = _horizon_check(nw, j + 1, samples, kernel, k, m, cap, amax)
                            horizon_cache[nw] = hf
                        if hf is not None:
                            continue
                        if j == n - 1 and s1 < incumbent_sum:
                            incumbent_sum = s1
                        _offer(nxt.setdefault(nw, []), s1, e1, pref + (v,))
        else:
            for win, labels in states.items():
                r = samples[j] - _past_tail(win, kernel, k)
                if abs(r) > cap:
                    continue
                nw = _slide(win, 0, w)
                expanded += 1
                if expanded > budget:
                    raise ComputationLimitReached("optimization state budget exceeded")
                hf = horizon_cache.get(nw)
                if hf is None:
                    hf = _horizon_check(nw, j + 1, samples, kernel, k, m, cap, amax)
                    horizon_cache[nw] = hf
                if hf is not None:
                    continue
                bucket = nxt.setdefault(nw, [])
                for s0, e0, pref in labels:
                    s1 = s0 + abs(r)
                    if s1 > incumbent_sum:
                        continue
                    if j == n - 1 and s1 < incumbent_sum:
                        incumbent_sum = s1
                    _offer(bucket, s1, e0, pref)

        states = nxt
        if not states:
            raise ComputationLimitReached("feasible set vanished during optimization")

    best: list | None = None
    for labels in states.values():
        for lab in labels:
            if best is None or (lab[0], lab[1], lab[2]) < (best[0], best[1], best[2]):
                best = lab
    assert best is not None
    return best[2], best[0], best[1], expanded


def solve(
    samples: list[int],
    kernel: list[int],
    residual_limit: int,
    amplitude_limit: int,
    max_events: int,
) -> DeconvResult:
    """Find the optimal feasible event sequence or report infeasibility."""
    n = len(samples)
    m = n - len(kernel) + 1

    def infeasible(reason: str, pos: int | None, explored: int) -> DeconvResult:
        return DeconvResult(
            feasible=False,
            reason=reason,
            amplitudes=tuple([0] * m),
            predicted=tuple([0] * n),
            residuals=tuple(samples),
            events=(),
            max_abs_residual=max((abs(s) for s in samples), default=0),
            sum_abs_residual=sum(abs(s) for s in samples),
            num_events=0,
            first_unexplainable_position=pos,
            explored_states=explored,
        )

    # Pass 1a: minimal attainable max residual ignoring the event cap.
    d_any, explored = _min_d_dp(
        samples, kernel, residual_limit, amplitude_limit, None
    )
    if d_any is None:
        # Nothing meets the residual cap: locate the first unexplainable row.
        block, ex = _first_blocking_row(samples, kernel, residual_limit, amplitude_limit)
        return infeasible("residual_limit", block, explored + ex)

    # Pass 1b: minimal attainable max residual under the event cap.
    d_star, ex = _min_d_dp(
        samples, kernel, residual_limit, amplitude_limit, max_events
    )
    explored += ex
    if d_star is None:
        return infeasible("max_events", None, explored)

    # Pass 2: optimize residual sum, then events, then lexicographic prefix.
    amps, _sum_res, num_events, ex = _optimal_pareto(
        samples, kernel, d_star, amplitude_limit, max_events
    )
    explored += ex

    pred = _convolve(amps, kernel, n)
    resid = tuple(samples[j] - pred[j] for j in range(n))
    events = tuple((p, v) for p, v in enumerate(amps) if v > 0)
    return DeconvResult(
        feasible=True,
        reason=None,
        amplitudes=tuple(amps),
        predicted=tuple(pred),
        residuals=resid,
        events=events,
        max_abs_residual=max((abs(r) for r in resid), default=0),
        sum_abs_residual=sum(abs(r) for r in resid),
        num_events=num_events,
        first_unexplainable_position=None,
        explored_states=explored,
    )
