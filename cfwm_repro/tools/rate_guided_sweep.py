"""Point selection for the rate-guided RD sweep (test_rate_guided.py).

Pure decision logic -- no torch, Lightning or MLflow -- so it can be unit-tested on synthetic
curves and replayed offline against finished dense sweeps (bin/replay_rate_guided_sweep.py).

next_lambda() is a PURE FUNCTION of the points measured so far. That is what makes resume
free: a requeued job, or a later launch on the same checkpoint, recomputes the same next lambda
from the results file and carries on. Nothing about the sweep's progress lives anywhere else.

Every lambda it proposes is rounded to float32 first. The module holds lmbda as a float32
tensor and tags its logged bpp with f'{lmbda.item():.3f}', so a proposal that already IS a
float32 value is recorded, keyed and logged identically -- and a point that failed is
recognised as tried on the next call instead of being proposed again until the budget runs out.
"""
import math
import struct
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

GAP_LOG2_BPP = 0.5            # a gap wider than sqrt(2)x in bpp is never clean
MIN_LAMBDA_RATIO = 2 ** 0.25  # resolution limit: a pair closer than this in lambda is never split
LADDER_STEP = 1.99            # "one x2 ladder step apart", with slack for float32 rounding


def lambda_key(value: float) -> str:
    """Stable string key for a lambda, so resume-matching doesn't depend on a float
    round-tripping bit-exactly through JSON."""
    return f'{float(value):.6g}'


def as_float32(value: float) -> float:
    return struct.unpack('f', struct.pack('f', float(value)))[0]


@dataclass(frozen=True)
class SweepConfig:
    lambda_ref: float
    floor_frac: float = 0.6
    max_points: int = 14
    min_points: int = 8
    plateau_tol: float = 0.015
    q_step: float = 0.03
    rate_stall: float = 0.10
    hi_frac: float = 0.85


@dataclass(frozen=True)
class Point:
    lam: float
    bpp: float
    q: float  # task score / uncompressed baseline


def as_point(p: dict, baseline: float) -> Optional[Point]:
    """The Point for one results-file entry, or None if it cannot inform a decision (a failed
    point: bpp or value None, or a non-positive bpp)."""
    lam, bpp, value = p.get('lambda'), p.get('bpp'), p.get('value')
    if lam is None or bpp is None or value is None:
        return None
    lam, bpp, value = float(lam), float(bpp), float(value)
    if not (lam > 0 and bpp > 0) or math.isnan(value):
        return None
    return Point(lam, bpp, value / baseline)

def usable_points(points: Sequence[dict], baseline: float) -> list:
    """Points that can inform a decision, ascending in lambda."""
    return sorted((pt for pt in (as_point(p, baseline) for p in points) if pt is not None),
                  key=lambda p: p.lam)


def plateau_window(pts: Sequence[Point]) -> list:
    """Every point within the top 2x of measured bpp."""
    r_top = max(p.bpp for p in pts)
    return [p for p in pts if p.bpp >= r_top / 2]


def plateau_reached(pts: Sequence[Point], cfg: SweepConfig) -> bool:
    """Every point in the top 2x of bpp within plateau_tol of the best score -- OR the top two
    ladder-spaced points gain nothing (within plateau_tol) over every point below them while
    sitting at or above floor_frac. The second clause stops a noise-driven
    runaway: one lucky high draw raises q_best for good, and later average plateau points then fail
    the first clause forever (a curve with 0.5% noise could otherwise climb to x2048)."""
    window = plateau_window(pts)
    q_best = max(p.q for p in pts)
    if len(window) >= 2 and all(p.q >= q_best - cfg.plateau_tol for p in window):
        return True
    top = max(pts, key=lambda p: p.lam)
    under = _step_below(sorted(pts, key=lambda p: p.lam), top)
    if under is None:
        return False
    below = [p for p in pts if p.lam < under.lam]
    if not below:
        return False
    top_best = max(top.q, under.q)
    return top_best >= cfg.floor_frac and top_best <= max(p.q for p in below) + cfg.plateau_tol


def select_points(points: Sequence[dict], baseline: float, cfg: SweepConfig) -> list:
    """The results-file entries the sweep decides from AND budgets against.

    Every entry this script placed (it carries a 'source') counts. An entry from an earlier
    sweep (no 'source') counts only where the rate-guided sweep would want a point:
      * inside the region of interest, q >= floor_frac;
      * plus the single one closest below the floor (highest lambda), which establishes it;
      * minus the deep plateau: of the old points, only the fewest, lowest-bpp ones that still
        let the plateau test pass are kept -- so a grid that ran far up the plateau does not
        spend the budget. The cut is only accepted where the kept points' best score is
        within plateau_tol of the best overall, so a flat stretch near the floor is not
        mistaken for the plateau.
    A failed old entry never counts. Nothing is removed from the file -- this only decides
    what the sweep looks at.
    """
    own = [p for p in points if p.get('source')]
    old = [(pt, p) for p in points if not p.get('source')
           for pt in [as_point(p, baseline)] if pt is not None]
    if not old:
        return list(own)
    in_region = [t for t in old if t[0].q >= cfg.floor_frac]
    below_floor = [t for t in old if t[0].q < cfg.floor_frac]
    if not in_region:
        # A grid entirely below the floor (a collapsed or weak run) already characterises the
        # run: count all of it, so the sweep stops on the budget instead of climbing past the
        # grid (counting only its top point could send it to x2000).
        keep = list(old)
    else:
        lowest_in = min(t[0].lam for t in in_region)
        under = [t for t in below_floor if t[0].lam < lowest_in]
        keep = in_region + ([max(under, key=lambda t: t[0].lam)] if under else [])

    own_pts = [pt for pt in (as_point(p, baseline) for p in own) if pt is not None]
    q_all = max([pt.q for pt in own_pts] + [t[0].q for t in keep])
    for cut, _ in sorted(in_region, key=lambda t: t[0].bpp):
        subset = own_pts + [t[0] for t in keep if t[0].bpp <= cut.bpp]
        if (len(subset) >= 2 and plateau_reached(subset, cfg)
                and max(x.q for x in subset) >= q_all - cfg.plateau_tol):
            # Drop only old points that are themselves ON the plateau, above the cut: an old
            # point on the slope always counts, however the plateau got established.
            keep = [t for t in keep if t[0].bpp <= cut.bpp or t[0].q < q_all - cfg.plateau_tol]
            break
    kept = {id(p) for _, p in keep}
    return [p for p in points if p.get('source') or id(p) in kept]

def gap_size(lo: Point, hi: Point, cfg: SweepConfig) -> float:
    """g <= 1 means the pair is close enough in both rate and quality ('clean')."""
    return max(abs(math.log2(hi.bpp / lo.bpp)) / GAP_LOG2_BPP, abs(hi.q - lo.q) / cfg.q_step)


def _step_below(pts, p):
    """The highest point at least one ladder step (x2) below p in lambda. pts ascending."""
    below = [x for x in pts if x.lam <= p.lam / LADDER_STEP]
    return below[-1] if below else None


def _step_above(pts, p):
    above = [x for x in pts if x.lam >= p.lam * LADDER_STEP]
    return above[0] if above else None


def _on_ladder(lam: float, cfg: SweepConfig) -> bool:
    """True for lambda_ref x 2^k -- the ladder points (exact in float32, so exact here)."""
    r = math.log2(lam / cfg.lambda_ref)
    return abs(r - round(r)) < 1e-4


def _stalled(lo: Point, hi: Point, cfg: SweepConfig) -> bool:
    return hi.bpp < lo.bpp * (1 + cfg.rate_stall)


def _midpoint(lo: Point, hi: Point) -> float:
    # Geometric mean: under R ~ lambda^d this is exactly the midpoint in log-bpp, no fit needed.
    return as_float32(math.sqrt(lo.lam * hi.lam))


def _splittable(lo: Point, hi: Point, tried: set) -> bool:
    return hi.lam / lo.lam > MIN_LAMBDA_RATIO and lambda_key(_midpoint(lo, hi)) not in tried


def _span(lo: Point, hi: Point, cfg: SweepConfig) -> str:
    return f'x{lo.lam / cfg.lambda_ref:.3g}-x{hi.lam / cfg.lambda_ref:.3g}'


def next_lambda(points: Sequence[dict], baseline: float,
                cfg: SweepConfig) -> tuple:
    """(next lambda, reason) or (None, reason) when the sweep is finished.

    points: the results file's rd_points (dicts with 'lambda', 'bpp', 'value'). Entries this
        script placed carry a 'source'; entries from an earlier sweep do not, and count only
        where select_points() says so.
    baseline: the uncompressed task score, in the same units as each point's 'value'.
    """
    if not baseline or baseline <= 0:
        raise ValueError(f'baseline must be a positive task score, got {baseline!r}')
    # Every lambda already in the file is "tried", counted or not: it is never measured twice.
    tried = {lambda_key(p['lambda']) for p in points if p.get('lambda') is not None}
    all_lams = [float(p['lambda']) for p in points if p.get('lambda') is not None]
    points = select_points(points, baseline, cfg)
    lam, reason = _propose(points, tried, all_lams, baseline, cfg)
    # The budget only binds when there is something left to do, so a sweep that finishes on
    # exactly max_points reports 'done', and a cut-short one says what it would have measured.
    if lam is not None and len(points) >= cfg.max_points:
        return None, (f'stop: budget spent ({len(points)} points, max_points={cfg.max_points}); '
                      f'next would have been {reason}')
    return lam, reason


def _propose(points: list, tried: set, all_lams: list, baseline: float,
             cfg: SweepConfig) -> tuple:
    """next_lambda's decision, ignoring the budget. `points` are the selected entries;
    `tried` / `all_lams` cover every entry in the file, counted or not."""
    # 1. The initial three: lambda_ref x 1, 2, 1/2.
    for rel in (1.0, 2.0, 0.5):
        lam = as_float32(cfg.lambda_ref * rel)
        if lambda_key(lam) not in tried:
            return lam, f'initial: relative lambda {rel:g}'

    pts = usable_points(points, baseline)
    if len(pts) < 2:
        return None, f'stop: only {len(pts)} usable point(s) after the initial three'

    # 2. Up the x2 ladder until the plateau shows, or the rate stops responding.
    top = pts[-1]
    if plateau_reached(pts, cfg):
        q_best = max(p.q for p in pts)
        # A flat curve far below the floor passes the plateau test too; name it for what it is.
        up_state = ('plateau' if q_best >= cfg.floor_frac
                    else f'collapsed (q_best={q_best:.3f})')
    else:
        prev = _step_below(pts, top)
        lam = as_float32(top.lam * 2)
        if prev is not None and _stalled(prev, top, cfg):
            up_state = f'rate stall (bpp {prev.bpp:.4g} -> {top.bpp:.4g})'
        elif lambda_key(lam) in tried:
            up_state = 'next lambda already tried and failed'
        else:
            window = plateau_window(pts)
            return lam, (f'up: no plateau yet (top-2x-bpp window: {len(window)} pt, '
                         f'min q {min(p.q for p in window):.3f} vs best {max(p.q for p in pts):.3f})')

    # 3. Down the ladder until quality drops below floor_frac, or the rate stops responding.
    bottom = pts[0]
    if bottom.q < cfg.floor_frac:
        down_state = f'floor (q={bottom.q:.3f})'
    else:
        nxt = _step_above(pts, bottom)
        lam = as_float32(bottom.lam / 2)
        if nxt is not None and _stalled(bottom, nxt, cfg):
            down_state = f'rate stall (bpp {nxt.bpp:.4g} -> {bottom.bpp:.4g})'
        elif lambda_key(lam) in tried:
            down_state = 'next lambda already tried and failed'
        else:
            return lam, f'down: q={bottom.q:.3f} still >= floor_frac {cfg.floor_frac}'

    # 4. Refine below the knee, top-down.
    knee = min(plateau_window(pts), key=lambda p: p.lam)
    below = [p for p in pts if p.lam <= knee.lam]

    # 4a. Ladder gaps 1 and 2 below the knee, bisected once each. Counted on the LADDER
    # points only: a bisection point can land inside the plateau window and move `knee`
    # down, which would renumber the gaps and skip the real gap 2.
    ladder = [p for p in pts if _on_ladder(p.lam, cfg)]
    ladder_window = [p for p in plateau_window(pts) if _on_ladder(p.lam, cfg)]
    hi = min(ladder_window, key=lambda p: p.lam) if ladder_window else knee
    for n in (1, 2):
        lo = _step_below(ladder, hi)
        if lo is None:
            break
        # Any entry inside the gap -- counted or not -- means it has already been split.
        bisected = any(lo.lam < lam < hi.lam for lam in all_lams)
        g = gap_size(lo, hi, cfg)
        # A gap whose two ends score the same (within plateau_tol) has nothing to resolve, however
        # wide it is in rate -- e.g. a collapsed curve.
        if not bisected and g > 1 and abs(hi.q - lo.q) > cfg.plateau_tol and _splittable(lo, hi, tried):
            return _midpoint(lo, hi), f'refine: gap {n} below the knee, {_span(lo, hi, cfg)}, g={g:.2f}'
        hi = lo

    # 4b. Then every open gap whose upper point is at or above hi_frac, largest g first.
    pairs = list(zip(below[:-1], below[1:]))

    def _pick(cands):
        return max(cands, key=lambda c: (gap_size(c[0], c[1], cfg), c[1].lam))

    upper = [(lo, hi) for lo, hi in pairs
             if hi.q >= cfg.hi_frac and gap_size(lo, hi, cfg) > 1 and _splittable(lo, hi, tried)]
    if upper:
        lo, hi = _pick(upper)
        return _midpoint(lo, hi), (f'refine: above hi_frac, {_span(lo, hi, cfg)}, '
                                   f'g={gap_size(lo, hi, cfg):.2f}')

    # 4c. Backstop to min_points: a sweep that would end with fewer than
    # min_points points steps DOWN past the floor (x1/2 below the lowest point) rather than
    # padding the range it already measured. Only an early end triggers it -- a collapsed run,
    # or a weak one whose curve plateaus below the floor -- so every run that reaches min_points
    # on its own is unchanged. It stops where the rate stops responding (the lowest point within
    # rate_stall of the point one ladder step above: the codec is sending almost nothing) or on
    # a lambda already tried.
    if len(points) < cfg.min_points:
        nxt = _step_above(pts, bottom)
        lam = as_float32(bottom.lam / 2)
        if nxt is not None and _stalled(bottom, nxt, cfg):
            down_state += f'; backstop stopped on a rate stall (bpp {nxt.bpp:.4g} -> {bottom.bpp:.4g})'
        elif lambda_key(lam) in tried:
            down_state += '; backstop stopped: next lambda already tried and failed'
        else:
            return lam, (f'down past floor: backstop to min_points ({len(points)}/{cfg.min_points}), '
                         f'lowest q={bottom.q:.3f}')

    # The normal end of a sweep: every phase above is finished (the initial three measured; the climb and the
    # descent ended; no gap left to split; the backstop not needed or stopped). The message says how the climb
    # and the descent ended.
    return None, f'done: up={up_state}; down={down_state}; {len(points)} points'


def run_sweep(doc: dict, baseline: float, cfg: SweepConfig,
              measure: Callable[[float], Optional[dict]],
              persist: Callable[[], None], log: Callable[[str], None] = print) -> bool:
    """Drive next_lambda until it returns None. Returns True if measurement was interrupted.

    measure(lam) -> the RD point dict ({'lambda', 'bpp', 'value', 'metric'}), or None if the
    run was interrupted. Each point is appended to doc['rd_points'] and persisted before the
    next decision, so a preempted job loses at most the point in flight.
    """
    while True:
        lam, reason = next_lambda(doc['rd_points'], baseline, cfg)
        if lam is None:
            log(f'[rate-guided] {reason}')
            return False
        log(f'[rate-guided] {reason} -> lambda={lam:.6g} (x{lam / cfg.lambda_ref:.4g})')
        point = measure(lam)
        if point is None:
            log('[rate-guided] interrupted before the point finished')
            return True
        point = dict(point, source='sweep', sweep_reason=reason)
        doc['rd_points'].append(point)
        persist()
        log(f"[rate-guided]   bpp={point['bpp']}  {point['metric']}={point['value']}")
