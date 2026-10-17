"""Rate-accuracy analysis: curve validity, pooled mean curves, task-metric BD-rate, cross-model summaries.

Quality is the task metric (mIoU or mAP); rate is bits per pixel (bpp) of the original image.
"""

import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import config as C
from .style import ANCHOR_ORDER


TASK_ANCHORS = {'mIoU': ('vtm_qpa1', 'vvenc_qpa1', 'bpg', 'jpeg'),
                'mAP': ('vtm_qpa1', 'vvenc_qpa1', 'bpg', 'jpeg')}


def anchors_for(metric: str) -> tuple:
    """Look up the anchors drawn and fitted for a task metric.

    Inputs:
        metric: task metric name ('mIoU' or 'mAP').

    Outputs:
        anchors: tuple of anchor keys from TASK_ANCHORS.
        fallback: ANCHOR_ORDER for a metric not listed in TASK_ANCHORS.
    """
    return TASK_ANCHORS.get(metric, ANCHOR_ORDER)


ANCHOR_RATE_CAP = 2.0


def cap_anchor_curves(anchors: dict, max_bpp_by_model: dict, factor: float = ANCHOR_RATE_CAP) -> dict:
    """Drop anchor points above `factor` x the model's highest learned bpp, so high-rate anchor points do not
    stretch a figure to the right.

    Inputs:
        anchors: {(model, metric): {anchor: [(bpp, value), ...]}}, as from load_anchor_curves.
        max_bpp_by_model: {model: highest learned bpp}.
        factor: the cap, as a multiple of that bpp; default ANCHOR_RATE_CAP.

    Outputs:
        capped: the same structure without the points above the cap.
        fallback: a model missing from `max_bpp_by_model` is dropped (there is nothing to compare its anchors
            with), and so is an anchor or a model left with no points.
    """
    out = {}
    for (model, metric), curves in anchors.items():
        # the model's highest learned bpp; without one, its anchors have nothing to be compared with
        top = max_bpp_by_model.get(model)
        if top is None:
            continue
        # per anchor, keep the points up to `factor` x that bpp, then drop anchors left with none
        kept = {anc: [pt for pt in pts if pt[0] <= factor * top] for anc, pts in curves.items()}
        kept = {anc: pts for anc, pts in kept.items() if pts}
        if kept:
            out[(model, metric)] = kept
    return out


def _max_learned_bpp(results: list[dict]) -> dict:
    """Find the highest learned bpp of each model, over all of its runs.

    Inputs:
        results: run dicts with 'model' and 'rd_points' ([{'bpp', 'value', ...}, ...]).

    Outputs:
        max_bpp: {model: highest bpp among the points that have both a bpp and a value}.
        fallback: a model with no such point is absent.
    """
    out = {}
    for run in results:
        # this run's rates, counting only points that also have a task score
        bpps = [pt['bpp'] for pt in (run.get('rd_points') or [])
                if pt.get('bpp') is not None and pt.get('value') is not None]
        # keep the highest over all of the model's runs
        if bpps:
            out[run['model']] = max(out.get(run['model'], 0.0), max(bpps))
    return out


def _model_baselines(results: list[dict]) -> dict:
    """Pick one uncompressed score per model: the first non-None 'baseline' among its runs (the per-model
    figure warns when a model's runs disagree).

    Inputs:
        results: run dicts with 'model' and 'baseline'.

    Outputs:
        baselines: {model: baseline}.
        fallback: a model without a baseline is absent.
    """
    out = {}
    for run in results:
        # setdefault keeps the first baseline seen for each model
        if run.get('baseline') is not None:
            out.setdefault(run['model'], run['baseline'])
    return out


def _group_by_metric(results: list[dict]) -> dict[str, list[dict]]:
    """Split results by task metric, so a mixed input gives one correctly labelled figure per metric.

    Inputs:
        results: run dicts with 'metric'.

    Outputs:
        groups: {metric: [runs]}, metrics in first-seen order.
        fallback: runs whose 'metric' is missing or None (no RD points) are left out.
    """
    groups: dict[str, list[dict]] = {}
    for run in results:
        metric = run.get('metric')
        if metric is None:
            continue  # no RD points
        groups.setdefault(metric, []).append(run)
    return groups


def _valid_points(pts: list[dict] | None, name: str, context: str) -> list[dict]:
    """Keep the points that have both a bpp and a value, sorted by bpp.

    A point can lack a value when the task metric could not be computed at an extreme rate (for example a
    detector with no detections at all); it is dropped and the rest of the series is kept. The points are
    sorted because every figure draws them as a connected line and a results file need not list them in rate
    order.

    Inputs:
        pts: a run's 'rd_points', dicts with 'bpp' and 'value'; None counts as empty.
        name: the series name, for the message.
        context: where the call comes from, for the message.

    Outputs:
        valid: the valid point dicts, sorted by bpp; possibly empty.
        printed: how many points were dropped, when any were.
    """
    pts = pts or []
    # keep the points with both a rate and a score, in rate order
    valid = sorted((pt for pt in pts if pt.get('bpp') is not None and pt.get('value') is not None),
                   key=lambda pt: pt['bpp'])
    # report what was dropped, so a missing point is never silent
    dropped = len(pts) - len(valid)
    if dropped:
        print(f"  {context}: dropped {dropped} point(s) with no value/bpp for {name} "
             f"(kept {len(valid)}/{len(pts)})")
    return valid


def pooled_mean_curve(series: dict, n: int = 240, coverage_power: float = 0.0) -> dict | None:
    """Fit the mean curve of a group of models: one cubic in log10(bpp), as in BD-rate, through all models'
    points pooled, so the curve stays smooth as models enter or leave the rate range.

    Each point is weighted (share of models whose measured rate range contains its rate) ** coverage_power /
    (that model's point count): every model counts equally, and rates reached by few models cannot bend the
    shared fit. The curve is made monotone in rate (its upper envelope, as in task_bd_rate) and capped at the
    best value any model measured.

    Inputs:
        series: {model: [(bpp, y), ...]}; y is any quality value (score, difference or ratio). Every model
            needs at least one point, and every bpp must be > 0.
        n: number of grid points, log-spaced from the lowest to the highest bpp of all models.
        coverage_power: exponent of the coverage share in the weights; 0 counts every model equally at every
            rate.

    Outputs:
        'grid': the rates (bpp).
        'mean': the pooled fit on the grid.
        fallback: None when fewer than 4 points have bpp > 0.
    """
    # point weights: coverage share ** coverage_power, divided by the model's point count
    spans = [(min(pt[0] for pt in mpts), max(pt[0] for pt in mpts)) for mpts in series.values()]
    # share of models whose measured rate range [first, last] contains a given rate
    share = lambda bpp: sum(first <= bpp <= last for first, last in spans) / len(spans)
    # pool every model's points, each with its weight
    pts, wts = [], []
    for name, mpts in series.items():
        pts += mpts; wts += [share(pt[0]) ** coverage_power / len(mpts) for pt in mpts]
    # the cubic needs at least 4 points with a positive rate
    good = [(pt, wt) for pt, wt in zip(pts, wts) if pt[0] > 0]
    if len(good) < 4:
        return None
    # weighted least-squares cubic in log10(bpp); np.polyfit squares its weights, hence the square root
    xx = np.log10([pt[0] for pt, _ in good]); yy = np.array([pt[1] for pt, _ in good])
    params = np.polyfit(xx, yy, 3, w=np.sqrt([wt for _, wt in good]))
    evaluate = lambda x, params: np.polyval(params, x)
    # evaluation grid: log-spaced over the union of all models' rates
    lo = min(pt[0] for pt in pts); hi = max(pt[0] for pt in pts)
    grid = np.logspace(np.log10(lo), np.log10(hi), n); grid[0], grid[-1] = lo, hi   # exact ends, free of rounding
    x = np.log10(grid)
    # Monotone envelope: accuracy cannot fall as the rate rises, but a cubic fitted to few points can
    # turn down at its high end. The task-metric BD-rate uses the same envelope.
    mean = np.fmax.accumulate(evaluate(x, params), axis=-1)
    # a tail fitted to few (or down-weighted) points could otherwise climb past every measured value,
    # even above uncompressed: cap it at the best value any model reached
    mean = np.minimum(mean, max(pt[1] for pt in pts))
    return {'grid': grid, 'mean': mean}


def run_lambda_ref(run: dict) -> float:
    """Read the lambda_ref a run's codec was trained with: the one place every module reads it from.

    Inputs:
        run: run dict, with 'train_params' and/or a top-level 'lambda_ref'.

    Outputs:
        lambda_ref: train_params['lambda_ref'] as a float.
        fallback: the run's top-level 'lambda_ref' when train_params has none; NaN when neither is present.
    """
    params = run.get('train_params') or {}
    return float(params.get('lambda_ref', run.get('lambda_ref', 'nan')))


def run_hash(run: dict) -> str:
    """Name a run: the id config.py uses to list runs, as (model, run hash). Every result file carries it.

    Inputs:
        run: run dict with a 'run_hash'.

    Outputs:
        hash: run['run_hash'] (as a string).
        error: ValueError for a run without one; check_run_hashes rejects duplicates.
    """
    if not run.get('run_hash'):
        raise ValueError(f"run of {run.get('model')} has no run_hash")
    return str(run['run_hash'])


def check_run_hashes(runs: list[dict]) -> None:
    """Make sure no two runs share a hash, since config.py names runs by it.

    Inputs:
        runs: every run dict of the analysis (results and overlay).

    Outputs:
        error: ValueError naming the duplicated hashes and their models.
    """
    seen = {}
    for run in runs:
        seen.setdefault(run_hash(run), []).append(run['model'])
    dups = {h: models for h, models in seen.items() if len(models) > 1}
    if dups:
        raise ValueError(f'runs share a hash: {dups}; give each run a unique run_hash')


def run_kind(run: dict) -> str | None:
    """Which list in config.py a run is in.

    Inputs:
        run: run dict.

    Outputs:
        kind: 'unified' (config.UNIFIED), 'rescued' (config.RESCUED) or 'fine-tuned' (config.FINETUNED).
        fallback: None for a run in none of them; select_listed_runs leaves such runs out.
    """
    key = (run['model'], run_hash(run))
    for kind, listed in (('unified', C.UNIFIED), ('rescued', C.RESCUED), ('fine-tuned', C.FINETUNED)):
        if key in listed:
            return kind
    return None


def is_retrain(run: dict) -> bool:
    """Whether a run is a retrain after the unified recipe, i.e. listed in config.RESCUED or config.FINETUNED.

    Inputs:
        run: run dict.

    Outputs:
        retrain: True for a rescued or fine-tuned run.
    """
    return run_kind(run) in ('rescued', 'fine-tuned')


def select_listed_runs(results: list[dict], overlay: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep only the runs listed in config.py (UNIFIED, RESCUED or FINETUNED), so an unclassified run can never
    change an output.

    Inputs:
        results: the main run dicts.
        overlay: the extra run dicts.

    Outputs:
        results: the listed main runs.
        overlay: the listed extra runs.
        warning: for every run in none of the lists, naming it and saying it is left out of every output (figures
            and tables); and for every listed run that is not in the data.
        error: ValueError for a run listed in more than one list.
    """
    lists = {'UNIFIED': C.UNIFIED, 'RESCUED': C.RESCUED, 'FINETUNED': C.FINETUNED}
    # a run may be in one list only
    seen = {}
    for name, listed in lists.items():
        for key in listed:
            seen.setdefault(key, []).append(name)
    twice = {key: names for key, names in seen.items() if len(names) > 1}
    if twice:
        raise ValueError(f'runs listed in more than one of config.UNIFIED, RESCUED, FINETUNED: {twice}')
    # unlisted runs are dropped, loudly
    dropped = [run for run in results + overlay if run_kind(run) is None]
    if dropped:
        names = ', '.join(f"{run['model']} (hash {run_hash(run)}, lambda_ref {run_lambda_ref(run):g})" for run in dropped)
        warnings.warn(f'{len(dropped)} run(s) are in none of config.UNIFIED, RESCUED, FINETUNED and are left out of '
                      f'every output (figures and tables): {names}', stacklevel=2)
    # listed runs missing from the data
    present = {(run['model'], run_hash(run)) for run in results + overlay}
    missing = sorted(key for key in seen if key not in present)
    if missing:
        warnings.warn(f'{len(missing)} run(s) listed in config.py are not in the data: {missing}', stacklevel=2)
    return ([run for run in results if run_kind(run) is not None],
            [run for run in overlay if run_kind(run) is not None])


def _one_run_per_model(group: list[dict], metric: str | None = None) -> list[dict]:
    """Choose the run that represents each model in a mean curve.

    Runs without valid points or a baseline, collapsed runs (is_collapsed) and the (model, metric) pairs in
    config.MEAN_EXCLUDE are left out. A model with one remaining run is represented by it; a model with several
    is represented by its one listed retrain (is_retrain).

    Inputs:
        group: run dicts of one task metric.
        metric: that metric, for the config.MEAN_EXCLUDE lookup.

    Outputs:
        runs: [(run, valid points), ...], one per model, in the order the models first appear.
        fallback: a model with no run left is absent.
        error: ValueError when a model has several remaining runs and not exactly one of them is listed in
            config.RESCUED or config.FINETUNED.
    """
    eligible = {}
    for run in group:
        # skip a run that cannot stand for its model: no points, no baseline, collapsed, or excluded
        pts = _valid_points(run['rd_points'], run.get('display_name', run['model']), 'mean')
        if not pts or run.get('baseline') is None or is_collapsed(run, pts):
            continue
        if (run['model'], metric) in C.MEAN_EXCLUDE:
            continue
        eligible.setdefault(run['model'], []).append((run, pts))
    chosen = []
    for model, runs in eligible.items():
        # several runs of one model: its listed retrain represents it, never a guess
        if len(runs) > 1:
            runs = [(run, pts) for run, pts in runs if is_retrain(run)]
            if len(runs) != 1:
                listed = [(run_hash(run), run_lambda_ref(run)) for run, _ in eligible[model]]
                raise ValueError(f'{model} ({metric}) has several runs (hash, lambda_ref): {listed}; exactly one '
                                 f'must be listed in config.RESCUED or config.FINETUNED')
        chosen.append(runs[0])
    return chosen


COLLAPSE_FRACTION = 0.10   # a run whose every point is below this fraction of uncompressed has collapsed


def is_collapsed(run: dict, pts: list[dict]) -> bool:
    """Decide whether a run's codec has collapsed: its task metric stays below COLLAPSE_FRACTION of
    uncompressed at every rate it was tested at.

    Inputs:
        run: run dict with 'baseline'.
        pts: its valid points (dicts with 'value').

    Outputs:
        collapsed: True when every point is below COLLAPSE_FRACTION x baseline.
        fallback: False with no points, or with a missing or non-positive baseline.
    """
    bl = run.get('baseline')
    return bool(pts) and bl is not None and bl > 0 and max(pt['value'] for pt in pts) < COLLAPSE_FRACTION * bl


def _crossing(grid, y, level):
    """Find the first rate at which a curve reaches a quality level.

    Inputs:
        grid: rates (bpp), ascending.
        y: the curve on `grid`; NaN entries are skipped.
        level: the quality level.

    Outputs:
        bpp: the first grid rate with y >= level.
        fallback: NaN when the curve never reaches it.
    """
    # ignore NaN entries, then take the first grid rate at or above the level
    ok = ~np.isnan(y)
    hit = np.where(y[ok] >= level)[0]
    return grid[ok][hit[0]] if len(hit) else float('nan')


def _monotone_envelope(points):
    """Reduce a rate-quality curve to its upper envelope, on which quality -> rate is a function: sorted by
    bpp, keeping only the points that raise the best quality so far.

    Inputs:
        points: [(bpp, quality), ...] in any order.

    Outputs:
        envelope: [(bpp, quality), ...], ascending in both.
        fallback: empty for empty input.
    """
    env = []
    # walk in rate order; keep a point only if it beats the best quality so far
    for bpp, qual in sorted(points):
        if not env or qual > env[-1][1]:
            env.append((bpp, qual))
    return env


def task_bd_rate(test, anchor, min_quality: float | None = None) -> dict:
    """Compute the Bjontegaard rate difference (%) of `test` against `anchor` in the task metric.

    Both curves are reduced to their monotone envelope, log10(bpp) is interpolated as a pchip function of
    quality, and the difference of the exact integrals is averaged over the quality range both curves cover.
    This is the JVET CTC BD-rate (as in the bjontegaard package, method 'pchip') on the monotone envelopes.

    Inputs:
        test: [(bpp, quality), ...] of the codec being measured.
        anchor: [(bpp, quality), ...] of the reference.
        min_quality: if given, only quality >= this counts (e.g. 90% of the uncompressed score); the full
            curves are still interpolated, and the average runs from exactly this quality.

    Outputs:
        'bd': BD-rate in %; negative when `test` needs fewer bits for the same quality.
        'lo', 'hi': the quality range integrated over.
        fallback: 'bd' is NaN when either envelope has fewer than 3 points or the range is empty (no overlap,
            or none above min_quality); 'lo' and 'hi' are then the bounds found, or NaN when a curve is empty.
    """
    # both curves as upper envelopes, so quality -> rate is a function
    anc_env, test_env = _monotone_envelope(anchor), _monotone_envelope(test)
    # the quality range both curves cover: from the higher of their lowest qualities to the lower of their
    # highest (NaN when a curve is empty)
    lo, hi = max(anc_env[0][1], test_env[0][1]) if anc_env and test_env else np.nan, min(anc_env[-1][1], test_env[-1][1]) if anc_env and test_env else np.nan
    # optionally start the range at a minimum quality (e.g. 90% of uncompressed)
    if min_quality is not None:
        lo = max(lo, min_quality)
    # no BD-rate with fewer than 3 envelope points per curve, or an empty range
    if len(anc_env) < 3 or len(test_env) < 3 or not hi > lo:
        return {'bd': float('nan'), 'lo': lo, 'hi': hi}
    # log10(rate) as a pchip function of quality, one per curve
    anc_fit = PchipInterpolator([qual for _, qual in anc_env], np.log10([bpp for bpp, _ in anc_env]))
    test_fit = PchipInterpolator([qual for _, qual in test_env], np.log10([bpp for bpp, _ in test_env]))
    # mean log-rate difference over the range (exact integral of the pchips), then as a rate change in %
    avg = (test_fit.integrate(lo, hi) - anc_fit.integrate(lo, hi)) / (hi - lo)
    return {'bd': float((10 ** avg - 1) * 100), 'lo': lo, 'hi': hi}


def cross_model_bd_matrix(docs: list[dict], own: dict, anchors: dict, models: list[str], metric: str,
                          anchor: str = 'vtm_qpa1', targets: list[str] | None = None) -> dict:
    """Compute the BD-rate matrices of one group of models tested on each other's codecs: rows are teachers
    (the model a codec was trained on), columns targets (the model it was tested on).

    Each model's own sweep fills the diagonal and is the matched reference of its column.

    Inputs:
        docs: cross-model result docs, each {'trained_on_task_model': teacher, 'results':
            [{'tested_on_task_model', 'metric', 'rd_points': [{'bpp', 'value'}, ...], 'baseline'}, ...]}.
            Entries of another metric, or with a teacher or target outside the group, are ignored; of a
            repeated pair, the last is kept.
        own: {model: ([(bpp, value), ...], baseline)}, each model's own-codec sweep; replaces any doc entry on
            the diagonal.
        anchors: {(model, metric): {anchor: [(bpp, value), ...]}}, as from load_anchor_curves.
        models: the teachers, one row each.
        metric: the task metric.
        anchor: the anchor key to compare against.
        targets: the columns; default `models`. Test-only targets (no codec of their own) may follow the
            teachers.

    Outputs:
        'vs_matched': (len(models), len(targets)) array, BD-rate vs the target's own codec; 0 on the diagonal.
        'vs_anchor': same shape, BD-rate vs the anchor on the target.
        'peak': same shape, the pair's best score / its baseline.
        'anchor_vs_matched': per target, the anchor's BD-rate vs the target's own codec, i.e. the cost of
            serving a collapsed pair with the anchor.
        'curves': {(teacher, target): ([(bpp, value), ...], baseline)}.
        'anchor_curves': {target: anchor points, or None}.
        fallback: NaN where the pair has no curve or the BD-rate is undefined, in a 'vs_matched' column (and
            'anchor_vs_matched' entry) missing from `own`, for a target without that anchor, and in 'peak'
            without a baseline.
    """
    targets = list(targets or models)       # columns: the teachers plus any test-only targets
    # row / column index of each model: rows are teachers (whose codec), columns are targets (tested on)
    row_of = {mdl: i for i, mdl in enumerate(models)}
    col_of = {mdl: j for j, mdl in enumerate(targets)}
    # the three result matrices start as NaN; a cell stays NaN when its pair has no data
    vs_matched, vs_anchor, peak = (np.full((len(models), len(targets)), np.nan) for _ in range(3))
    # 1. Collect each (teacher, target) pair's rate-accuracy curve from the cross-model docs. A doc holds
    #    one teacher's codec tested on several targets; keep only pairs within this group, in this task metric.
    curves = {}
    for doc in docs:
        teacher = doc.get('trained_on_task_model')
        for entry in doc.get('results') or []:
            target = entry.get('tested_on_task_model')
            if teacher in row_of and target in col_of and entry.get('metric') == metric:
                # the pair's points that have both a rate and a score, with the target's uncompressed score
                pts = [(pt['bpp'], pt['value']) for pt in entry.get('rd_points') or []
                       if pt.get('bpp') is not None and pt.get('value') is not None]
                curves[(teacher, target)] = (pts, entry.get('baseline'))
    # 2. The diagonal (a codec on its own model) comes from each model's own sweep, not from the docs.
    for mdl in models:
        if mdl in own:
            curves[(mdl, mdl)] = own[mdl]
    # 3. Per target column: the anchor's BD-rate vs the target's own codec, i.e. the cost of a pair served
    #    by the anchor. NaN when the target has no anchor curve or no codec of its own.
    anchor_vs_matched = np.full(len(targets), np.nan)
    for target, j in col_of.items():
        anc_pts, ref = (anchors.get((target, metric)) or {}).get(anchor), own.get(target)
        if anc_pts and ref:
            anchor_vs_matched[j] = task_bd_rate(anc_pts, ref[0])['bd']
    # 4. Fill the matrices, pair by pair.
    for (teacher, target), (pts, bl) in curves.items():
        i, j = row_of[teacher], col_of[target]
        # peak: the best score the pair reached, as a fraction of the target's uncompressed score
        if pts and bl:
            peak[i, j] = max(val for _, val in pts) / bl
        # vs_matched: BD-rate vs the target's own codec; 0 on the diagonal by definition, NaN for a test-only
        # target (it has no codec of its own)
        ref = own.get(target)
        if ref and pts:
            vs_matched[i, j] = 0.0 if teacher == target else task_bd_rate(pts, ref[0])['bd']
        # vs_anchor: BD-rate vs the anchor on the target
        anc_pts = (anchors.get((target, metric)) or {}).get(anchor)
        if anc_pts and pts:
            vs_anchor[i, j] = task_bd_rate(pts, anc_pts)['bd']
    return {'vs_matched': vs_matched, 'vs_anchor': vs_anchor, 'peak': peak, 'anchor_vs_matched': anchor_vs_matched,
            # the raw curves: {(teacher, target): ([(bpp, value)], baseline)} and the anchor's per target
            'curves': dict(curves), 'anchor_curves': {target: (anchors.get((target, metric)) or {}).get(anchor) for target in targets}}


def _stat(vals) -> tuple:
    """Summarise a list of values, ignoring NaN: the statistic every cross-model summary reports.

    Inputs:
        vals: numbers, possibly NaN.

    Outputs:
        mean: the mean of the non-NaN values.
        sd: their sample s.d. (ddof=1).
        n_ok: how many values are not NaN.
        n: how many values there are.
        fallback: the mean is NaN with no value, the s.d. with fewer than 2.
    """
    arr = np.array(vals, dtype=float)
    # drop NaN values; the mean and sample s.d. are over the rest
    ok = arr[~np.isnan(arr)] if len(arr) else arr
    return (float(ok.mean()) if len(ok) else float('nan'),
            float(ok.std(ddof=1)) if len(ok) > 1 else float('nan'), int(len(ok)), int(len(arr)))


def summarize_cross_group(mats: dict, models: list[str], targets: list[str], cells=None) -> dict:
    """Summarise a group's cross-model BD-rates, or a selection of its cells, with the anchor as fallback.

    A cross pair whose BD-rate vs the anchor is undefined (the codec collapses on that target) is served by
    the anchor instead: it counts 0 against the anchor, and the anchor's BD-rate against the target's own
    codec ('anchor_vs_matched'). Each statistic is (mean, sample s.d., values that are not NaN, values), over
    the values that are not NaN.

    Inputs:
        mats: output of cross_model_bd_matrix.
        models: its rows (teachers).
        targets: its columns.
        cells: boolean array of shape (len(models), len(targets)) selecting the cells to summarise; default
            every cell. summarize_cross_roles passes one model's row or column.

    Outputs:
        'matched_vs_anchor': statistic of the diagonal, i.e. each model's own codec vs the anchor.
        'cross_vs_anchor': statistic of the off-diagonal pairs vs the anchor, after the fallback.
        'cross_vs_matched': statistic of the off-diagonal pairs vs the target's own codec, after the fallback.
        'fallbacks': (pairs served by the anchor, off-diagonal pairs).
        fallback: a mean is NaN with no value and an s.d. with fewer than 2; a test-only target has no codec
            of its own, so its pairs have no 'cross_vs_matched' value.
    """
    # walk every selected cell: diagonal cells are each model's own codec; off-diagonal cells are the cross pairs
    diag, vs_anc, vs_own, n_fallback, n_pairs = [], [], [], 0, 0
    for i, teacher in enumerate(models):
        for j, target in enumerate(targets):
            if cells is not None and not cells[i, j]:
                continue
            bd_anc, bd_own = mats['vs_anchor'][i, j], mats['vs_matched'][i, j]
            if teacher == target:
                diag.append(bd_anc)
                continue
            n_pairs += 1
            if bd_anc != bd_anc:                       # collapsed: served by the anchor
                n_fallback += 1
                bd_anc, bd_own = 0.0, mats['anchor_vs_matched'][j]
            # the pair's two BD-rates, after the fallback, enter the group's statistics
            vs_anc.append(bd_anc)
            vs_own.append(bd_own)
    return {'matched_vs_anchor': _stat(diag), 'cross_vs_anchor': _stat(vs_anc), 'cross_vs_matched': _stat(vs_own),
            'fallbacks': (n_fallback, n_pairs)}


def summarize_cross_roles(mats: dict, models: list[str], targets: list[str], name: str) -> dict:
    """Summarise one model's cross-model BD-rates in its two roles: summarize_cross_group on the model's
    row (as teacher) and on its column (as target), with the same anchor fallback. The model's own cell (its
    codec on itself) is in neither, so 'matched_vs_anchor' stays empty. Nothing is excluded.

    Inputs:
        mats: output of cross_model_bd_matrix.
        models: its rows (teachers).
        targets: its columns.
        name: the model.

    Outputs:
        'source': the model as teacher (its codec on every other target): the dict of
            summarize_cross_group, with an empty 'matched_vs_anchor' (NaN, NaN, 0, 0).
        'target': the model as target (every other codec on it), the same.
        fallback: 'source' is absent when `name` is not a row and 'target' when it is not a column.
    """
    shape = (len(models), len(targets))
    out = {}
    # as teacher: the model's row, without its own cell
    if name in models:
        cells = np.zeros(shape, dtype=bool)
        cells[models.index(name), :] = True
        if name in targets:
            cells[models.index(name), targets.index(name)] = False
        out['source'] = summarize_cross_group(mats, models, targets, cells)
    # as target: the model's column, without its own cell
    if name in targets:
        cells = np.zeros(shape, dtype=bool)
        cells[:, targets.index(name)] = True
        if name in models:
            cells[models.index(name), targets.index(name)] = False
        out['target'] = summarize_cross_group(mats, models, targets, cells)
    return out
