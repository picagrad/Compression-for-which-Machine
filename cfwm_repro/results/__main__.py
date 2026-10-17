"""Write the paper's rate-accuracy results from result files.

    python -m cfwm_repro.results --results R.json [R.json ...] [--overlay X.json ...] --anchors ANCHORS.jsonl
                                 [--anchor-bpp mean_image|aggregate] [--cross-collated DIR] --out OUT [--name NAME] [--only GROUP ...]

scripts/run_paper_results.sh runs it on the published result files of this repository.

Groups: per_model, summaries, backbones, bd_tables, cross_matrices, cross_scatter.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from . import config as C
from .analysis import (_group_by_metric, _one_run_per_model, _valid_points, check_run_hashes, cross_model_bd_matrix,
                       is_collapsed, run_hash, run_kind, run_lambda_ref, select_listed_runs, summarize_cross_group,
                       summarize_cross_roles, task_bd_rate)
from .figures import (save_all_models_figure, save_cross_matrix_figure,
                      save_cross_scatter_figure, save_mean_figure,
                      save_per_model_figures, save_relative_figure)
from .inputs import ANCHOR_BPP, disambiguate_model_collisions, load_anchor_curves, load_runs, save_csv, save_json, write_csv
from .style import ANCHOR_LABEL, TASK_SHORT

GROUPS = ['per_model', 'summaries', 'backbones', 'bd_tables', 'cross_matrices', 'cross_scatter']


def per_model(results, overlay, anchors, out, experiment):
    """Draw the per-model figures of every run in `results` and `overlay`.

    Inputs:
        results: run dicts.
        overlay: more run dicts, drawn alongside. Both lists are copied, not changed; with an overlay, display
            names are disambiguated across all runs.
        anchors: as from load_anchor_curves.
        out: the output root.
        experiment: passed on; not used.

    Outputs:
        files: <out>/per_model/<model>_rd.png.
    """
    # copies, so labelling the runs does not change the caller's lists
    runs = copy.deepcopy(results) + copy.deepcopy(overlay)
    if overlay:
        disambiguate_model_collisions(runs)
    save_per_model_figures(runs, out / 'per_model', experiment, anchors=anchors)


def summaries(results, anchors, folder, title):
    """Draw the all-models figures (absolute, delta and ratio) of one set of runs.

    Inputs:
        results: run dicts.
        anchors: as from load_anchor_curves.
        folder: the output folder.
        title: used in the skip messages.

    Outputs:
        files: <folder>/<task>_models_{absolute,delta,ratio}.png.
    """
    save_all_models_figure(results, folder / 'models_absolute.png', f'RD curves — {title}')
    save_relative_figure(results, folder / 'models_delta.png', f'RD curves relative to baseline — {title}', anchors=anchors)
    save_relative_figure(results, folder / 'models_ratio.png', f'RD curves as a fraction of the uncompressed score — {title}',
                         anchors=anchors, mode='ratio')


def backbones(results, anchors, out, experiment):
    """Write the per-backbone outputs of config.BACKBONES.

    Inputs:
        results: run dicts.
        anchors: as from load_anchor_curves.
        out: the output root.
        experiment: the name of the result set (--name); it names each group's JSON and CSV ('_all_' replaced by
            '_<folder>_', else '_<folder>' appended) and is used in the summaries' titles.

    Outputs:
        figure: <out>/summaries/<task>_mean.png, the mean curves per backbone
            with the all-backbones curve.
        printed: per task and group, the bpp at which the mean reaches 90% and 95% of uncompressed, for ours
            and MEAN_REPORT_ANCHOR, and our saving.
        per group: <out>/summaries/<folder>/ with the group's summaries (summaries) and its results JSON and
            CSV; a group with no runs is skipped.
    """
    # {group label: models} from config.BACKBONES
    groups = {label: members for label, members in C.BACKBONES.values()}
    kw = dict(anchor_list=C.MEAN_ANCHORS, report_anchor=C.MEAN_REPORT_ANCHOR,
              coverage_power=C.COVERAGE_POWER, xmax=C.MEAN_XMAX)
    # the published by-backbone figure; it also returns the 90/95% crossings printed below
    table = save_mean_figure(results, groups, out / 'summaries' / 'mean.png',
                                         anchors, overall=True, colours=C.MEAN_COLOURS,
                                         overall_colour=C.MEAN_ALL_COLOUR, **kw)
    print(f"{'task':5s} {'backbone':8s} {'n':>2s}   bpp at 90% / 95%: ours | {C.MEAN_REPORT_ANCHOR} | saving")
    for (metric, label), row in table.items():
        cells = []
        for level in (90, 95):
            ours_bpp, anc_bpp = row.get(f'ours_{level}', float('nan')), row.get(f'anchor_{level}', float('nan'))
            cells.append(f'{ours_bpp:.3f} | {anc_bpp:.3f} | {100 * (1 - ours_bpp / anc_bpp):3.0f}%')
        print(f"{metric:5s} {label:8s} {row['n']:2d}   " + '   '.join(cells))
    # per group: its all-models figures, and its own results JSON and CSV
    for folder, (label, members) in C.BACKBONES.items():
        subset = [run for run in results if run['model'] in members]
        if subset:
            summaries(subset, anchors, out / 'summaries' / folder, f'{experiment} — {label}')
            stem = experiment.replace('_all_', f'_{folder}_') if '_all_' in experiment else f'{experiment}_{folder}'
            save_json(subset, out / 'summaries' / folder / f'{stem}.json')
            save_csv(subset, out / 'summaries' / folder / f'{stem}.csv')


def _sweep_label(row: dict) -> str:
    """Label a per-sweep BD-table row by what kind of sweep it is, checked in this order: a collapsed or
    excluded (MEAN_EXCLUDE) row keeps its label; a run listed in UNIFIED is 'healthy', one in FINETUNED
    'fine-tuned' and one in RESCUED 'rescued'.

    Inputs:
        row: a per-sweep row dict with 'kind' (run_kind) and 'note' (its label so far).

    Outputs:
        label: the row's label.
    """
    if row['note'] in ('collapsed', 'excluded'):
        return row['note']
    return {'unified': 'healthy', 'fine-tuned': 'fine-tuned', 'rescued': 'rescued'}[row['kind']]


def bd_tables(results, overlay, anchors, out):
    """Write both BD-rate tables, against every anchor in BD_ANCHORS: one row per sweep, and one row per recipe
    group.

    Per-sweep table: every run of `results` and `overlay` with valid points, a baseline and a curve of the main
    anchor (BD_ANCHORS[0]), except the runs in BD_EXCLUDE. Rows are labelled by _sweep_label: healthy, rescued,
    fine-tuned, collapsed or excluded.

    Recipe table, per task: 'Unified recipe' (every run in UNIFIED, plus UNSWEPT_COLLAPSES),
    'Fine-tuned' (the mean-curve run of each model not in RECIPE_OUTLIERS), then one row per RECIPE_OUTLIERS
    model. Against each anchor, a run without a BD-rate (collapsed) or with a positive one (worse than the
    anchor) is served by that anchor: it counts 0, and is counted in served_by_<anchor>.

    Inputs:
        results: the main run dicts.
        overlay: extra run dicts (e.g. superseded or collapsed sweeps), used in the tables but not as
            mean-curve runs.
        anchors: as from load_anchor_curves.
        out: the output root.

    Outputs:
        per-sweep table: <out>/summaries/bd_rates_per_sweep.csv (_write_sweep_table).
        recipe table: <out>/summaries/bd_rates.csv (_write_recipe_table).
        fallback: collapsed runs are left out of the per-sweep table unless BD_SHOW_COLLAPSED, which also adds
            UNSWEPT_COLLAPSES. A never-swept collapse counts 0 against every anchor.
        error: ValueError for a per-sweep row that fits no group.
    """
    main = C.BD_ANCHORS[0]
    # lookups: each model's backbone label, the backbone order, and the runs used in the mean curves
    backbone_of = {mdl: label for label, mdls in C.BACKBONES.values() for mdl in mdls}
    order = {label: i for i, (label, _) in enumerate(C.BACKBONES.values())}
    paper = {id(run) for metric, group in _group_by_metric(results).items() for run, _ in _one_run_per_model(group, metric)}

    def bd_pair(run, anchor):
        """Compute a run's BD-rates against an anchor on its own model.

        Inputs:
            run: run dict.
            anchor: the anchor key.

        Outputs:
            bd: BD-rate, in %.
            bd_q: BD-rate above BD_QUALITY_FRAC of uncompressed.
            fallback: (NaN, NaN) without valid points, a baseline or that anchor curve; either value is NaN
                when undefined.
        """
        bl = run.get('baseline')
        pts = _valid_points(run.get('rd_points'), run['model'], 'bd table')
        anc_pts = (anchors.get((run['model'], run['metric'])) or {}).get(anchor)
        test = [(pt['bpp'], pt['value']) for pt in pts]
        if not pts or not anc_pts or not bl:
            return float('nan'), float('nan')
        return task_bd_rate(test, anc_pts)['bd'], task_bd_rate(test, anc_pts, min_quality=C.BD_QUALITY_FRAC * bl)['bd']

    # per-sweep table: one row per sweep with valid points, a baseline and the main anchor curve
    rows = []
    for run in results + overlay:
        metric, bl = run.get('metric'), run.get('baseline')
        pts = _valid_points(run.get('rd_points'), run['model'], 'bd table')
        if not pts or not bl or not (anchors.get((run['model'], metric)) or {}).get(main):
            continue
        # runs left out by hand (config.BD_EXCLUDE)
        if (run['model'], run_hash(run)) in C.BD_EXCLUDE:
            continue
        # a first label, from what kind of run it is
        if is_collapsed(run, pts):
            note = 'collapsed'
        elif id(run) in paper:
            note = 'in mean'
        elif (run['model'], metric) in C.MEAN_EXCLUDE:
            note = 'excluded'
        elif run in overlay:
            note = 'superseded'
        else:
            note = 'extra'
        rows.append({'task': metric, 'backbone': backbone_of.get(run['model'], 'other'), 'model': run['model'],
                     'hash': run_hash(run), 'kind': run_kind(run), 'lambda_ref': run_lambda_ref(run), 'points': len(pts), 'note': note,
                     'peak': max(pt['value'] for pt in pts) / bl,
                     'bd': {anc: bd_pair(run, anc) for anc in C.BD_ANCHORS}})
    # collapsed runs that were never swept are only shown with BD_SHOW_COLLAPSED
    for metric, model, lam in (C.UNSWEPT_COLLAPSES if C.BD_SHOW_COLLAPSED else []):
        rows.append({'task': metric, 'backbone': backbone_of.get(model, 'other'), 'model': model, 'hash': '', 'kind': 'unified', 'lambda_ref': lam,
                     'points': 0, 'note': 'collapsed', 'peak': float('nan'),
                     'bd': {anc: (float('nan'), float('nan')) for anc in C.BD_ANCHORS}})
    if not C.BD_SHOW_COLLAPSED:
        rows = [row for row in rows if row['note'] != 'collapsed']
    # segmentation first, then backbone order, model and lambda_ref
    rows.sort(key=lambda row: (row['task'] != 'mIoU', order.get(row['backbone'], len(order)), row['model'], row['lambda_ref']))
    # the sweep's label in the table: healthy, rescued or fine-tuned (other rows keep theirs)
    for row in rows:
        row['note'] = _sweep_label(row)
    _write_sweep_table(rows, out / 'summaries' / 'bd_rates_per_sweep.csv')

    def band(run):
        """Read a run's training lambda band.

        Inputs:
            run: run dict with 'train_params'.

        Outputs:
            band: (lambda_ref, lambda_ref x lambda_min, lambda_ref x lambda_max), lambda_min and lambda_max
                being ratios to lambda_ref.
            fallback: lambda_ref (run_lambda_ref) is NaN when missing; a missing lambda_min or lambda_max ratio
                is taken from UNSWEPT_BAND.
        """
        params = run.get('train_params') or {}
        lam = run_lambda_ref(run)
        return (lam, lam * float(params.get('lambda_min', C.UNSWEPT_BAND[0])),
                lam * float(params.get('lambda_max', C.UNSWEPT_BAND[1])))

    def group(task, label, runs, unswept=()):
        """Build one recipe-table row: against each anchor, the mean and sample s.d. over its runs, with runs
        served by the anchor counted 0.

        Inputs:
            task: the task metric.
            label: the row name.
            runs: run dicts.
            unswept: [(model, lambda_ref)] of collapsed runs that were never swept; each is served by every
                anchor (0 everywhere), with UNSWEPT_BAND as its lambda band.

        Outputs:
            row: {'task', 'label', 'n' (runs), 'bands' ([(lambda_ref, lambda_min, lambda_max)] per run), 'anchors':
                {anchor: {'bd', 'bd_sd', 'bd_q', 'bd_q_sd', 'served', 'q_served'}}}, as _write_recipe_table writes
                it. 'served' counts the runs served by the anchor; 'q_served' the further runs whose value above
                BD_QUALITY_FRAC was served by it.
            fallback: an s.d. is None with fewer than 2 values, a mean NaN with none.
        """
        per = {anc: {'vals': [], 'q_vals': [], 'served': 0, 'q_served': 0} for anc in C.BD_ANCHORS}
        bands = []
        for run in runs:
            for anc, acc in per.items():
                bd, bd_q = bd_pair(run, anc)
                # served by the anchor: no BD-rate (collapsed) or a positive one (worse than the anchor); counts 0
                served, q_served = bd != bd or bd > 0, bd_q != bd_q or bd_q > 0
                acc['served'] += served
                acc['q_served'] += q_served and not served
                acc['vals'].append(0.0 if served else bd)
                acc['q_vals'].append(0.0 if q_served else bd_q)
            # its training lambda band, for the table's lambda range
            bands.append(band(run))
        # a collapse that was never swept is served by every anchor
        for model, lam in unswept:
            for acc in per.values():
                acc['vals'].append(0.0); acc['q_vals'].append(0.0); acc['served'] += 1
            bands.append((lam, lam * C.UNSWEPT_BAND[0], lam * C.UNSWEPT_BAND[1]))
        # mean and sample s.d. over the row's runs (NaN entries ignored)
        sd = lambda vals: float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) > 1 else None
        mean = lambda vals: float(np.nanmean(vals)) if (~np.isnan(vals)).any() else float('nan')
        stats = {}
        for anc, acc in per.items():
            vals, q_vals = np.array(acc['vals'], dtype=float), np.array(acc['q_vals'], dtype=float)
            stats[anc] = {'bd': mean(vals), 'bd_sd': sd(vals), 'bd_q': mean(q_vals), 'bd_q_sd': sd(q_vals),
                          'served': acc['served'], 'q_served': acc['q_served']}
        return {'task': task, 'label': label, 'n': len(runs) + len(unswept), 'bands': bands, 'anchors': stats}

    # the recipe table, per task: the unified recipe (config.UNIFIED), the runs as finally trained, and the
    # named outliers, each its own row
    recipe = []
    for task in ('mIoU', 'mAP'):
        unified = [run for run in results + overlay if run.get('metric') == task and run_kind(run) == 'unified']
        unswept = [(mdl, lam) for mtr, mdl, lam in C.UNSWEPT_COLLAPSES if mtr == task]
        recipe.append(group(task, 'Unified recipe', unified, unswept))
        final = [run for run in results if run.get('metric') == task and id(run) in paper and run['model'] not in C.RECIPE_OUTLIERS]
        recipe.append(group(task, 'Fine-tuned', final))
        for mdl, out_label in C.RECIPE_OUTLIERS.items():
            out_runs = [run for run in results if run.get('metric') == task and run['model'] == mdl]
            if out_runs:
                recipe.append(group(task, out_label, out_runs))
    _write_recipe_table(recipe, out / 'summaries' / 'bd_rates.csv')


def _pct() -> int:
    """The BD-rate quality threshold as a whole percentage, for column names (e.g. 90 for BD_QUALITY_FRAC 0.9)."""
    return int(round(C.BD_QUALITY_FRAC * 100))


def _write_sweep_table(rows: list[dict], path: Path) -> None:
    """Write the per-sweep BD-rate table as a CSV, one line per sweep, values at full precision.

    Inputs:
        rows: dicts with 'task', 'backbone', 'model', 'hash' (run_hash), 'note' (the sweep label), 'lambda_ref',
            'points', 'peak'
            (best score / uncompressed) and 'bd' ({anchor: (BD-rate, BD-rate above BD_QUALITY_FRAC)}).
        path: the .csv file; parent directories are created.

    Outputs:
        file: columns task, backbone, model, run_hash, sweep, lambda_ref, points, peak_fraction, then per anchor of
            BD_ANCHORS: bd_rate_vs_<anchor> and bd_rate_vs_<anchor>_ge<pct> (BD-rates in %).
        printed: the path.
        fallback: an undefined BD-rate or peak is written as nan.
    """
    pct = _pct()
    headers = ['task', 'backbone', 'model', 'run_hash', 'sweep', 'lambda_ref', 'points', 'peak_fraction']
    for anc in C.BD_ANCHORS:
        headers += [f'bd_rate_vs_{anc}', f'bd_rate_vs_{anc}_ge{pct}']
    lines = []
    for row in rows:
        line = [row['task'], row['backbone'], row['model'], row['hash'], row['note'], row['lambda_ref'], row['points'],
                row['peak']]
        for anc in C.BD_ANCHORS:
            line += list(row['bd'][anc])
        lines.append(line)
    write_csv(headers, lines, path)


def _write_recipe_table(groups: list[dict], path: Path) -> None:
    """Write the recipe-level BD-rate table as a CSV, one line per row group, values at full precision.

    Inputs:
        groups: row dicts from bd_tables.group.
        path: the .csv file; parent directories are created.

    Outputs:
        file: columns task, row, runs, lambda_min and lambda_max (the lowest and highest training lambda of the
            row's runs), then per anchor of BD_ANCHORS: bd_rate_vs_<anchor> and its _sd, its _ge<pct> and
            _ge<pct>_sd (the BD-rate above BD_QUALITY_FRAC of uncompressed), served_by_<anchor> (runs without a
            BD-rate against it, or worse than it, counted 0) and served_by_<anchor>_ge<pct> (the further runs whose
            thresholded value was served by it).
        printed: the path.
        fallback: an undefined mean is written as nan, and an s.d. over fewer than 2 runs is left empty.
    """
    pct = _pct()
    headers = ['task', 'row', 'runs', 'lambda_min', 'lambda_max']
    for anc in C.BD_ANCHORS:
        headers += [f'bd_rate_vs_{anc}', f'bd_rate_vs_{anc}_sd', f'bd_rate_vs_{anc}_ge{pct}', f'bd_rate_vs_{anc}_ge{pct}_sd',
                    f'served_by_{anc}', f'served_by_{anc}_ge{pct}']
    blank = lambda val: '' if val is None else val
    lines = []
    for grp in groups:
        line = [grp['task'], grp['label'], grp['n'], min(band[1] for band in grp['bands']),
                max(band[2] for band in grp['bands'])]
        for anc in C.BD_ANCHORS:
            st = grp['anchors'][anc]
            line += [st['bd'], blank(st['bd_sd']), st['bd_q'], blank(st['bd_q_sd']), st['served'], st['q_served']]
        lines.append(line)
    write_csv(headers, lines, path)


def _hide(mats, keep):
    """Keep only some target columns of a set of matrices.

    Inputs:
        mats: output of cross_model_bd_matrix.
        keep: indices of the targets to keep.

    Outputs:
        mats: a new dict in which 2-D arrays keep those columns, 1-D arrays those entries, and anything else
            is unchanged.
    """
    return {key: (mat[:, keep] if getattr(mat, 'ndim', 0) == 2 else mat[keep] if getattr(mat, 'ndim', 0) == 1 else mat)
            for key, mat in mats.items()}


def cross_matrices(results, overlay, anchors, collated, out, draw=True):
    """Compute the cross-model BD-rate matrices of every backbone group, task and calibration setting, and
    draw them.

    For each setting in CROSS_SETTINGS, reads <collated>/<setting>/*.json and keeps one doc per teacher: the
    one from the run in the mean curves, else the model's only doc. Each teacher's own sweep is found by the
    doc's 'source_run_hash' (the run_hash of the sweep of the teacher's codec). A group needs at least 2 teachers with results in the task; CROSS_EXTRA_TARGETS
    with results are added as test-only targets.

    Inputs:
        results: the main run dicts.
        overlay: extra run dicts, also searched for the teachers' own sweeps.
        anchors: as from load_anchor_curves.
        collated: the folder of collated cross-model docs.
        out: the output root.
        draw: write the figures.

    Outputs:
        matrices: {(metric, group folder, setting): (mats, models, targets, group label)}, including the
            hidden targets.
        figures: with `draw`, <out>/summaries/cross_model/<task>_<group>_<setting>_bd_matrix.png per group and
            setting, without the CROSS_MATRIX_HIDE targets.
        fallback: a teacher with no doc, or with several and none from the mean-curve run, is left out.
    """
    # each sweep by its run hash, and the run in the mean curves of each model
    by_run = {run_hash(run): run for run in results + overlay}
    paper_run = {run['model']: run_hash(run) for metric, group in _group_by_metric(results).items()
                 for run, _ in _one_run_per_model(group, metric)}
    grid_mats = {}
    for setting, setting_label in C.CROSS_SETTINGS.items():
        # the collated cross-model docs of this calibration setting
        all_docs = [json.load(open(fpath)) for fpath in sorted((collated / setting).glob('*.json'))]
        by_model = {}
        for doc in all_docs:
            by_model.setdefault(doc.get('trained_on_task_model'), []).append(doc)
        docs = []
        for mdl, doc_list in sorted(by_model.items()):   # one codec per model: the paper's run, else the only one
            pick = [doc for doc in doc_list if doc.get('source_run_hash') == paper_run.get(mdl)] or (doc_list if len(doc_list) == 1 else [])
            if pick:
                docs.append(pick[0])
        # each teacher's own sweep (its codec on itself), found through the doc's source run hash
        own = {}
        for doc in docs:
            run = by_run.get(doc.get('source_run_hash'))
            if run is not None:
                mdl = doc.get('trained_on_task_model')
                own[mdl] = ([(pt['bpp'], pt['value']) for pt in _valid_points(run['rd_points'], mdl, 'own')], run['baseline'])
        teachers = {doc.get('trained_on_task_model') for doc in docs}
        # one matrix per backbone group and task: the group's teachers with results in that task, plus the
        # test-only targets that have results
        for folder, (label, members) in C.BACKBONES.items():
            for metric in ('mIoU', 'mAP'):
                models = [mdl for mdl in members
                          if mdl in teachers and any(run['model'] == mdl and run['metric'] == metric for run in results)]
                if len(models) < 2:
                    continue
                targets = models + [tgt for tgt in C.CROSS_EXTRA_TARGETS.get(folder, [])
                                    if any(entry.get('tested_on_task_model') == tgt and entry.get('metric') == metric
                                           for doc in docs for entry in doc.get('results') or [])]
                mats = cross_model_bd_matrix(docs, {mdl: own[mdl] for mdl in models if mdl in own}, anchors, models, metric,
                                             anchor=C.BD_ANCHORS[0], targets=targets)
                grid_mats[(metric, folder, setting)] = (mats, models, targets, label)
                # the per-group figure, without the hidden targets
                if draw:
                    keep = [j for j, tgt in enumerate(targets) if tgt not in C.CROSS_MATRIX_HIDE]
                    save_cross_matrix_figure(
                        _hide(mats, keep), models,
                        out / 'summaries' / 'cross_model' / f'{TASK_SHORT[metric]}_{folder}_{setting}_bd_matrix.png',
                        f"Cross-model compatibility: {'segmentation' if metric == 'mIoU' else 'detection'}, "
                        f'{label}, {setting_label}', ANCHOR_LABEL[C.BD_ANCHORS[0]], clip=C.CROSS_CLIP,
                        targets=[targets[j] for j in keep])
    return grid_mats


def _scatter_label(key, label):
    """Name a cross-model group in the scatter legend, e.g. 'Seg, ResNet (cal.)'.

    Inputs:
        key: (metric, group folder, setting), a CROSS_GRID entry.
        label: the group's backbone label.

    Outputs:
        name: 'Seg' or 'Det', the backbone label, and ' (cal.)' for the calibrated setting. In CROSS_GRID the ResNet
            groups are the calibrated ones.
    """
    metric, _, setting = key
    return f"{'Seg' if metric == 'mIoU' else 'Det'}, {label}" + (' (cal.)' if setting == 'cal' else '')


def cross_scatter(grid_mats, out):
    """Draw the cross-model scatters, calibrated and uncalibrated: each group's means with the anchor fallback,
    each CROSS_HIGHLIGHT model in its two roles, and every pair behind its group's mean.

    Inputs:
        grid_mats: output of cross_matrices.
        out: the output root.

    Outputs:
        calibrated: <out>/summaries/cross_model/cross_model_scatter.png, the CROSS_GRID groups.
        uncalibrated: <out>/summaries/cross_model/cross_model_scatter_uncal.png, the same groups with every setting
            'nocal'.
        fallback: in a group where the highlighted model is not a teacher, it has no codec of its own, so its
            points have no y value.
    """
    fig_dir = out / 'summaries' / 'cross_model'
    uncal = [(metric, folder, 'nocal') for metric, folder, _ in C.CROSS_GRID]
    _cross_scatter(grid_mats, C.CROSS_GRID, fig_dir / 'cross_model_scatter.png', 'Cross-model compatibility (VTM fallback)',
                   ymax=390)
    # two uncalibrated pairs reach y ~ 420 and ~ 540 (2026-10-16): a higher cap, so they are drawn where they are
    _cross_scatter(grid_mats, uncal, fig_dir / 'cross_model_scatter_uncal.png',
                   'Cross-model compatibility (VTM fallback, uncalibrated)', ymax=560)


def _cross_scatter(grid_mats, grid, path, title, ymax):
    """One cross-model scatter of the groups in `grid` ((metric, group folder, setting) keys of grid_mats), its y axis
    from -75 to `ymax` (values above it are drawn at it)."""
    # one point per group: its means with the anchor fallback
    entries = []
    for key in grid:
        if key in grid_mats:
            mats, models, targets, label = grid_mats[key]
            stats = summarize_cross_group(mats, models, targets)
            n_fallback, n_pairs = stats['fallbacks']
            entries.append({'label': _scatter_label(key, label), 'grid_key': key, 'stats': stats,
                            'note': f'{n_fallback}/{n_pairs} VTM' if n_fallback else None})
    # the highlighted models (CROSS_HIGHLIGHT) in their two roles, in every group
    for key in grid:
        if key not in grid_mats:
            continue
        mats, models, targets, _ = grid_mats[key]
        for outlier, outlier_label in C.CROSS_HIGHLIGHT.items():
            for role, stats in summarize_cross_roles(mats, models, targets, outlier).items():
                # not a teacher in this group: it has no own codec here, so no y value
                if outlier not in models:
                    stats['cross_vs_matched'] = (float('nan'), float('nan'), 0, 0)
                n_fallback, n_pairs = stats['fallbacks']
                entries.append({'label': outlier_label, 'name': f'{n_fallback}/{n_pairs} VTM' if n_fallback else '', 'stats': stats, 'outlier': True,
                                'role': role, 'offset': C.CROSS_LABEL_OFFSET.get((outlier, role))})
    # every pair behind its group's mean: a collapsed pair sits at (0, the anchor's BD-rate vs the target's own
    # codec); pairs of a highlighted model are marked by role
    with_pairs = []
    for entry in entries:
        pair_entry = dict(entry)
        if not entry.get('outlier'):
            mats, models, targets, _ = grid_mats[entry['grid_key']]
            pairs = []
            for i, teacher in enumerate(models):
                for j, target in enumerate(targets):
                    if teacher == target:
                        continue
                    bd_anc, bd_own = mats['vs_anchor'][i, j], mats['vs_matched'][i, j]
                    role = 'source' if teacher in C.CROSS_HIGHLIGHT else ('target' if target in C.CROSS_HIGHLIGHT else None)
                    pairs.append((0.0, float(mats['anchor_vs_matched'][j]), True, role) if bd_anc != bd_anc
                                 else (float(bd_anc), float(bd_own), False, role))
            pair_entry['pairs'] = pairs
        with_pairs.append(pair_entry)
    save_cross_scatter_figure(with_pairs, path, ANCHOR_LABEL[C.BD_ANCHORS[0]], title=title, figsize=(12.5, 4.4),
                              ylim=(-75, ymax), ymax=ymax, xlim=(-80, 5),
                              xlabel=f'BD-Rate (%) vs. {ANCHOR_LABEL[C.BD_ANCHORS[0]]}',
                              ylabel='BD-Rate (%) vs. matched codecs', text_scale=1.3, outlier_name='Mask2Former-R50',
                              size_note=False)


def main(argv=None):
    """Run the command line; see the module docstring for the arguments.

    Inputs:
        argv: the arguments; default sys.argv[1:].

    Outputs:
        files: the groups chosen with --only, under --out.
        error: a file given as --results or --overlay that is not a results file exits with an error, and asking for
            the cross-model outputs without --cross-collated exits with a usage error.
    """
    # command-line arguments (see the module docstring)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', required=True, nargs='+', type=Path,
                        help='results files (one run each, or a list of runs), in order: the runs of the mean curves')
    parser.add_argument('--overlay', nargs='*', default=[], type=Path,
                    help='more results files: drawn on the per-model figures and counted in the BD tables, not in the means')
    parser.add_argument('--anchors', required=True, type=Path, help='anchor results JSONL')
    parser.add_argument('--anchor-bpp', default='mean_image', choices=ANCHOR_BPP,
                        help="the anchors' rate: mean_image (default), the mean of per-image bpp, as our codec's bpp "
                             "is computed; aggregate, total bits / total pixels, as in the submitted paper's tables")
    parser.add_argument('--cross-collated', type=Path, help='collated cross-model docs: <dir>/{cal,nocal}/*.json')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--name', default=None,
                        help="name of the result set, used in titles and file names (default: the --results file's "
                             'stem when there is one file)')
    parser.add_argument('--only', nargs='+', default=GROUPS, choices=GROUPS)
    args = parser.parse_args(argv)

    # inputs: the results, the overlay runs and the anchor curves
    results = load_runs(args.results)
    overlay = load_runs(args.overlay)
    check_run_hashes(results + overlay)
    # only runs listed in config.py (UNIFIED, RESCUED, FINETUNED) take part; any other run is dropped with a warning
    results, overlay = select_listed_runs(results, overlay)
    anchors = load_anchor_curves(args.anchors, bpp=args.anchor_bpp)
    if args.name is None and len(args.results) != 1:
        parser.error('--name is needed with several --results files')
    experiment = args.name or args.results[0].stem
    # each requested output group in turn
    if 'per_model' in args.only:
        per_model(results, overlay, anchors, args.out, experiment)
    if 'summaries' in args.only:
        summaries(results, anchors, args.out / 'summaries', experiment)
    if 'backbones' in args.only:
        backbones(results, anchors, args.out, experiment)
    if 'bd_tables' in args.only:
        bd_tables(results, overlay, anchors, args.out)
    if {'cross_matrices', 'cross_scatter'} & set(args.only):
        if args.cross_collated is None:
            parser.error('--cross-collated is needed for the cross-model outputs')
        grid_mats = cross_matrices(results, overlay, anchors, args.cross_collated, args.out, draw='cross_matrices' in args.only)
        if 'cross_scatter' in args.only:
            cross_scatter(grid_mats, args.out)


if __name__ == '__main__':
    main()
