"""Reading result files (results JSON, cross-model docs, anchor JSONL) and writing JSON and CSV."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import os
import sys
import tempfile



def _atomic_write_json(doc, path: Path) -> None:
    """Write JSON so a reader never sees a partial file: to a temporary file in the same folder, then renamed
    over `path`.

    Inputs:
        doc: any JSON-serialisable object.
        path: the target file; parent directories are created.

    Outputs:
        file: `doc` as indented JSON at `path`.
        fallback: on any error the temporary file is removed and the error raised again.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # write to a temporary file next to the target, then rename it over the target in one step
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_json(results: list[dict], path: Path) -> None:
    """Write a results list as indented JSON, atomically.

    Inputs:
        results: run dicts.
        path: the .json file.

    Outputs:
        file: the JSON at `path`.
        printed: the path.
    """
    _atomic_write_json(results, path)
    print(f"Saved: {path}")


def is_cross_model_json(doc) -> bool:
    """Tell a cross-model results doc from a results file by shape alone.

    Inputs:
        doc: the loaded JSON document.

    Outputs:
        is_cross_model: True for a dict with 'results' (per-target entries) and 'trained_on_task_model'; False
            otherwise, e.g. for a results file (a list of run dicts).
    """
    return isinstance(doc, dict) and 'results' in doc and 'trained_on_task_model' in doc


def load_runs(paths) -> list[dict]:
    """Read the runs of one or more results files, in the order given. A file holds one run (a published results
    file, which names its model 'task_model') or a list of runs.

    Inputs:
        paths: results files.

    Outputs:
        runs: run dicts, each with 'model'.
        printed: how many runs each file held.
        error: exits the program with an error message for a cross-model doc or a file of neither shape.
    """
    runs = []
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        # one run (a dict with 'rd_points') or a list of runs; never a cross-model doc
        if is_cross_model_json(doc) or not (isinstance(doc, list) or (isinstance(doc, dict) and 'rd_points' in doc)):
            sys.exit(f"Error: {path} is not a results file (one run with 'rd_points', or a list of runs)")
        file_runs = [doc] if isinstance(doc, dict) else doc
        # a published results file names the model 'task_model'
        for run in file_runs:
            if 'model' not in run and 'task_model' in run:
                run = {'model': run['task_model'], **{k: v for k, v in run.items() if k != 'task_model'}}
            runs.append(run)
        print(f"Loaded {len(file_runs)} result(s) from {path}")
    return runs


_VARIANT_KEYS = ('lr_scheduler', 'codec_family', 'var_rate', 'lambda_ref')


_MAX_VARIANT_KEYS = 2


def _variant_value(run: dict, key: str):
    """Read one candidate variant key of a run, for telling apart several runs of a model. The training
    parameters come first, since the codec and schedule hyperparameters live there.

    Inputs:
        run: run dict, optionally with 'train_params'.
        key: one of _VARIANT_KEYS.

    Outputs:
        value: run['train_params'][key] when present and not None.
        fallback: run.get(key), which is None when absent.
    """
    params = run.get('train_params') or {}
    if key in params and params[key] is not None:
        return params[key]
    return run.get(key)


def disambiguate_model_collisions(results: list[dict]) -> None:
    """Give every result a 'display_name' that tells apart several runs of the same model, so each gets its
    own legend entry and CSV rows.

    A model with a single run keeps its name. Otherwise the label lists whichever of _VARIANT_KEYS differ
    across that model's runs (up to _MAX_VARIANT_KEYS, in that order), e.g. 'SegKNetSWIN (lr_scheduler=wsd)'.
    'model' itself is never changed: use 'display_name' for labels and 'model' for identity and file names.

    Inputs:
        results: run dicts with 'model'; changed in place.

    Outputs:
        display_name: set on every run.
        fallback: a missing variant value is labelled 'unknown'. When no variant key differs, the label is the
            run's 'run_hash'.
    """
    # group the runs by model
    by_name: dict[str, list[dict]] = defaultdict(list)
    for run in results:
        by_name[run['model']].append(run)

    for name, group in by_name.items():
        # a single run keeps the model's name
        if len(group) <= 1:
            group[0]['display_name'] = name
            continue

        # Keys whose value isn't the same across the whole group, in priority order.
        differing = []
        for key in _VARIANT_KEYS:
            values = {_variant_value(run, key) for run in group}
            if len(values) > 1:
                differing.append(key)
        differing = differing[:_MAX_VARIANT_KEYS]

        # label each run by the keys that differ, e.g. 'Model (lr_scheduler=wsd)'
        if differing:
            for run in group:
                parts = [f'{key}={"unknown" if (val := _variant_value(run, key)) is None else val}'
                         for key in differing]
                run['display_name'] = f"{name} ({', '.join(parts)})"
            continue

        # none of _VARIANT_KEYS differ: label by run hash
        for run in group:
            run['display_name'] = f"{name} ({run.get('run_hash', '')})"


_ANCHOR_DATASET_METRIC = {'coco_val2017': 'mAP', 'ade_val': 'mIoU'}   # same scales as our results


ANCHOR_BPP = ('mean_image', 'aggregate')   # which anchor bpp to read; mean_image is the one our codec's bpp matches


def load_anchor_curves(path, bpp: str = 'mean_image') -> dict:
    """Read the anchors' rate-accuracy curves from the anchor-evaluation JSONL.

    Inputs:
        path: JSONL file, one record per (model, anchor, quality setting) with 'model', 'anchor', 'dataset'
            ('coco_val2017' for mAP, 'ade_val' for mIoU), 'bpp_mean_image' (the mean over the test images of each
            image's bits / original pixels), 'bpp_aggregate' (the total bits / total original pixels of the test set)
            and 'metric' (the task score).
        bpp: which of the two rates to read. 'mean_image' (default) is computed the way our codec's bpp is (the
            mean of per-image bpp); 'aggregate' is what the submitted paper's tables used.

    Outputs:
        curves: {(model, metric): {anchor: [(bpp, score), ...] sorted by bpp}}.
        fallback: blank or unparsable lines are skipped, as are smoke records ('limit' set: a subset, not a
            publishable point) and records with another dataset or without a score or bpp. A missing file
            gives {}: anchors are optional, never a failure.
    """
    if bpp not in ANCHOR_BPP:
        raise ValueError(f'bpp must be one of {ANCHOR_BPP}, got {bpp!r}')
    field = f'bpp_{bpp}'
    path = Path(path)
    out: dict = {}
    if not path.exists():
        return out
    # one JSON record per line; blank or unparsable lines are skipped
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # the task metric follows from the dataset; smoke records and records without a score or rate are
        # skipped
        metric = _ANCHOR_DATASET_METRIC.get(rec.get('dataset'))
        if rec.get('limit') or metric is None or rec.get('metric') is None or rec.get(field) is None:
            continue
        out.setdefault((rec['model'], metric), {}).setdefault(rec['anchor'], []).append(
            (float(rec[field]), float(rec['metric'])))
    # each anchor curve in rate order
    for curves in out.values():
        for anc in curves:
            curves[anc].sort()
    return out


def write_csv(headers: list[str], rows: list[list], path: Path) -> None:
    """Write a CSV through the stdlib writer, so a field containing a comma (e.g. a display_name listing two
    hyperparameters) stays one field.

    Inputs:
        headers: column names.
        rows: one list of values per row.
        path: the .csv file; parent directories are created.

    Outputs:
        file: the CSV at `path`.
        printed: the path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_csv(results: list[dict], path: Path) -> None:
    # one row per RD point, plus one per baseline (empty lambda and bpp). 'metric' names the task score in
    # 'value' (mAP, mIoU, ...); 'model' is the canonical name and 'display_name' the run's label.
    """Write the results as a CSV: one row per RD point, plus one per baseline with empty lambda and bpp.

    Inputs:
        results: run dicts with 'model' and 'rd_points' (each with 'lambda', 'bpp' and 'value') and,
            optionally, 'display_name', 'metric', 'baseline' and 'run_hash'.
        path: the .csv file.

    Outputs:
        file: the CSV at `path`, with columns model, display_name, metric, run_hash, lambda, bpp and value.
        fallback: a missing optional field is written empty; display_name defaults to the model.
    """
    # columns of the CSV
    headers = ['model', 'display_name', 'metric', 'run_hash', 'lambda', 'bpp', 'value']
    rows = []
    # per run: its baseline row (when known), then one row per RD point
    for run in results:
        metric = run.get('metric') or ''
        display_name = run.get('display_name', run['model'])
        if run.get('baseline') is not None:
            rows.append([run['model'], display_name, metric, run.get('run_hash', ''), '', '', run['baseline']])
        for pt in run['rd_points']:
            rows.append([run['model'], display_name, metric, run.get('run_hash', ''), pt['lambda'], pt['bpp'], pt['value']])
    write_csv(headers, rows, path)
