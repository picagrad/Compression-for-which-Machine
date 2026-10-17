"""Figures: per-model rate-accuracy curves, all-model summaries, mean curves per backbone, and the
cross-model BD-rate matrices and scatter. Every figure is skipped, never an error, when matplotlib is
missing."""
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .analysis import _crossing, _group_by_metric, _max_learned_bpp, _model_baselines, _one_run_per_model, _valid_points, anchors_for, cap_anchor_curves, is_collapsed, pooled_mean_curve
from .style import ANCHOR_LABEL, ANCHOR_STYLE, BACKBONE_COLOURS, MEAN_STYLE, QUALITY_FRACTIONS, SUMMARY_FIG_SIZE, SUMMARY_FONT_SCALE, TASK_SHORT, _STYLE_COLORS, _STYLE_LINESTYLES, _STYLE_MARKERS, _TASK_NAME, our_codec_label


def _build_style_map(results: list[dict]) -> dict[str, dict]:
    """Assign one (color, marker, linestyle) per plotted series, so a series looks the same in every figure
    whichever others are skipped.

    Styles go in sorted display_name order, cycling through the tab10 colours, then the markers, then the line
    styles. Keying by display_name rather than model gives several runs of one model distinct styles.

    Inputs:
        results: run dicts with 'model' and, optionally, 'display_name' (default: the model).

    Outputs:
        styles: {display_name: {'color', 'marker', 'linestyle'}}.
    """
    n_colors, n_markers = len(_STYLE_COLORS), len(_STYLE_MARKERS)
    # one style per series name, in sorted order so the assignment does not depend on input order
    names = sorted({run.get('display_name', run['model']) for run in results})
    style_map = {}
    for i, name in enumerate(names):
        # cycle the colours first, then the markers, then the line styles
        style_map[name] = {
            'color': _STYLE_COLORS[i % n_colors],
            'marker': _STYLE_MARKERS[(i // n_colors) % n_markers],
            'linestyle': _STYLE_LINESTYLES[(i // (n_colors * n_markers)) % len(_STYLE_LINESTYLES)],
        }
    return style_map


FONT_SCALE = 1.0          # multiplies every font size the figure helpers set


FIG_SIZE = (10, 7)        # default figure size; the summary figures use SUMMARY_FIG_SIZE


MEAN_RATE_LIMIT = 1.5     # mean curves end at this multiple of the highest learned bpp in the figure


def _fs(size: float) -> float:
    """Scale a font size by FONT_SCALE.

    Inputs:
        size: the font size at scale 1.

    Outputs:
        size: the scaled font size.
    """
    return size * FONT_SCALE


def _fig_setup(title: str) -> tuple:
    """Start a figure of FIG_SIZE with a title and a light grid.

    Inputs:
        title: the axes title.

    Outputs:
        fig: the figure.
        ax: its axes.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title(title, fontsize=_fs(13))
    ax.grid(True, alpha=0.3)
    return fig, ax


def _save_fig(fig, ax, path: Path, xlabel: str, ylabel: str, legend_inside: bool = False) -> None:
    """Label the axes, add the legend, save the figure at 150 dpi and close it.

    Inputs:
        fig: the figure, from _fig_setup.
        ax: its axes.
        path: the output file; parent directories are created.
        xlabel: x-axis label.
        ylabel: y-axis label.
        legend_inside: put the legend in the lower-right corner of the axes, in two columns once it has more
            than 6 entries (summary-size figures) or 10 (others); False puts it outside, to the right.

    Outputs:
        file: the PNG at `path`.
        printed: the path.
    """
    # axis labels and tick sizes, scaled by FONT_SCALE
    ax.set_xlabel(xlabel, fontsize=_fs(12))
    ax.set_ylabel(ylabel, fontsize=_fs(12))
    ax.tick_params(labelsize=_fs(11))
    if legend_inside:   # paper figures: over the (usually empty) lower-right corner, no extra width
        n_entries = len(ax.get_legend_handles_labels()[1])
        two_cols = n_entries > (6 if FIG_SIZE == SUMMARY_FIG_SIZE else 10)   # landscape summaries: two columns sooner
        ax.legend(fontsize=_fs(10 if n_entries <= 10 else 9), ncol=2 if two_cols else 1, loc='lower right',
                  framealpha=0.85, columnspacing=1.0, handlelength=2.2)
    else:
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    # save at 150 dpi and free the figure
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _metric_ylabel(metric: str) -> str:
    """Name a task metric for the y axis: mAP with its IoU range, any other metric as itself.

    Inputs:
        metric: the task metric.

    Outputs:
        label: the axis label.
    """
    return 'mAP  (IoU=0.50:0.95)' if metric == 'mAP' else metric


def _figure_path(base_path: Path, metric: str, all_metrics: set[str]) -> Path:
    """Add the task prefix to a per-task figure's file name: <task>_<kind>_<y-space>[_<fit>], e.g.
    seg_models_ratio.png. The prefix is added even when a folder holds a single task, so file names do not
    depend on the folder.

    Inputs:
        base_path: the path without the prefix.
        metric: the task metric; TASK_SHORT gives the prefix.
        all_metrics: not used.

    Outputs:
        path: the prefixed path, in the same folder.
        fallback: a metric not in TASK_SHORT is used as its own prefix.
    """
    task = TASK_SHORT.get(metric, metric)
    return base_path.with_name(f'{task}_{base_path.stem}{base_path.suffix}')


def _paper_title(metric: str) -> str:
    """Title a figure 'Rate-Accuracy for <task> models'.

    Inputs:
        metric: the task metric; _TASK_NAME names the task.

    Outputs:
        title: the figure title.
        fallback: a metric not in _TASK_NAME is used as the task name.
    """
    return f"Rate-Accuracy for {_TASK_NAME.get(metric, metric)} models"


def _ref_line(ax, y: float, colour: str, text: str, width: float = 1.0) -> None:
    """Draw a dashed horizontal reference line, labelled in place at the right edge (on a white backing) and
    kept out of the legend.

    Inputs:
        ax: the axes.
        y: the line's height, in data units.
        colour: line and text colour.
        text: the short label.
        width: line width.

    Outputs:
        drawn: the line and its label on `ax`.
    """
    ax.axhline(y, color=colour, linewidth=width, linestyle='--', alpha=0.8, label='_nolegend_')
    ax.text(0.995, y, text, transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=_fs(9),
            color=colour, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))


def save_all_models_figure(results: list[dict], path: Path, title: str) -> None:
    """Draw every run's measured curve, in absolute task score, one figure per task metric.

    Inputs:
        results: run dicts with 'model', 'metric', 'rd_points' and, optionally, 'display_name'.
        path: base output path; the task prefix is added (_figure_path).
        title: used only in the skip messages.

    Outputs:
        files: one PNG per metric.
        fallback: a run without valid RD points is skipped with a message; a metric with no curve left writes
            nothing, also with a message.
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    # one figure per task metric; nothing to draw without any metric
    groups = _group_by_metric(results)
    if not groups:
        print(f"  Skipping '{title}': no result has any RD points.")
        return
    style_map = _build_style_map(results)
    for metric, group in groups.items():
        fig, ax = _fig_setup(_paper_title(metric))
        plotted = 0
        # each run's measured curve, skipping runs with no valid points
        for run in group:
            name = run.get('display_name', run['model'])
            pts = _valid_points(run['rd_points'], name, f"'{title}' ({metric})")
            if not pts:
                print(f"  '{title}' ({metric}): skipping {name} (no valid RD points)")
                continue
            ax.plot([pt['bpp'] for pt in pts], [pt['value'] for pt in pts],
                    linewidth=1.5, markersize=6, label=name, **style_map[name])
            plotted += 1
        # a metric with nothing drawn writes no file
        if plotted == 0:
            plt.close(fig)
            print(f"  Skipping '{title}' ({metric}): no curves available.")
            continue
        _save_fig(fig, ax, _figure_path(path, metric, set(groups)),
                  'Bits per pixel (bpp)', _metric_ylabel(metric), legend_inside=True)


def save_relative_figure(results: list[dict], path: Path, title: str, anchors: dict | None = None,
                         mode: str = 'diff', coverage_power: float = 2.0) -> None:
    """Draw every run's measured curve relative to its uncompressed score, with each anchor's mean curve, one
    figure per task metric.

    Mode 'diff' plots value - baseline and mode 'ratio' value / baseline; the ratio figure adds dashed lines
    at QUALITY_FRACTIONS of uncompressed. Each anchor's mean is the pooled fit (pooled_mean_curve, as in the
    by-backbone figure) over this figure's models that have that anchor, up to MEAN_RATE_LIMIT x the highest
    learned bpp. The per-model anchor curves are on the per-model figures.

    Inputs:
        results: run dicts with 'model', 'metric', 'rd_points', 'baseline' and, optionally, 'display_name'.
        path: base output path; the task prefix is added (_figure_path).
        title: used only in the skip messages.
        anchors: {(model, metric): {anchor: [(bpp, value), ...]}}, as from load_anchor_curves; None or empty
            for no anchor curves.
        mode: 'diff' or 'ratio'.
        coverage_power: passed to pooled_mean_curve for the anchor means.

    Outputs:
        files: one PNG per metric.
        fallback: a run missing valid RD points or a baseline is skipped with a message; a metric with no
            curve left writes nothing; an anchor that no model has, or whose fit is not possible, is not
            drawn.
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    # one figure per task metric; nothing to draw without any metric
    groups = _group_by_metric(results)
    if not groups:
        print(f"  Skipping '{title}': no result has any RD points.")
        return
    style_map = _build_style_map(results)
    for metric, group in groups.items():
        fig, ax = _fig_setup(_paper_title(metric))
        skipped = []
        plotted = 0
        # each run's curve relative to its uncompressed score; runs without points or a baseline are skipped
        for run in group:
            baseline = run.get('baseline')
            name = run.get('display_name', run['model'])
            pts = _valid_points(run['rd_points'], name, f"'{title}' ({metric})")
            if not pts or baseline is None:
                skipped.append(name)
                continue
            delta = ([pt['value'] - baseline for pt in pts] if mode == 'diff'
                     else [pt['value'] / baseline for pt in pts])
            ax.plot([pt['bpp'] for pt in pts], delta,
                    linewidth=1.5, markersize=6, label=name, **style_map[name])
            plotted += 1
        if skipped:
            print(f"  '{title}' ({metric}): skipped {skipped} (missing RD points or baseline)")
        if plotted == 0:
            plt.close(fig)
            print(f"  Skipping '{title}' ({metric}): no curves available (all missing "
                 f"RD points or baseline).")
            continue
        # one mean curve per anchor, over this figure's models that have it, in the same relative space
        if anchors:
            top = max(_max_learned_bpp(group).values(), default=None)
            rel = (lambda val, bl: val - bl) if mode == 'diff' else (lambda val, bl: val / bl)
            for anc in anchors_for(metric):
                # each model's anchor curve relative to its baseline; models without the anchor are dropped
                series = {mdl: [(bpp, rel(val, bl)) for bpp, val in (anchors.get((mdl, metric)) or {}).get(anc, [])]
                          for mdl, bl in _model_baselines(group).items()}
                series = {mdl: pts for mdl, pts in series.items() if pts}
                fit = pooled_mean_curve(series, coverage_power=coverage_power) if series else None
                if fit is None:
                    continue
                # end the mean at MEAN_RATE_LIMIT x the highest learned bpp, so the anchors don't stretch the
                # x axis
                keep = fit['grid'] <= MEAN_RATE_LIMIT * top if top else slice(None)
                style = {k: v for k, v in ANCHOR_STYLE[anc].items() if k in ('color', 'linestyle')}
                ax.plot(fit['grid'][keep], fit['mean'][keep], linewidth=2.6, label=f'{ANCHOR_LABEL[anc]}, mean', **style)
        # reference lines: uncompressed, and (ratio only) the QUALITY_FRACTIONS of it
        _ref_line(ax, 0 if mode == 'diff' else 1, 'gray', 'Uncomp.', width=1.2)
        if mode == 'ratio':
            for frac, colour in QUALITY_FRACTIONS:
                _ref_line(ax, frac, colour, f'{frac:.0%}')
        ylabel = (f'Δ{metric} vs. uncompressed' if mode == 'diff'
                  else f'{metric} / {metric} uncompressed')
        _save_fig(fig, ax, _figure_path(path, metric, set(groups)),
                  'Bits per pixel (bpp)', ylabel, legend_inside=True)


def _safe_filename(name: str) -> str:
    """Make a model name safe as a file name: every run of characters other than word characters and . = + -
    becomes '_', and leading or trailing '_' are removed, so a separator cannot turn the name into a path.

    Inputs:
        name: the model name.

    Outputs:
        name: the safe name.
        fallback: 'model' when nothing is left.
    """
    return re.sub(r'[^\w.=+-]+', '_', name).strip('_') or 'model'


def save_per_model_figures(results: list[dict], out_dir: Path, experiment: str,
                           anchors: dict | None = None) -> None:
    """Draw one figure per model, overlaying all of its runs and its anchor curves, in absolute task score.

    Our curve is named by its codec and '(ours)' (our_codec_label), with a run suffix when the model has
    several runs. A collapsed run (is_collapsed) gets a circle and a 'collapsed' callout. Dashed reference
    lines mark the uncompressed score and QUALITY_FRACTIONS of it. Series styles come from _build_style_map,
    so a run looks the same in every figure.

    Inputs:
        results: run dicts with 'model', 'metric', 'rd_points', 'baseline' and, optionally, 'display_name' and
            'train_params'.
        out_dir: the output folder.
        experiment: not used.
        anchors: {(model, metric): {anchor: [(bpp, value), ...]}}, as from load_anchor_curves; the model's
            anchors for its task (anchors_for) are drawn in grey up to ANCHOR_RATE_CAP x its highest learned
            bpp (cap_anchor_curves). None for no anchors.

    Outputs:
        files: <out_dir>/<model>_rd.png per model.
        fallback: a run without valid RD points or a metric is skipped with a message, and a model with no run
            left writes nothing. A run whose codec cannot be named keeps its display_name. The task metric is
            taken from the model's first drawn run. When the runs disagree on the baseline, the lowest is
            drawn, with a warning; without a baseline there are no reference lines.
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    # group the runs by model: one figure per model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for run in results:
        by_model[run['model']].append(run)

    style_map = _build_style_map(results)
    for model, group in sorted(by_model.items()):
        # the model's runs that have valid points and a metric
        drawable = []
        for run in group:
            display_name = run.get('display_name', run['model'])
            pts = _valid_points(run['rd_points'], display_name,
                                f"{model} per-model figure")
            if not pts or run.get('metric') is None:
                print(f"  {model} per-model figure: skipping {display_name} "
                      f"(no valid RD points).")
                continue
            drawable.append((run, display_name, pts))

        if not drawable:
            print(f"  Skipping per-model figure for {model}: no valid RD points.")
            continue

        metric = drawable[0][0]['metric']
        # "Rate-Accuracy for <model>": our curve named by its codec + "(ours)" (with a run suffix only when
        # the model has several runs), anchors by codec name, and dashed reference lines at QUALITY_FRACTIONS
        # of uncompressed.
        fig, ax = _fig_setup(f"Rate-Accuracy for {model}")
        for run, display_name, pts in drawable:
            # legend label: our codec's name with '(ours)', plus a run suffix only when the model has several
            # runs
            codec = our_codec_label(run)
            suffix = display_name[len(model):].strip().strip('()') if display_name.startswith(model) else ''
            label = (display_name if codec is None else
                     f'{codec} (ours{", " + suffix if len(drawable) > 1 and suffix else ""})')
            ax.plot([pt['bpp'] for pt in pts], [pt['value'] for pt in pts],
                    linewidth=1.5, markersize=6, label=label,
                    **style_map[display_name])
            if is_collapsed(run, pts):
                # A collapsed codec collapses in rate too (~0.005 bpp, ~0 metric), so its points pile up in
                # the corner and are easy to miss: circle the cluster and point a 'collapsed' callout at it.
                xs, ys = [pt['bpp'] for pt in pts], [pt['value'] for pt in pts]
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                ax.plot([cx], [cy], linestyle='none', marker='o', markersize=20, markerfacecolor='none',
                        markeredgecolor='tab:red', markeredgewidth=1.5, zorder=10, label='_nolegend_')
                ax.annotate('collapsed', xy=(cx, cy), xytext=(0.16, 0.16), textcoords='axes fraction',
                            fontsize=11, color='tab:red', zorder=11,
                            arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.2, shrinkB=10),
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        # the model's anchor curves for its task, cut at ANCHOR_RATE_CAP x its highest learned bpp, in the
        # order of anchors_for
        capped = cap_anchor_curves(anchors or {}, {model: max(pt['bpp'] for _, _, pts in drawable
                                                              for pt in pts)})
        shown = anchors_for(metric)
        for anc, anc_pts in sorted(((anc, anc_pts) for anc, anc_pts in (capped.get((model, metric)) or {}).items() if anc in shown),
                               key=lambda item: shown.index(item[0])):
            ax.plot([pt[0] for pt in anc_pts], [pt[1] for pt in anc_pts], linewidth=1.3, markersize=6,
                    label=ANCHOR_LABEL.get(anc, anc), **ANCHOR_STYLE.get(anc, {}))
        # the uncompressed score; runs of one model should agree, so a disagreement is only warned about
        baselines = {run['baseline'] for run, _, _ in drawable if run.get('baseline') is not None}
        if len(baselines) > 1:
            print(f"  Warning: {model} has disagreeing baselines {sorted(baselines)} — "
                  f"drawing the first; the uncompressed score shouldn't depend on the codec.")
        if baselines:
            # reference lines are labelled in place (short text at the right edge), not in the legend
            bl = sorted(baselines)[0]
            _ref_line(ax, bl, 'gray', 'Uncomp.', width=1.2)
            for frac, colour in QUALITY_FRACTIONS:
                _ref_line(ax, frac * bl, colour, f'{frac:.0%}')

        _save_fig(fig, ax, out_dir / f"{_safe_filename(model)}_rd.png",
                  'Bits per pixel (bpp)', _metric_ylabel(metric), legend_inside=True)


def _save_mean_figure(results: list[dict], groups: dict, path: Path, anchors: dict,
                                 anchor_list: tuple | None = None, report_anchor: str = 'vtm_qpa1',
                                 coverage_power: float = 2.0, xmax: dict | None = None,
                                 ylim: tuple = (0.65, 1.02), individual: bool = True,
                                 overall: bool = False, overall_colour: str = '#5E9C8F',
                                 overall_alpha: float = 0.6, colours: tuple | None = None) -> dict:
    """Draw ratio-space mean curves per backbone group, all groups on one figure per task: colour = backbone,
    line style = codec.

    Each group's curves are pooled_mean_curve fits over its models, one run per model (_one_run_per_model),
    for our codec and each anchor in `anchor_list`.

    Inputs:
        results: run dicts with 'model', 'metric', 'rd_points' and 'baseline'.
        groups: {group label: [model names]}, in drawing order.
        path: base output path; the task prefix is added (_figure_path).
        anchors: {(model, metric): {anchor: [(bpp, value), ...]}}, as from load_anchor_curves.
        anchor_list: the anchors to draw; default the task's anchors (anchors_for).
        report_anchor: the anchor whose 90/95% crossings are returned, and whose all-backbones mean is drawn
            with `overall`.
        coverage_power: passed to pooled_mean_curve.
        xmax: {metric: x-axis limit}; a metric not listed ends at MEAN_RATE_LIMIT x the highest learned bpp of
            the drawn groups.
        ylim: y-axis limits.
        individual: also draw each of our models' measured curve, thin, in its group colour.
        overall: also draw the all-backbones mean of ours and `report_anchor`, over every model in `groups`.
        overall_colour: colour of the all-backbones curves.
        overall_alpha: their opacity.
        colours: one colour per group, in order; default BACKBONE_COLOURS.

    Outputs:
        files: one PNG per metric.
        crossings: {(metric, group label): {'n': models, 'ours_90', 'ours_95', 'anchor_90', 'anchor_95'}}, the
            bpp at which the mean reaches 90% and 95% of uncompressed, for ours and for `report_anchor`.
        fallback: a crossing is NaN when the mean never reaches the level, and absent when that curve was not
            drawn. A group with fewer than 2 usable models, or beyond the last colour, is not drawn; an anchor
            is drawn for a group only when all its models have it; a metric with no group drawn writes no
            figure. {} without matplotlib.
    """
    if not MATPLOTLIB_AVAILABLE:
        return {}
    summary = {}
    for metric, group in _group_by_metric(results).items():
        fig, ax = _fig_setup(_paper_title(metric) + ' (mean, cubic fit)')   # per-backbone is in the legend
        # the codecs drawn: ours, then the anchors
        codecs = ['ours'] + list(anchor_list or anchors_for(metric))
        top, drawn, seen = 0.0, [], set()
        # one backbone group at a time, each in its own colour
        for (label, members), colour in zip(groups.items(), colours or BACKBONE_COLOURS):
            # one run per model of this group; a mean needs at least 2 models
            runs = _one_run_per_model([run for run in group if run['model'] in members], metric)
            if len(runs) < 2:
                continue
            # the highest learned bpp over the drawn groups sets the default x limit
            top = max(top, max(max(pt['bpp'] for pt in pts) for _, pts in runs))
            # each codec's per-model curves as a fraction of the model's uncompressed score; an anchor is kept
            # only when every model of the group has it
            series = {'ours': {run['model']: [(pt['bpp'], pt['value'] / run['baseline']) for pt in pts] for run, pts in runs}}
            for anc in codecs[1:]:
                fam = {run['model']: [(bpp, val / run['baseline']) for bpp, val in (anchors.get((run['model'], metric)) or {}).get(anc, [])]
                       for run, _ in runs}
                if all(fam.values()):
                    series[anc] = fam
            # our models' measured curves, thin, behind the means
            if individual:
                for pts in series['ours'].values():
                    ax.plot(*zip(*sorted(pts)), color=colour, alpha=0.47, linewidth=1.0, label='_nolegend_')
            # the pooled mean of each codec, and for ours and the reported anchor the bpp where it reaches
            # 90% and 95% of uncompressed
            row = {'n': len(runs)}
            for codec in codecs:
                if codec not in series:
                    continue
                fit = pooled_mean_curve(series[codec], coverage_power=coverage_power)
                if fit is None:
                    continue
                style = MEAN_STYLE.get(codec, dict(linestyle='--', linewidth=1.8))
                # full opacity: how many models stand behind each stretch is shown by the thin model curves
                ax.plot(fit['grid'], fit['mean'], color=colour, label='_nolegend_', **style)
                seen.add(codec)
                key = 'ours' if codec == 'ours' else ('anchor' if codec == report_anchor else None)
                if key:
                    for level in (0.90, 0.95):
                        row[f'{key}_{int(level * 100)}'] = _crossing(fit['grid'], fit['mean'], level)
            summary[(metric, label)] = row
            drawn.append((label, len(runs), colour))
        # a task with no group drawn writes no figure
        if not drawn:
            plt.close(fig)
            continue
        if overall:   # the all-backbones mean, for ours and the reported anchor
            # every model of every group, pooled into one mean for ours and the reported anchor
            members = {mdl for mdls in groups.values() for mdl in mdls}
            runs = _one_run_per_model([run for run in group if run['model'] in members], metric)
            series = {'ours': {run['model']: [(pt['bpp'], pt['value'] / run['baseline']) for pt in pts] for run, pts in runs}}
            fam = {run['model']: [(bpp, val / run['baseline']) for bpp, val in (anchors.get((run['model'], metric)) or {}).get(report_anchor, [])]
                   for run, _ in runs}
            if all(fam.values()):
                series[report_anchor] = fam
            for codec, curves in series.items():
                fit = pooled_mean_curve(curves, coverage_power=coverage_power)
                if fit is not None:
                    style = MEAN_STYLE.get(codec, dict(linestyle='--', linewidth=1.8))
                    ax.plot(fit['grid'], fit['mean'], color=overall_colour, alpha=overall_alpha, label='_nolegend_',
                            zorder=5, **style)
            drawn.append(('All backbones', len(runs), overall_colour))
        # reference lines, axis limits, then the legend
        _ref_line(ax, 1, 'gray', 'Uncomp.', width=1.2)
        for frac, colour in QUALITY_FRACTIONS:
            _ref_line(ax, frac, colour, f'{int(frac * 100)}%', width=1.0)
        ax.set_xlim(0, (xmax or {}).get(metric, MEAN_RATE_LIMIT * top))
        ax.set_ylim(*ylim)
        # two-part legend: colour = backbone, line style = codec
        for label, n_models, colour in drawn:
            name = 'All backbones' if label == 'All backbones' else f'{label} backbone'
            ax.plot([], [], color=colour, linewidth=2.6, label=f'{name} \u00d7{n_models}',
                    alpha=overall_alpha if (overall and label == 'All backbones') else 1.0)
        for codec in codecs:
            if codec in seen:
                ax.add_line(Line2D([], [], color='0.3', label='ELFC-GQVRF (ours)' if codec == 'ours' else ANCHOR_LABEL[codec],
                                   **MEAN_STYLE.get(codec, {})))
        _save_fig(fig, ax, _figure_path(path, metric, set()), 'Bits per pixel (bpp)',
                  f'{metric} / {metric} uncompressed', legend_inside=True)
    return summary


def save_mean_figure(*args, **kwargs):
    """Run _save_mean_figure at SUMMARY_FONT_SCALE and SUMMARY_FIG_SIZE.

    Inputs:
        args, kwargs: the arguments of _save_mean_figure.

    Outputs:
        files, crossings: as _save_mean_figure.
        state: FONT_SCALE and FIG_SIZE are restored afterwards, also on an error.
    """
    # switch the module's font scale and figure size to the summary ones, and restore them even on error
    global FONT_SCALE, FIG_SIZE
    old_scale, FONT_SCALE = FONT_SCALE, SUMMARY_FONT_SCALE
    old_size, FIG_SIZE = FIG_SIZE, SUMMARY_FIG_SIZE
    try:
        return _save_mean_figure(*args, **kwargs)
    finally:
        FONT_SCALE, FIG_SIZE = old_scale, old_size


def save_cross_matrix_figure(mats: dict, models: list[str], path: Path, title: str, anchor_label: str,
                             clip: float = 100.0, targets: list[str] | None = None) -> None:
    """Draw two cross-model BD-rate matrices side by side: vs the target's own (matched) codec, and vs the
    anchor on the target.

    Rows are targets (tested on) and columns teachers (codec trained on). Colours diverge around 0 (blue =
    fewer bits) and are clipped at +-clip; values beyond +-999 print as '>+999' or '<-999'. A model's own cell
    is outlined and its value bold.

    Inputs:
        mats: output of cross_model_bd_matrix; columns may have been removed from it.
        models: the teachers, as in `mats`.
        path: the output file; parent directories are created.
        title: the figure title.
        anchor_label: the anchor's display name, for the panel title.
        clip: colour-scale limit, in %.
        targets: the targets, as in `mats`; default `models`.

    Outputs:
        file: the PNG at `path`.
        printed: the path.
        fallback: a cell without a BD-rate shows the pair's peak score (% of uncompressed) on grey, or is
            blank when the pair has no data.
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    # display each matrix transposed: targets as rows, teachers as columns
    targets = list(targets or models)
    rows, cols = targets, models                      # displayed: targets down, teachers across
    n_rows, n_cols = len(rows), len(cols)
    # figure size grows with the number of models; cell is in inches
    cell = 0.62
    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, 3.2 + 2 * cell * n_cols + 1.0), 3.6 + cell * n_rows), constrained_layout=True)
    panels = (('vs_matched', "vs. the target's own codec"), ('vs_anchor', f'vs. {anchor_label} on the target'))
    # one panel per comparison: vs the target's own codec, and vs the anchor
    for ax, (key, sub) in zip(axes, panels):
        bd, peak = mats[key].T, mats['peak'].T          # stored as [teacher, target]
        im = ax.imshow(np.clip(bd, -clip, clip), cmap='RdBu_r', vmin=-clip, vmax=clip)
        # label each cell: the BD-rate when there is one, else the pair's peak score on grey
        for i in range(n_rows):
            for j in range(n_cols):
                val, pk_val = bd[i, j], peak[i, j]
                if val == val:
                    txt = ('0' if abs(val) < 0.5 else f'{val:+.0f}') if abs(val) < 1000 else ('>+999' if val > 0 else '<-999')
                    ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                            color='white' if abs(val) > 0.6 * clip else 'black',
                            fontweight='bold' if rows[i] == cols[j] else 'normal')
                elif pk_val == pk_val:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color='0.85'))
                    ax.text(j, i, f'pk {100 * pk_val:.0f}%', ha='center', va='center', fontsize=7.5, color='0.25')
        # outline each model's own cell (its codec on itself)
        for j, mdl in enumerate(cols):
            if mdl in rows:
                i = rows.index(mdl)
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor='0.2', linewidth=1.2))
        # ticks and labels; only the left panel names the rows
        ax.set_xticks(range(n_cols), cols, rotation=40, ha='right', fontsize=9)
        ax.set_yticks(range(n_rows), rows if ax is axes[0] else [''] * n_rows, fontsize=9)
        ax.set_xlabel('codec trained on (teacher)', fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel('tested on (target)', fontsize=10)
        ax.set_title(f'BD-rate (%)\n{sub}', fontsize=11)
    # one shared colour bar, title and save
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label(f'BD-rate (%), clipped at ±{clip:.0f}', fontsize=9)
    fig.suptitle(title, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def save_cross_scatter_figure(entries: list[dict], path: Path, anchor_label: str, ymax: float = 100.0,
                              show_matched: bool = False, title: str = 'Cross-model compatibility',
                              xlabel: str | None = None, ylabel: str | None = None, size_note: bool | str = True,
                              text_scale: float = 1.0, outlier_name: str = 'outlier', figsize: tuple = (7.6, 5.6),
                              ylim: tuple | None = None, xlim: tuple | None = None, jitter: float = 0.0) -> None:
    """Draw the cross-model summary as a scatter: x = BD-rate vs the anchor on the target, y = BD-rate vs the
    target's own codec.

    Each group is one filled point at its cross-model mean, its area growing with the spread (RMS of the two
    axes' s.d., capped at SD_CAP). Outlier entries, one model in one role, are grey triangles (up: as teacher;
    down: as target) with a label, joined when both of the model's roles are drawn.

    Inputs:
        entries: dicts with 'label' and 'stats' (from summarize_cross_group, or
            summarize_cross_roles for an outlier). Optional 'note': text attached to the point;
            default 'k/n pairs work' when some pairs have no value. Optional 'pairs': [(x, y, collapsed,
            role), ...] drawn behind the group's mean, role being 'source' or 'target' for a highlighted
            model's pair, or left out. An outlier entry adds 'outlier': True, 'role' ('source' or 'target';
            default 'target'), 'name' (its label text; default 'label') and 'offset' (label offset, in
            points).
        path: the output file; parent directories are created.
        anchor_label: the anchor's display name, for the default x label and the notes.
        ymax: values above it are drawn at it, an outlier's noted as off scale.
        show_matched: also draw each group's own codecs at (own vs anchor, 0), with an arrow to the mean.
        title: the figure title.
        xlabel: x-axis label; default names the anchor.
        ylabel: y-axis label; default names the target's own codec.
        size_note: True for the default note on what the marker area means, a string for that text, False for
            none.
        text_scale: multiplies the text and pair-marker sizes.
        outlier_name: the outliers' name in the legend.
        figsize: figure size, in inches.
        ylim: y-axis limits; default from just below the bottom strip to ymax + 10.
        xlim: x-axis limits; default from -100 to just past the largest positive mean, or to 0.
        jitter: half-width, in BD-rate %, of a random x offset for collapsed pairs (fixed seed); 0 for none.

    Outputs:
        file: the PNG at `path`.
        printed: the path.
        fallback: a group without a cross mean vs the anchor is in the legend but not drawn, and one without a
            y value is listed in a note. An outlier with no own codec sits in a strip at the bottom, and one
            whose pairs all collapse is listed in a note. A pair without a y value is not drawn.
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    # one colour per group; marker area grows with the spread, capped at SD_CAP
    colours = ['#1A6599', '#D86B0B', '#258825', '#7B5BA6', '#8C564B', '#C2185B', '#5D6D7E', '#B7950B']
    SD_CAP = 30.0
    area = lambda spread: 20 + 4 * min(spread if spread == spread else 0, SD_CAP) ** 1.5   # area grows with the s.d., capped
    fig, ax = plt.subplots(figsize=figsize)
    # y of the bottom strip for outliers with no own codec (no y value)
    strip_y = -55
    notes, handles = [], []
    colour_idx = 0
    role_marker = {'source': '^', 'target': 'v'}
    rng = np.random.default_rng(0)                 # fixed seed: the same jitter on every run
    placed = {}                                  # outlier name -> {role: (x, y)}, to join a model's two roles
    collapsed = {}
    # one entry at a time: either an outlier (one model in one role) or a group mean
    for entry in entries:
        # each statistic is (mean, s.d., values, pairs); x is vs the anchor, y vs the target's own codec
        stats = entry['stats']
        own_x, own_sd, _, own_n = stats['matched_vs_anchor']
        x_mean, x_sd, x_ok, x_n = stats['cross_vs_anchor']
        y_mean, y_sd, y_ok, y_n = stats['cross_vs_matched']
        # an outlier: noted when all its pairs collapse, else a grey triangle with a label
        if entry.get('outlier'):
            name, role = entry.get('name', entry['label']), entry.get('role', 'target')
            if x_mean != x_mean and x_n:
                collapsed.setdefault(name, []).append((role, x_n))
                continue
            if x_mean != x_mean:
                continue
            # no own codec: draw it in the bottom strip; values above ymax are drawn at ymax
            y = y_mean if y_n else strip_y
            y_drawn = min(y, ymax)
            outlier_spread = math.sqrt(((x_sd if x_sd == x_sd else 0) ** 2 + (y_sd if y_sd == y_sd else 0) ** 2) / 2)
            ax.scatter(x_mean, y_drawn, s=min(max(area(outlier_spread), 45), area(15)), marker=role_marker.get(role, 'D'), color='0.25',
                       edgecolor='white', linewidth=0.6, zorder=4)
            placed.setdefault(name, {})[role] = (x_mean, y_drawn)
            # the label: its name, how many pairs work, and its value when off scale
            tag = (name + (f' ({x_ok}/{x_n} pairs work)' if x_n > 1 and x_ok < x_n else '')
                   + (f'\ny = {y_mean:+.0f}, off scale' if y_n and y_mean > ymax else ''))   # s.d. is shown by the marker size
            right = x_mean > -45
            offset = entry.get('offset') or ((-12, 0) if right else (9, 3))
            ax.annotate(tag, (x_mean, y_drawn), xytext=offset, textcoords='offset points', fontsize=text_scale * 7, color='0.15',
                        ha='right' if offset[0] < 0 else 'left', va='center')
            continue
        # a group: its colour, then its pairs (if given) behind its mean
        colour = colours[colour_idx % len(colours)]
        colour_idx += 1
        # every cross pair behind the mean: working pairs as faint dots, collapsed ones (served by the anchor:
        # x = 0, y = the anchor vs the target's own codec) as hollow crosses
        for pair_x, pair_y, fell, *rest in entry.get('pairs', ()):
            if pair_y != pair_y:
                continue
            if fell and jitter:   # spread the stacked x = 0 collapses a little (seeded above)
                pair_x = pair_x + rng.uniform(-jitter, jitter)
            role = rest[0] if rest else None
            if role:   # a highlighted model's pair: its own triangle (up = as teacher, down = as target), drawn on top
                marker = role_marker.get(role, 'D')
                if fell:
                    ax.scatter(pair_x, pair_y, s=50 * text_scale, marker=marker, facecolor='none', edgecolor='0.25', linewidth=1.3,
                               alpha=0.95, zorder=3.5)
                else:
                    ax.scatter(pair_x, pair_y, s=34 * text_scale, marker=marker, color='0.25', alpha=0.75, linewidth=0, zorder=3.5)
            elif fell:
                ax.scatter(pair_x, pair_y, s=38 * text_scale, marker='X', facecolor='none', edgecolor=colour, linewidth=1.0,
                           alpha=0.85, zorder=2)
            else:
                ax.scatter(pair_x, pair_y, s=16 * text_scale, color=colour, alpha=0.35, linewidth=0, zorder=2)
        # a group mean without a y value is listed in a note instead of drawn
        if x_n and x_mean == x_mean and y_mean != y_mean:      # no value vs its own codecs: cannot be placed on y
            notes.append(f"{entry['label']}: vs {anchor_label} {x_mean:+.0f}, no overlap with its own codecs")
        if own_n and show_matched:   # own codecs at y = 0 (off by default)
            ax.scatter(own_x, 0, s=area(own_sd), facecolor='white', edgecolor=colour, linewidth=1.6, zorder=3)
        # the group mean, sized by its spread, with a note when some pairs have no value
        if x_n and x_mean == x_mean and y_mean == y_mean:
            spread = math.sqrt(((x_sd if x_sd == x_sd else 0) ** 2 + (y_sd if y_sd == y_sd else 0) ** 2) / 2)
            ax.scatter(x_mean, min(y_mean, ymax), s=area(spread), color=colour, edgecolor='white', linewidth=0.6, zorder=4)
            if own_n and show_matched:
                ax.annotate('', (x_mean, min(y_mean, ymax)), (own_x, 0),
                            arrowprops=dict(arrowstyle='->', color=colour, lw=1.0, shrinkA=6, shrinkB=6, alpha=0.8))
            if entry.get('note') or x_ok < x_n:
                ax.annotate(entry.get('note') or f'{x_ok}/{x_n} pairs work', (x_mean, min(y_mean, ymax)), xytext=(10, 9),
                            textcoords='offset points', fontsize=text_scale * 7, color=colour, ha='left',
                            arrowprops=dict(arrowstyle='-', color=colour, lw=0.5))
        handles.append(plt.Line2D([], [], marker='o', linestyle='', color=colour, label=entry['label'], markersize=7))
    # join each outlier's two roles with a dotted line; list fully collapsed outliers in the note
    for name, roles in placed.items():           # the same model in its two roles
        if len(roles) == 2:
            (x1, y1), (x2, y2) = roles['source'], roles['target']
            ax.plot([x1, x2], [y1, y2], color='0.45', lw=0.8, ls=':', zorder=2)
    for name, role_counts in collapsed.items():
        notes.append(f"{name}: all pairs collapse " + ' and '.join(
            f"{'as teacher' if role == 'source' else 'as target'} (0/{n_pairs})" for role, n_pairs in role_counts))
    # axes: y = 0 line, the bottom strip when used, then the limits
    ax.axhline(0, color='0.4', linewidth=0.8)
    if any(entry.get('outlier') and entry['stats']['cross_vs_matched'][3] == 0 and entry['stats']['cross_vs_anchor'][3]
           for entry in entries):                   # only when something sits in the no-own-codec strip
        ax.axhline(strip_y + 7, color='0.75', linewidth=0.6, linestyle=':')
        ax.text(-99, strip_y + 1.5, 'no own codec (y n/a)', fontsize=text_scale * 6.5, color='0.45', va='bottom')
    xs_all = [entry['stats']['cross_vs_anchor'][0] for entry in entries if entry['stats']['cross_vs_anchor'][0] == entry['stats']['cross_vs_anchor'][0]]
    ax.set_xlim(-100, 10 * math.ceil((max(xs_all) + 12) / 10) if xs_all and max(xs_all) > 0 else 0)
    if xlim:
        ax.set_xlim(*xlim)
    if ax.get_xlim()[1] > 0:
        ax.axvline(0, color='0.4', linewidth=0.8)
    ax.set_ylim(*(ylim or (strip_y - 7, ymax + 10)))
    ax.set_xlabel(xlabel or f'BD-rate (%) vs. {anchor_label} on the target  (left = fewer bits)', fontsize=text_scale * 9.5)
    ax.set_ylabel(ylabel or "BD-rate (%) vs. the target's own codec", fontsize=text_scale * 9.5)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=text_scale * 10)
    # legend: groups, then pair markers and outlier roles when drawn
    if any(entry.get('pairs') for entry in entries):
        handles += [plt.Line2D([], [], marker='o', linestyle='', color='0.4', alpha=0.4, markersize=4,
                               label='single pair'),
                    plt.Line2D([], [], marker='X', linestyle='', markerfacecolor='none', markeredgecolor='0.3',
                               markersize=6, label='Collapsed - VTM fallback')]
    handles += ([plt.Line2D([], [], marker='o', linestyle='', markerfacecolor='white', markeredgecolor='0.3',
                            label='own codec (matched)', markersize=7)] if show_matched else []) + ([
                plt.Line2D([], [], marker='^', linestyle='', color='0.25', label=f'{outlier_name} as TEACHER', markersize=6),
                plt.Line2D([], [], marker='v', linestyle='', color='0.25', label=f'{outlier_name} as TARGET', markersize=6)]
                if any(entry.get('outlier') for entry in entries) else [])
    ax.legend(handles=handles, fontsize=text_scale * 6.5, loc='upper left', frameon=True, markerscale=0.8,
              title='cross-model mean', title_fontsize=text_scale * 6.5, borderpad=0.4, labelspacing=0.3, handletextpad=0.3)
    # notes box, marker-area note, title and save
    if notes:
        ax.text(0.99, 0.66, '\n'.join(notes), transform=ax.transAxes, ha='right', va='top', fontsize=text_scale * 7,
                color='0.15', bbox=dict(boxstyle='round', facecolor='0.95', edgecolor='0.7'))
    if size_note:
        ax.text(1.0, -0.115, size_note if isinstance(size_note, str) else
                f'marker area ∝ spread (s.d.) across pairs, capped at s.d. {SD_CAP:.0f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=text_scale * 6.5, color='0.4')
    ax.set_title(title, fontsize=text_scale * 11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
