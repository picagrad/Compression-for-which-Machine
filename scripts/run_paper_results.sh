#!/usr/bin/env bash
# Reproduce the paper's results figures and tables from the result files in results/.
#
#     scripts/run_paper_results.sh [OUT] [OPTION ...]
#
# Run from the repository root, with this package installed. OUT is the output folder (default results/paper); any
# further options go to python -m cfwm_repro.results (e.g. --anchor-bpp aggregate: the submitted paper's tables).
# The runs are listed here, in one place; REPRODUCE.md lists the same runs. Set PYTHON to choose the interpreter.
set -euo pipefail

# one file per selected sweep (the codec that represents its task model), in this order: the runs of the mean
# curves and the BD tables
SELECTED=(
    results/selected/SegKNetUPerResNet.json
    results/selected/SegMask2FormerResNet_lambda30.json
    results/selected/SegMaskFormerResNet_lambda30.json
    results/selected/SegUperNetResNet.json
    results/selected/SegMaskFormerSWIN.json
    results/selected/SegMask2FormerSWIN.json
    results/selected/SegKNetSWIN.json
    results/selected/SegUperNetSWIN.json
    results/selected/DINOSWIN-L.json
    results/selected/DDQDETRSWIN-L.json
    results/selected/ATSSSWIN-L.json
    results/selected/GroundingDINO.json
    results/selected/GLIP.json
    results/selected/Mask2FormerSWIN.json
    results/selected/RepPointsV2.json
    results/selected/ATSSResNet.json
    results/selected/RetinaNet.json
    results/selected/FasterRCNN.json
    results/selected/CascadeMaskRCNN.json
    results/selected/DeformableDETR.json
    results/selected/DINOResNet.json
    results/selected/Mask2FormerResNet.json
    results/selected/DDQDETRResNet.json
    results/selected/FCOS.json
    results/selected/MaskRCNN.json
)
# the replaced sweeps, used in the per-model figures and the BD tables only: the unified-recipe runs that a
# retrain replaced, the collapsed ones and the suboptimal ones. Order matters (here and in SELECTED): the BD-rate
# means are summed in this order, so another order changes their last digits.
REPLACED=(
    results/replaced/DINOResNet_lambda5.json
    results/replaced/DDQDETRResNet_lambda5.json
    results/replaced/DeformableDETR_lambda5.json
    results/replaced/SegUperNetResNet_lambda5.json
    results/replaced/GLIP_lambda5.json
    results/replaced/SegMask2FormerResNet_lambda5.json
    results/replaced/SegMaskFormerResNet_lambda5.json
)
# the anchor codecs' (VTM, VVenC, BPG, JPEG) task scores per model and quality setting
ANCHORS=results/anchors/anchor_eval.jsonl
# the cross-model results: <dir>/{cal,nocal}/<teacher codec>.json
CROSS=results/cross_model

OUT=${1:-results/paper}
if [ $# -gt 0 ]; then shift; fi

"${PYTHON:-python}" -m cfwm_repro.results --results "${SELECTED[@]}" --overlay "${REPLACED[@]}" \
    --anchors "$ANCHORS" --cross-collated "$CROSS" --out "$OUT" --name rate_guided_all_2026-10-06 "$@"
