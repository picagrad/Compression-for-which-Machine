# Commands for every result file

<!-- generated from the private run table; do not edit by hand -->

Every command behind the published results, with the settings of each run. The methods are explained in [REPRODUCE.md](REPRODUCE.md), which also defines the shell variables (`$DATA_COCO`, `$DATA_ADE`, `$WEIGHTS`, `$CKPT`).

## The selected sweeps

25 sweeps, one per task model, of the codec selected to represent it, in `results/selected/`. Each command writes `results/mine/<task_model>__<weights file stem>.json`; compare it with `results/selected/<weights file stem>.json`. The published file keeps only the fields listed in REPRODUCE.md, part 1; yours also has the run's settings and the reason the sweep chose each point.

| model | weights file | lambda_ref | metric | published points |
|---|---|---|---|---|
| `SegKNetUPerResNet` | `SegKNetUPerResNet.ckpt` | 5.0 | mIoU | 14 |
| `SegMask2FormerResNet` | `SegMask2FormerResNet_lambda30.ckpt` | 30.0 | mIoU | 14 |
| `SegMaskFormerResNet` | `SegMaskFormerResNet_lambda30.ckpt` | 30.0 | mIoU | 14 |
| `SegUperNetResNet` | `SegUperNetResNet.ckpt` | 50.0 | mIoU | 13 |
| `SegMaskFormerSWIN` | `SegMaskFormerSWIN.ckpt` | 5.0 | mIoU | 13 |
| `SegMask2FormerSWIN` | `SegMask2FormerSWIN.ckpt` | 5.0 | mIoU | 12 |
| `SegKNetSWIN` | `SegKNetSWIN.ckpt` | 5.0 | mIoU | 10 |
| `SegUperNetSWIN` | `SegUperNetSWIN.ckpt` | 5.0 | mIoU | 11 |
| `DINOSWIN-L` | `DINOSWIN-L.ckpt` | 5.0 | mAP | 11 |
| `DDQDETRSWIN-L` | `DDQDETRSWIN-L.ckpt` | 50.0 | mAP | 14 |
| `ATSSSWIN-L` | `ATSSSWIN-L.ckpt` | 50.0 | mAP | 14 |
| `GroundingDINO` | `GroundingDINO.ckpt` | 5.0 | mAP | 14 |
| `GLIP` | `GLIP.ckpt` | 10.0 | mAP | 14 |
| `Mask2FormerSWIN` | `Mask2FormerSWIN.ckpt` | 5.0 | mAP | 13 |
| `RepPointsV2` | `RepPointsV2.ckpt` | 5.0 | mAP | 14 |
| `ATSSResNet` | `ATSSResNet.ckpt` | 5.0 | mAP | 12 |
| `RetinaNet` | `RetinaNet.ckpt` | 5.0 | mAP | 14 |
| `FasterRCNN` | `FasterRCNN.ckpt` | 5.0 | mAP | 14 |
| `CascadeMaskRCNN` | `CascadeMaskRCNN.ckpt` | 5.0 | mAP | 14 |
| `DeformableDETR` | `DeformableDETR.ckpt` | 30.0 | mAP | 13 |
| `DINOResNet` | `DINOResNet.ckpt` | 30.0 | mAP | 13 |
| `Mask2FormerResNet` | `Mask2FormerResNet.ckpt` | 5.0 | mAP | 12 |
| `DDQDETRResNet` | `DDQDETRResNet.ckpt` | 50.0 | mAP | 14 |
| `FCOS` | `FCOS.ckpt` | 5.0 | mAP | 14 |
| `MaskRCNN` | `MaskRCNN.ckpt` | 5.0 | mAP | 12 |

<details>
<summary>The 25 commands</summary>

`SegKNetUPerResNet` (`SegKNetUPerResNet.ckpt`): compare `results/mine/SegKNetUPerResNet__SegKNetUPerResNet.json` with `results/selected/SegKNetUPerResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model SegKNetUPerResNet --data_root ${DATA_ADE} --checkpoint ${WEIGHTS}/SegKNetUPerResNet.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`SegMask2FormerResNet` (`SegMask2FormerResNet_lambda30.ckpt`): compare `results/mine/SegMask2FormerResNet__SegMask2FormerResNet_lambda30.json` with `results/selected/SegMask2FormerResNet_lambda30.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model SegMask2FormerResNet --data_root ${DATA_ADE} --checkpoint ${WEIGHTS}/SegMask2FormerResNet_lambda30.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`SegMaskFormerResNet` (`SegMaskFormerResNet_lambda30.ckpt`): compare `results/mine/SegMaskFormerResNet__SegMaskFormerResNet_lambda30.json` with `results/selected/SegMaskFormerResNet_lambda30.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model SegMaskFormerResNet --data_root ${DATA_ADE} --checkpoint ${WEIGHTS}/SegMaskFormerResNet_lambda30.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`SegUperNetResNet` (`SegUperNetResNet.ckpt`): compare `results/mine/SegUperNetResNet__SegUperNetResNet.json` with `results/selected/SegUperNetResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model SegUperNetResNet --data_root ${DATA_ADE} --checkpoint ${WEIGHTS}/SegUperNetResNet.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`SegMaskFormerSWIN` (`SegMaskFormerSWIN.ckpt`): compare `results/mine/SegMaskFormerSWIN__SegMaskFormerSWIN.json` with `results/selected/SegMaskFormerSWIN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model SegMaskFormerSWIN --data_root ${DATA_ADE} --checkpoint ${WEIGHTS}/SegMaskFormerSWIN.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`SegMask2FormerSWIN` (`SegMask2FormerSWIN.ckpt`): compare `results/mine/SegMask2FormerSWIN__SegMask2FormerSWIN.json` with `results/selected/SegMask2FormerSWIN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model SegMask2FormerSWIN --data_root ${DATA_ADE} --checkpoint ${WEIGHTS}/SegMask2FormerSWIN.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`SegKNetSWIN` (`SegKNetSWIN.ckpt`): compare `results/mine/SegKNetSWIN__SegKNetSWIN.json` with `results/selected/SegKNetSWIN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model SegKNetSWIN --data_root ${DATA_ADE} --checkpoint ${WEIGHTS}/SegKNetSWIN.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`SegUperNetSWIN` (`SegUperNetSWIN.ckpt`): compare `results/mine/SegUperNetSWIN__SegUperNetSWIN.json` with `results/selected/SegUperNetSWIN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model SegUperNetSWIN --data_root ${DATA_ADE} --checkpoint ${WEIGHTS}/SegUperNetSWIN.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`DINOSWIN-L` (`DINOSWIN-L.ckpt`): compare `results/mine/DINOSWIN-L__DINOSWIN-L.json` with `results/selected/DINOSWIN-L.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model DINOSWIN-L --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/DINOSWIN-L.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`DDQDETRSWIN-L` (`DDQDETRSWIN-L.ckpt`): compare `results/mine/DDQDETRSWIN-L__DDQDETRSWIN-L.json` with `results/selected/DDQDETRSWIN-L.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model DDQDETRSWIN-L --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/DDQDETRSWIN-L.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`ATSSSWIN-L` (`ATSSSWIN-L.ckpt`): compare `results/mine/ATSSSWIN-L__ATSSSWIN-L.json` with `results/selected/ATSSSWIN-L.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model ATSSSWIN-L --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/ATSSSWIN-L.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`GroundingDINO` (`GroundingDINO.ckpt`): compare `results/mine/GroundingDINO__GroundingDINO.json` with `results/selected/GroundingDINO.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model GroundingDINO --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/GroundingDINO.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`GLIP` (`GLIP.ckpt`): compare `results/mine/GLIP__GLIP.json` with `results/selected/GLIP.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model GLIP --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/GLIP.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`Mask2FormerSWIN` (`Mask2FormerSWIN.ckpt`): compare `results/mine/Mask2FormerSWIN__Mask2FormerSWIN.json` with `results/selected/Mask2FormerSWIN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model Mask2FormerSWIN --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/Mask2FormerSWIN.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`RepPointsV2` (`RepPointsV2.ckpt`): compare `results/mine/RepPointsV2__RepPointsV2.json` with `results/selected/RepPointsV2.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model RepPointsV2 --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/RepPointsV2.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`ATSSResNet` (`ATSSResNet.ckpt`): compare `results/mine/ATSSResNet__ATSSResNet.json` with `results/selected/ATSSResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model ATSSResNet --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/ATSSResNet.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`RetinaNet` (`RetinaNet.ckpt`): compare `results/mine/RetinaNet__RetinaNet.json` with `results/selected/RetinaNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model RetinaNet --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/RetinaNet.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`FasterRCNN` (`FasterRCNN.ckpt`): compare `results/mine/FasterRCNN__FasterRCNN.json` with `results/selected/FasterRCNN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model FasterRCNN --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/FasterRCNN.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`CascadeMaskRCNN` (`CascadeMaskRCNN.ckpt`): compare `results/mine/CascadeMaskRCNN__CascadeMaskRCNN.json` with `results/selected/CascadeMaskRCNN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model CascadeMaskRCNN --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/CascadeMaskRCNN.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`DeformableDETR` (`DeformableDETR.ckpt`): compare `results/mine/DeformableDETR__DeformableDETR.json` with `results/selected/DeformableDETR.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model DeformableDETR --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/DeformableDETR.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`DINOResNet` (`DINOResNet.ckpt`): compare `results/mine/DINOResNet__DINOResNet.json` with `results/selected/DINOResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model DINOResNet --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/DINOResNet.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`Mask2FormerResNet` (`Mask2FormerResNet.ckpt`): compare `results/mine/Mask2FormerResNet__Mask2FormerResNet.json` with `results/selected/Mask2FormerResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model Mask2FormerResNet --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/Mask2FormerResNet.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`DDQDETRResNet` (`DDQDETRResNet.ckpt`): compare `results/mine/DDQDETRResNet__DDQDETRResNet.json` with `results/selected/DDQDETRResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model DDQDETRResNet --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/DDQDETRResNet.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`FCOS` (`FCOS.ckpt`): compare `results/mine/FCOS__FCOS.json` with `results/selected/FCOS.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model FCOS --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/FCOS.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`MaskRCNN` (`MaskRCNN.ckpt`): compare `results/mine/MaskRCNN__MaskRCNN.json` with `results/selected/MaskRCNN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model MaskRCNN --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/MaskRCNN.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

</details>

## The replaced sweeps

7 sweeps, of the unified-recipe runs that a retrain replaced, in `results/replaced/`. Each command writes `results/mine/<task_model>__<weights file stem>.json`; compare it with `results/replaced/<weights file stem>.json`.

| model | weights file | lambda_ref | metric | published points | why it was replaced |
|---|---|---|---|---|---|
| `DINOResNet` | `DINOResNet_lambda5.ckpt` | 5.0 | mAP | 3 | collapsed |
| `DDQDETRResNet` | `DDQDETRResNet_lambda5.ckpt` | 5.0 | mAP | 3 | collapsed |
| `DeformableDETR` | `DeformableDETR_lambda5.ckpt` | 5.0 | mAP | 3 | collapsed |
| `SegUperNetResNet` | `SegUperNetResNet_lambda5.ckpt` | 5.0 | mIoU | 3 | collapsed |
| `GLIP` | `GLIP_lambda5.ckpt` | 5.0 | mAP | 11 | suboptimal |
| `SegMask2FormerResNet` | `SegMask2FormerResNet_lambda5.ckpt` | 5.0 | mIoU | 12 | suboptimal |
| `SegMaskFormerResNet` | `SegMaskFormerResNet_lambda5.ckpt` | 5.0 | mIoU | 10 | suboptimal |

The 4 collapsed sweeps were run with an earlier version of the sweep. It stopped after the three starting points once the codec showed it had collapsed, while this version keeps halving lambda towards `--min_points`. Your sweep may therefore have a few more points, all near zero score; compare the three it shares.

<details>
<summary>The 7 commands</summary>

`DINOResNet` (`DINOResNet_lambda5.ckpt`): compare `results/mine/DINOResNet__DINOResNet_lambda5.json` with `results/replaced/DINOResNet_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model DINOResNet --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/DINOResNet_lambda5.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`DDQDETRResNet` (`DDQDETRResNet_lambda5.ckpt`): compare `results/mine/DDQDETRResNet__DDQDETRResNet_lambda5.json` with `results/replaced/DDQDETRResNet_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model DDQDETRResNet --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/DDQDETRResNet_lambda5.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`DeformableDETR` (`DeformableDETR_lambda5.ckpt`): compare `results/mine/DeformableDETR__DeformableDETR_lambda5.json` with `results/replaced/DeformableDETR_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model DeformableDETR --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/DeformableDETR_lambda5.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`SegUperNetResNet` (`SegUperNetResNet_lambda5.ckpt`): compare `results/mine/SegUperNetResNet__SegUperNetResNet_lambda5.json` with `results/replaced/SegUperNetResNet_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model SegUperNetResNet --data_root ${DATA_ADE} --checkpoint ${WEIGHTS}/SegUperNetResNet_lambda5.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`GLIP` (`GLIP_lambda5.ckpt`): compare `results/mine/GLIP__GLIP_lambda5.json` with `results/replaced/GLIP_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model GLIP --data_root ${DATA_COCO} --checkpoint ${WEIGHTS}/GLIP_lambda5.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`SegMask2FormerResNet` (`SegMask2FormerResNet_lambda5.ckpt`): compare `results/mine/SegMask2FormerResNet__SegMask2FormerResNet_lambda5.json` with `results/replaced/SegMask2FormerResNet_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model SegMask2FormerResNet --data_root ${DATA_ADE} --checkpoint ${WEIGHTS}/SegMask2FormerResNet_lambda5.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

`SegMaskFormerResNet` (`SegMaskFormerResNet_lambda5.ckpt`): compare `results/mine/SegMaskFormerResNet__SegMaskFormerResNet_lambda5.json` with `results/replaced/SegMaskFormerResNet_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model SegMaskFormerResNet --data_root ${DATA_ADE} --checkpoint ${WEIGHTS}/SegMaskFormerResNet_lambda5.ckpt --results_dir results/mine --max_points 14 --min_points 8 --floor_frac 0.6 --plateau_tol 0.015 --q_step 0.03 --rate_stall 0.1 --hi_frac 0.85
```

</details>

## The cross-model commands

57 commands, one per file in `results/cross_model/`. Each writes `results/mine/cross_model/<setting>/<task_model>__<weights file stem>.json` (each setting into its own folder: a codec's file has the same name in every setting); compare each target's entry with the published file's at matched bpp.

<details>
<summary>Segmentation, ResNet-50, calibrated with training-image statistics (6 files)</summary>

`SegKNetUPerResNet.ckpt`: compare `results/mine/cross_model/cal/SegKNetUPerResNet__SegKNetUPerResNet.json` with `results/cross_model/cal/SegKNetUPerResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegKNetUPerResNet --checkpoint ${WEIGHTS}/SegKNetUPerResNet.ckpt --target_task_models SegFCNResNet,SegMask2FormerResNet,SegMaskFormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`SegMask2FormerResNet_lambda30.ckpt`: compare `results/mine/cross_model/cal/SegMask2FormerResNet__SegMask2FormerResNet_lambda30.json` with `results/cross_model/cal/SegMask2FormerResNet_lambda30.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMask2FormerResNet --checkpoint ${WEIGHTS}/SegMask2FormerResNet_lambda30.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMaskFormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`SegMask2FormerResNet_lambda5.ckpt`: compare `results/mine/cross_model/cal/SegMask2FormerResNet__SegMask2FormerResNet_lambda5.json` with `results/cross_model/cal/SegMask2FormerResNet_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMask2FormerResNet --checkpoint ${WEIGHTS}/SegMask2FormerResNet_lambda5.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMaskFormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`SegMaskFormerResNet_lambda30.ckpt`: compare `results/mine/cross_model/cal/SegMaskFormerResNet__SegMaskFormerResNet_lambda30.json` with `results/cross_model/cal/SegMaskFormerResNet_lambda30.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMaskFormerResNet --checkpoint ${WEIGHTS}/SegMaskFormerResNet_lambda30.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMask2FormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`SegMaskFormerResNet_lambda5.ckpt`: compare `results/mine/cross_model/cal/SegMaskFormerResNet__SegMaskFormerResNet_lambda5.json` with `results/cross_model/cal/SegMaskFormerResNet_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMaskFormerResNet --checkpoint ${WEIGHTS}/SegMaskFormerResNet_lambda5.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMask2FormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`SegUperNetResNet.ckpt`: compare `results/mine/cross_model/cal/SegUperNetResNet__SegUperNetResNet.json` with `results/cross_model/cal/SegUperNetResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegUperNetResNet --checkpoint ${WEIGHTS}/SegUperNetResNet.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMask2FormerResNet,SegMaskFormerResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal --calibration_stats train
```

</details>

<details>
<summary>Segmentation, ResNet-50, calibrated with test-set statistics, as submitted (6 files)</summary>

`SegKNetUPerResNet.ckpt`: compare `results/mine/cross_model/cal_test/SegKNetUPerResNet__SegKNetUPerResNet.json` with `results/cross_model/cal_test/SegKNetUPerResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegKNetUPerResNet --checkpoint ${WEIGHTS}/SegKNetUPerResNet.ckpt --target_task_models SegFCNResNet,SegMask2FormerResNet,SegMaskFormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`SegMask2FormerResNet_lambda30.ckpt`: compare `results/mine/cross_model/cal_test/SegMask2FormerResNet__SegMask2FormerResNet_lambda30.json` with `results/cross_model/cal_test/SegMask2FormerResNet_lambda30.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMask2FormerResNet --checkpoint ${WEIGHTS}/SegMask2FormerResNet_lambda30.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMaskFormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`SegMask2FormerResNet_lambda5.ckpt`: compare `results/mine/cross_model/cal_test/SegMask2FormerResNet__SegMask2FormerResNet_lambda5.json` with `results/cross_model/cal_test/SegMask2FormerResNet_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMask2FormerResNet --checkpoint ${WEIGHTS}/SegMask2FormerResNet_lambda5.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMaskFormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`SegMaskFormerResNet_lambda30.ckpt`: compare `results/mine/cross_model/cal_test/SegMaskFormerResNet__SegMaskFormerResNet_lambda30.json` with `results/cross_model/cal_test/SegMaskFormerResNet_lambda30.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMaskFormerResNet --checkpoint ${WEIGHTS}/SegMaskFormerResNet_lambda30.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMask2FormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`SegMaskFormerResNet_lambda5.ckpt`: compare `results/mine/cross_model/cal_test/SegMaskFormerResNet__SegMaskFormerResNet_lambda5.json` with `results/cross_model/cal_test/SegMaskFormerResNet_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMaskFormerResNet --checkpoint ${WEIGHTS}/SegMaskFormerResNet_lambda5.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMask2FormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`SegUperNetResNet.ckpt`: compare `results/mine/cross_model/cal_test/SegUperNetResNet__SegUperNetResNet.json` with `results/cross_model/cal_test/SegUperNetResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegUperNetResNet --checkpoint ${WEIGHTS}/SegUperNetResNet.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMask2FormerResNet,SegMaskFormerResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

</details>

<details>
<summary>Segmentation, ResNet-50, uncalibrated (6 files)</summary>

`SegKNetUPerResNet.ckpt`: compare `results/mine/cross_model/nocal/SegKNetUPerResNet__SegKNetUPerResNet.json` with `results/cross_model/nocal/SegKNetUPerResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegKNetUPerResNet --checkpoint ${WEIGHTS}/SegKNetUPerResNet.ckpt --target_task_models SegFCNResNet,SegMask2FormerResNet,SegMaskFormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`SegMask2FormerResNet_lambda30.ckpt`: compare `results/mine/cross_model/nocal/SegMask2FormerResNet__SegMask2FormerResNet_lambda30.json` with `results/cross_model/nocal/SegMask2FormerResNet_lambda30.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMask2FormerResNet --checkpoint ${WEIGHTS}/SegMask2FormerResNet_lambda30.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMaskFormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`SegMask2FormerResNet_lambda5.ckpt`: compare `results/mine/cross_model/nocal/SegMask2FormerResNet__SegMask2FormerResNet_lambda5.json` with `results/cross_model/nocal/SegMask2FormerResNet_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMask2FormerResNet --checkpoint ${WEIGHTS}/SegMask2FormerResNet_lambda5.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMaskFormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`SegMaskFormerResNet_lambda30.ckpt`: compare `results/mine/cross_model/nocal/SegMaskFormerResNet__SegMaskFormerResNet_lambda30.json` with `results/cross_model/nocal/SegMaskFormerResNet_lambda30.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMaskFormerResNet --checkpoint ${WEIGHTS}/SegMaskFormerResNet_lambda30.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMask2FormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`SegMaskFormerResNet_lambda5.ckpt`: compare `results/mine/cross_model/nocal/SegMaskFormerResNet__SegMaskFormerResNet_lambda5.json` with `results/cross_model/nocal/SegMaskFormerResNet_lambda5.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMaskFormerResNet --checkpoint ${WEIGHTS}/SegMaskFormerResNet_lambda5.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMask2FormerResNet,SegUperNetResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`SegUperNetResNet.ckpt`: compare `results/mine/cross_model/nocal/SegUperNetResNet__SegUperNetResNet.json` with `results/cross_model/nocal/SegUperNetResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegUperNetResNet --checkpoint ${WEIGHTS}/SegUperNetResNet.ckpt --target_task_models SegFCNResNet,SegKNetUPerResNet,SegMask2FormerResNet,SegMaskFormerResNet --data_root ${DATA_ADE} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

</details>

<details>
<summary>Segmentation, Swin-T, uncalibrated (4 files)</summary>

`SegKNetSWIN.ckpt`: compare `results/mine/cross_model/nocal/SegKNetSWIN__SegKNetSWIN.json` with `results/cross_model/nocal/SegKNetSWIN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegKNetSWIN --checkpoint ${WEIGHTS}/SegKNetSWIN.ckpt --target_task_models SegMask2FormerSWIN,SegMaskFormerSWIN,SegUperNetSWIN --data_root ${DATA_ADE} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`SegMask2FormerSWIN.ckpt`: compare `results/mine/cross_model/nocal/SegMask2FormerSWIN__SegMask2FormerSWIN.json` with `results/cross_model/nocal/SegMask2FormerSWIN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMask2FormerSWIN --checkpoint ${WEIGHTS}/SegMask2FormerSWIN.ckpt --target_task_models SegKNetSWIN,SegMaskFormerSWIN,SegUperNetSWIN --data_root ${DATA_ADE} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`SegMaskFormerSWIN.ckpt`: compare `results/mine/cross_model/nocal/SegMaskFormerSWIN__SegMaskFormerSWIN.json` with `results/cross_model/nocal/SegMaskFormerSWIN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegMaskFormerSWIN --checkpoint ${WEIGHTS}/SegMaskFormerSWIN.ckpt --target_task_models SegKNetSWIN,SegMask2FormerSWIN,SegUperNetSWIN --data_root ${DATA_ADE} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`SegUperNetSWIN.ckpt`: compare `results/mine/cross_model/nocal/SegUperNetSWIN__SegUperNetSWIN.json` with `results/cross_model/nocal/SegUperNetSWIN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model SegUperNetSWIN --checkpoint ${WEIGHTS}/SegUperNetSWIN.ckpt --target_task_models SegKNetSWIN,SegMask2FormerSWIN,SegMaskFormerSWIN --data_root ${DATA_ADE} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

</details>

<details>
<summary>Detection, ResNet-50, calibrated with training-image statistics (9 files)</summary>

`ATSSResNet.ckpt`: compare `results/mine/cross_model/cal/ATSSResNet__ATSSResNet.json` with `results/cross_model/cal/ATSSResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model ATSSResNet --checkpoint ${WEIGHTS}/ATSSResNet.ckpt --target_task_models CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`CascadeMaskRCNN.ckpt`: compare `results/mine/cross_model/cal/CascadeMaskRCNN__CascadeMaskRCNN.json` with `results/cross_model/cal/CascadeMaskRCNN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model CascadeMaskRCNN --checkpoint ${WEIGHTS}/CascadeMaskRCNN.ckpt --target_task_models ATSSResNet,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`DDQDETRResNet.ckpt`: compare `results/mine/cross_model/cal/DDQDETRResNet__DDQDETRResNet.json` with `results/cross_model/cal/DDQDETRResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model DDQDETRResNet --checkpoint ${WEIGHTS}/DDQDETRResNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`DINOResNet.ckpt`: compare `results/mine/cross_model/cal/DINOResNet__DINOResNet.json` with `results/cross_model/cal/DINOResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model DINOResNet --checkpoint ${WEIGHTS}/DINOResNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`DeformableDETR.ckpt`: compare `results/mine/cross_model/cal/DeformableDETR__DeformableDETR.json` with `results/cross_model/cal/DeformableDETR.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model DeformableDETR --checkpoint ${WEIGHTS}/DeformableDETR.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`FCOS.ckpt`: compare `results/mine/cross_model/cal/FCOS__FCOS.json` with `results/cross_model/cal/FCOS.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model FCOS --checkpoint ${WEIGHTS}/FCOS.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`FasterRCNN.ckpt`: compare `results/mine/cross_model/cal/FasterRCNN__FasterRCNN.json` with `results/cross_model/cal/FasterRCNN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model FasterRCNN --checkpoint ${WEIGHTS}/FasterRCNN.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`Mask2FormerResNet.ckpt`: compare `results/mine/cross_model/cal/Mask2FormerResNet__Mask2FormerResNet.json` with `results/cross_model/cal/Mask2FormerResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model Mask2FormerResNet --checkpoint ${WEIGHTS}/Mask2FormerResNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal --calibration_stats train
```

`RetinaNet.ckpt`: compare `results/mine/cross_model/cal/RetinaNet__RetinaNet.json` with `results/cross_model/cal/RetinaNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model RetinaNet --checkpoint ${WEIGHTS}/RetinaNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal --calibration_stats train
```

</details>

<details>
<summary>Detection, ResNet-50, calibrated with test-set statistics, as submitted (9 files)</summary>

`ATSSResNet.ckpt`: compare `results/mine/cross_model/cal_test/ATSSResNet__ATSSResNet.json` with `results/cross_model/cal_test/ATSSResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model ATSSResNet --checkpoint ${WEIGHTS}/ATSSResNet.ckpt --target_task_models CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`CascadeMaskRCNN.ckpt`: compare `results/mine/cross_model/cal_test/CascadeMaskRCNN__CascadeMaskRCNN.json` with `results/cross_model/cal_test/CascadeMaskRCNN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model CascadeMaskRCNN --checkpoint ${WEIGHTS}/CascadeMaskRCNN.ckpt --target_task_models ATSSResNet,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`DDQDETRResNet.ckpt`: compare `results/mine/cross_model/cal_test/DDQDETRResNet__DDQDETRResNet.json` with `results/cross_model/cal_test/DDQDETRResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model DDQDETRResNet --checkpoint ${WEIGHTS}/DDQDETRResNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`DINOResNet.ckpt`: compare `results/mine/cross_model/cal_test/DINOResNet__DINOResNet.json` with `results/cross_model/cal_test/DINOResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model DINOResNet --checkpoint ${WEIGHTS}/DINOResNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`DeformableDETR.ckpt`: compare `results/mine/cross_model/cal_test/DeformableDETR__DeformableDETR.json` with `results/cross_model/cal_test/DeformableDETR.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model DeformableDETR --checkpoint ${WEIGHTS}/DeformableDETR.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`FCOS.ckpt`: compare `results/mine/cross_model/cal_test/FCOS__FCOS.json` with `results/cross_model/cal_test/FCOS.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model FCOS --checkpoint ${WEIGHTS}/FCOS.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`FasterRCNN.ckpt`: compare `results/mine/cross_model/cal_test/FasterRCNN__FasterRCNN.json` with `results/cross_model/cal_test/FasterRCNN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model FasterRCNN --checkpoint ${WEIGHTS}/FasterRCNN.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`Mask2FormerResNet.ckpt`: compare `results/mine/cross_model/cal_test/Mask2FormerResNet__Mask2FormerResNet.json` with `results/cross_model/cal_test/Mask2FormerResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model Mask2FormerResNet --checkpoint ${WEIGHTS}/Mask2FormerResNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

`RetinaNet.ckpt`: compare `results/mine/cross_model/cal_test/RetinaNet__RetinaNet.json` with `results/cross_model/cal_test/RetinaNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model RetinaNet --checkpoint ${WEIGHTS}/RetinaNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/cal_test --calibration_stats test
```

</details>

<details>
<summary>Detection, ResNet-50, uncalibrated (9 files)</summary>

`ATSSResNet.ckpt`: compare `results/mine/cross_model/nocal/ATSSResNet__ATSSResNet.json` with `results/cross_model/nocal/ATSSResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model ATSSResNet --checkpoint ${WEIGHTS}/ATSSResNet.ckpt --target_task_models CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`CascadeMaskRCNN.ckpt`: compare `results/mine/cross_model/nocal/CascadeMaskRCNN__CascadeMaskRCNN.json` with `results/cross_model/nocal/CascadeMaskRCNN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model CascadeMaskRCNN --checkpoint ${WEIGHTS}/CascadeMaskRCNN.ckpt --target_task_models ATSSResNet,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`DDQDETRResNet.ckpt`: compare `results/mine/cross_model/nocal/DDQDETRResNet__DDQDETRResNet.json` with `results/cross_model/nocal/DDQDETRResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model DDQDETRResNet --checkpoint ${WEIGHTS}/DDQDETRResNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`DINOResNet.ckpt`: compare `results/mine/cross_model/nocal/DINOResNet__DINOResNet.json` with `results/cross_model/nocal/DINOResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model DINOResNet --checkpoint ${WEIGHTS}/DINOResNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`DeformableDETR.ckpt`: compare `results/mine/cross_model/nocal/DeformableDETR__DeformableDETR.json` with `results/cross_model/nocal/DeformableDETR.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model DeformableDETR --checkpoint ${WEIGHTS}/DeformableDETR.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,FCOS,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`FCOS.ckpt`: compare `results/mine/cross_model/nocal/FCOS__FCOS.json` with `results/cross_model/nocal/FCOS.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model FCOS --checkpoint ${WEIGHTS}/FCOS.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FasterRCNN,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`FasterRCNN.ckpt`: compare `results/mine/cross_model/nocal/FasterRCNN__FasterRCNN.json` with `results/cross_model/nocal/FasterRCNN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model FasterRCNN --checkpoint ${WEIGHTS}/FasterRCNN.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,Mask2FormerResNet,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`Mask2FormerResNet.ckpt`: compare `results/mine/cross_model/nocal/Mask2FormerResNet__Mask2FormerResNet.json` with `results/cross_model/nocal/Mask2FormerResNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model Mask2FormerResNet --checkpoint ${WEIGHTS}/Mask2FormerResNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,RetinaNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`RetinaNet.ckpt`: compare `results/mine/cross_model/nocal/RetinaNet__RetinaNet.json` with `results/cross_model/nocal/RetinaNet.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model RetinaNet --checkpoint ${WEIGHTS}/RetinaNet.ckpt --target_task_models ATSSResNet,CascadeMaskRCNN,DDQDETRResNet,DINOResNet,DeformableDETR,FCOS,FasterRCNN,Mask2FormerResNet --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

</details>

<details>
<summary>Detection, Swin-T, uncalibrated (5 files)</summary>

`GLIP.ckpt`: compare `results/mine/cross_model/nocal/GLIP__GLIP.json` with `results/cross_model/nocal/GLIP.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model GLIP --checkpoint ${WEIGHTS}/GLIP.ckpt --target_task_models GroundingDINO,Mask2FormerSWIN,MaskRCNN,RepPointsV2 --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`GroundingDINO.ckpt`: compare `results/mine/cross_model/nocal/GroundingDINO__GroundingDINO.json` with `results/cross_model/nocal/GroundingDINO.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model GroundingDINO --checkpoint ${WEIGHTS}/GroundingDINO.ckpt --target_task_models GLIP,Mask2FormerSWIN,MaskRCNN,RepPointsV2 --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`Mask2FormerSWIN.ckpt`: compare `results/mine/cross_model/nocal/Mask2FormerSWIN__Mask2FormerSWIN.json` with `results/cross_model/nocal/Mask2FormerSWIN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model Mask2FormerSWIN --checkpoint ${WEIGHTS}/Mask2FormerSWIN.ckpt --target_task_models GLIP,GroundingDINO,MaskRCNN,RepPointsV2 --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`MaskRCNN.ckpt`: compare `results/mine/cross_model/nocal/MaskRCNN__MaskRCNN.json` with `results/cross_model/nocal/MaskRCNN.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model MaskRCNN --checkpoint ${WEIGHTS}/MaskRCNN.ckpt --target_task_models GLIP,GroundingDINO,Mask2FormerSWIN,RepPointsV2 --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`RepPointsV2.ckpt`: compare `results/mine/cross_model/nocal/RepPointsV2__RepPointsV2.json` with `results/cross_model/nocal/RepPointsV2.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model RepPointsV2 --checkpoint ${WEIGHTS}/RepPointsV2.ckpt --target_task_models GLIP,GroundingDINO,Mask2FormerSWIN,MaskRCNN --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

</details>

<details>
<summary>Detection, Swin-L, uncalibrated (3 files)</summary>

`ATSSSWIN-L.ckpt`: compare `results/mine/cross_model/nocal/ATSSSWIN-L__ATSSSWIN-L.json` with `results/cross_model/nocal/ATSSSWIN-L.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model ATSSSWIN-L --checkpoint ${WEIGHTS}/ATSSSWIN-L.ckpt --target_task_models DDQDETRSWIN-L,DINOSWIN-L --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`DDQDETRSWIN-L.ckpt`: compare `results/mine/cross_model/nocal/DDQDETRSWIN-L__DDQDETRSWIN-L.json` with `results/cross_model/nocal/DDQDETRSWIN-L.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model DDQDETRSWIN-L --checkpoint ${WEIGHTS}/DDQDETRSWIN-L.ckpt --target_task_models ATSSSWIN-L,DINOSWIN-L --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

`DINOSWIN-L.ckpt`: compare `results/mine/cross_model/nocal/DINOSWIN-L__DINOSWIN-L.json` with `results/cross_model/nocal/DINOSWIN-L.json`.

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model DINOSWIN-L --checkpoint ${WEIGHTS}/DINOSWIN-L.ckpt --target_task_models ATSSSWIN-L,DDQDETRSWIN-L --data_root ${DATA_COCO} --results_dir results/mine/cross_model/nocal --calibration_stats none
```

</details>

## The training commands

Every command saves into its own `--checkpoint_dir` under `${CKPT}`; relaunching the same command resumes it.

### The unified recipe (lambda_ref 5)

Every task model was first trained with this recipe. On the models marked `collapses` the codec collapses: its sweep stays near zero task score at every rate. Those models were retrained with a raised lambda_ref (below).

| model | data | lambda_ref | lambda range | epochs | LR schedule | stage-1 epochs | outcome | note |
|---|---|---|---|---|---|---|---|---|
| `SegKNetUPerResNet` | ADE20K | 5.0 | 0.05-10 | 200 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | |
| `SegMask2FormerResNet` | ADE20K | 5.0 | 0.05-10 | 200 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | |
| `SegMaskFormerResNet` | ADE20K | 5.0 | 0.05-10 | 200 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | |
| `SegUperNetResNet` | ADE20K | 5.0 | 0.05-10 | 200 | wsd (decay 0.3, to 0.05 x lr) | 5 | collapses | |
| `SegMaskFormerSWIN` | ADE20K | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | (a) |
| `SegMask2FormerSWIN` | ADE20K | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | (a) |
| `SegKNetSWIN` | ADE20K | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | (a) |
| `SegUperNetSWIN` | ADE20K | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | (a) |
| `DINOSWIN-L` | COCO | 5.0 | 0.05-10 | 75 | wsd (decay 0.3, to 0.05 x lr) | 0 | works | (a) |
| `DDQDETRSWIN-L` | COCO | 5.0 | 0.05-10 | 75 | wsd (decay 0.3, to 0.05 x lr) | 0 | collapses | not swept: validation mAP 0.000 when stopped at epoch 64 of 75 (a) |
| `ATSSSWIN-L` | COCO | 5.0 | 0.05-10 | 75 | wsd (decay 0.3, to 0.05 x lr) | 0 | collapses | not swept: validation mAP 0.001 when stopped at epoch 58 of 75 (a) |
| `GroundingDINO` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | (a) |
| `GLIP` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | (a) |
| `Mask2FormerSWIN` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | (a) |
| `RepPointsV2` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | (a) |
| `ATSSResNet` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | |
| `RetinaNet` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | |
| `FasterRCNN` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | |
| `CascadeMaskRCNN` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | |
| `DeformableDETR` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | collapses | |
| `DINOResNet` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | collapses | |
| `Mask2FormerResNet` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | tops out at 0.82 of the uncompressed mAP |
| `DDQDETRResNet` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | collapses | |
| `FCOS` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | |
| `MaskRCNN` | COCO | 5.0 | 0.05-10 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | works | (a) |

(a) Trained with an earlier normalisation of the rate term; retraining with this code gives close but not identical weights.

<details>
<summary>The 25 commands</summary>

`SegKNetUPerResNet`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model SegKNetUPerResNet --data_root ${DATA_ADE} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 200 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/SegKNetUPerResNet_lambda5
```

`SegMask2FormerResNet`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model SegMask2FormerResNet --data_root ${DATA_ADE} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 200 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/SegMask2FormerResNet_lambda5
```

`SegMaskFormerResNet`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model SegMaskFormerResNet --data_root ${DATA_ADE} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 200 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/SegMaskFormerResNet_lambda5
```

`SegUperNetResNet`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model SegUperNetResNet --data_root ${DATA_ADE} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 200 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/SegUperNetResNet_lambda5
```

`SegMaskFormerSWIN`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model SegMaskFormerSWIN --data_root ${DATA_ADE} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/SegMaskFormerSWIN_lambda5
```

`SegMask2FormerSWIN`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model SegMask2FormerSWIN --data_root ${DATA_ADE} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/SegMask2FormerSWIN_lambda5
```

`SegKNetSWIN`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model SegKNetSWIN --data_root ${DATA_ADE} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/SegKNetSWIN_lambda5
```

`SegUperNetSWIN`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model SegUperNetSWIN --data_root ${DATA_ADE} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/SegUperNetSWIN_lambda5
```

`DINOSWIN-L`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model DINOSWIN-L --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 75 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 0 --checkpoint_dir ${CKPT}/DINOSWIN-L_lambda5
```

`DDQDETRSWIN-L`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model DDQDETRSWIN-L --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 75 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 0 --checkpoint_dir ${CKPT}/DDQDETRSWIN-L_lambda5
```

`ATSSSWIN-L`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model ATSSSWIN-L --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 75 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 0 --checkpoint_dir ${CKPT}/ATSSSWIN-L_lambda5
```

`GroundingDINO`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model GroundingDINO --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/GroundingDINO_lambda5
```

`GLIP`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model GLIP --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/GLIP_lambda5
```

`Mask2FormerSWIN`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model Mask2FormerSWIN --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/Mask2FormerSWIN_lambda5
```

`RepPointsV2`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model RepPointsV2 --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/RepPointsV2_lambda5
```

`ATSSResNet`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model ATSSResNet --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/ATSSResNet_lambda5
```

`RetinaNet`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model RetinaNet --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/RetinaNet_lambda5
```

`FasterRCNN`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model FasterRCNN --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/FasterRCNN_lambda5
```

`CascadeMaskRCNN`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model CascadeMaskRCNN --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/CascadeMaskRCNN_lambda5
```

`DeformableDETR`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model DeformableDETR --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/DeformableDETR_lambda5
```

`DINOResNet`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model DINOResNet --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/DINOResNet_lambda5
```

`Mask2FormerResNet`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model Mask2FormerResNet --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/Mask2FormerResNet_lambda5
```

`DDQDETRResNet`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model DDQDETRResNet --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/DDQDETRResNet_lambda5
```

`FCOS`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model FCOS --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/FCOS_lambda5
```

`MaskRCNN`, lambda_ref 5.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model MaskRCNN --data_root ${DATA_COCO} --lambda_ref 5.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/MaskRCNN_lambda5
```

</details>

### The rescued and fine-tuned runs

The published codecs trained with a raised lambda_ref: the models whose unified run collapsed (rescued), and the models whose unified run worked but was suboptimal (fine-tuned).

| model | data | lambda_ref | lambda range | epochs | LR schedule | stage-1 epochs | note |
|---|---|---|---|---|---|---|---|
| `SegMask2FormerResNet` | ADE20K | 30.0 | 0.3-60 | 200 | wsd (decay 0.3, to 0.05 x lr) | 5 | improves on the unified run |
| `SegMaskFormerResNet` | ADE20K | 30.0 | 0.3-60 | 200 | wsd (decay 0.3, to 0.05 x lr) | 5 | improves on the unified run |
| `SegUperNetResNet` | ADE20K | 50.0 | 0.5-500 | 200 | wsd (decay 0.3, to 0.05 x lr) | 5 | the unified run collapses |
| `DDQDETRSWIN-L` | COCO | 50.0 | 0.5-100 | 75 | wsd (decay 0.2, to 0.1 x lr) | 0 | the unified run collapses (a) |
| `ATSSSWIN-L` | COCO | 50.0 | 0.5-100 | 75 | wsd (decay 0.2, to 0.1 x lr) | 0 | the unified run collapses (a) |
| `GLIP` | COCO | 10.0 | 0.1-80 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | improves on the unified run; the published codec was warm-started for 30 epochs from the unified recipe checkpoint (a) |
| `DeformableDETR` | COCO | 30.0 | 0.3-60 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | the unified run collapses |
| `DINOResNet` | COCO | 30.0 | 0.3-60 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | the unified run collapses |
| `DDQDETRResNet` | COCO | 50.0 | 0.5-100 | 100 | wsd (decay 0.3, to 0.05 x lr) | 5 | the unified run collapses |

(a) Trained with an earlier normalisation of the rate term; retraining with this code gives close but not identical weights.

<details>
<summary>The 9 commands</summary>

`SegMask2FormerResNet`, lambda_ref 30.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model SegMask2FormerResNet --data_root ${DATA_ADE} --lambda_ref 30.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 200 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/SegMask2FormerResNet_lambda30
```

`SegMaskFormerResNet`, lambda_ref 30.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model SegMaskFormerResNet --data_root ${DATA_ADE} --lambda_ref 30.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 200 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/SegMaskFormerResNet_lambda30
```

`SegUperNetResNet`, lambda_ref 50.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model SegUperNetResNet --data_root ${DATA_ADE} --lambda_ref 50.0 --lambda_min 0.01 --lambda_max 10.0 --epochs 200 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/SegUperNetResNet_lambda50
```

`DDQDETRSWIN-L`, lambda_ref 50.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model DDQDETRSWIN-L --data_root ${DATA_COCO} --lambda_ref 50.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 75 --lr_scheduler wsd --lr_decay_frac 0.2 --lr_min_factor 0.1 --lambda_warmup_epochs 0 --checkpoint_dir ${CKPT}/DDQDETRSWIN-L_lambda50
```

`ATSSSWIN-L`, lambda_ref 50.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model ATSSSWIN-L --data_root ${DATA_COCO} --lambda_ref 50.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 75 --lr_scheduler wsd --lr_decay_frac 0.2 --lr_min_factor 0.1 --lambda_warmup_epochs 0 --checkpoint_dir ${CKPT}/ATSSSWIN-L_lambda50
```

`GLIP`, lambda_ref 10.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model GLIP --data_root ${DATA_COCO} --lambda_ref 10.0 --lambda_min 0.01 --lambda_max 8.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/GLIP_lambda10
```

`DeformableDETR`, lambda_ref 30.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model DeformableDETR --data_root ${DATA_COCO} --lambda_ref 30.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/DeformableDETR_lambda30
```

`DINOResNet`, lambda_ref 30.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model DINOResNet --data_root ${DATA_COCO} --lambda_ref 30.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/DINOResNet_lambda30
```

`DDQDETRResNet`, lambda_ref 50.0:

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model DDQDETRResNet --data_root ${DATA_COCO} --lambda_ref 50.0 --lambda_min 0.01 --lambda_max 2.0 --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 --checkpoint_dir ${CKPT}/DDQDETRResNet_lambda50
```

</details>
