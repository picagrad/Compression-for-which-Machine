"""The paper's curated choices: model groups, which run stands for each model, and figure settings.

Input and output paths are given on the command line."""

COVERAGE_POWER = 2.0               # weight of a point: (share of models measured at its rate) ** power / model's points

# backbone groups: output folder -> (label, models)
BACKBONES = {
    'resnet': ('ResNet', ['ATSSResNet', 'CascadeMaskRCNN', 'DDQDETRResNet', 'DINOResNet', 'DeformableDETR', 'FCOS',
                          'FasterRCNN', 'Mask2FormerResNet', 'RetinaNet', 'SegKNetUPerResNet',
                          'SegMask2FormerResNet', 'SegMaskFormerResNet', 'SegUperNetResNet']),
    'swin_t': ('SWIN-T', ['GLIP', 'GroundingDINO', 'Mask2FormerSWIN', 'MaskRCNN', 'RepPointsV2', 'SegKNetSWIN',
                          'SegMask2FormerSWIN', 'SegMaskFormerSWIN', 'SegUperNetSWIN']),
    'swin_l': ('SWIN-L', ['ATSSSWIN-L', 'DDQDETRSWIN-L', 'DINOSWIN-L']),
}

# mean curves per backbone
MEAN_REPORT_ANCHOR = 'vtm_qpa1'                     # anchor in the printed bpp-at-90/95% table
MEAN_ANCHORS = ('vtm_qpa1', 'bpg', 'jpeg')   # anchors drawn per backbone
MEAN_XMAX = {'mAP': 1.6}                     # x limit per task; unset: 1.5x the highest learned bpp
MEAN_COLOURS = ('#1F4E99', '#C8642D', '#5F6F7F')   # backbone colours on the figure with the all-backbones curve
MEAN_ALL_COLOUR = '#2A9D8F'                    # the all-backbones curve (drawn at alpha 0.6)

# BD-rate tables
# (model, task metric) pairs left out of the mean curves and the cross-model results, with the reason
MEAN_EXCLUDE = {
    ('Mask2FormerResNet', 'mAP'): 'semi-collapsed detector: tops out at 0.82 of uncompressed at 3.6 bpp; '
                                  'every other detector reaches >=0.93',
}
BD_ANCHORS = ('vtm_qpa1', 'bpg')   # BD-rate references; the first is the main one (per-sweep rows, cross-model)
BD_QUALITY_FRAC = 0.90             # second column: task scores >= this fraction of uncompressed only
# Runs are named by (model, run hash): see analysis.run_hash. Every run is listed in exactly one of UNIFIED,
# RESCUED and FINETUNED; a run in none of them is left out of every output, with a warning.
# The unified recipe: every model trained with the same settings.
UNIFIED = {
    ('ATSSResNet', '6c5188e6b7'),
    ('CascadeMaskRCNN', '643b063c9c'),
    ('DDQDETRResNet', '4fc36120ca'),           # collapsed
    ('DINOResNet', '5f1af6e55a'),              # collapsed
    ('DINOSWIN-L', '3ac1855142'),
    ('DeformableDETR', '6b8fb2955c'),          # collapsed
    ('FCOS', 'c324749169'),
    ('FasterRCNN', 'f2d4f8b22b'),
    ('GLIP', 'e1b5ef39b5'),
    ('GroundingDINO', 'f07196114a'),
    ('Mask2FormerResNet', '788e54b6c1'),
    ('Mask2FormerSWIN', '9a4d0ade61'),
    ('MaskRCNN', '7ea6bf0f75'),
    ('RepPointsV2', '972306c588'),
    ('RetinaNet', '16a49ef2fc'),
    ('SegKNetSWIN', '4afbaf90a9'),
    ('SegKNetUPerResNet', 'e1d87272c2'),
    ('SegMask2FormerResNet', 'f62e46886f'),
    ('SegMask2FormerSWIN', '6e231498d8'),
    ('SegMaskFormerResNet', 'db3c9f3ae4'),
    ('SegMaskFormerSWIN', 'fe8609b0ef'),
    ('SegUperNetResNet', '987e4f0462'),        # collapsed
    ('SegUperNetSWIN', 'a493872b94'),
}
# Retrains after the unified recipe; a model's retrain represents it (mean curves, the BD tables' 'Fine-tuned'
# rows, cross-model results). RESCUED: the model's unified run collapsed. FINETUNED: it worked, and the retrain
# superseded it.
RESCUED = {
    ('SegUperNetResNet', 'a3dc3a1ac2'),    # lambda_ref 50
    ('DDQDETRResNet', '2ed6991833'),       # lambda_ref 50
    ('DINOResNet', 'ab981f325f'),          # lambda_ref 30
    ('DeformableDETR', 'b371fa29eb'),      # lambda_ref 30
    ('ATSSSWIN-L', 'c9d91e6484'),          # lambda_ref 50
    ('DDQDETRSWIN-L', 'ff8d63fca3'),       # lambda_ref 50
}
FINETUNED = {
    ('SegMask2FormerResNet', '8eec8dc9a2'),  # lambda_ref 30
    ('SegMaskFormerResNet', '2f2428c2f6'),   # lambda_ref 30
    ('GLIP', '98777b2bae'),                  # lambda_ref 10
}
# runs left out of the per-sweep BD table, chosen by hand so newer runs never change it
BD_EXCLUDE = {
    ('GLIP', 'e1b5ef39b5'),                  # lambda_ref 5, superseded by its fine-tuned run
    ('SegMask2FormerResNet', 'f62e46886f'),  # lambda_ref 5, superseded by its fine-tuned run
    ('SegMaskFormerResNet', 'db3c9f3ae4'),   # lambda_ref 5, superseded by its fine-tuned run
}
BD_SHOW_COLLAPSED = False
# collapsed unified runs that were never swept: (task metric, model, lambda_ref), each counted as served by
# every anchor
UNSWEPT_COLLAPSES = [('mAP', 'ATSSSWIN-L', 5.0), ('mAP', 'DDQDETRSWIN-L', 5.0)]
RECIPE_OUTLIERS = {'Mask2FormerResNet': 'Mask2Former-R50'}
UNSWEPT_BAND = (0.01, 2.0)         # lambda_min / lambda_max ratios assumed for a run that was never swept

# cross-model tests
CROSS_SETTINGS = {'cal': 'calibrated', 'nocal': 'uncalibrated'}
CROSS_EXTRA_TARGETS = {'resnet': ['SegFCNResNet']}     # tested on, but no codec of their own
CROSS_MATRIX_HIDE = {'SegFCNResNet'}                   # left out of the matrix figures
CROSS_HIGHLIGHT = {'Mask2FormerResNet': 'Mask2Former-R50'}
CROSS_LABEL_OFFSET = {('Mask2FormerResNet', 'source'): (8, -16), ('Mask2FormerResNet', 'target'): (12, 0),
                               ('SegFCNResNet', 'target'): (12, 0)}
CROSS_CLIP = 100                                       # matrix colour scale +-%
CROSS_GRID = [('mIoU', 'resnet', 'cal'), ('mIoU', 'swin_t', 'nocal'),
              ('mAP', 'resnet', 'cal'), ('mAP', 'swin_t', 'nocal'), ('mAP', 'swin_l', 'nocal')]
