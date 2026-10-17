"""Display names, colours and line styles shared by the figures and tables."""

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False



ANCHOR_ORDER = ('vtm', 'vvenc', 'bpg', 'jpeg')


ANCHOR_LABEL = {'vtm': 'VTM', 'vvenc': 'VVenC', 'bpg': 'BPG', 'jpeg': 'JPEG',
                'vtm_qpa1': 'VTM (QPA)', 'vvenc_qpa1': 'VVenC (QPA)'}


ANCHOR_STYLE = {'vtm': dict(color='0.0', linestyle='-', marker='d', markerfacecolor='none'),
                'vtm_qpa1': dict(color='0.0', linestyle='-', marker='d', markerfacecolor='none'),
                'vvenc_qpa1': dict(color='0.25', linestyle='--', marker='x'),
                'vvenc': dict(color='0.25', linestyle='--', marker='x'),
                'bpg': dict(color='0.45', linestyle=':', marker='+'),
                'jpeg': dict(color='0.65', linestyle='-.', marker='1')}


_STYLE_COLORS = plt.get_cmap('tab10').colors if MATPLOTLIB_AVAILABLE else ()


_STYLE_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


_STYLE_LINESTYLES = ['-', '--', '-.', ':']


SUMMARY_FONT_SCALE = 1.35  # summary (mean-curve) figures: larger fonts, legible at half text width


SUMMARY_FIG_SIZE = (10, 5)  # summary figures: landscape, to save vertical space


TASK_SHORT = {'mIoU': 'seg', 'mAP': 'det'}


_TASK_NAME = {'mIoU': 'segmentation', 'mAP': 'detection'}





_CODEC_FAMILY_NAME = {'elfc': 'ELFC'}


_VAR_RATE_NAME = {'gqvrf': 'GQVRF', 'qvrf': 'QVRF'}


QUALITY_FRACTIONS = ((0.95, 'tab:green'), (0.90, 'tab:red'))   # dashed reference lines at these fractions of uncompressed


def our_codec_label(run: dict) -> str | None:
    """Name our codec in a run as the paper does, e.g. 'ELFC-GQVRF': the codec family (_CODEC_FAMILY_NAME) and
    the variable-rate scheme (_VAR_RATE_NAME) from its training parameters.

    Inputs:
        run: run dict with 'train_params' holding 'codec_family' and 'var_rate'.

    Outputs:
        label: '<family>-<variable-rate scheme>'.
        fallback: None when either is missing or not listed; the caller then keeps the run's display_name
            rather than mislabel it.
    """
    # the codec family and the variable-rate scheme, each by its paper name
    params = run.get('train_params') or {}
    fam, var_rate = _CODEC_FAMILY_NAME.get(params.get('codec_family')), _VAR_RATE_NAME.get(params.get('var_rate'))
    return f'{fam}-{var_rate}' if fam and var_rate else None


BACKBONE_COLOURS = ('#0B2E59', '#A0522D', '#6B6B6B', '#7B5BA6', '#7F6A4F')   # per backbone group, in order: navy, sienna, grey, then spares


MEAN_STYLE = {'ours': dict(linestyle='-', linewidth=2.6),
                     'vtm_qpa1': dict(linestyle=(0, (4, 2)), linewidth=1.8), 'vtm': dict(linestyle=(0, (4, 2)), linewidth=1.8),
                     'vvenc_qpa1': dict(linestyle=(0, (10, 3)), linewidth=1.8),
                     'vvenc': dict(linestyle=(0, (10, 3)), linewidth=1.8),
                     'bpg': dict(linestyle=':', linewidth=1.8), 'jpeg': dict(linestyle='-.', linewidth=1.8)}
