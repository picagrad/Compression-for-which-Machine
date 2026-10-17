import torch
import torch.nn.functional as F
import math
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from typing import Optional
from .utils import DelayedLRScheduler
from cfm_model_zoo.split_utils import SplitResNetFeatures, SplitSwinFeatures, task_model_mode
from cfm_model_zoo.directory import build_split_model
from ..codecs.base_codecs_vr import QVRFELIC
from mmengine.model import BaseDataPreprocessor
from .tracking import tracking_errors
import traceback

rank_zero_print = rank_zero_only(print)

DEFAULT_LAMBDA = 0.0483


def true_num_pixels(data_samples):
    """Original-image pixels the coded content represents (the denominator of bits per pixel).

    The batched tensor's N*H*W would count padding as image. After a RandomCrop the content is
    `img_shape` (which RandomCrop updates) while `scale_factor` still records the resize only, so
    `img_shape_area / scale_factor_area` is the original-image area each sample covers.

    Inputs:
        data_samples: the batch's mmdet/mmseg data samples; each needs 'img_shape' and 'scale_factor'
            in its metainfo (every mm test and train pipeline sets both).
    Outputs:
        The total original-image area of the batch, in pixels.
    """
    total = 0.0
    for d in data_samples:
        meta = d.metainfo
        if 'img_shape' not in meta or 'scale_factor' not in meta:
            raise KeyError("bits per pixel needs 'img_shape' and 'scale_factor' in every data sample's metainfo; "
                           f'got {sorted(meta)}')
        ish = meta['img_shape']
        ish = tuple(int(x) for x in (ish.tolist() if hasattr(ish, 'tolist') else ish))
        sf = meta['scale_factor']
        total += ish[0] * ish[1] / (float(sf[0]) * float(sf[1]))
    return total


class FixedRateCFM(pl.LightningModule):
    """
    Trainer for fixed-rate codecs. Uses a single lambda value throughout.

    The task model stays frozen; the codec is trained on lambda x distortion + bpp, where the distortion
    is the MSE between the task model's backbone features computed from the original features at the cut
    and from the codec's reconstruction.

    Stage semantics (see the ChangeModelStage callback):
       -1  uncompressed baseline: test_step skips the codec and runs the task model's own frontend and
           backend. Never set by ChangeModelStage; a test script sets it around one trainer.test() call.
        0  warm-up: one lambda (lambda_ref), no validation inference, no LR schedule changes
        1  variable-rate warm-up (VariableRateCFM): lambda is sampled per image while the lambda range widens
        2  bulk training: LR schedule, validation inference and checkpointing active
        3  testing: test_step compresses and decompresses for real (entropy-coded bitstreams)

    A fixed-rate run skips stage 1: ChangeModelStage with intermediate_stage_epochs=0 goes from 0 to 2.

    This repository ships only variable-rate codecs (QVRFELIC and its subclasses), which this class refuses: given
    no lambda, they sample one per image while the loss here is weighted by a fixed one. Bring your own
    fixed-rate codec, or use VariableRateCFM.
    """

    accepts_variable_rate_codec = False   # VariableRateCFM: True

    def __init__(
        self,
        task_model_name: str,
        task_model_args: dict,
        codec_cls: type,
        codec_args: dict,
        epsilon: float = 1e-6,
        stage: int = 0,
        val_inference: bool = False,
        lmbda: float = DEFAULT_LAMBDA,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 1e-4,
        sched_dict: Optional[dict] = None,
    ):
        super().__init__()
        if not self.accepts_variable_rate_codec and isinstance(codec_cls, type) and issubclass(codec_cls, QVRFELIC):
            raise TypeError(f'{type(self).__name__} needs a fixed-rate codec; {codec_cls.__name__} is one of this '
                            "repository's variable-rate codecs: use VariableRateCFM, or supply your own fixed-rate codec")
        self.task_model_input = task_model_name
        self.task_model_args = task_model_args
        self.task_model = build_split_model(task_model_name, **task_model_args)

        self.task_model.requires_grad_(False)
        # Freeze every norm that tracks running statistics, here in __init__: the automatic batch-size
        # probe runs forward passes before training starts, and live norms would update their running
        # statistics from its dummy batches. The split model re-applies the freeze on every mode change
        # (cfm_model_zoo.split_utils), including Lightning's own mode switches.
        self.task_model.freeze_norms = True
        # build_split_model returns the model in eval mode; the loss path runs it in train mode (frozen
        # norms), so start there.
        self.task_model.train()
        self.codec = codec_cls(**codec_args)
        self.lambda_ref = nn.Parameter(torch.tensor(lmbda, dtype=torch.float32), requires_grad=False)
        object.__setattr__(self, 'lmbda', self.lambda_ref)  # plain alias, not a second Parameter
        self.epsilon = epsilon
        # The distortion: MSE between the task model's full set of backbone features computed from the
        # original features at the cut and from the codec's reconstruction (see _compute_losses).
        self.distillation_loss = nn.MSELoss()
        self.stage = stage
        self.val_inference = val_inference
        self.updated_entropy_bottlenecks = False
        self.optim = optimizer
        self.lr = lr
        self.sched_dict = sched_dict

    def _apply(self, fn, *args, **kwargs):
        # mmengine's data preprocessor caches its target device in `_device`, and cast_data() moves every batch
        # there. Only mmengine's own .to()/.cuda()/.cpu() update that cache; nn.Module._apply -- which is how
        # LightningModule.to(), the strategy's model_to_device(), teardown's .cpu() and the batch-size probe reach
        # children -- moves the preprocessor's buffers but not the cache. Re-sync it after every move, so the
        # preprocessor always follows the module.
        module = super()._apply(fn, *args, **kwargs)
        device = next(self.codec.parameters()).device     # read off what was moved
        for m in self.task_model.modules():
            if isinstance(m, BaseDataPreprocessor):
                m._device = device
        return module

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _module_spatial_downsample_factor(module: nn.Module) -> int:
        """Total spatial downsampling factor of a submodule, from its stride-2
        Conv2d/ConvTranspose2d layers.
        """
        factor = 1
        for m in module.modules():
            if isinstance(m, nn.Conv2d) and m.stride[0] > 1:
                factor *= m.stride[0]
            elif isinstance(m, nn.ConvTranspose2d) and m.stride[0] > 1:
                factor //= m.stride[0]
        return factor

    def _codec_spatial_divisor(self):
        """The multiple the codec's input height and width must be padded to.

        The analysis transform g_a and the hyper-analysis transform h_a both downsample with stride-2
        convolutions; the input must be divisible by the product of their factors so that every stride-2
        stage sees even dimensions. Our codecs use 4 in each (a divisor of 16), read here from the modules
        themselves. ELIC-style codecs keep h_a inside their latent codec (codec.latent_codec.hyper.h_a);
        a codec with neither attribute is an error, since guessing the divisor would pad it wrongly.

        Outputs:
            The divisor (an int).
        """
        g_a = getattr(self.codec, 'g_a', None)
        h_a = getattr(self.codec, 'h_a', None)
        if h_a is None:
            try:
                h_a = self.codec.get_submodule('latent_codec.hyper.h_a')
            except AttributeError:
                h_a = None
        if not isinstance(g_a, nn.Module) or not isinstance(h_a, nn.Module):
            raise AttributeError(f'{type(self.codec).__name__}: the padding divisor needs the analysis transform '
                                 'codec.g_a and the hyper-analysis transform (codec.h_a or '
                                 'codec.latent_codec.hyper.h_a)')
        return self._module_spatial_downsample_factor(g_a) * self._module_spatial_downsample_factor(h_a)

    def _apply_calibration(self, feat_tensor):
        """Optional cross-model feature calibration, applied to the codec's input (ConvNet (B, C, H, W) only).

        Only test_cross_model.py sets `self.calibration` (a test_common.FeatureCalibration: anything with
        forward(x) and invert(x_hat)); without it this is a no-op.
        """
        calibration = getattr(self, 'calibration', None)
        return feat_tensor if calibration is None else calibration.forward(feat_tensor)

    def _invert_calibration(self, x_hat):
        """Inverse of _apply_calibration, applied to the codec's reconstruction before the task model's backend."""
        calibration = getattr(self, 'calibration', None)
        return x_hat if calibration is None else calibration.invert(x_hat)

    def _prepare_codec_input(self, feat_tensor):
        """Calibrate (if enabled), then pad feat_tensor spatially to the codec's required
        multiple to prevent checkerboard shape mismatches (see _codec_spatial_divisor for
        the derivation). Every codec call (_compute_codec_output, _compress_decompress) goes through here.

        Returns (prepared_feat_tensor, pad_info) where pad_info encodes what to undo in
        _finalize_codec_output. pad_info is None if no padding was needed.
        """
        return self._pad_feat_tensor(self._apply_calibration(feat_tensor))

    def _pad_feat_tensor(self, feat_tensor):
        """Pad feat_tensor spatially to the codec's required multiple to prevent checkerboard
        shape mismatches (see _codec_spatial_divisor for the derivation).

        Returns (padded_feat_tensor, pad_info) where pad_info encodes what to undo.
        pad_info is None if no padding was needed.

        SWIN codecs: feat_tensor is (B, N_h*N_w, D); padded by reshaping to spatial grid.
        Spatial codecs: feat_tensor is (B, C, H, W); padded with F.pad.
        """
        divisor = self._codec_spatial_divisor()
        if hasattr(self.codec, 'unproject') and self.codec.unproject.hw_shape is not None:
            N_h, N_w = self.codec.unproject.hw_shape
            N_h_pad = math.ceil(N_h / divisor) * divisor
            N_w_pad = math.ceil(N_w / divisor) * divisor
            if N_h_pad == N_h and N_w_pad == N_w:
                return feat_tensor, None
            B, _N, D = feat_tensor.shape
            x = feat_tensor.view(B, N_h, N_w, D)
            x = F.pad(x, (0, 0, 0, N_w_pad - N_w, 0, N_h_pad - N_h))
            self.codec.unproject.hw_shape = (N_h_pad, N_w_pad)
            return x.reshape(B, N_h_pad * N_w_pad, D), ('swin', N_h, N_w, N_h_pad, N_w_pad)
        elif feat_tensor.dim() == 4:
            h, w = feat_tensor.shape[2], feat_tensor.shape[3]
            pad_h = (-h) % divisor
            pad_w = (-w) % divisor
            if pad_h == 0 and pad_w == 0:
                return feat_tensor, None
            return F.pad(feat_tensor, [0, pad_w, 0, pad_h]), ('spatial', h, w, pad_h, pad_w)
        return feat_tensor, None

    def _finalize_codec_output(self, comp_out, pad_info):
        """Crop x_hat back to its original extent (_unpad_codec_output), then invert the calibration
        (if any): the counterpart of _prepare_codec_input.
        """
        comp_out = self._unpad_codec_output(comp_out, pad_info)
        if 'x_hat' in comp_out:
            comp_out = {**comp_out, 'x_hat': self._invert_calibration(comp_out['x_hat'])}
        return comp_out

    def _unpad_codec_output(self, comp_out, pad_info):
        """Crop x_hat back to the original spatial extent, reversing _pad_feat_tensor."""
        if pad_info is None:
            return comp_out
        if pad_info[0] == 'swin':
            _, N_h, N_w, N_h_pad, N_w_pad = pad_info
            self.codec.unproject.hw_shape = (N_h, N_w)  # restore before any early return
            N_h_out, N_w_out = comp_out['hw_shape']
            # Infer effective spatial expansion ratio (handles shuffle_ratio != 1).
            # round() works because the PatchEmbed offset (-3) is small vs N_h_pad (>= 64).
            r_h = max(round(N_h_out / N_h_pad), 1)
            r_w = max(round(N_w_out / N_w_pad), 1)
            N_h_out_orig = N_h_out - (N_h_pad - N_h) * r_h
            N_w_out_orig = N_w_out - (N_w_pad - N_w) * r_w
            B = comp_out['x_hat'].shape[0]
            x_hat = comp_out['x_hat'].view(B, N_h_out, N_w_out, -1)
            x_hat = x_hat[:, :N_h_out_orig, :N_w_out_orig, :].reshape(B, N_h_out_orig * N_w_out_orig, -1)
            return {**comp_out, 'x_hat': x_hat, 'hw_shape': (N_h_out_orig, N_w_out_orig)}
        if pad_info[0] == 'spatial':
            _, h, w, pad_h, pad_w = pad_info
            x_hat = comp_out.get('x_hat')
            if x_hat is not None and x_hat.dim() == 4:
                H_out, W_out = x_hat.shape[2], x_hat.shape[3]
                r_h = max(round(H_out / (h + pad_h)), 1)
                r_w = max(round(W_out / (w + pad_w)), 1)
                return {**comp_out, 'x_hat': x_hat[:, :, :H_out - pad_h * r_h, :W_out - pad_w * r_w]}
        return comp_out

    def _compute_codec_output(self, feat_tensor, mode: str = 'train') -> dict:
        """Run the codec forward pass, with spatial padding to avoid checkerboard mismatches."""
        feat_tensor, pad_info = self._prepare_codec_input(feat_tensor)
        return self._finalize_codec_output(self.codec(feat_tensor), pad_info)

    def _compress_decompress(self, feat_tensor):
        """Run compress then decompress for stage-3 test. Returns (comp_out, decomp_out)."""
        feat_tensor, pad_info = self._prepare_codec_input(feat_tensor)
        comp_out = self.codec.compress(feat_tensor)
        decomp_out = self.codec.decompress(comp_out['strings'], comp_out['shape'])
        decomp_out = self._finalize_codec_output(decomp_out, pad_info)
        return comp_out, decomp_out

    # ------------------------------------------------------------------
    # Loss computation (scalar losses; FixedRateCFM is always stage 0)
    # ------------------------------------------------------------------

    def _compute_losses(self, comp_out, data, feats, mode):
        """The training loss: lambda x distortion + bpp.

        Inputs:
            comp_out: the codec's forward output (x_hat, likelihoods, ...).
            data: the preprocessed batch (its data_samples give the bpp denominator).
            feats: the frontend output at the cut (the distortion target).
            mode: 'train', 'val' or 'test'.
        Outputs:
            (loss, distortion, bpp_loss, z_bpp_loss); z_bpp_loss is the hyperprior's share of bpp_loss.
        """
        try:
            backend_input = self._make_backend_input(comp_out, data)
            distill_target = self.task_model.feature_backbone(feats)
            distill_hat = self.task_model.feature_backbone(backend_input)
            if isinstance(distill_target, (tuple, list)):
                distillation_loss = sum([self.distillation_loss(d_h, d_t) for d_h, d_t in zip(distill_hat, distill_target)]) / len(distill_hat)
            else:
                distillation_loss = self.distillation_loss(distill_hat, distill_target)

        except RuntimeError as e:
            print('Got a Runtime error calculating losses, most likely culprits are the codec strides ("strides","analysis_strides" and/or "synthesis_strides") or feature_chs/embed_dims')
            raise e

        num_pixels = true_num_pixels(data['data_samples'])
        bpp_loss = sum(
            torch.log(lkl + self.epsilon).sum() / (-math.log(2) * num_pixels)
            for lkl in comp_out['likelihoods'].values()
        )
        z_bpp_loss = (torch.log(comp_out['likelihoods']['z'] + self.epsilon).sum() / (-math.log(2) * num_pixels)) if 'z' in comp_out['likelihoods'] else 0

        loss = self.lmbda * distillation_loss + bpp_loss
        return loss, distillation_loss, bpp_loss, z_bpp_loss

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.codec(x)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return {k: v for k, v in state_dict.items() if not k.startswith('task_model')}

    def load_state_dict(self, state_dict, strict=True):
        state_dict = {k.removeprefix('codec.'): v for k, v in state_dict.items()}
        # Load module-level lambda_ref if present (only for fixed-rate; variable-rate codec owns it)
        if 'lambda_ref' in state_dict and 'lambda_ref' not in self.codec.state_dict():
            with torch.no_grad():
                self.lambda_ref.copy_(state_dict.pop('lambda_ref'))
        # A checkpoint saved after update_codec_bottlenecks() holds built entropy tables; the codec's own
        # load_state_dict (compressai's CompressionModel) resizes its table buffers to match.
        self.codec.load_state_dict(state_dict, strict=strict)

    def _prepare_batch(self, batch, mode: str):
        """Preprocess the batch and run the frontend up to the cut.

        Outputs:
            (data, feats, feat_tensor, N): the preprocessed batch, the frontend output, the cut-point
            feature tensor the codec codes, and the batch size.
        """
        with torch.no_grad():
            data = self.task_model.frontend_preprocessor(batch, mode == 'train')
            N = data['inputs'].shape[0]
        with torch.no_grad():
            feats = self.task_model.feature_frontend(data, preprocessed=True)
            if isinstance(feats['inputs'], SplitSwinFeatures):
                if hasattr(self.codec, 'unproject'):
                    self.codec.unproject.hw_shape = feats['inputs'].hw_shape
        feat_tensor = feats['inputs'].outs[0]
        return data, feats, feat_tensor, N

    def _log_step_losses(self, mode: str, distortion, bpp_loss, z_bpp_loss, loss, N: int):
        """Log {mode}_bpp_loss, {mode}_z_bpp_loss (the hyperprior's share of the rate), {mode}_loss and
        {mode}_distortion_loss (lambda x distortion, averaged over the batch).

        Training logs every step (thinned by log_every_n_steps) and the epoch mean; validation and test log
        the epoch mean only, under the bare name ('val_loss'), since logging every validation step to a
        tracking server is slow.
        """
        on_step = mode == 'train'
        try:
            self.log(f'{mode}_bpp_loss', bpp_loss, on_step=on_step, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=N)
            self.log(f'{mode}_z_bpp_loss', z_bpp_loss, on_step=on_step, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=N)
            self.log(f'{mode}_loss', loss, on_step=on_step, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=N)
            self.log(f'{mode}_distortion_loss', (self.lmbda * distortion).mean(), on_step=on_step, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=N)
        except tracking_errors():
            print(f'Could not log to mlflow, continuing without logging. For record purposes this is the error trace: {traceback.format_exc()}')

    def _make_backend_input(self, comp_out, data):
        """Wrap the codec output in the container the task model's backend expects.

        SWIN codecs carry an 'hw_shape'; conv codecs do not.
        """
        if 'hw_shape' in comp_out:
            feats = SplitSwinFeatures(outs=[comp_out['x_hat']],
                                      hw_shape=comp_out['hw_shape'], device=self.device)
        else:
            feats = SplitResNetFeatures(outs=[comp_out['x_hat']], device=self.device)
        return {'inputs': feats, 'data_samples': data['data_samples']}

    def _run_inference(self, comp_out, data):
        """Metric forward: the task model in true eval mode (dropout off, norms on running statistics).

        task_model_mode restores the previous mode on exit, also if anything raises, and the split model
        re-applies the norm freeze on that mode change. The predictions go back to the caller; the task metric
        is computed from them by the ModelEvaluator callback.
        """
        with task_model_mode(self.task_model, training=False):
            return self.task_model.backend(self._make_backend_input(comp_out, data), mode='inference')

    def _step_return(self, mode: str, loss, comp_out, data):
        """Run backend inference (if applicable) and return the step result.

        A validation step with inference and a test step also return the predictions ('out', 'data_samples'),
        which the ModelEvaluator callback passes to its evaluators.
        """
        if mode == 'train':
            return loss
        elif 'val' in mode:
            if self.val_inference and self.stage >= 1:
                self.val_batch_size = len(data['data_samples'])
                out = self._run_inference(comp_out, data)
                return {'loss': loss, 'out': out, 'data_samples': data['data_samples']}
            return loss
        elif mode == 'test':
            self.test_batch_size = len(data['data_samples'])
            out = self._run_inference(comp_out, data)
            return loss, {'loss': loss, 'out': out, 'data_samples': data['data_samples']}
        else:
            raise ValueError(f'mode must be one of train, val, or test. Got {mode} instead')

    def training_step(self, batch, batch_idx, dataloader_idx=0, mode: str = 'train'):
        """One loss step (also used by validation, and by test before stage 3).

        The whole loss path runs the frozen task model in train mode with its norms frozen (and requires_grad=False):
          * train mode, because some heads branch on self.training inside loss() (RepPointsV2, KNet,
            DDQ-DETR, VFNet, GuidedAnchor), so an eval-mode loss would be a different computation;
          * frozen norms, because a train-mode BatchNorm fails on UperNet's [1, 512, 1, 1] global-pooling
            branch (a batch variance over one element).
        For validation and test this leaves Lightning's eval mode for the loss and restores it on exit, so
        the inference forward in _step_return still runs in true eval mode.
        Keep the whole loss path (_prepare_batch, _compute_codec_output, _compute_losses)
        inside the task_model_mode(training=True) block.
        """
        with task_model_mode(self.task_model, training=True):
            data, feats, feat_tensor, N = self._prepare_batch(batch, mode)
            comp_out = self._compute_codec_output(feat_tensor, mode=mode)
            loss, distortion, bpp_loss, z_bpp_loss = self._compute_losses(comp_out, data, feats, mode)
        self._log_step_losses(mode, distortion, bpp_loss, z_bpp_loss, loss, N)
        return self._step_return(mode, loss, comp_out, data)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.training_step(batch, batch_idx, mode='val')

    def update_codec_bottlenecks(self, update_quantiles: bool = False):
        """Build the codec's entropy-coding tables (needed before compress/decompress).

        Keep update_quantiles False (compressai's default): the hyper-decoder was trained on z rounded
        around the trained medians (no auxiliary loss; the hyper codec quantises with a straight-through
        estimator), and re-fitting the medians moves that lattice, which raises the real rate without
        improving accuracy.
        """
        self.codec.update(update_quantiles=update_quantiles)
        self.updated_entropy_bottlenecks = True

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.stage == -1:
            # Uncompressed baseline: run the task model's own frontend and backend, without the codec.
            # It goes through trainer.test() on the same dataloader as the compressed sweep: a separate
            # loop over the dataloader can deadlock for models that load a HuggingFace tokenizer (GLIP,
            # GroundingDINO) when the dataloader first forks its workers.
            self.task_model.eval()
            with torch.no_grad():
                data = self.task_model.frontend_preprocessor(batch, False)
                feats = self.task_model.feature_frontend(data, preprocessed=True)
                out = self.task_model.backend(feats, mode='inference')
            loss = torch.tensor(0.0)
            return loss, {'loss': loss, 'out': out, 'data_samples': data['data_samples']}
        elif self.stage < 3:
            return self.training_step(batch, batch_idx, mode='test')
        else:
            # The entropy tables are built here, on the device the test runs on (the GPU
            # for a sweep). Tables built on the CPU differ in a few rows, so a sweep's bitstreams depend on it.
            if not self.updated_entropy_bottlenecks:
                self.update_codec_bottlenecks()
            self.task_model.eval()
            assert len(batch['data_samples']) == 1, 'Test batch size must be 1'
            with torch.no_grad():
                data = self.task_model.frontend_preprocessor(batch, False)
                num_pixels = true_num_pixels(data['data_samples'])
            feats = self.task_model.feature_frontend(data, preprocessed=True)
            if isinstance(feats['inputs'], SplitSwinFeatures):
                if hasattr(self.codec, 'unproject'):
                    self.codec.unproject.hw_shape = feats['inputs'].hw_shape
            feat_tensor = feats['inputs'].outs[0]
            comp_out, decomp_out = self._compress_decompress(feat_tensor)
            if 'hw_shape' in decomp_out:
                backend_input = {'inputs': SplitSwinFeatures(outs=[decomp_out['x_hat']], hw_shape=decomp_out['hw_shape'], device=self.device), 'data_samples': data['data_samples']}
            else:
                backend_input = {'inputs': SplitResNetFeatures(outs=[decomp_out['x_hat']], device=self.device), 'data_samples': data['data_samples']}
            out = self.task_model.backend(backend_input, mode='inference')
            loss = torch.tensor(0.0)
            try:
                bpp_val = (sum(len(s[0]) for s in comp_out['strings']) * 8 + 32 + 24 * len(comp_out['strings'])) / num_pixels
                self.log(f'full_test_bpp_{self.lmbda.item():.3f}', bpp_val, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=1)
            except tracking_errors():
                print(f'Could not log to mlflow, continuing without logging. For record purposes this is the error trace: {traceback.format_exc()}')
            strings = comp_out.get('strings')
            # The bitstream is returned too, for callers that want to inspect or save it (ModelEvaluator does not use it).
            return loss, {'loss': loss, 'out': out, 'data_samples': data['data_samples'], 'strings': strings}

    @property
    def val_monitor_key(self) -> str:
        """Metric that ReduceLROnPlateau tracks: the plain epoch-mean val_loss."""
        return 'val_loss'

    def configure_optimizers(self, lr: Optional[float] = None):
        # Lightning calls this with no arguments: fall back to self.lr (set from --lr).
        lr = self.lr if lr is None else lr
        optimizer = self.optim(self.codec.parameters(), lr=lr)
        if self.sched_dict is not None:
            # Keys starting with '_' are directives for this builder, not scheduler kwargs.
            # '_active_stage' = the stage at which the schedule starts running; 0 means "from
            # epoch 0", in which case we do NOT wrap at all (DelayedLRScheduler exists purely
            # to hold a scheduler dormant, and it also rebases the epoch counter on activation,
            # which would corrupt a whole-run schedule such as WSD).
            kwargs = {k: v for k, v in self.sched_dict.items() if not k.startswith('_')}
            scheduler_class = kwargs.pop('scheduler')
            active_stage = int(self.sched_dict.get('_active_stage', 2))
            if active_stage <= 0:
                delayed_scheduler = scheduler_class(optimizer, **kwargs)
            else:
                delayed_scheduler = DelayedLRScheduler(0, active_stage)(scheduler_class)(
                    optimizer, **kwargs)
            if scheduler_class == torch.optim.lr_scheduler.ReduceLROnPlateau:
                return [optimizer], [{"scheduler": delayed_scheduler,
                                      "monitor": self.val_monitor_key, "interval": "epoch"}]
            return [optimizer], [{"scheduler": delayed_scheduler, "interval": "epoch"}]
        return optimizer


class VariableRateCFM(FixedRateCFM):
    """
    Trainer for variable-rate codecs (QVRF / GQVRF). Inherits from FixedRateCFM
    and overrides codec call hooks to pass lambda, and resets lambda before
    validation and test.
    """

    accepts_variable_rate_codec = True

    def __init__(self, task_model_name, task_model_args, codec_cls, codec_args, n_val_lambdas: int = 10, **kwargs):
        super().__init__(task_model_name, task_model_args, codec_cls, codec_args, **kwargs)
        # Codec owns lambda_ref — drop the module-level lambda_ref from FixedRateCFM
        del self._parameters['lambda_ref']
        self.lmbda = torch.as_tensor(self.codec.lambda_ref.detach(), dtype=torch.float32)
        self.variable_rate = True
        # Per-image distortion: from stage 1 on, every image in a batch has its own lambda.
        self.distillation_loss = nn.MSELoss(reduction='none')
        # Validation cycles through a fixed log-spaced set of lambdas (by batch_idx).
        # Inference (AP) is handled by PeriodicInferenceValidation / ChangeModelStage.
        self.val_inference = False
        # lmbda starts at lambda_ref, and the validation lambda grid is fixed here, from the
        # codec's target lambda range, before LambdaRangeWarmup narrows codec.lambda_ratio and widens it again.
        lm_min = float(self.codec.lambda_ref.item() * self.codec.lambda_ratio[0])
        lm_max = float(self.codec.lambda_ref.item() * self.codec.lambda_ratio[1])
        self._val_lambdas = torch.exp(torch.linspace(math.log(lm_min), math.log(lm_max), n_val_lambdas))

    def _compute_codec_output(self, feat_tensor, mode: str = 'train') -> dict:
        # At stage 0 or during val/test, pass the current lmbda to the codec.
        # At stage > 0 during training, pass None so the codec samples lambda.
        lmbda_arg = self.lmbda if (self.stage == 0 or mode in ('val', 'test')) else None
        feat_tensor, pad_info = self._prepare_codec_input(feat_tensor)
        return self._finalize_codec_output(self.codec(feat_tensor, lmbda_arg), pad_info)

    def _compress_decompress(self, feat_tensor):
        feat_tensor, pad_info = self._prepare_codec_input(feat_tensor)
        comp_out = self.codec.compress(feat_tensor, lmbda=self.lmbda)
        decomp_out = self.codec.decompress(comp_out['strings'], comp_out['shape'], lmbda=self.lmbda)
        decomp_out = self._finalize_codec_output(decomp_out, pad_info)
        return comp_out, decomp_out

    def _compute_losses(self, comp_out, data, feats, mode):
        """As FixedRateCFM._compute_losses, with a per-image distortion: lmbda is one value at stage 0 and in
        validation and test, and the codec's per-image sampled values from stage 1 on (returned in comp_out).
        """
        if 'lmbda' in comp_out:
            self.lmbda = comp_out['lmbda']

        try:
            backend_input = self._make_backend_input(comp_out, data)
            # Per-image distortion (reduction='none'), averaged over the batch at stage 0 (one lambda).
            distill_target = self.task_model.feature_backbone(feats)
            distill_hat = self.task_model.feature_backbone(backend_input)
            if isinstance(distill_target, (tuple, list)):
                distill_losses = [self.distillation_loss(d_h, d_t).flatten(start_dim=1).mean(dim=1) for d_h, d_t in zip(distill_hat, distill_target)]
                distillation_loss = sum(distill_losses) / len(distill_losses)
            else:
                distillation_loss = self.distillation_loss(distill_hat, distill_target).flatten(start_dim=1).mean(dim=1)
            if self.stage == 0:
                distillation_loss = distillation_loss.mean()

        except RuntimeError as e:
            print('Got a Runtime error calculating losses, most likely culprits are the codec strides ("strides","analysis_strides" and/or "synthesis_strides") or feature_chs/embed_dims')
            raise e

        num_pixels = true_num_pixels(data['data_samples'])
        bpp_loss = sum(
            torch.log(lkl + self.epsilon).sum() / (-math.log(2) * num_pixels)
            for lkl in comp_out['likelihoods'].values()
        )
        z_bpp_loss = (torch.log(comp_out['likelihoods']['z'] + self.epsilon).sum() / (-math.log(2) * num_pixels)) if 'z' in comp_out['likelihoods'] else 0

        # lmbda is per-image ([N], sampled in training) or one scalar (stage 0, val, test); broadcasting
        # covers both, and a single element's mean is the element itself.
        loss = (self.lmbda * distillation_loss).mean() + bpp_loss
        return loss, distillation_loss, bpp_loss, z_bpp_loss

    def load_state_dict(self, state_dict, strict=True):
        state_dict = {k.removeprefix('codec.'): v for k, v in state_dict.items()}
        # QVRF key fixup: add lambda_ref if loading from a checkpoint that predates it being
        # registered as a parameter (only relevant for QVRF variable-rate codecs)
        named_params = dict(self.codec.named_parameters())
        if 'QVRF' in self.codec.__class__.__name__:
            if 'lambda_ref' not in state_dict and 'lambda_ref' in named_params:
                state_dict['lambda_ref'] = self.codec.lambda_ref
            if 'scaler.lambda_ref' not in state_dict and 'lambda_ref' in named_params:
                state_dict['scaler.lambda_ref'] = state_dict['lambda_ref']
        super().load_state_dict(state_dict, strict=strict)

    def _reset_lmbda(self):
        """Reset lmbda to the codec's reference value (called before val and test)."""
        self.lmbda = torch.as_tensor(self.codec.lambda_ref.detach(), dtype=torch.float32)

    @property
    def val_monitor_key(self) -> str:
        """Metric that ReduceLROnPlateau tracks: the plain epoch-mean val_loss."""
        return 'val_loss'

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Pass 1: cycling lambda — regular loss and logging (always runs).
        lmbda_idx = batch_idx % len(self._val_lambdas)
        self.lmbda = self._val_lambdas[lmbda_idx]
        val_inf = self.val_inference
        self.val_inference = False
        loss = super().validation_step(batch, batch_idx, dataloader_idx)
        self.val_inference = val_inf

        # Pass 2: reference lambda — inference metrics + bpp at the fixed lambda (not the cycling average).
        # lmbda is left at the reference value after this so ModelEvaluator labels correctly.
        if val_inf:
            self._reset_lmbda()
            if batch_idx == 0:
                rank_zero_print(f'Running inference validation pass (lambda={float(self.lmbda):.3f}) ...')
            data, feats, feat_tensor, N = self._prepare_batch(batch, 'val')
            comp_out = self._compute_codec_output(feat_tensor, mode='val')
            num_pixels = true_num_pixels(data['data_samples'])
            bpp_inference = sum(
                torch.log(lkl + self.epsilon).sum() / (-math.log(2) * num_pixels)
                for lkl in comp_out['likelihoods'].values()
            )
            try:
                self.log('val_bpp_loss_inference', bpp_inference, on_step=False, on_epoch=True,
                         prog_bar=False, logger=True, sync_dist=True, batch_size=N)
            except tracking_errors():
                print(f'Could not log val_bpp_loss_inference to mlflow: {traceback.format_exc()}')
            return self._step_return('val', loss, comp_out, data)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.stage < 3:
            self._reset_lmbda()
        return super().test_step(batch, batch_idx, dataloader_idx)

