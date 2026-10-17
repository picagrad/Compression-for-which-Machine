from compressai.ops import quantize_ste
from torch import Tensor, tensor

from typing import Any, Dict, List, Tuple


def scale_quantize_ste(x: Tensor, scale: Tensor = tensor(1.), unscale: Tensor = tensor(1.)) -> Tensor:
    """
    Rounding with non-zero gradients. Gradients are approximated by replacing
    the derivative by the identity function.

    Used in `"Lossy Image Compression with Compressive Autoencoders"
    <https://arxiv.org/abs/1703.00395>`_

    .. note::

        Implemented with the pytorch `detach()` reparametrization trick:

        `x_round = x_round - x.detach() + x`
    """

    assert scale.shape == unscale.shape, 'Scale and unscale must be inverse of each other and thus have the same shape'
    if scale.numel() == 1:
        return quantize_ste(x * scale) * unscale
    elif scale.numel() == x.shape[0]:
        return quantize_ste(x * scale.view(-1, 1, 1, 1)) * unscale.view(-1, 1, 1, 1)
    else:
        raise ValueError('Scale and unscale must be either scalar or have the the same length as the batch size')

    return quantize_ste(x * scale) * unscale


   
def checkerboard_forward_twopass(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
    """Drop-in replacement for compressai's CheckerboardLatentCodec._forward_twopass.

    Identical to the original except quantize_ste is replaced with scale_quantize_ste
    so that self.scale / self.unscale (set by QVRFELIC.set_scale) are applied to both
    the anchor and non-anchor passes during training.
    """
    B, C, H, W = y.shape
    params = y.new_zeros((B, C * 2, H, W))
    y_hat_ = []
    for step in ("anchor", "non_anchor"):
        if step == "anchor":
            y_ctx = self._y_ctx_zero(y)
        else:
            y_hat_anchors = y_hat_[0]
            y_ctx = self.context_prediction(y_hat_anchors)
        params_i = self.entropy_parameters(self.merge(y_ctx, side_params))
        self._copy(params, params_i, step)
        params_i = self._keep_only(params_i, step)
        y_i = self._keep_only(y, step)
        _, means_i = self.latent_codec["y"]._chunk(params_i)
        y_hat_i = self._keep_only(scale_quantize_ste(y_i - means_i, self.scale, self.unscale) + means_i, step)
        y_hat_.append(y_hat_i)
    [y_hat_anchors, y_hat_non_anchors] = y_hat_
    y_hat = y_hat_anchors + y_hat_non_anchors
    gaussian_params = self.latent_codec["y"].entropy_parameters(params)
    scales_hat, means_hat = self.latent_codec["y"]._chunk(gaussian_params)
    scale = self.scale.view(-1, 1, 1, 1) if self.scale.numel() > 1 else self.scale
    _, y_likelihoods = self.latent_codec["y"].gaussian_conditional(
        (y - means_hat) * scale, scales_hat * scale
    )
    return {"likelihoods": {"y": y_likelihoods}, "y_hat": y_hat}

def gaussian_conditional_compress(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = self._chunk(gaussian_params)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * self.scale)
        y_strings = self.gaussian_conditional.compress((y-means_hat)*self.scale, indexes, 0)
        y_hat = self.gaussian_conditional.decompress(
            y_strings, indexes) * self.unscale + means_hat
        return {"strings": [y_strings], "shape": y.shape[2:4], "y_hat": y_hat}

def gaussian_conditional_decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, int],
        ctx_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        (y_strings,) = strings
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = self._chunk(gaussian_params)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * self.scale)
        y_hat = self.gaussian_conditional.decompress(
            y_strings, indexes) * self.unscale + means_hat
        assert y_hat.shape[2:4] == shape
        return {"y_hat": y_hat}
