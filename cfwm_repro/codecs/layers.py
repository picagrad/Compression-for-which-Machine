import torch
import torch.nn as nn
from torch import Tensor, tensor
from typing import Optional, Tuple

from compressai.layers import conv3x3, conv1x1


EPS = tensor(1e-7)

class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block.

    Introduced by [He2016], this block sandwiches a 3x3 convolution
    between two 1x1 convolutions which reduce and then restore the
    number of channels. This reduces the number of parameters required.

    [He2016]: `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/abs/1512.03385>`_, by Kaiming He, Xiangyu Zhang,
    Shaoqing Ren, and Jian Sun, CVPR 2016.

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)

        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out + identity

class Scaler(nn.Module):
    def __init__(self, lambda_ref: float, n_groups: int = 1, eps: float = EPS) -> None:
        super().__init__()
        if n_groups == 1:
            # Scalar params — preserves checkpoint compatibility with old single-scale models.
            self.a0 = nn.Parameter(tensor(0.))
            self.a1 = nn.Parameter(tensor(1.))
        else:
            # One independent (a0, a1) pair per group.
            self.a0 = nn.Parameter(torch.zeros(n_groups))
            self.a1 = nn.Parameter(torch.ones(n_groups))
        self.n_groups = n_groups
        self.lambda_ref = lambda_ref
        self.eps = float(eps)  # keep as Python float so torch.clamp works on any device

    def forward(self, lmbda: torch.Tensor) -> torch.Tensor:
        ratio = (lmbda / self.lambda_ref) ** 0.5
        # When lmbda is batched (B,) and a0/a1 are per-group (n_groups,), broadcasting would
        # collapse the batch into the group dimension.  Unsqueeze so the result is (B, n_groups).
        # For n_groups == 1, a0/a1 are scalars so no unsqueeze is needed (output stays (B,)).
        if self.n_groups > 1 and ratio.ndim > 0:
            ratio = ratio.unsqueeze(-1)          # (B,) → (B, 1)
        return torch.clamp(self.a0 + self.a1 * ratio, min=self.eps)


    


class UnProject(nn.Module):
    '''
    UnProject is a class that undoes the projection operation at the end of a PatchEmbed Layer performed in a SwinTransformer
    Useful for using regular codecs on B X P X C patch features rather than B X C X H X W image features
    '''
    def __init__(self, hw_shape: Optional[Tuple[int,int]] = None):
        super().__init__()
        self.hw_shape = hw_shape
    
    def forward(self, x):
        assert x.dim()==3, f'Input should be 3D output of SwINTransformerBlock but got {x.dim()} dimensions'
        assert self.hw_shape is not None, 'hw_shape must be set to the shape of the original image features before running the forward pass'
        return x.transpose(1,2).reshape(x.shape[0], -1, *self.hw_shape).contiguous()
