import inspect
from compressai.models import Elic2022Official
from mmdet.models.backbones.swin import PatchEmbed
from compressai.layers import AttentionBlock
from compressai.models.utils import deconv, conv
from torch import nn
from torch import Tensor
from typing import List, Optional, Dict, Any, Union
from .layers import ResidualBottleneckBlock, UnProject

from .base_codecs_vr import QVRFELIC


def make_elic_syn_block(in_chs:int , out_chs:int, stride:int = 2, atten_loc:int = 0, num_residuals:int = 3):
    '''
    Make a single ELIC synthesis transform block

    Args:
        in_chs (int): Number of input channels
        out_chs (int): Number of output channels
        stride (int): Stride of the deconvolutional layer
        atten_loc (int): Location of the attention block: -1 for before the deconvolution, 1 for after the deconvolution, 0 for no attention block.
        num_residuals (int): Number of residual blocks
    '''
    assert atten_loc in [-1,0,1], f'atten_loc must be -1, 0 or 1 but got {atten_loc}'
    if atten_loc == -1:
        return nn.Sequential(
            AttentionBlock(in_chs),
            deconv(in_chs, out_chs, kernel_size=5, stride=stride),
            *[ResidualBottleneckBlock(out_chs, out_chs) for i in range(num_residuals)],
        )
    if atten_loc == 1:
        return nn.Sequential(
            deconv(in_chs, out_chs, kernel_size=5, stride=stride),
            AttentionBlock(out_chs),
            *[ResidualBottleneckBlock(out_chs, out_chs) for i in range(num_residuals)],
        )
    return nn.Sequential(deconv(in_chs, out_chs, kernel_size=5, stride=stride),
                            *[ResidualBottleneckBlock(out_chs, out_chs) for i in range(num_residuals)])

def make_elic_syn(strides:List[int], in_chs:int, mid_chs:int, out_chs:int):
    '''
    Make the entire ELIC synthesis transform

    Args:
        strides (List[int]): List of strides for each deconvolutional layer
    '''
    if len(strides) == 1:
        return make_elic_syn_block(in_chs, out_chs, stride = strides[0], atten_loc=-1)
    
    return nn.Sequential(
        *[make_elic_syn_block(in_chs if i==0 else mid_chs, mid_chs, stride, atten_loc = (2*i-1)*(i<=1), ) for i,stride in enumerate(strides[:-1])],
        deconv(mid_chs, out_chs, kernel_size=5, stride=strides[-1])
    )


def make_elic_analysis_block(in_chs:int , out_chs:int, stride:int = 2, with_atten:bool = False, num_residuals:int = 3):
    '''
    Make a single ELIC analysis transform block

    Args:
        in_chs (int): Number of input channels
        out_chs (int): Number of output channels
        stride (int): Stride of the strided convolution
        with_atten (bool): put an AttentionBlock on the block's input, before the strided convolution
        num_residuals (int): Number of residual blocks
    '''
    return nn.Sequential(
            AttentionBlock(in_chs) if with_atten else nn.Identity(),
            conv(in_chs, out_chs, kernel_size=5, stride=stride),
            *[ResidualBottleneckBlock(out_chs, out_chs) for i in range(num_residuals)],
        )
    

def make_elic_analysis(strides:List[int], in_chs:int, mid_chs:int, out_chs:int):
    '''
    Make the entire ELIC analysis transform (g_a)

    Args:
        strides (List[int]): List of strides, one per strided convolution
    '''
    if len(strides) == 1:
        return nn.Sequential(conv(in_chs, out_chs, kernel_size=5, stride=strides[0]), AttentionBlock(out_chs))
    return nn.Sequential(
        *[make_elic_analysis_block(in_chs if i==0 else mid_chs, mid_chs, stride, with_atten = i == len(strides)-2, ) for i,stride in enumerate(strides[:-1])],
        conv(mid_chs, out_chs, kernel_size=5, stride=strides[-1]),
        AttentionBlock(out_chs)
    )


class QVRFELICFMForSWIN(QVRFELIC):
    def __init__(self, N:int = 192, M:int = 320, groups: Optional[List[int]] = None, 
                 feature_chs:int  = 128, 
                 strides: List[int]=[1,1,1,1],
                 patch_embed_checkpoint : str = '',
                 shuffle_ratio:int  = 1,
                 embed_dims:int  = 96,
                 lambda_ref:float = 0.0483, 
                 lambda_ratio: List[float] = [0.01, 10],
                 **kwargs):
        self.groups = groups
        self.feature_chs = feature_chs
        self.strides = strides
        self.shuffle_ratio = shuffle_ratio
        self.embed_dims = embed_dims
        self.patch_embed_checkpoint = patch_embed_checkpoint

        super().__init__(N=N, M = M, groups = groups, lambda_ref = lambda_ref, lambda_ratio = lambda_ratio, **kwargs)
        self.g_s = nn.Sequential(
            make_elic_syn(strides, M, N, feature_chs),
            nn.PixelShuffle(shuffle_ratio),
            PatchEmbed(
            in_channels=feature_chs//shuffle_ratio**2,
            embed_dims=embed_dims   ,
            conv_type='Conv2d',
            kernel_size=4,
            stride=1,
            norm_cfg=dict(type='LN'),
            init_cfg=dict(
            checkpoint=
            patch_embed_checkpoint,
            type='Pretrained') if patch_embed_checkpoint else None)
        )
    
    def forward(self, x: Tensor, lmbda: Optional[Union[Tensor, float]] = None):
        out = super().forward(x, lmbda)
        assert isinstance(out['x_hat'], tuple), 'x_hat should be a tuple in any codec for swin'
        out['hw_shape'] = out['x_hat'][1]
        out['x_hat'] = out['x_hat'][0]
        return out


    def decompress(self, strings, shape, lmbda: Optional[Tensor] = None, return_y_hat: bool = False)->Dict[str, Any]:
        out = super().decompress(strings, shape, lmbda, clamp = False, return_y_hat= return_y_hat)
        x_hat, hw_shape = out['x_hat']
        if return_y_hat:
            return {
                "x_hat": x_hat,
                "hw_shape": hw_shape,
                "y_hat": out['y_hat']
            }
        return {
            "x_hat": x_hat,
            "hw_shape": hw_shape
        }

class QVRFELICFM(QVRFELIC):
    def __init__(self, N:int = 192, M:int = 320, groups: Optional[List[int]]=None, 
                 feature_chs:int = 128, 
                 strides:List[int] = [1,1,1,1],
                 lambda_ref: float = 0.0483,
                 lambda_ratio: List[float] = [0.01, 10], 
                 **kwargs):
        super().__init__(N=N, M = M, groups = groups, lambda_ref=lambda_ref, lambda_ratio = lambda_ratio, **kwargs)
        self.groups = groups
        self.feature_chs = feature_chs
        self.strides = strides

        self.g_s = make_elic_syn(strides, M, N, feature_chs)

    def decompress(self, strings, shape, lmbda: Optional[Tensor] = None, return_y_hat: bool = False)->Dict[str, Any]:
        return super().decompress(strings, shape, lmbda, clamp = False, return_y_hat= return_y_hat)


def FeatureCompressionConverter():
    '''
    Decorator for automatically converting a direct-compression for machines codec to a feature-compression codec
    Replace the analysis layers with different input dimensions and user controllable number of blocks based on the original codec
    For ELIC based codecs - user controls the number of residual blocks in the analysis transform with no conditions
    Synthesis transform is handled as in the regular CFM codec
    '''
    def decorator(cls):
        assert issubclass(cls, Elic2022Official), 'The class must be a subclass of Elic2022Official'
        new_cls = type(cls.__name__.replace('ICFM','FC'), (cls,), {})
        # Class names drive behaviour ('ForSWIN' -> the UnProject input layer here; 'QVRF' -> the
        # lambda_ref fill-in of VariableRateCFM.load_state_dict); keep the published names.
        use_unproject = 'ForSWIN' in cls.__name__
        original_init = cls.__bases__[0].__init__ if cls.__name__ == new_cls.__name__ else cls.__init__
        def __init__(self, *args, analysis_strides:List[int] = [1,1,1,1], synthesis_strides:List[int] = [1,1,1,1], hw_shape:Optional[tuple[int,int]] = None, **kwargs):
           # Get default keyword arguments from the original __init__ method
            original_defaults = inspect.getfullargspec(original_init).defaults
            original_kwargs =  inspect.getfullargspec(original_init).args[-len(original_defaults):]
            # Update kwargs with default values if missing
            for key, value in zip(original_kwargs, original_defaults):
                if key == 'strides':
                    assert key not in kwargs, 'in the FC version of the model, strides must be provided as analysis_strides and synthesis_strides'
                    kwargs['strides'] = synthesis_strides
                    continue
                if key not in kwargs:
                    kwargs[key] = value

            # Call the original __init__ method with the updated kwargs
            original_init(self, *args, **kwargs)

            self.analysis_strides = analysis_strides
            self.synthesis_strides = synthesis_strides
            self.use_unproject = use_unproject
            # assert self.synthesis_strides == self.strides
            self.g_a = make_elic_analysis(in_chs=kwargs['feature_chs'], mid_chs=kwargs['N'], out_chs=kwargs['M'], strides=analysis_strides)
            if use_unproject:
                self.unproject = UnProject(hw_shape)
                # A 1x1 lift from the Swin embedding dims to feature_chs; it is followed directly by g_a's first
                # (linear) conv, so the pair is one conv with extra parameters. Kept: the published SWIN
                # checkpoints contain it.
                embed_to_feat = conv(kwargs['embed_dims'], kwargs['feature_chs'], kernel_size=1, stride = 1)
                self.g_a = nn.Sequential(self.unproject, embed_to_feat, *self.g_a)

        new_cls.__init__ = __init__
        return new_cls

    return decorator

@FeatureCompressionConverter()
class QVRFELFCForSWIN(QVRFELICFMForSWIN):
    def __init__(self):
        pass

@FeatureCompressionConverter()
class QVRFELFC(QVRFELICFM):
    def __init__(self):
        pass
