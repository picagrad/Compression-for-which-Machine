import math
import torch
from torch import Tensor, tensor
from typing import Optional, List, Union
import torch.nn as nn

from compressai.models import Elic2022Official
from compressai.latent_codecs import CheckerboardLatentCodec, GaussianConditionalLatentCodec
from functools import partial
from .utils import checkerboard_forward_twopass, gaussian_conditional_compress, gaussian_conditional_decompress
from .layers import Scaler

class QVRFELIC(Elic2022Official):
    def __init__(self, N:int = 192, M:int = 320, groups:List[int]=None, lambda_ref:float = 0.0483, lambda_ratio:List[float] = [0.01,10], scale_per_group:bool = True, **kwargs):
        super().__init__(N=N, M=M, groups=groups, **kwargs)
        self.lambda_ref = nn.Parameter(torch.tensor(lambda_ref), requires_grad=False) # make lambda_ref a parameter but not trainable so it stays as part of the model
        self.lambda_ratio = lambda_ratio
        self.checkerboard_latent_codecs = []
        self.gaussian_conditionals = []
        for name, module in self.named_modules():
            if isinstance(module, CheckerboardLatentCodec):
                # Variable rate: the two-pass checkerboard forward with this codec's per-lambda quantisation scale.
                module._forward_twopass = partial(checkerboard_forward_twopass, module)
                setattr(module, 'scale', tensor(1.))
                setattr(module, 'unscale', tensor(1.))
                module.forward_method = 'twopass'
                self.checkerboard_latent_codecs.append(module)
            if isinstance(module, GaussianConditionalLatentCodec):
                module.compress = partial(gaussian_conditional_compress, module)
                module.decompress = partial(gaussian_conditional_decompress, module)
                setattr(module, 'scale', tensor(1.))
                setattr(module, 'unscale', tensor(1.))
                self.gaussian_conditionals.append(module)
        # Scaler created after module discovery so n_groups matches the actual slice count.
        n_groups = len(self.checkerboard_latent_codecs) if scale_per_group else 1
        self.scaler = Scaler(self.lambda_ref, n_groups=n_groups)

    def set_scale(self, q_scale, q_unscale):
        n_groups = self.scaler.n_groups
        for k, module in enumerate(self.checkerboard_latent_codecs):
            s_k = q_scale[..., k] if n_groups > 1 else q_scale
            u_k = q_unscale[..., k] if n_groups > 1 else q_unscale
            module.scale = s_k
            module.unscale = u_k
        for k, module in enumerate(self.gaussian_conditionals):
            gi = min(k, n_groups - 1)
            s_k = q_scale[..., gi] if n_groups > 1 else q_scale
            u_k = q_unscale[..., gi] if n_groups > 1 else q_unscale
            module.scale = s_k
            module.unscale = u_k
    
    def sample_lmb(self, n:int, device:str = 'cpu', low_ratio:float = 0.01, high_ratio:float = 10.0):
        low, high = low_ratio*self.lambda_ref, high_ratio*self.lambda_ref
        low, high = math.log(low), math.log(high)
        transformed_lmb = low + (high-low) * torch.rand(n,device=device)
        return torch.exp(transformed_lmb)
        
        
    def forward(self, x: Tensor, lmbda: Optional[Union[Tensor, float]] = None) -> dict:
        if isinstance(lmbda, Tensor):
            lmbda = lmbda.to(x.device)
        elif isinstance(lmbda, (float,int)):
            lmbda = torch.tensor(lmbda, device = x.device)
        else:
            lmbda = self.sample_lmb(x.shape[0], device = x.device, low_ratio=self.lambda_ratio[0], high_ratio=self.lambda_ratio[1])
        q_scale = self.scaler(lmbda)
        q_unscale = 1/q_scale.clone().detach()
        self.set_scale(q_scale, q_unscale)
        out = super().forward(x)
        return out | {"lmbda": lmbda}
    
    @torch.no_grad()
    def compress(self, x, lmbda: Optional[Union[float,Tensor]] = None) -> dict:
        if isinstance(lmbda, Tensor):
            lmbda = lmbda.to(x.device)
        elif isinstance(lmbda, float):
            lmbda = torch.tensor(lmbda, device = x.device)
        else:
            lmbda = self.lambda_ref.to(x.device)
        q_scale = self.scaler(lmbda)
        q_unscale = 1/q_scale
        self.set_scale(q_scale, q_unscale)

        return super().compress(x)  
    
    @torch.no_grad()
    def decompress(self, strings, shape, lmbda: Optional[Union[float,Tensor]] = None, clamp:bool = True, return_y_hat: bool = False):
        device = self.scaler.a0.device
        if isinstance(lmbda, Tensor):
            lmbda = lmbda.to(device)
        elif isinstance(lmbda, float):
            lmbda = torch.tensor(lmbda, device = device)
        else:
            lmbda = self.lambda_ref.to(device)
        q_scale = self.scaler(lmbda)
        q_unscale = 1/q_scale
        self.set_scale(q_scale, q_unscale)
        
        y_out = self.latent_codec.decompress(strings, shape)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat).clamp_(0, 1) if clamp else self.g_s(y_hat)
        if return_y_hat:
            return {"x_hat": x_hat, "y_hat": y_hat}    
        return {
            "x_hat": x_hat,
        }

