import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F

class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.):
        super().__init__()
        d_hidden = int(d_model*mult)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden*2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model)
        )
    def forward(self, x, **kwargs):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, expand, dropout)

    def forward(self, x, mask=None):
        x_norm = self.norm(x)
        out = self.mamba(x_norm)
        x =  out + x
        
        x_norm = self.norm_ff(x)
        out_ff = self.ff(x_norm)
        x = out_ff + x
        return x
    
class MambaModel(nn.Module):
    def __init__(
            self, 
            d_model=512, 
            n_layer=24, 
            d_state=16, 
            d_conv=4, 
            expand=2,
            rc=False
    ):
        super(MambaModel, self).__init__()
        self.rc = rc
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_state, d_conv, expand) for _ in range(n_layer)])
    
    def forward(self, x, mask=None, x_rc=None, mask_rc=None):
        for i in range(len(self.layers)):
            x = self.layers[i](x, mask)
            if self.rc:
                x_rc = self.layers[i](x_rc, mask_rc)
        return x, x_rc