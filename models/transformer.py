import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

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

class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads=8,
        d_head=16,
        dropout=0.,
    ):
        super().__init__()

        d_inner = d_head * n_heads
        self.n_heads = n_heads

        self.scale = d_head ** -0.5

        self.to_q = nn.Linear(d_model, d_inner, bias=False)
        self.to_k = nn.Linear(d_model, d_inner, bias=False)
        self.to_v = nn.Linear(d_model, d_inner, bias=False)

        self.to_out = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.n_heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.dropout(self.to_out(out)) , sim

class Encoder(nn.Module):
    def __init__(
            self,
            d_model,
            n_heads,
            d_head,
            mult,
            dropout
    ):
        super().__init__()

        self.norm_attn = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, n_heads, d_head, dropout)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mult, dropout)

    def forward(self, x, mask=None):
        x_norm = self.norm_attn(x)
        out_attn, sim = self.attention(x_norm)
        x = out_attn + x
        # out_attn  = self.attention(x)
        # x = self.norm_attn(out_attn + x)

        x_norm = self.norm_ff(x)
        out_ff = self.ff(x_norm)
        x = out_ff + x
        # out_ff = self.ff(x)
        # x = self.norm_ff(out_ff + x)

        return x 

class Transformer(nn.Module):
    def __init__(
        self,
        d_model=128,
        n_layer=6,
        n_heads=8,
        d_head=32,
        mult=2,
        dropout=0.1,
        rc=False
    ):
        super().__init__()
        self.rc = rc
        self.layers = nn.ModuleList([
            Encoder(
                d_model=d_model,
                n_heads=n_heads,
                d_head=d_head,
                mult=mult,
                dropout=dropout,
            ) for _ in range(n_layer)
        ])

    def forward(
            self, 
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
            x_rc: torch.Tensor | None = None,
            mask_rc: torch.Tensor | None = None
        ) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x, mask)
            if self.rc:
                x_rc = layer(x_rc, mask_rc)
        return x, x_rc