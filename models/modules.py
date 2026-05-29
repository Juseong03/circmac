from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, fusion: str = 'concat'):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** 0.5
        self.fusion = fusion

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        if self.fusion == 'concat':
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif self.fusion != 'add':
            raise ValueError("fusion must be 'concat' or 'add'")

    def forward(
        self,
        query_input: torch.Tensor,      # [B, L_q, D] — e.g., E_circ
        key_input: torch.Tensor,        # [B, L_kv, D] — e.g., E_mirna
        value_input: torch.Tensor,      # [B, L_kv, D]
        key_padding_mask: torch.Tensor = None  # [B, L_kv]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_q, _ = query_input.size()
        L_kv = key_input.size(1)

        Q = self.q_proj(query_input).view(B, L_q, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, d]
        K = self.k_proj(key_input).view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value_input).view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)
        

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L_q, L_kv]

        # if key_padding_mask is not None:
        #     key_padding_mask = key_padding_mask.bool()
        #     scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)  # [B, H, L_q, d]
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.d_model)  # [B, L_q, D]

        if self.fusion == 'concat':
            fused = self.fusion_proj(torch.cat([query_input, context], dim=-1))  # [B, L_q, D]
        else:  # 'add'
            fused = query_input + context
        return fused, attn_weights
    

class Embeddings(nn.Module):
    def __init__(
        self, 
        d_model, 
        vocab_size=500,
        max_position_embeddings=2048, 
        model_name='transformer',
        dropout=0.1
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.model_name = model_name
        if self.model_name in ['transformer']:
            self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
            self.LayerNorm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embeddings = self.word_embeddings(x)
        if self.model_name in ['transformer']:
            batch_size, seq_len = x.size()
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
        return embeddings

class CONVBlock(nn.Module):
    def __init__(
        self, 
        dim_in, 
        dim_hidden=64,         # Default number of hidden dimensions
        kernel_size=3,         # Default kernel size for local pattern detection
        stride=1,              # Default stride to preserve sequence length
        padding=1,             # Padding to maintain input-output dimensions
        dropout=0.1,           # Default dropout for regularization
        activation=nn.ReLU()   # Default activation function
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(dim_in, dim_hidden, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(dim_hidden)
        self.act = activation
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(dim_hidden, dim_in, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(1, 2)  # Transpose to match Conv1d input dimensions
        out = self.conv_in(out)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv_out(out)
        return out.transpose(1, 2)  # Transpose back to original dimensions

class Attention(nn.Module):
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

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h = self.n_heads
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x2)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.dropout(self.to_out(out)) 

class TokenDropout(nn.Module):
    def __init__(self, active: bool, mask_ratio: float, mask_tkn_prob: float, mask_tkn_idx: int, pad_tkn_idx: int):
        super().__init__()
        self.active = active
        self.mask_ratio_train = mask_ratio * mask_tkn_prob
        self.mask_tkn_idx = mask_tkn_idx
        self.pad_tkn_idx = pad_tkn_idx

    def forward(self, x, tokens):
        if self.active:
            pad_mask = tokens.eq(self.pad_tkn_idx)
            src_lens = (~pad_mask).sum(dim=-1).float()
            mask_token_mask = tokens.eq(self.mask_tkn_idx).unsqueeze(dim=-1)

            x = torch.where(mask_token_mask, torch.tensor(0.0, device=x.device), x)
            mask_ratio_observed = (mask_token_mask.squeeze(-1).sum(dim=-1).float() / src_lens).clamp(min=1e-9)
            scaling_factor = (1 - self.mask_ratio_train) / (1 - mask_ratio_observed[..., None, None])

            x = x * scaling_factor

        return x

