"""
CircMAC
Architecture:
    CircMACBlock:
        ├─ Attention (global)
        ├─ Mamba (sequential)
        └─ CNN (circular padding)
        └─ 3-branch Router
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba


# =========================
# Configuration
# =========================
@dataclass
class CircMACConfig:
    """Configuration for CircMAC v3 model."""
    d_model: int = 512
    n_layer: int = 12
    n_heads: int = 8
    rc: bool = False
    circular: bool = True
    use_multiscale: bool = True  # Global down/up (HyMBA style)
    d_state: int = 16
    d_conv: int = 4
    mamba_expand: int = 2
    conv_kernel_size: int = 7


# =========================
# Normalization & Utils
# =========================
class HymbaRMSNorm(nn.Module):
    """Root Mean Square Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            return (self.weight.to(input_dtype) * x.to(input_dtype))
        else:
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * x


def circular_rel_bias(L: int, device: torch.device, slope: float = 1.0) -> torch.Tensor:
    """
    Circular relative position bias for attention.
    d_circular(i,j) = min(|i-j|, L-|i-j|)
    """
    idx = torch.arange(L, device=device)
    d = (idx[None, :] - idx[:, None]).abs()
    d = torch.minimum(d, L - d).float()
    return -slope * d


# =========================
# Attention Branch
# =========================
class AttentionBranch(nn.Module):
    """
    Multi-head attention with circular relative bias.
    """
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_size: int,
        circular: bool = True
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.circular = circular

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query_states: [B, L, H*D]
            key_states: [B, L, H*D]
            value_states: [B, L, H*D]
            mask: [B, L]
        Returns:
            [B, L, H*D]
        """
        bsz, q_len, _ = query_states.size()

        # Reshape to multi-head: [B, L, H, D]
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size)
        key_states = key_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size)
        value_states = value_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size)

        # [B, H, L, D]
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        # Attention scores: [B, H, L, L]
        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        # Add circular relative bias
        if self.circular:
            bias = circular_rel_bias(q_len, query_states.device)
            attn_scores = attn_scores + bias.unsqueeze(0).unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill((mask == 0).view(bsz, 1, 1, q_len), -1e9)

        # Softmax + weighted sum
        attn_weights = attn_scores.softmax(dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)  # [B, H, L, D]

        # Reshape back: [B, L, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        return attn_output


# =========================
# CNN Branch (Circular)
# =========================
class CircularCNNBranch(nn.Module):
    """
    Depthwise CNN with circular padding for local pattern extraction.
    """
    def __init__(self, d_model: int, kernel_size: int = 7, circular: bool = True):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.circular = circular

        # Depthwise convolution
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=0,
            groups=d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            [B, L, D]
        """
        pad_size = (self.kernel_size - 1) // 2

        # [B, L, D] → [B, D, L]
        x = x.transpose(1, 2)

        # Apply circular or zero padding
        if self.circular:
            x = F.pad(x, (pad_size, pad_size), mode='circular')
        else:
            x = F.pad(x, (pad_size, pad_size), mode='constant', value=0)

        # Convolution
        x = self.conv(x)

        # [B, D, L] → [B, L, D]
        x = x.transpose(1, 2)

        return x


# =========================
# CircMAC Block
# =========================
class CircMACBlock(nn.Module):
    """
    CircMAC Block

    Architecture:
        Input → in_proj → split to (Q, K, V, base)
        ├─ Attention(Q, K, V) - global with circular bias
        ├─ Mamba(base) - sequential
        └─ CircularCNN(base) - local patterns
        └─ Router fusion → output

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        mamba_expand: Mamba expansion factor
        circular: Use circular features
        d_state: Mamba state dimension
        d_conv: Mamba convolution kernel
        conv_kernel_size: CNN kernel size
        use_attn: Enable attention branch (ablation)
        use_mamba: Enable mamba branch (ablation)
        use_conv: Enable conv branch (ablation)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        mamba_expand: int = 2,
        circular: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        conv_kernel_size: int = 7,
        use_attn: bool = True,
        use_mamba: bool = True,
        use_conv: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.circular = circular

        # Ablation flags
        self.use_attn = use_attn
        self.use_mamba = use_mamba
        self.use_conv = use_conv
        self.n_active_branches = sum([use_attn, use_mamba, use_conv])

        if self.n_active_branches == 0:
            raise ValueError("At least one branch must be enabled!")

        # Input projection: QKV + base
        self.qkv_dim = d_model * 3  # Q, K, V each d_model
        self.in_proj = nn.Linear(d_model, self.qkv_dim + d_model, bias=True)

        # Branch 1: Attention
        if use_attn:
            self.self_attn = AttentionBranch(
                num_attention_heads=n_heads,
                attention_head_size=self.head_dim,
                circular=circular
            )
            self.norm_attn = HymbaRMSNorm(d_model)

        # Branch 2: Mamba
        if use_mamba:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=mamba_expand
            )
            self.norm_mamba = HymbaRMSNorm(d_model)

        # Branch 3: Circular CNN
        if use_conv:
            self.cnn = CircularCNNBranch(
                d_model=d_model,
                kernel_size=conv_kernel_size,
                circular=circular
            )
            self.norm_cnn = HymbaRMSNorm(d_model)

        # Router: n-branch fusion (dynamic based on active branches)
        if self.n_active_branches > 1:
            self.router = nn.Sequential(
                nn.Linear(d_model * self.n_active_branches, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.n_active_branches),
                nn.Softmax(dim=-1)
            )

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with ablation support.

        Args:
            x: [B, L, D]
            mask: [B, L]

        Returns:
            [B, L, D]
        """
        B, L, D = x.shape

        # Input projection
        full_proj = self.in_proj(x)  # [B, L, qkv_dim + d_model]

        # Split into Q, K, V, base
        qkv, base = torch.split(full_proj, [self.qkv_dim, self.d_model], dim=-1)
        q, k, v = torch.split(qkv, [self.d_model, self.d_model, self.d_model], dim=-1)

        # Collect active branch outputs
        branch_outputs = []

        # Branch 1: Attention
        if self.use_attn:
            attn_out = self.self_attn(q, k, v, mask)
            attn_out = self.norm_attn(attn_out)
            branch_outputs.append(attn_out)

        # Branch 2: Mamba
        if self.use_mamba:
            mamba_out = self.mamba(base)
            mamba_out = self.norm_mamba(mamba_out)
            branch_outputs.append(mamba_out)

        # Branch 3: Circular CNN
        if self.use_conv:
            cnn_out = self.cnn(base)
            cnn_out = self.norm_cnn(cnn_out)
            branch_outputs.append(cnn_out)

        # Fusion
        if self.n_active_branches == 1:
            # Single branch: no routing needed
            fused = branch_outputs[0]
        else:
            # Multi-branch: use router
            router_input = torch.cat([
                out.mean(dim=1, keepdim=True).expand(-1, L, -1)
                for out in branch_outputs
            ], dim=-1)  # [B, L, n_branches * D]

            gates = self.router(router_input)  # [B, L, n_branches]

            # Weighted fusion
            fused = sum(
                out * gates[..., i:i+1]
                for i, out in enumerate(branch_outputs)
            )

        return self.out_proj(fused)


# =========================
# Down/Up Sampling
# =========================
class Downsample1D(nn.Module):
    """Global downsampling"""
    def __init__(self, d_model: int, factor: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=3, stride=factor, padding=1,
            groups=d_model
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.norm(x)


class Upsample1D(nn.Module):
    """Global upsampling."""
    def __init__(self, d_model: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.interpolate(x, scale_factor=self.factor, mode='linear', align_corners=False)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return self.norm(x)


# =========================
# CircMAC Main Model
# =========================
class CircMAC(nn.Module):
    """
    Args:
        d_model: Model dimension
        n_layer: Number of layers
        n_heads: Number of attention heads
        rc: Process reverse complement
        circular: Use circular features
        use_multiscale: Use global down/up sampling
        d_state: Mamba state dimension
        d_conv: Mamba conv kernel
        expand: Mamba expansion
        conv_kernel_size: CNN kernel size
    """
    def __init__(
        self,
        d_model: int = 512,
        n_layer: int = 12,
        n_heads: int = 8,
        rc: bool = False,
        circular: bool = True,
        use_multiscale: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        conv_kernel_size: int = 7,
        # Ablation flags
        use_attn: bool = True,
        use_mamba: bool = True,
        use_conv: bool = True,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.rc = rc
        self.circular = circular
        self.use_multiscale = use_multiscale

        # Global down/up sampling
        if use_multiscale:
            self.down = Downsample1D(d_model, factor=2)
            self.up = Upsample1D(d_model, factor=2)

        # Input normalization
        self.input_norm = HymbaRMSNorm(d_model)

        # Stack of CircMACBlock layers
        self.encoder = nn.ModuleList([
            CircMACBlock(
                d_model=d_model,
                n_heads=n_heads,
                mamba_expand=expand,
                circular=circular,
                d_state=d_state,
                d_conv=d_conv,
                conv_kernel_size=conv_kernel_size,
                use_attn=use_attn,
                use_mamba=use_mamba,
                use_conv=use_conv
            )
            for _ in range(n_layer)
        ])

    def _forward_core(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Core forward pass.

        Args:
            x: [B, L, D]
            mask: [B, L]

        Returns:
            [B, L, D]
        """
        x = self.input_norm(x)
        skip = None

        # Global downsampling
        if self.use_multiscale:
            skip = x
            x = self.down(x)

            # Downsample mask — match exact length of downsampled x (odd inputs need padding)
            if mask is not None:
                mask_down = F.avg_pool1d(mask.float().unsqueeze(1), 2, 2).squeeze(1)
                if mask_down.size(1) < x.size(1):
                    mask_down = F.pad(mask_down, (0, x.size(1) - mask_down.size(1)))
                mask = mask_down > 0.5

        # Encoder layers with residual connections
        for layer in self.encoder:
            x = x + layer(x, mask)

        # Global upsampling + skip connection
        if self.use_multiscale and skip is not None:
            x = self.up(x)

            # Align with original length
            if x.size(1) != skip.size(1):
                x = F.interpolate(
                    x.transpose(1, 2),
                    size=skip.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            x = x + skip

        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        x_rc: Optional[torch.Tensor] = None,
        mask_rc: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input [B, L, D]
            mask: Mask [B, L]
            x_rc: Reverse complement [B, L, D]
            mask_rc: RC mask [B, L]

        Returns:
            Tuple of (output, output_rc)
        """
        out_x = self._forward_core(x, mask)
        out_rc = None

        if self.rc and x_rc is not None:
            out_rc = self._forward_core(x_rc, mask_rc)

        return out_x, out_rc

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> CircMACConfig:
        """Get model configuration."""
        return CircMACConfig(
            d_model=self.d_model,
            n_layer=self.n_layer,
            rc=self.rc,
            circular=self.circular,
            use_multiscale=self.use_multiscale
        )
