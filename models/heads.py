import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn import CNN1D

class MultiKernelCNN(nn.Module):
    def __init__(self, d_model, kernels=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=k, padding=k//2)
            for k in kernels
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # x: [B, L, D]
        x_in = x
        x = x.permute(0, 2, 1)  # [B, D, L]
        conv_outs = [conv(x) for conv in self.convs]  # list of [B, D, L]
        x = sum(conv_outs) / len(conv_outs)
        x = x.permute(0, 2, 1)  # [B, L, D]
        x = self.norm(x + x_in)  # Residual + Norm
        x = self.dropout(x)
        return x

class UnifiedSiteHead(nn.Module):
    """
    Unified head for binding site prediction.
    - Main task: per-position binding site prediction
    - Derived task: binding prediction via mean pooling of site probabilities

    No CLS token needed - aligns with circular nature of circRNA.
    """
    def __init__(self, d_model: int, d_hidden: int = None, binding_pooling: str = 'mean'):
        super(UnifiedSiteHead, self).__init__()
        if d_hidden is None:
            d_hidden = d_model // 2

        self.binding_pooling = binding_pooling  # 'mean', 'max', or 'attention'

        # Site prediction layers
        self.feature_enhancer = MultiKernelCNN(d_model)
        self.site_classifier = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 1)  # Binary per-position
        )

        # Optional: attention-based pooling for binding
        if binding_pooling == 'attention':
            self.attn_pool = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.Tanh(),
                nn.Linear(d_hidden, 1)
            )

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, D] - sequence embeddings (without CLS token)
            mask: [B, L] - padding mask (1 for valid, 0 for padding)

        Returns:
            sites_logits: [B, L, 1] - per-position binding site logits
            binding_logits: [B, 1] - sequence-level binding logits (derived)
        """
        # Enhance features
        x_enhanced = self.feature_enhancer(x)  # [B, L, D]

        # Per-position site prediction
        sites_logits = self.site_classifier(x_enhanced)  # [B, L, 1]

        # Derive binding prediction from sites
        sites_probs = torch.sigmoid(sites_logits)  # [B, L, 1]

        if mask is not None:
            # Mask out padding positions
            mask = mask.unsqueeze(-1).float()  # [B, L, 1]
            sites_probs_masked = sites_probs * mask

            if self.binding_pooling == 'mean':
                binding_score = sites_probs_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            elif self.binding_pooling == 'max':
                sites_probs_masked = sites_probs_masked + (1 - mask) * (-1e9)
                binding_score = sites_probs_masked.max(dim=1)[0]
            elif self.binding_pooling == 'attention':
                attn_weights = self.attn_pool(x_enhanced)  # [B, L, 1]
                attn_weights = attn_weights + (1 - mask) * (-1e9)
                attn_weights = F.softmax(attn_weights, dim=1)
                binding_score = (sites_probs * attn_weights).sum(dim=1)
        else:
            if self.binding_pooling == 'mean':
                binding_score = sites_probs.mean(dim=1)  # [B, 1]
            elif self.binding_pooling == 'max':
                binding_score = sites_probs.max(dim=1)[0]  # [B, 1]
            elif self.binding_pooling == 'attention':
                attn_weights = F.softmax(self.attn_pool(x_enhanced), dim=1)
                binding_score = (sites_probs * attn_weights).sum(dim=1)

        # Convert binding score to logit (inverse sigmoid)
        binding_score = binding_score.clamp(1e-7, 1 - 1e-7)
        binding_logits = torch.log(binding_score / (1 - binding_score))

        return {
            'sites_logits': sites_logits,      # [B, L, 1]
            'binding_logits': binding_logits,  # [B, 1]
            'sites_probs': sites_probs,        # [B, L, 1] - for visualization
        }
    
class SSLHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, x):  # x: [B, L, D]
        return self.proj(x)

class EnhancedSSLHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.enhancer = CNN1D(d_model)  # optional
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, x):  # x: [B, L, D]
        x, _ = self.enhancer(x)  # CNN1D returns (x, mask)
        return self.proj(x)


class PairingHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)

    def forward(self, x):  # x: [B, L, D]
        Q = self.linear_q(x)  # [B, L, D]
        K = self.linear_k(x)  # [B, L, D]
        return torch.matmul(Q, K.transpose(1, 2))  # [B, L, L]


class CircularPairingHead(nn.Module):
    """
    Circular-aware Pairing Head for circRNA.
    Adds circular distance bias to account for BSJ connectivity.
    """
    def __init__(self, d_model: int, use_circular_bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_circular_bias = use_circular_bias

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)

        # Learnable circular distance embedding
        if use_circular_bias:
            self.dist_embed = nn.Embedding(512, 1)  # max distance bins
            self.bias_scale = nn.Parameter(torch.ones(1))

    def _circular_distance(self, L: int, device: torch.device) -> torch.Tensor:
        """Compute circular distance matrix: min(|i-j|, L-|i-j|)"""
        idx = torch.arange(L, device=device)
        d = (idx[None, :] - idx[:, None]).abs()
        d_circular = torch.minimum(d, L - d)
        return d_circular

    def forward(self, x, mask=None):  # x: [B, L, D]
        B, L, D = x.shape

        Q = self.linear_q(x)  # [B, L, D]
        K = self.linear_k(x)  # [B, L, D]

        # Normalize for stable dot product
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        # Base pairing score
        pairing_score = torch.matmul(Q, K.transpose(1, 2))  # [B, L, L]

        # Add circular distance bias
        if self.use_circular_bias:
            dist = self._circular_distance(L, x.device)  # [L, L]
            dist_clamped = dist.clamp(max=511).long()
            dist_bias = self.dist_embed(dist_clamped).squeeze(-1)  # [L, L]
            pairing_score = pairing_score + self.bias_scale * dist_bias

        return pairing_score  # [B, L, L]