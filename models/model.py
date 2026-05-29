from typing import Tuple, Optional
import torch
import torch.nn as nn

from .mamba import MambaModel
from .circmac import CircMAC
from .hymba import Hymba
from .transformer import Transformer
from .lstm import LSTM
from .pretrainedmodel import PretrainedModel

from .modules import TokenDropout, Embeddings, CrossAttention
from .heads import BindingHead, SiteHead, SSLHead, EnhancedSSLHead, PairingHead, CircularPairingHead, UnifiedSiteHead
from .cnn import CNN1D

class ModelWrapper(nn.Module):
    def __init__(
        self, 
        config, 
        name: str = 'mamba',
        device: str = 'cpu',
        pooling_mode_target: str = 'cls'
    ) -> None:
        super(ModelWrapper, self).__init__()
        self.name = name
        self.config = config
        self.device = device
        self.pooling_mode_target = pooling_mode_target
        self.is_convblock = False  
        self.is_cross_attention = False  
        
        self._init_embedding()
        
        self._set_proj_target(d_target=config.d_model)
        
        self.backbone = self._get_backbone()

        if hasattr(self.config, 'ssp_vocab_size'):
            self._set_ssp_head()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        x_rc: torch.Tensor | None = None,
        mask_rc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.name.lower() in ['rnabert', 'rnaernie', 'rnafm', 'rnamsm']:
            emb = x
            if x_rc is not None:
                emb_rc = x_rc
            else:
                emb_rc = None
        else:
            emb = self.embedding(x)
            if x_rc is not None:
                emb_rc = self.embedding(x_rc)
            else:
                emb_rc = None

        emb, emb_rc = self.backbone(emb, mask, emb_rc, mask_rc)

        return emb, emb_rc

    def decoder(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_proj = self.get_target_projected(target)
        interaction_map = x * target_proj.unsqueeze(1)
        return self.to_out(interaction_map)
    
    def forward_pretrained(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_proj = self.get_target_projected(target)
        interaction_map = x * target_proj.unsqueeze(1)
        return self.to_out(interaction_map)

    def get_target_projected(self, emb_target: torch.Tensor, mode=None) -> torch.Tensor:
        pooling_mode = mode if mode is not None else self.pooling_mode_target
        if pooling_mode == 'cls':
            pooled = emb_target[:, 0, :]
        elif pooling_mode == 'mean':
            pooled = emb_target[:, 1:-1, :].mean(dim=1)
        elif pooling_mode == 'None':
            pooled = emb_target
        else:
            raise ValueError(f"Invalid pooling mode: {pooling_mode}")
        return self.proj_target(pooled)

    def to_out(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_convblock:
            return self.convblock(x)
        else:
            return x

    def _init_embedding(self) -> None:
        self.embedding = Embeddings(
            d_model=self.config.d_model,
            vocab_size=self.config.vocab_size,
        ).to(self.device)

        self.sigmoid_rc = nn.Sigmoid().to(self.device)
        self.proj_rc = nn.Linear(self.config.d_model*2, self.config.d_model).to(self.device)

    def _set_cross_attention(self) -> None:
        self.cross_attention = CrossAttention(
            d_model=self.config.d_model,
            n_heads=4,
            dropout=0.1,
            fusion='concat'
        ).to(self.device)
        self.is_cross_attention = True

    def _set_convblock(self) -> None:
        self.convblock = CNN1D(d_model=self.config.d_model).to(self.device)
        self.is_convblock = True

    def _set_proj_target(self, d_target: int) -> None:
        self.proj_target = nn.Sequential(
            nn.Linear(d_target, self.config.d_model),
            nn.LayerNorm(self.config.d_model)
        ).to(self.device)

    def _set_mlp_weights(self) -> None:
        self.mlp_weights = nn.Sequential(
            nn.Linear(self.config.d_model * 2, self.config.d_model),
            nn.SiLU(),
            nn.Linear(self.config.d_model, 1),
            nn.Sigmoid()
        ).to(self.device)

    def _set_unified_site_head(self, binding_pooling: str = 'mean', interaction: str = 'concat') -> None:
        """
        Set unified site head for site-first approach.
        Sites prediction is main task, binding is derived from sites.
        """
        if interaction == 'concat':
            d_input = self.config.d_model * 2
        else:  # 'elementwise' or 'cross_attention'
            d_input = self.config.d_model

        self.unified_site_head = UnifiedSiteHead(
            d_model=d_input,
            binding_pooling=binding_pooling
        ).to(self.device)
    
    def _set_mlm_head(self) -> None:
        self.mlm_head = SSLHead(
            d_model=self.config.d_model, 
            vocab_size=self.config.vocab_size
        ).to(self.device)

    def _set_ntp_head(self) -> None:
        self.ntp_head = SSLHead(
            d_model=self.config.d_model, 
            vocab_size=self.config.vocab_size
        ).to(self.device)

    def _set_ss_embedding(self, ss_vocab_size: int = 4) -> None:
        """SS token embedding for SS-pair contrastive learning.
        Adds structure information to sequence embeddings.
        ss_vocab: 0=PAD, 1=(, 2=), 3=.
        """
        self.ss_embedding = nn.Embedding(
            ss_vocab_size, self.config.d_model, padding_idx=0
        ).to(self.device)

    def _set_proj_contrastive(self) -> None:
        # 2-layer MLP projection head (SimCLR-style)
        d = self.config.d_model
        self.proj_contrastive = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        ).to(self.device)
    
    def _set_token_dropout(
        self,
        active: bool = True,
        mask_ratio: float = 0.15,
        mask_tkn_prob: float = 0.8,
        mask_tkn_idx: int = 1,
        pad_tkn_idx: int = 0
    ) -> None:
        self.token_dropout = TokenDropout(
            active=active,
            mask_ratio=mask_ratio,
            mask_tkn_prob=mask_tkn_prob,
            mask_tkn_idx=mask_tkn_idx,
            pad_tkn_idx=pad_tkn_idx
        ).to(self.device)

    def _set_ssp_head(self) -> None:
        ssp_vocab_size = getattr(self.config, 'ssp_vocab_size', 4)

        self.ssp_head = SSLHead(
                d_model=self.config.d_model, vocab_size=ssp_vocab_size
            ).to(self.device)

    def _set_ss_labels_head(self):
        self.ss_labels_head = EnhancedSSLHead(
            d_model=self.config.d_model, vocab_size=2
        ).to(self.device)


    def _set_ss_labels_multi_head(self):
        num_classes = getattr(self.config, 'ss_labels_multi_classes', 8)
        self.ss_labels_multi_head = EnhancedSSLHead(
            d_model=self.config.d_model, vocab_size=num_classes
        ).to(self.device)


    def _set_pairing_head(self, circular: bool = False):
        if circular:
            self.pairing_head = CircularPairingHead(
                d_model=self.config.d_model,
                use_circular_bias=True
            ).to(self.device)
        else:
            self.pairing_head = PairingHead(
                d_model=self.config.d_model
            ).to(self.device)

    def _get_backbone(self) -> nn.Module:
        if self.name == 'mamba':
            backbone = MambaModel(
                d_model=self.config.d_model,
                n_layer=self.config.n_layer,
                d_state=self.config.d_state,
                d_conv=self.config.d_conv,
                expand=self.config.expand,
                rc=self.config.rc
            )
        elif self.name == 'transformer':
            backbone = Transformer(
                d_model=self.config.d_model,
                n_layer=self.config.n_layer,
                n_heads=self.config.n_heads,
                d_head=self.config.d_head,
                mult=self.config.mult,
                dropout=self.config.dropout,
                rc=self.config.rc
            )
        elif self.name == 'hymba':
            backbone = Hymba(
                d_model=self.config.d_model,
                n_layer=self.config.n_layer,
                n_heads=self.config.n_heads,
                d_state=self.config.d_state,
                d_conv=self.config.d_conv,
                expand=self.config.expand,
                rc=self.config.rc
            )
        elif self.name == 'circmac':
            backbone = CircMAC(
                d_model=self.config.d_model,
                n_layer=self.config.n_layer,
                n_heads=getattr(self.config, 'n_heads', 8),
                d_state=self.config.d_state,
                d_conv=self.config.d_conv,
                expand=self.config.expand,
                rc=self.config.rc,
                circular=getattr(self.config, 'circular', True),
                use_multiscale=getattr(self.config, 'use_multiscale', True),
                conv_kernel_size=getattr(self.config, 'conv_kernel_size', 7),
                use_attn=not getattr(self.config, 'no_attn', False),
                use_mamba=not getattr(self.config, 'no_mamba', False),
                use_conv=not getattr(self.config, 'no_conv', False)
            )
        elif self.name in ['rnabert', 'rnaernie', 'rnafm', 'rnamsm']:
            backbone = PretrainedModel(
                rna_model=self.name,
                d_model=self.config.d_model,
                trainable=self.config.trainable,
                rc=self.config.rc
            )
        elif self.name == 'lstm':
            backbone = LSTM(
                d_model=self.config.d_model,
                n_layer=self.config.n_layer,
                dropout=self.config.dropout,
                bidirectional=False,
                cnn=False,
                rc=self.config.rc
            )
        else:
            raise ValueError(f'Invalid model name: {self.name}')

        return backbone.to(self.device)
