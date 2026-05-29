from dataclasses import dataclass

@dataclass
class BasicConfig:
    d_model: int
    n_layer: int = 6
    dropout: float = 0.1
    vocab_size: int = 500
    rc: bool = False
    cross_attention_multihead: int = 1

@dataclass
class MambaConfig(BasicConfig):
    d_state: int = 32  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    pad_vocab_size_multiple: int = 16
    n_heads: int = 8

@dataclass
class Transformer2Config(BasicConfig):
    n_heads: int = 8
    d_head: int = 32
    mult: int = 4
    def __post_init__(self):
        self.d_inner = self.mult * self.d_model

@dataclass
class PretrainedConfig(BasicConfig):
    n_layer: int = 0
    trainable: bool = False

@dataclass
class GRUConfig(BasicConfig):
    pass

@dataclass
class LSTMConfig(BasicConfig):
    pass

@dataclass
class CircMACConfig(BasicConfig):
    """Configuration for CircMAC model."""
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 8
    circular: bool = False
    use_multiscale: bool = True
    conv_kernel_size: int = 7
    # Ablation flags
    use_attn: bool = True
    use_mamba: bool = True
    use_conv: bool = True

def get_model_config(
        model_name: str, 
        d_model: int,
        n_layer: int = None,
        verbose: bool = True,
        rc: bool = False,
        vocab_size: int = 500
    ):
    if model_name == 'transformer':
        config = Transformer2Config(d_model=d_model)
    elif model_name == 'circmac':
        config = CircMACConfig(d_model=d_model)
    elif model_name in ['mamba', 'hymba']:
        config = MambaConfig(d_model=d_model)
    elif model_name in ['rnabert', 'rnaernie', 'rnafm', 'rnamsm']:
        config = PretrainedConfig(d_model=d_model)
    elif model_name in ['lstm']:
        config = LSTMConfig(d_model=d_model)
    else:
        raise ValueError(f"\t Model '{model_name}' not recognized.")
    
    if n_layer is not None:
        config.n_layer = n_layer
    
    config.rc = rc
    config.vocab_size = vocab_size

    if verbose:
        print(f'- Model: {model_name}')
        print(f'- d_model: {d_model}')
        print(f'- n_layer: {config.n_layer}')

    return config