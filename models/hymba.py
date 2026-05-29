from mamba_ssm import Mamba
from einops import rearrange
from torch import einsum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from transformers.activations import ACT2FN


def repeat_kv(tensor, num_repeats):
    return tensor.repeat_interleave(num_repeats, dim=1)


class HymbaMLP(nn.Module):
    def __init__(self, act_fn_name="silu", d_model=128, hidden_dim=None):
        super().__init__()
        self.act_fn_name = act_fn_name
        self.act_fn = ACT2FN[self.act_fn_name]
        self.d_model = d_model
        self.d_hidden = hidden_dim if hidden_dim is not None else d_model * 2

        if self.act_fn_name == "silu":
            self.gate_proj = nn.Linear(self.d_model, self.d_hidden, bias=False)
        self.down_proj = nn.Linear(self.d_hidden, self.d_model, bias=False)
        self.up_proj = nn.Linear(self.d_model, self.d_hidden, bias=False)

    def forward(self, x):
        if self.act_fn_name == "silu":
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        elif self.act_fn_name == "relu2":
            return self.down_proj(self.act_fn(self.up_proj(x)))
        else:
            raise NotImplementedError(f"No such hidden_act: {self.act_fn_name}")


class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# class AttentionBranch(nn.Module):
#     def __init__(
#         self, 
#         num_attention_heads, 
#         num_key_value_heads, 
#         attention_head_size, 
#         attention_window_size=None, 
#         modify_attention_mask=False, 
#         num_meta_tokens=None, 
#         seq_length=None, 
#         use_positional_embedding=False, 
#         rope_base=None
#     ):
#         super().__init__()

#         self.num_attention_heads = num_attention_heads
#         self.num_key_value_heads = num_key_value_heads
#         self.attention_head_size = attention_head_size
#         self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

#         self.attention_window_size = attention_window_size
#         self.modify_attention_mask = modify_attention_mask
#         self.num_meta_tokens = num_meta_tokens
#         self.seq_length = seq_length

#         self.use_positional_embedding = use_positional_embedding
#         self.rope_base = rope_base

#         if self.modify_attention_mask:
#             assert num_meta_tokens is not None
#             assert self.attention_window_size is not None

#             try:
#                 from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks, or_masks
#             except ImportError:
#                 print("Please install PyTorch>=2.5.0 to use flex_attention if you want to use modify_attention_mask=True")

#             self.create_block_mask = create_block_mask

#             def sliding_window(b, h, q_idx, kv_idx):
#                 return q_idx - kv_idx <= self.attention_window_size

#             def causal_mask(b, h, q_idx, kv_idx):
#                 return q_idx >= kv_idx

#             attn_mask = and_masks(causal_mask, sliding_window)

#             def prefix_mask(b, h, q_idx, kv_idx):
#                 return kv_idx < self.num_meta_tokens

#             register_mask = and_masks(causal_mask, prefix_mask)
#             self.attn_mask = or_masks(attn_mask, register_mask)

#             qk_length = self.seq_length + self.num_meta_tokens
#             self.block_mask = self.create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=qk_length, KV_LEN=qk_length)
#             self.flex_attention = torch.compile(flex_attention)

#         if self.use_positional_embedding:
#             self.rotary_emb = RotaryEmbedding(dim=self.attention_head_size, base=self.rope_base)

#     def forward(self, query_states, key_states, value_states):
#         bsz, q_len, _ = query_states.size()

#         query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2).contiguous()
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1, 2).contiguous()
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1, 2).contiguous()

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         if self.use_positional_embedding:
#             cos, sin = self.rotary_emb(query_states)
#             query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         # Ensure FlashAttention dtype compatibility
#         query_states = query_states.to(torch.float16)
#         key_states = key_states.to(torch.float16)
#         value_states = value_states.to(torch.float16)

#         if not self.modify_attention_mask:
#             query_states = query_states.transpose(1, 2)
#             key_states = key_states.transpose(1, 2)
#             value_states = value_states.transpose(1, 2)

#             if self.attention_window_size is not None:
#                 attn_outputs = flash_attn_func(query_states, key_states, value_states, causal=True, window_size=(self.attention_window_size, self.attention_window_size))
#             else:
#                 attn_outputs = flash_attn_func(query_states, key_states, value_states, causal=True)

#             attn_outputs = attn_outputs.reshape(bsz, q_len, int(self.num_attention_heads * self.attention_head_size)).contiguous()
#         else:
#             if key_states.shape[-2] <= self.block_mask.shape[-2] - 128 or key_states.shape[-2] > self.block_mask.shape[-2]:
#                 block_mask = self.create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=key_states.shape[-2], KV_LEN=key_states.shape[-2])
#             else:
#                 block_mask = self.block_mask

#             attn_outputs = self.flex_attention(query_states, key_states, value_states, block_mask=block_mask)
#             attn_outputs = attn_outputs.transpose(1, 2).contiguous()
#             attn_outputs = attn_outputs.reshape(bsz, q_len, int(self.num_attention_heads * self.attention_head_size)).contiguous()

#         return attn_outputs


class AttentionBranch(nn.Module):
    def __init__(
        self, 
        num_attention_heads, 
        num_key_value_heads, 
        attention_head_size, 
        attention_window_size=None, 
        modify_attention_mask=False, 
        num_meta_tokens=None, 
        seq_length=None, 
        use_positional_embedding=False, 
        rope_base=None
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_size = attention_head_size
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.attention_window_size = attention_window_size
        self.modify_attention_mask = modify_attention_mask
        self.num_meta_tokens = num_meta_tokens
        self.seq_length = seq_length

        self.use_positional_embedding = use_positional_embedding
        self.rope_base = rope_base

        if self.modify_attention_mask:
            assert num_meta_tokens is not None
            assert self.attention_window_size is not None

            try:
                from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks, or_masks
            except ImportError:
                print("Please install PyTorch>=2.5.0 to use flex_attention if you want to use modify_attention_mask=True")

            self.create_block_mask = create_block_mask

            def sliding_window(b, h, q_idx, kv_idx):
                return q_idx - kv_idx <= self.attention_window_size

            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

            attn_mask = and_masks(causal_mask, sliding_window)

            def prefix_mask(b, h, q_idx, kv_idx):
                return kv_idx < self.num_meta_tokens

            register_mask = and_masks(causal_mask, prefix_mask)
            self.attn_mask = or_masks(attn_mask, register_mask)

            qk_length = self.seq_length + self.num_meta_tokens
            self.block_mask = self.create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=qk_length, KV_LEN=qk_length)
            self.flex_attention = torch.compile(flex_attention)

        if self.use_positional_embedding:
            self.rotary_emb = RotaryEmbedding(dim=self.attention_head_size, base=self.rope_base)

    def forward(self, query_states, key_states, value_states):
        bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1, 2).contiguous()

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.use_positional_embedding:
            cos, sin = self.rotary_emb(query_states)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states.to(torch.float16)
        key_states = key_states.to(torch.float16)
        value_states = value_states.to(torch.float16)

        if not self.modify_attention_mask:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            # 일반 attention
            # reshape: (B, L, H, D) → (B*H, L, D)
            B, L, H, D = query_states.shape
            q = query_states.reshape(B * H, L, D)
            k = key_states.reshape(B * H, L, D)
            v = value_states.reshape(B * H, L, D)

            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)  # (B*H, L, L)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)  # (B*H, L, D)

            # 원래 shape로 복원
            attn_output = attn_output.reshape(B, H, L, D).transpose(1, 2).contiguous()  # (B, L, H, D)
            attn_outputs = attn_output.reshape(B, L, H * D).contiguous()

        return attn_outputs


class HymbaBlock(nn.Module):
    def __init__(
        self, 
        mamba_expand=2, 
        hidden_size=768, 
        num_attention_heads=12, 
        num_key_value_heads=4, 
        conv_kernel_size=3, 
        time_step_rank=8, 
        ssm_state_size=16, 
        attention_window_size=None, 
        modify_attention_mask=False, 
        num_meta_tokens=None, 
        seq_length=None, 
        use_positional_embedding=False, 
        rope_base=None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.mamba_expand = mamba_expand
        self.conv_kernel_size = conv_kernel_size
        self.time_step_rank = time_step_rank
        self.ssm_state_size = ssm_state_size

        self.intermediate_size = int(self.mamba_expand * self.hidden_size)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_size = int(self.intermediate_size / self.num_attention_heads)

        self.latent_dim = self.intermediate_size * 2 + self.attention_head_size * self.num_key_value_heads * 2

        self.pre_avg_layernorm1 = HymbaRMSNorm(self.intermediate_size)
        self.pre_avg_layernorm2 = HymbaRMSNorm(self.intermediate_size)

        self.in_proj = nn.Linear(self.hidden_size, self.latent_dim + self.intermediate_size, bias=True)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)

        self.self_attn = AttentionBranch(self.num_attention_heads, self.num_key_value_heads, self.attention_head_size, attention_window_size, modify_attention_mask, num_meta_tokens, seq_length, use_positional_embedding, rope_base)

        self.mamba = Mamba(d_model=self.intermediate_size)  # use MambaSimple or update to your actual Mamba version

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        hidden_states, gate = projected_states.tensor_split((self.latent_dim,), dim=1)
        query_states, key_states, value_states, hidden_states = hidden_states.tensor_split((self.intermediate_size, self.intermediate_size + self.attention_head_size * self.num_key_value_heads, self.intermediate_size + self.attention_head_size * self.num_key_value_heads * 2,), dim=1)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_outputs = self.self_attn(query_states=query_states, key_states=key_states, value_states=value_states)
        mamba_outputs = self.mamba(hidden_states.transpose(1, 2))

        assert attn_outputs.shape == mamba_outputs.shape
        hidden_states = (self.pre_avg_layernorm1(attn_outputs) + self.pre_avg_layernorm2(mamba_outputs)) / 2
        contextualized_states = self.out_proj(hidden_states)

        return contextualized_states


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.block = HymbaBlock(hidden_size=d_model, num_attention_heads=n_heads, num_key_value_heads=max(1, n_heads // 2), conv_kernel_size=d_conv, time_step_rank=8, ssm_state_size=d_state, mamba_expand=expand)
        self.mlp = HymbaMLP(act_fn_name="silu", d_model=d_model, hidden_dim=d_model * 2)
        self.input_layernorm = HymbaRMSNorm(d_model)
        self.pre_mlp_layernorm = HymbaRMSNorm(d_model)

    def forward(self, x, mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.block(x)
        x = x + residual

        residual = x
        x = self.pre_mlp_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x


class Hymba(nn.Module):
    def __init__(self, d_model=512, n_layer=24, n_heads=8, d_state=16, d_conv=4, expand=2, rc=False):
        super().__init__()
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_state, d_conv, expand) for _ in range(n_layer)])
        self.rc = rc

    def forward(self, x, x_rc=None, mask=None, mask_rc=None):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x, mask)
            if self.rc and x_rc is not None:
                x_rc = self.encoder[i](x_rc, mask_rc)
        return x, x_rc
