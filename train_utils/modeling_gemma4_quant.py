from typing import Optional

import torch
import torch.nn as nn
from train_utils.quant_linear import QuantizeLinear
from transformers.activations import ACT2FN
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig
from transformers.utils import logging
from vllm.model_executor.layers.layernorm import RMSNorm

logger = logging.get_logger(__name__)

class Gemma4MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = QuantizeLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = QuantizeLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = QuantizeLinear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        activation_name = getattr(config, "hidden_activation", "gelu")
        self.act_fn = ACT2FN[activation_name]

    def forward(self, x, R1):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x, R1)) * self.up_proj(x, R1),
            R1,
            transpose=True,
        )
        return down_proj

class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        # Gemma4 uses scaling=1.0.
        # Unlike Gemma2/3, query_pre_attn_scalar is NOT used here;
        # Q/K norms with learnable weights handle scaling implicitly.
        self.scaling = 1.0
        self.attn_scale = self.scaling * (self.head_dim ** -0.5)

        self.q_proj = QuantizeLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = QuantizeLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = QuantizeLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = QuantizeLinear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )

        # Q/K norms: output = norm(x) * weight (learnable pre-head scale)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.R2 = None

        self.rotary_emb = None

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, R1=None):
        batch_size, seq_length, _ = hidden_states.size()
        r2_weight = self.R2.weight if self.R2 is not None else None

        # project to q/k/v
        query = self.q_proj(hidden_states, R1)
        key = self.k_proj(hidden_states, R1)
        value = self.v_proj(hidden_states, R1, R2=r2_weight)

        # reshape to (batch_size, num_heads, seq_length, head_dim)
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # apply RMSNorm and scaling
        query = self.q_norm(query) * self.attn_scale
        key = self.k_norm(key) * self.attn_scale
        value = self.v_norm(value)

        # apply rotary embeddings
        if self.rotary_emb is not None:
            query = self.rotary_emb(query, seq_len=seq_length)
            key = self.rotary_emb(key, seq_len=seq_length)

        # compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1))

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # compute attention output
        attn_output = torch.matmul(attn_weights, value)

        # reshape back to (batch_size, seq_length, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        # project output
        attn_output = self.o_proj(attn_output, R1, R2=r2_weight, transpose=True)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs

# TODO implement Gemma4FlashAttention2, notice that the attention pattern is different from Gemma2/3, we need to apply the rotation after the q/k/v projections and before the attention computation, and we need to use the same rotation for q/k/v in the same layer. We can also share the rotation matrices across layers to save memory, but we will implement it with separate rotation matrices for each layer first.
class Gemma4FlashAttention2(Gemma4Attention):
    def __init__(self, config: Gemma4Config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        # Per-layer rotation (can be shared across layers in future optimization).
        self.qkv_rotation = nn.Parameter(
            torch.eye(self.head_dim, dtype=torch.float32),
            requires_grad=False,
        )

    def _apply_shared_qkv_rotation(self, states: torch.Tensor) -> torch.Tensor:
        rotation = self.qkv_rotation.to(device=states.device, dtype=states.dtype)
        return torch.matmul(states, rotation)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        R1=None,
    ):
        batch_size, seq_length, _ = hidden_states.size()
        r2_weight = self.R2.weight if self.R2 is not None else None

        query = self.q_proj(hidden_states, R1)
        key = self.k_proj(hidden_states, R1)
        value = self.v_proj(hidden_states, R1, R2=r2_weight)

        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Gemma4-specific pattern: rotate after q/k/v projections and before attention.
        query = self._apply_shared_qkv_rotation(query)
        key = self._apply_shared_qkv_rotation(key)
        value = self._apply_shared_qkv_rotation(value)

        query = self.q_norm(query) * self.attn_scale
        key = self.k_norm(key) * self.attn_scale
        value = self.v_norm(value)

        if self.rotary_emb is not None:
            query = self.rotary_emb(query, seq_len=seq_length)
            key = self.rotary_emb(key, seq_len=seq_length)

        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output, R1, R2=r2_weight, transpose=True)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
