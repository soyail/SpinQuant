import torch
import triton
import triton.language as tl
from typing import List, Optional, Union
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEActivationFormat,
    FusedMoEMethodBase,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
)
import dataclasses
import copy




# ================================================================= #
# 1. Triton Kernel Implementation
# ================================================================= #

@triton.jit
def sefp_power_of_2_kernel(
    x_ptr,              
    out_ptr,            
    scale_ptr,          
    n_elements,         
    stride_n,           
    stride_k,           
    qmax: tl.constexpr, 
    qmin: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start_offset = pid * BLOCK_SIZE
    
    offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据并立即转换为 FP32（关键！）
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)  # 强制转换为 FP32 进行计算
    
    # 计算 Scale（在 FP32 下进行）
    x_abs = tl.abs(x)
    max_val = tl.max(x_abs, axis=0)
    
    # 添加更大的 epsilon 以应对 FP16/BF16 的精度问题
    scale_linear = max_val / qmax
    scale_linear = tl.maximum(scale_linear, 1e-5)  # 从 1e-6 改为 1e-5
    
    # 安全的 log2 计算
    scale_log2 = tl.math.log2(scale_linear)
    scale_exp_idx = tl.math.ceil(scale_log2)
    
    # 限制指数范围，防止 FP16/BF16 溢出
    # FP16 范围: 2^-24 到 2^15
    # BF16 范围: 2^-126 到 2^127
    scale_exp_idx = tl.clamp(scale_exp_idx, -15.0, 15.0)  # 保守范围
    
    scale = tl.math.exp2(scale_exp_idx)
    
    # 量化（在 FP32 下进行）
    x_scaled = x / scale
    x_clamped = tl.clamp(x_scaled, float(qmin), float(qmax))
    x_rounded = tl.floor(x_clamped + 0.5)
    
    out = x_rounded * scale
    
    # 存储时转回原始数据类型（通过 ptr 的类型自动处理）
    tl.store(out_ptr + offsets, out, mask=mask)
    
    if scale_ptr is not None:
        tl.store(scale_ptr + pid, scale)
def sefp_quantize_activation(x: torch.Tensor, group_size: int = 64, bits: int = 8):
    original_dtype = x.dtype
    original_shape = x.shape
    
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    
    x_flat = x.flatten()
    num_elements = x_flat.numel()
    
    # 对齐处理 (Padding)
    pad_len = (group_size - (num_elements % group_size)) % group_size
    if pad_len > 0:
        x_flat = torch.cat([x_flat, torch.zeros(pad_len, device=x.device, dtype=original_dtype)])
    
    x_reshaped = x_flat.view(-1, group_size).contiguous()
    num_groups = x_reshaped.shape[0]
    out = torch.empty_like(x_reshaped)
    
    grid = (num_groups,)
    sefp_power_of_2_kernel[grid](
        x_ptr=x_reshaped,
        out_ptr=out,
        scale_ptr=None,
        n_elements=x_reshaped.numel(),
        stride_n=x_reshaped.stride(0),
        stride_k=x_reshaped.stride(1),
        qmax=qmax,
        qmin=qmin,
        BLOCK_SIZE=group_size,
    )
    
    out_flat = out.view(-1)
    if pad_len > 0:
        out_flat = out_flat[:-pad_len]
    
    return out_flat.reshape(original_shape).to(original_dtype)

# ================================================================= #
# 2. Weight Quantization Logic
# ================================================================= #

def quantize_weights_groupwise(w: torch.Tensor, group_size: int = 64, bits: int = 8):
    """权重 Group-wise 对称量化伪量化"""
    original_shape = w.shape
    original_dtype = w.dtype
    
    # 假设权重是 [Out, In]，在 In 维度做 group
    w_flat = w.view(-1, group_size)
    qmax = (1 << (bits - 1)) - 1
    
    # 计算每个 group 的 scale: max(abs(w)) / qmax
    max_val = torch.max(torch.abs(w_flat), dim=-1, keepdim=True)[0]
    scales = max_val / qmax
    scales = scales.clamp(min=1e-8)
    
    # 量化反量化
    w_quant = torch.round(w_flat / scales).clamp(-(qmax+1), qmax)
    w_dequant = w_quant * scales
    
    return w_dequant.view(original_shape).to(original_dtype)

# ================================================================= #
# 3. vLLM Plugin Implementation
# ================================================================= #

class SEFPLinearMethod(UnquantizedLinearMethod):
    """
    自定义线性层方法：实现激活值 SEFP 量化和权重 Group-wise 量化。
    """
    def __init__(self, group_size: int = 64, bits: int = 8):
        self.group_size = group_size
        self.bits = bits
        self.weight_bits = 4
        self._weights_ready = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """在权重加载完成后执行一次权重伪量化。"""
        with torch.no_grad():
            q_weight = quantize_weights_groupwise(
                layer.weight.data, self.group_size, self.weight_bits
            )
            layer.weight.data.copy_(q_weight)
        self._weights_ready = True

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # 1. 激活值量化 (SEFP)
        x_q = sefp_quantize_activation(x, self.group_size, self.bits)

        # 2. 兼容 fallback：如果宿主未调用 post-load hook，则在首次 forward 时补做一次
        if not self._weights_ready:
            self.process_weights_after_loading(layer)

        # 3. 执行线性运算（仅激活做伪量化；权重已提前处理）
        return torch.nn.functional.linear(x_q, layer.weight, bias)
        # return torch.nn.functional.linear(x, layer.weight, bias)

class SEFPMoEMethod(FusedMoEMethodBase):
    def __init__(self, group_size: int, bits: int, moe_config):
        super().__init__(moe_config)
        self.group_size = group_size
        self.bits = bits
        self.weight_bits = 4
        self._weights_ready = False
        # 内部缓存 config 对象
        self._moe_quant_config: Optional[FusedMoEQuantConfig] = None

    def process_weights_after_loading(self, layer: "FusedMoE") -> None:
        """在权重加载完成后执行一次 MoE 权重伪量化。"""
        with torch.no_grad():
            q_w13 = quantize_weights_groupwise(
                layer.w13_weight.data, self.group_size, self.weight_bits
            )
            q_w2 = quantize_weights_groupwise(
                layer.w2_weight.data, self.group_size, self.weight_bits
            )
            layer.w13_weight.data.copy_(q_w13)
            layer.w2_weight.data.copy_(q_w2)
        self._weights_ready = True

    def get_fused_moe_quant_config(self, layer: "FusedMoE") -> Optional[FusedMoEQuantConfig]:
        """
        构造包含 Bias 的配置对象。
        vLLM 的 fused_experts 算子会从这个对象的 .w1_bias 和 .w2_bias 属性中提取数据。
        """
        if self._moe_quant_config is None:
            # 这里的属性名必须和 FusedMoEQuantConfig 的定义匹配
            # 通常 w1_bias 对应 w13, w2_bias 对应 w2
            # self._moe_quant_config = dataclasses.replace(FUSED_MOE_UNQUANTIZED_CONFIG)
            #self._moe_quant_config._w1.bias = getattr(layer, "w13_bias", None)
            #self._moe_quant_config._w2.bias = getattr(layer, "w2_bias", None)

            new_config = copy.copy(FUSED_MOE_UNQUANTIZED_CONFIG)
            new_config._w1 = copy.copy(new_config._w1)
            new_config._w2 = copy.copy(new_config._w2)
            new_config._w1.bias = getattr(layer, "w13_bias", None)
            new_config._w2.bias = getattr(layer, "w2_bias", None)
            self._moe_quant_config = new_config
        return self._moe_quant_config

    def create_weights(self, layer, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int, params_dtype: torch.dtype,
                       **kwargs):
        # 1. 注册 W13 和 W2 权重
        layer.register_parameter("w13_weight", torch.nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size_per_partition, hidden_size, dtype=params_dtype),
            requires_grad=False))
        layer.register_parameter("w2_weight", torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size_per_partition, dtype=params_dtype),
            requires_grad=False))

        # 2. 根据模型配置注册 Bias
        #if self.moe_config.has_bias:
        if True:
            layer.register_parameter("w13_bias", torch.nn.Parameter(
                torch.empty(num_experts, 2 * intermediate_size_per_partition, dtype=params_dtype),
                requires_grad=False))
            layer.register_parameter("w2_bias", torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False))
        else:
            # 即使没有 bias，也要显式设为 None 避免属性访问错误
            layer.w13_bias = None
            layer.w2_bias = None

        layer.weight_loader = kwargs.get("weight_loader")

    def apply(self, layer, router, x: torch.Tensor,
              router_logits: torch.Tensor) -> torch.Tensor:
        
        # 1. 激活值 SEFP 处理
        x_q = sefp_quantize_activation(x, self.group_size, self.bits)

        # 2. 兼容 fallback：如果宿主未调用 post-load hook，则在首次 forward 时补做一次
        if not self._weights_ready:
            self.process_weights_after_loading(layer)

        # 3. 路由选择
        topk_weights, topk_ids = router.select_experts(x_q, router_logits)

        # 4. 获取包含 Bias 的 Config
        q_config = self.get_fused_moe_quant_config(layer)

        # 5. 调用算子
        return fused_experts(
            x_q,
            # x,
            # layer.w13_weight,
            # layer.w2_weight,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            inplace=True,
            activation=layer.activation,
            # 核心修改：将包含 Bias 的 config 传进去
            quant_config=q_config,
            # 处理专家并行（EP）的情况
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map
        )
@register_quantization_config("sefp")
class SEFPQuantConfig(QuantizationConfig):
    """SEFP 量化配置。"""

    def __init__(self, group_size: int = 64, bits: int = 5):
        self.group_size = group_size
        self.bits = bits

    def get_name(self) -> str:
        return "sefp"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # 需要 Ampere 架构及以上以支持高性能 BF16/Triton

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "SEFPQuantConfig":
        # 如果 config 中定义了这些参数则读取，否则默认 64/8
        group_size = config.get("group_size", 64)
        bits = config.get("bits", 8)
        return cls(group_size, bits)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            return SEFPLinearMethod(self.group_size, self.bits)
        elif isinstance(layer, FusedMoE):
            return SEFPMoEMethod(self.group_size, self.bits, layer.moe_config)
        return None