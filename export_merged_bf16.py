# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
思路二：将 SpinQuant 旋转矩阵（R1/R2）永久融合进原始模型权重，
        存储为标准 HuggingFace bf16 格式，然后计算 PPL 对比验证正确性。

用法示例：
  # 单卡运行
  python export_merged_bf16.py \
      --input_model meta-llama/Llama-2-7b-hf \
      --optimized_rotation_path ./output_rotation/R.bin \
      --output_dir ./merged_bf16_model \
      --eval_ppl \
      --access_token <your_hf_token>

  # 验证时同时评估原始模型 PPL
  python export_merged_bf16.py \
      --input_model meta-llama/Llama-2-7b-hf \
      --optimized_rotation_path ./output_rotation/R.bin \
      --output_dir ./merged_bf16_model \
      --eval_ppl \
      --eval_baseline \
      --access_token <your_hf_token>

说明：
  - 此脚本只做 "旋转融合"，不做量化，输出是全精度 bf16 模型。
  - 旋转融合后，模型的数值精度（PPL）应与 ptq.py --rotate --w_bits 16 的结果一致。
  - LayerNorm 权重被置为全 1（已融合进 linear 层），仍保留 layernorm 结构以兼容 HF。
"""

import argparse
import os
import sys
import math
import logging
from logging import Logger

import torch
import tqdm
import transformers
from transformers import LlamaTokenizerFast

# 确保可以 import 本仓库内的模块
sys.path.insert(0, os.path.dirname(__file__))

from eval_utils.modeling_llama import LlamaForCausalLM
from utils.fuse_norm_utils import fuse_layer_norms
from utils.hadamard_utils import apply_exact_had_to_linear
from utils import data_utils
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log: Logger = logging.getLogger("export_merged_bf16")


# ---------------------------------------------------------------------------
# 旋转融合函数（复用 eval_utils/rotation_utils.py 的逻辑，但不依赖 args）
# ---------------------------------------------------------------------------

def rotate_embeddings(model, R1: torch.Tensor) -> None:
    """embed_tokens.weight @= R1"""
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    """lm_head.weight @= R1"""
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1: torch.Tensor) -> None:
    """q_proj / k_proj / v_proj .weight @= R1"""
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1: torch.Tensor) -> None:
    """o_proj.weight = R1.T @ o_proj.weight"""
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1: torch.Tensor) -> None:
    """up_proj / gate_proj .weight @= R1"""
    for W in [layer.mlp.up_proj, layer.mlp.gate_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1: torch.Tensor) -> None:
    """down_proj.weight = R1.T @ down_proj.weight
    
    注意：eval/ptq 路径还会同时设置 ActQuantWrapper.online_full_had=True，
    将 Hadamard (R4) 分成"权重侧静态 H"和"激活侧在线 H"两半，两者相消。
    因此在 bf16 全精度导出（无在线 H）时，不能对权重施加 H，否则 H 无法抵消，
    导致推理结果错误（PPL 爆炸）。只需将 R1.T 融合进权重即可。
    """
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_dim: int, R2: torch.Tensor) -> None:
    """v_proj / o_proj 施加 R2（per-head Hadamard/Random 旋转）"""
    apply_exact_had_to_linear(layer.self_attn.v_proj, had_dim=head_dim, output=True, R2=R2)
    apply_exact_had_to_linear(layer.self_attn.o_proj, had_dim=head_dim, output=False, R2=R2)


@torch.inference_mode()
def fuse_rotations_into_weights(model, rotation_path: str) -> None:
    """
    将 R.bin 中保存的 R1 / R2 永久融合进模型权重。

    R.bin 结构（由 optimize_rotation.py 保存）：
      {
        "R1":                              Tensor[hidden, hidden],
        "model.layers.{i}.self_attn.R2":  Tensor[head_dim, head_dim],
        ...
      }
    """
    log.info(f"Loading rotation matrices from: {rotation_path}")
    R_dict = torch.load(rotation_path, map_location="cpu")

    R1 = R_dict["R1"].to(device="cuda", dtype=torch.float64)
    log.info(f"R1 shape: {R1.shape}, norm: {R1.norm():.4f}")

    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    # Step 1: 融合 LayerNorm
    log.info("Step 1/3: Fusing LayerNorms into adjacent linear layers...")
    fuse_layer_norms(model)

    # Step 2: 应用 R1 到所有层
    log.info("Step 2/3: Applying R1 rotation to embeddings, head, and all layers...")
    rotate_embeddings(model, R1)
    rotate_head(model, R1)

    layers = list(model.model.layers)
    for idx, layer in enumerate(tqdm.tqdm(layers, desc="Applying R1 + R2")):
        # R1 作用于注意力输入/输出、MLP 输入/输出
        rotate_attention_inputs(layer, R1)
        rotate_attention_output(layer, R1)
        rotate_mlp_input(layer, R1)
        rotate_mlp_output(layer, R1)

        # R2 作用于 v_proj / o_proj 的 per-head 维度
        key = f"model.layers.{idx}.self_attn.R2"
        if key in R_dict:
            R2 = R_dict[key].to(device="cuda", dtype=torch.float64)
        else:
            # 若 R.bin 中没有 R2（非 SpinQuant 优化模式），使用随机正交矩阵占位（理论上不应发生）
            log.warning(f"R2 for layer {idx} not found in R.bin, using identity.")
            R2 = torch.eye(head_dim, dtype=torch.float64, device="cuda")
        rotate_ov_proj(layer, head_dim, R2)

    log.info("Step 3/3: Rotation fusion complete.")


# ---------------------------------------------------------------------------
# PPL 评估
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_ppl(model, tokenizer, seqlen: int = 2048, device: str = "cuda") -> float:
    """
    在 wikitext-2 测试集上计算 PPL。
    显存充足时直接整体推理（无需逐层卸载），实现简单且速度快。
    """
    log.info("Loading wikitext-2 test set...")
    testloader = data_utils.get_wikitext2(
        seed=0,
        seqlen=seqlen,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    model.eval()
    model.to(device)
    model.config.use_cache = False

    input_ids = testloader.input_ids  # (1, total_len)
    nsamples = input_ids.numel() // seqlen
    # 截断到整数倍
    input_ids = input_ids[:, : nsamples * seqlen].view(nsamples, seqlen)

    loss_fct = torch.nn.CrossEntropyLoss()
    nlls = []

    for i in tqdm.tqdm(range(nsamples), desc="Evaluating PPL"):
        batch = input_ids[i : i + 1].to(device)   # (1, seqlen)
        with torch.no_grad():
            logits = model(batch).logits            # (1, seqlen, vocab)
        # shift：用前 seqlen-1 个 token 预测后 seqlen-1 个
        shift_logits = logits[:, :-1, :].contiguous().float()  # (1, seqlen-1, vocab)
        shift_labels = batch[:, 1:].contiguous()               # (1, seqlen-1)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.item())

    ppl = math.exp(sum(nlls) / len(nlls))
    log.info(f"WikiText-2 PPL: {ppl:.4f}")
    return ppl


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export SpinQuant rotated model to standard HF bf16 format")
    parser.add_argument("--input_model", type=str, required=True,
                        help="Path or HF hub name of the original model")
    parser.add_argument("--optimized_rotation_path", type=str, required=True,
                        help="Path to the R.bin file from optimize_rotation.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the merged bf16 model")
    parser.add_argument("--access_token", type=str, default=None,
                        help="HuggingFace access token for gated models")
    parser.add_argument("--eval_ppl", action="store_true",
                        help="Evaluate PPL of the merged model on wikitext-2")
    parser.add_argument("--eval_baseline", action="store_true",
                        help="Also evaluate PPL of the original (un-rotated) model for comparison")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for PPL evaluation")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Load model in bf16 (default: True)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    # ------------------------------------------------------------------
    # 1. （可选）评估原始模型 baseline PPL
    # ------------------------------------------------------------------
    baseline_ppl = None
    if args.eval_baseline:
        log.info("=" * 60)
        log.info("Loading ORIGINAL (baseline) model for PPL evaluation...")
        log.info("=" * 60)
        config_base = transformers.AutoConfig.from_pretrained(
            args.input_model, token=args.access_token
        )
        if config_base.tie_word_embeddings:
            config_base.tie_word_embeddings = False
        model_base = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.input_model,
            config=config_base,
            torch_dtype=dtype,
            token=args.access_token,
        )
        if config_base.tie_word_embeddings:
            model_base.lm_head.weight.data = model_base.model.embed_tokens.weight.data.clone()
        tokenizer_base = LlamaTokenizerFast.from_pretrained(
            args.input_model, token=args.access_token
        )
        log.info("Evaluating baseline PPL...")
        baseline_ppl = evaluate_ppl(model_base, tokenizer_base, seqlen=args.seqlen, device=device)
        log.info(f"[Baseline] WikiText-2 PPL = {baseline_ppl:.4f}")
        del model_base
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. 加载原始模型
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Loading original model for rotation fusion...")
    log.info("=" * 60)
    config = transformers.AutoConfig.from_pretrained(
        args.input_model, token=args.access_token
    )
    # Llama v3.2: tie_word_embeddings 不兼容 SpinQuant
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.input_model,
        config=config,
        torch_dtype=dtype,
        token=args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()
    model.eval()

    tokenizer = LlamaTokenizerFast.from_pretrained(
        args.input_model,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=args.access_token,
    )

    # ------------------------------------------------------------------
    # 3. 融合旋转矩阵进权重
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Fusing rotation matrices into model weights...")
    log.info("=" * 60)
    fuse_rotations_into_weights(model, args.optimized_rotation_path)

    # ------------------------------------------------------------------
    # 4. （可选）评估融合后模型 PPL
    # ------------------------------------------------------------------
    merged_ppl = None
    if args.eval_ppl:
        log.info("=" * 60)
        log.info("Evaluating PPL of the MERGED model...")
        log.info("=" * 60)
        merged_ppl = evaluate_ppl(model, tokenizer, seqlen=args.seqlen, device=device)
        log.info(f"[Merged bf16] WikiText-2 PPL = {merged_ppl:.4f}")

    # 对比摘要
    if baseline_ppl is not None or merged_ppl is not None:
        log.info("=" * 60)
        log.info("PPL Summary")
        log.info("=" * 60)
        if baseline_ppl is not None:
            log.info(f"  Original (no rotation) : {baseline_ppl:.4f}")
        if merged_ppl is not None:
            log.info(f"  Merged bf16 (rotated)  : {merged_ppl:.4f}")
        if baseline_ppl is not None and merged_ppl is not None:
            log.info(
                f"  PPL difference         : {merged_ppl - baseline_ppl:+.4f} "
                f"({'↑ worse' if merged_ppl > baseline_ppl else '↓ better / equal'})"
            )
            # 旋转是正交变换，理论上 PPL 应基本不变（< 0.01 差异属于正常浮点误差）
            if abs(merged_ppl - baseline_ppl) < 0.05:
                log.info("  ✅ PPL difference is negligible — rotation fusion is CORRECT.")
            else:
                log.warning("  ⚠️  PPL difference is large — please check the rotation fusion!")

    # ------------------------------------------------------------------
    # 5. 保存为标准 HF 格式
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info(f"Saving merged model to: {args.output_dir}")
    log.info("=" * 60)
    os.makedirs(args.output_dir, exist_ok=True)

    # 将模型移回 CPU 再保存，节省显存
    model = model.cpu()

    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,   # 使用 safetensors 格式
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(args.output_dir)

    # 更新 config：移除 tie_word_embeddings=False 可能引起的混淆
    config.save_pretrained(args.output_dir)

    log.info("Done! The merged bf16 model has been saved.")
    log.info(f"You can now load it with:")
    log.info(f"  from transformers import AutoModelForCausalLM")
    log.info(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}')")
    log.info(f"Or serve it with vLLM:")
    log.info(f"  vllm serve {args.output_dir}")

    # ------------------------------------------------------------------
    # 6. 打印最终摘要
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Export Summary")
    log.info("=" * 60)
    log.info(f"  Input model     : {args.input_model}")
    log.info(f"  Rotation path   : {args.optimized_rotation_path}")
    log.info(f"  Output dir      : {args.output_dir}")
    log.info(f"  Dtype           : {dtype}")
    if baseline_ppl:
        log.info(f"  Baseline PPL    : {baseline_ppl:.4f}")
    if merged_ppl:
        log.info(f"  Merged PPL      : {merged_ppl:.4f}")


if __name__ == "__main__":
    main()
