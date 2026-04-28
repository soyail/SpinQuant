# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
思路三：对已融合旋转的 bf16 模型，使用 AutoGPTQ 进行 int4 量化并保存为
        标准 GPTQ HuggingFace 格式，可直接被 vLLM 加载推理。

前置条件：
  1. 先运行 export_merged_bf16.py 得到融合旋转的 bf16 模型（--output_dir）。
  2. 安装 AutoGPTQ：pip install auto-gptq optimum

用法示例：
  python export_gptq_hf.py \
      --input_model ./merged_bf16_model \
      --output_dir ./gptq_model_w4g128 \
      --bits 4 \
      --group_size 128 \
      --nsamples 128

  # 量化后用 vLLM 推理
  vllm serve ./gptq_model_w4g128 --quantization gptq

注意事项：
  - 由于旋转已永久融合进权重，此处做的是标准 GPTQ（无需再感知旋转）。
  - 推荐 bits=4, group_size=128，与 SpinQuant 论文设置一致。
  - desc_act=False 速度更快，与 SpinQuant 的 act_order=False 对应。
"""

import argparse
import logging
import os
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("export_gptq_hf")


def check_autogptq():
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # noqa: F401
    except ImportError:
        log.error(
            "AutoGPTQ is not installed. Please install it with:\n"
            "  pip install auto-gptq optimum\n"
            "or for faster CUDA kernels:\n"
            "  pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/"
        )
        sys.exit(1)


def load_calibration_data(tokenizer, nsamples: int = 128, seqlen: int = 2048):
    """
    加载 wikitext-2 训练集作为 GPTQ 校准数据，格式为 AutoGPTQ 期望的 list[dict]。
    """
    import datasets

    log.info("Loading wikitext-2 calibration data...")
    traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]
    text = "\n\n".join(traindata["text"])
    enc = tokenizer(text, return_tensors="pt")

    import random
    random.seed(42)

    calibration_data = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        inp = enc.input_ids[:, i : i + seqlen]
        attention_mask = torch.ones_like(inp)
        calibration_data.append({"input_ids": inp, "attention_mask": attention_mask})

    log.info(f"Prepared {len(calibration_data)} calibration samples (seqlen={seqlen})")
    return calibration_data


def main():
    parser = argparse.ArgumentParser(
        description="Quantize SpinQuant merged bf16 model to GPTQ format using AutoGPTQ"
    )
    parser.add_argument(
        "--input_model",
        type=str,
        required=True,
        help="Path to the merged bf16 model (output of export_merged_bf16.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the GPTQ quantized model",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Number of quantization bits (default: 4)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="GPTQ group size (default: 128, use -1 for per-column)",
    )
    parser.add_argument(
        "--desc_act",
        action="store_true",
        default=False,
        help="Use activation ordering in GPTQ (slower but sometimes better, default: False)",
    )
    parser.add_argument(
        "--sym",
        action="store_true",
        default=False,
        help="Use symmetric quantization (default: False = asymmetric)",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples for GPTQ (default: 128)",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length for calibration (default: 2048)",
    )
    parser.add_argument(
        "--damp_percent",
        type=float,
        default=0.01,
        help="GPTQ dampening percent (default: 0.01)",
    )
    parser.add_argument(
        "--use_triton",
        action="store_true",
        default=False,
        help="Use Triton backend for faster inference (requires triton installed)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 0. 检查依赖
    # ------------------------------------------------------------------
    check_autogptq()
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    # ------------------------------------------------------------------
    # 1. 配置量化参数
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("GPTQ Quantization Configuration")
    log.info("=" * 60)
    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        sym=args.sym,
        damp_percent=args.damp_percent,
    )
    log.info(f"  bits       : {args.bits}")
    log.info(f"  group_size : {args.group_size}")
    log.info(f"  desc_act   : {args.desc_act}")
    log.info(f"  sym        : {args.sym}")
    log.info(f"  damp_percent: {args.damp_percent}")

    # ------------------------------------------------------------------
    # 2. 加载 tokenizer
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info(f"Loading tokenizer from: {args.input_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.input_model, use_fast=True)

    # ------------------------------------------------------------------
    # 3. 加载已融合旋转的 bf16 模型（用 AutoGPTQ 封装）
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info(f"Loading merged bf16 model from: {args.input_model}")
    log.info("=" * 60)
    model = AutoGPTQForCausalLM.from_pretrained(
        args.input_model,
        quantize_config=quantize_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",  # 自动分配 GPU
    )

    # ------------------------------------------------------------------
    # 4. 准备校准数据
    # ------------------------------------------------------------------
    calibration_data = load_calibration_data(
        tokenizer, nsamples=args.nsamples, seqlen=args.seqlen
    )

    # ------------------------------------------------------------------
    # 5. 执行 GPTQ 量化
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Running GPTQ quantization (this may take 15-60 minutes for 7B models)...")
    log.info("=" * 60)
    model.quantize(
        calibration_data,
        batch_size=1,
        use_triton=args.use_triton,
    )
    log.info("GPTQ quantization complete.")

    # ------------------------------------------------------------------
    # 6. 保存为标准 GPTQ HF 格式
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info(f"Saving GPTQ quantized model to: {args.output_dir}")
    log.info("=" * 60)
    os.makedirs(args.output_dir, exist_ok=True)

    model.save_quantized(
        args.output_dir,
        use_safetensors=True,
    )
    tokenizer.save_pretrained(args.output_dir)

    log.info("Done! GPTQ model saved.")
    log.info("")
    log.info("=" * 60)
    log.info("How to load and serve")
    log.info("=" * 60)
    log.info("  # With AutoGPTQ:")
    log.info("  from auto_gptq import AutoGPTQForCausalLM")
    log.info(f"  model = AutoGPTQForCausalLM.from_quantized('{args.output_dir}', device='cuda:0')")
    log.info("")
    log.info("  # With vLLM:")
    log.info(f"  vllm serve {args.output_dir} --quantization gptq")
    log.info("")
    log.info("  # With transformers (requires optimum):")
    log.info("  from transformers import AutoModelForCausalLM")
    log.info(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}', device_map='auto')")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 7. 打印文件大小对比
    # ------------------------------------------------------------------
    def dir_size_gb(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total / (1024 ** 3)

    input_size = dir_size_gb(args.input_model)
    output_size = dir_size_gb(args.output_dir)
    log.info(f"Model size comparison:")
    log.info(f"  Input  (bf16) : {input_size:.2f} GB")
    log.info(f"  Output (int{args.bits} GPTQ): {output_size:.2f} GB")
    log.info(f"  Compression   : {input_size / output_size:.1f}x")


if __name__ == "__main__":
    main()
