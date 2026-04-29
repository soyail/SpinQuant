# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import torch
import tqdm

from train_utils.quant_linear import QuantizeLinear
from utils import quant_utils, utils


@torch.no_grad()
def ssfp_fwrd(model, dev, args):
    # 类似于llama.cpp中的q4_0, q8_0，对称量化，每64个值共享一个FP16的缩放因子(scale)，无零点
    # group_size = 64
    assert args.w_groupsize == 64, "Groupsize must be 64 in SSFP!"
    assert args.w_asym == False, "Only symmetric quantization is supported in SSFP!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="Inserting weight quantizer"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(
            layer, layers=[torch.nn.Linear, QuantizeLinear]
        )

        for name in subset:
            layer_weight_bits = args.w_bits
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            #if args.int8_down_proj and "down_proj" in name:
            #   layer_weight_bits = 8

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=False,
                sym=not (args.w_asym),
                mse=False,
                weight_groupsize=args.w_groupsize,
            )
            subset[name].quantizer = quantizer

            quantizers["model.layers.%d.%s" % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
    return quantizers

