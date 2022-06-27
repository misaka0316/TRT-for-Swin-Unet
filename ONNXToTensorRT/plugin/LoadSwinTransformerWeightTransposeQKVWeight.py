# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import math
import tensorrt as trt
#import onnx_graphsurgeon as gs
import onnx

#from FasterTransformer.examples.pytorch.swin.SwinTransformerWeightTransposeQKVWeight import SwinTransformerWeightTransposeQKVWeight

def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def swin_transformer(network, input, weights_dict, th_path, kind_num):
    depths = [2, 2, 2, 2]
    num_heads = [3, 6, 12, 24]
    qk_scale = 1.0
    ape = 0
    patch_norm = 1
    qkv_bias = 1

    #SwinTransformer = gs.Node("CustomSwinTransformerPlugin", "CustomSwinTransformerPlugin", inputs=[input], outputs=[output])

    max_batch_size = trt.PluginField("max_batch_size", np.array([1]).astype(np.int32), trt.PluginFieldType.INT32)
    img_size = trt.PluginField("img_size", np.array([224]).astype(np.int32), trt.PluginFieldType.INT32)
    patch_size = trt.PluginField("patch_size", np.array([4]).astype(np.int32), trt.PluginFieldType.INT32)
    in_chans = trt.PluginField("in_chans", np.array([1]).astype(np.int32), trt.PluginFieldType.INT32)
    embed_dim = trt.PluginField("embed_dim", np.array([96]).astype(np.int32), trt.PluginFieldType.INT32)
    window_size = trt.PluginField("window_size", np.array([7]).astype(np.int32), trt.PluginFieldType.INT32)
    ape = trt.PluginField("ape", np.array([ape]).astype(np.int32), trt.PluginFieldType.INT32)
    patch_norm = trt.PluginField("patch_norm", np.array([patch_norm]).astype(np.int32), trt.PluginFieldType.INT32)
    layer_num = trt.PluginField("layer_num", np.array([len(depths)]).astype(np.int32), trt.PluginFieldType.INT32)
    mlp_ratio = trt.PluginField("mlp_ratio", np.array([4.0]).astype(np.float32), trt.PluginFieldType.FLOAT32)
    qkv_bias = trt.PluginField("qkv_bias", np.array([qkv_bias]).astype(np.int32), trt.PluginFieldType.INT32)
    qk_scale = trt.PluginField("qk_scale", np.array([qk_scale]).astype(np.float32), trt.PluginFieldType.FLOAT32)
    #kind = trt.PluginField("kind", np.array([kind_num]).astype(np.float32), trt.PluginFieldType.INT32)
    depths_f = trt.PluginField("depths", np.array(depths).astype(np.int32), trt.PluginFieldType.INT32)
    num_heads_f = trt.PluginField("num_heads", np.array(num_heads).astype(np.int32), trt.PluginFieldType.INT32)
    #print("111")
    sw_weights = SwinTransformerWeightTransposeQKVWeight(len(depths), 7, depths, num_heads, th_path, weights_dict)   
    #print("What")
    part_fc = []
    weight_idx = 0
    for l in range(len(depths)):
        for b in range(depths[l]):
            part_fc.append(trt.PluginField("attention_qkv_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_qkv_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_proj_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_proj_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear2_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear2_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm_gamma_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm_beta_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm2_gamma_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm2_beta_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_relative_pos_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx].cpu()).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1

        # delete merging weights
        part_fc.append(trt.PluginField("patchMerge_norm_gamma_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
        weight_idx += 1
        part_fc.append(trt.PluginField("patchMerge_norm_beta_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
        weight_idx += 1
        part_fc.append(trt.PluginField("patchMerge_linear_kernel_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
        weight_idx += 1
        part_fc.append(trt.PluginField("attn_mask_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
        weight_idx += 1

    part_fc.append(trt.PluginField("patchEmbed_proj_kernel", np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
    weight_idx += 1
    part_fc.append(trt.PluginField("patchEmbed_proj_bias", np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
    weight_idx += 1
    part_fc.append(trt.PluginField("patchEmbed_norm_gamma", np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
    weight_idx += 1
    part_fc.append(trt.PluginField("patchEmbed_norm_beta", np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
    weight_idx += 1
    part_fc.append(trt.PluginField("norm_gamma", np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
    weight_idx += 1
    part_fc.append(trt.PluginField("norm_beta", np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
    weight_idx += 1
    
    plg_registry = trt.get_plugin_registry()
    swinTransformer_plg_creator = plg_registry.get_plugin_creator("CustomSwinTransformerPlugin", "1", "")
    #print("1")
    pfc = trt.PluginFieldCollection([max_batch_size, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale, depths_f, num_heads_f] + part_fc)
    #print("222")
    fn = swinTransformer_plg_creator.create_plugin("swin_transformer" + str(kind_num), pfc)
    inputs = [input]
    #print("???????")
    sw = network.add_plugin_v2(inputs, fn) 
    
    set_output_name(sw, "swin_transformer_", "fuck" + str(kind_num))
    return sw

def swin_transformer_fp16(network, input, weights_dict, th_path, kind_num):
    depths = [2, 2, 2, 2]
    num_heads = [3, 6, 12, 24]
    qk_scale = 1.0
    ape = 0
    patch_norm = 1
    qkv_bias = 1

    #SwinTransformer = gs.Node("CustomSwinTransformerPlugin", "CustomSwinTransformerPlugin", inputs=[input], outputs=[output])

    max_batch_size = trt.PluginField("max_batch_size", np.array([1]).astype(np.int32), trt.PluginFieldType.INT32)
    img_size = trt.PluginField("img_size", np.array([224]).astype(np.int32), trt.PluginFieldType.INT32)
    patch_size = trt.PluginField("patch_size", np.array([4]).astype(np.int32), trt.PluginFieldType.INT32)
    in_chans = trt.PluginField("in_chans", np.array([1]).astype(np.int32), trt.PluginFieldType.INT32)
    embed_dim = trt.PluginField("embed_dim", np.array([96]).astype(np.int32), trt.PluginFieldType.INT32)
    window_size = trt.PluginField("window_size", np.array([7]).astype(np.int32), trt.PluginFieldType.INT32)
    ape = trt.PluginField("ape", np.array([ape]).astype(np.int32), trt.PluginFieldType.INT32)
    patch_norm = trt.PluginField("patch_norm", np.array([patch_norm]).astype(np.int32), trt.PluginFieldType.INT32)
    layer_num = trt.PluginField("layer_num", np.array([len(depths)]).astype(np.int32), trt.PluginFieldType.INT32)
    mlp_ratio = trt.PluginField("mlp_ratio", np.array([4.0]).astype(np.float16), trt.PluginFieldType.FLOAT16)
    qkv_bias = trt.PluginField("qkv_bias", np.array([qkv_bias]).astype(np.int32), trt.PluginFieldType.INT32)
    qk_scale = trt.PluginField("qk_scale", np.array([qk_scale]).astype(np.float16), trt.PluginFieldType.FLOAT16)
    #kind = trt.PluginField("kind", np.array([kind_num]).astype(np.float32), trt.PluginFieldType.INT32)
    depths_f = trt.PluginField("depths", np.array(depths).astype(np.int32), trt.PluginFieldType.INT32)
    num_heads_f = trt.PluginField("num_heads", np.array(num_heads).astype(np.int32), trt.PluginFieldType.INT32)
    #print("111")
    sw_weights = SwinTransformerWeightTransposeQKVWeight(len(depths), 7, depths, num_heads, th_path, weights_dict)   
    #print("What")
    part_fc = []
    weight_idx = 0
    for l in range(len(depths)):
        for b in range(depths[l]):
            part_fc.append(trt.PluginField("attention_qkv_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_qkv_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_proj_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_proj_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear2_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear2_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm_gamma_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm_beta_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm2_gamma_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm2_beta_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_relative_pos_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx].cpu()).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1

        # delete merging weights
        part_fc.append(trt.PluginField("patchMerge_norm_gamma_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
        weight_idx += 1
        part_fc.append(trt.PluginField("patchMerge_norm_beta_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
        weight_idx += 1
        part_fc.append(trt.PluginField("patchMerge_linear_kernel_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
        weight_idx += 1
        part_fc.append(trt.PluginField("attn_mask_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
        weight_idx += 1

    part_fc.append(trt.PluginField("patchEmbed_proj_kernel", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
    weight_idx += 1
    part_fc.append(trt.PluginField("patchEmbed_proj_bias", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
    weight_idx += 1
    part_fc.append(trt.PluginField("patchEmbed_norm_gamma", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
    weight_idx += 1
    part_fc.append(trt.PluginField("patchEmbed_norm_beta", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
    weight_idx += 1
    part_fc.append(trt.PluginField("norm_gamma", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
    weight_idx += 1
    part_fc.append(trt.PluginField("norm_beta", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
    weight_idx += 1
    
    plg_registry = trt.get_plugin_registry()
    swinTransformer_plg_creator = plg_registry.get_plugin_creator("CustomSwinTransformerPlugin", "1", "")
    #print("1")
    pfc = trt.PluginFieldCollection([max_batch_size, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale, depths_f, num_heads_f] + part_fc)
    #print("222")
    fn = swinTransformer_plg_creator.create_plugin("swin_transformer" + str(kind_num), pfc)
    inputs = [input]
    #print("???????")
    sw = network.add_plugin_v2(inputs, fn) 
    
    set_output_name(sw, "swin_transformer_", "fuck" + str(kind_num))
    return sw

def load_weights(inputbase):
    weights_dict = dict()
    try:
        tensor_dict = torch.load(inputbase,
                                 map_location='cpu') #加载权重
        #tensor_dict = tensor_dict['model']
        # remove training-related variables in the checkpoint
        param_names = [key for key in sorted(tensor_dict)]
        for pn in param_names:
            if isinstance(tensor_dict[pn], np.ndarray):
                tensor = tensor_dict[pn]
            else:
                tensor = tensor_dict[pn].numpy()

            shape = tensor.shape

            ##to be compatible with SwinTransformerWeightTransposeQKVWeight
            if "index" in pn:
                flat_tensor = tensor.flatten().astype(dtype=np.int64)
                weights_dict[pn] = torch.tensor(flat_tensor, dtype=torch.int64).cuda()
            elif "table" in pn:
                flat_tensor = tensor.flatten().astype(dtype=np.float32)
                weights_dict[pn] = torch.tensor(flat_tensor, dtype=torch.float32).cuda()
            else:
                flat_tensor = tensor.flatten().astype(dtype=np.float32)
                weights_dict[pn] = torch.tensor(flat_tensor, dtype=torch.float32)

            shape_str = "{} ".format(len(shape)) + " ".join([str(d) for d in shape])
            #print("TensorRT name: {:}, shape: {:}".format(pn, shape_str))

    except Exception as error:
        TRT_LOGGER.log(TRT_LOGGER.ERROR, str(error))

    return weights_dict



class SwinTransformerWeightTransposeQKVWeight(object):
    def __init__(self, layer_num, window_size, depths, num_heads, ths_path, weights=None):
        """weights need be a state_dict of swin transformer model"""
        block_weight_suffixes = ['attn.qkv.weight',
                                 'attn.qkv.bias',
                                 'attn.proj.weight',
                                 'attn.proj.bias',
                                 'mlp.fc1.weight',
                                 'mlp.fc1.bias',
                                 'mlp.fc2.weight',
                                 'mlp.fc2.bias',
                                 'norm1.weight',
                                 'norm1.bias',
                                 'norm2.weight',
                                 'norm2.bias']
        # merging weights
        layer_weight_suffixes = ['downsample.norm.weight',
                                  'downsample.norm.bias',
                                  'downsample.reduction.weight']
        sw_weight_suffixes = ['patch_embed.proj.weight',
                              'patch_embed.proj.bias',
                              'patch_embed.norm.weight',
                              'patch_embed.norm.bias',
                              'norm.weight',
                              'norm.bias']
        self.layer_num = layer_num
        self.depths = depths
        self.weights = []
        torch.classes.load_library(ths_path)
        gen_relative_pos_bias = torch.ops.fastertransformer.gen_relative_pos_bias
        if weights is None:
            print("[ERROR][SwinTransformerWeights::__init__] weights should not be empty!")
            exit(-1)
        else:
            self._generated_weights = False
            #loop over layers
            for layer_idx in range(layer_num):
                ##loop over blocks
                for block_idx in range(depths[layer_idx]):
                    ###block_weight_suffixes
                    for block_weight_suffix in block_weight_suffixes:
                        weight_name = 'swin_unet.layers.{}.blocks.{}.{}'.format(layer_idx, block_idx, block_weight_suffix)
                        if weight_name in weights:
                            #transpose qkv weight [3*head*size, k] --> [k, head*3*size]
                            if "attn.qkv.weight" in weight_name:
                                shape = weights[weight_name].shape
                                #in case we flatten this weight
                                if len(shape) == 1:
                                    dim = int(math.sqrt(shape[0]/3))
                                    weights[weight_name] = weights[weight_name].reshape([3*dim, dim])
                                    shape = weights[weight_name].shape
                                weights[weight_name] = weights[weight_name].reshape([3, num_heads[layer_idx], int(shape[0]/3/num_heads[layer_idx]), -1]).permute(3, 1, 0, 2).reshape(shape[1], -1)
                            #transpose qkv bias
                            if "attn.qkv.bias" in weight_name:
                                shape = weights[weight_name].shape
                                weights[weight_name] = weights[weight_name].reshape([3, num_heads[layer_idx], int(shape[0]/3/num_heads[layer_idx])]).permute(1, 0, 2).reshape(-1)
                            self.weights.append(weights[weight_name])
                        else:
                            print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(weight_name))
                            exit(-1)
                    ###get relative position bias
                    index_name = 'swin_unet.layers.{}.blocks.{}.attn.relative_position_index'.format(layer_idx, block_idx)
                    table_name = 'swin_unet.layers.{}.blocks.{}.attn.relative_position_bias_table'.format(layer_idx, block_idx)
                    if index_name in weights and table_name in weights:
                        relative_position_bias = gen_relative_pos_bias(weights[table_name], weights[index_name], window_size, num_heads[layer_idx])
                        self.weights.append(relative_position_bias)
                    else:
                        print("[ERROR][SwinTransformerWeights::__init__] missing weight {} or {}.".format(index_name, table_name))
                        exit(-1)
                ##deal with layer weights
                ###loop over layer_weight_suffixes
                for layer_weight_suffix in layer_weight_suffixes:
                    weight_name = 'swin_unet.layers.{}.{}'.format(layer_idx, layer_weight_suffix)
                    if weight_name in weights:
                        self.weights.append(weights[weight_name])
                    else:
                        ####the last layer has not dowmsample weight
                        if layer_idx == layer_num - 1:
                            self.weights.append(torch.Tensor())
                        else:
                            print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(weight_name))
                            exit(-1)
                ###get attn_mask (same for each layer, some layer may not has one)
                attn_mask_name = 'swin_unet.layers.{}.blocks.1.attn_mask'.format(layer_idx)
                if attn_mask_name in weights:
                    self.weights.append(weights[attn_mask_name])
                else:
                    self.weights.append(torch.Tensor())
            #deal with sw weights
            for sw_weight_suffix in sw_weight_suffixes:
                weight_name = 'swin_unet.{}'.format(sw_weight_suffix)
                if weight_name in weights:
                    self.weights.append(weights[weight_name])
                else:
                    print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(weight_name))
                    exit(-1)



    def to_cuda(self):
        for idx, v in enumerate(self.weights):
            self.weights[idx] = v.cuda()

    def to_half(self):
        for idx, v in enumerate(self.weights):
            self.weights[idx] = v.half()

