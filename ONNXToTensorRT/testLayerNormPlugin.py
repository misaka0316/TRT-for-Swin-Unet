#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#

from ast import Num
import os
import ctypes
import numpy as np
from cuda import cudart  # 使用 cuda runtime API
import tensorrt as trt
import torch
from torch import nn

soFilePath      = './LayerNorm.so'
nBS             = 4
nSL             = 64
nEmbedding      = 256
epsilon         = 6e-6

np.random.seed(38)

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def layerNormCPU(bufferH):
    _x = bufferH[0]
    # print(_x.shape)
    nEmbed = bufferH[0].shape[2]
    _10 = _x.reshape(1,1,50176)
    _0  = np.mean(_10,2)[:,:,np.newaxis]
    _1  = _10 - _0
    # print(_1[0,0,1])
    _2  = _1 * _1
    _3  = np.mean(_2,2)[:,:,np.newaxis]
    print(_3.shape)
    _4  = np.array(epsilon,dtype=np.float32)
    _5  = _4.reshape(1,1,1)
    _6  = _3 + _5
    print(_6.shape)
    _7  = np.sqrt(_6)
    print(_7.shape)
    _8  = 1 / _7                # 1/sqrt(...)
    print(_8.shape)
    _9  = _1 * _8
    # print(_9[0,0,2])
    _11 =  _9.reshape(1,1,224,224)
    # print(_11[0,0,0,1] , "test")
    return _11

def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'LayerNorm':
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<0)
    config          = builder.create_builder_config()
    config.max_workspace_size = 4 << 30
    config.flags    = 0

    inputTensorList = []
    inputTensorList.append( network.add_input('inputT', trt.float32, [1,1,224,224]) )

    profile = builder.create_optimization_profile()
    profile.set_shape('inputT', [1, 1, 224, 224], [1, 1, 224, 224], [1, 1, 224, 224])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())
    network.mark_output(pluginLayer.get_output(0))
    #生成引擎
    engineString = builder.build_serialized_network(network, config)
    #生成引擎实例
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0,[1, 1, 224, 224])

    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    
    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput

    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    bufferH.append( np.random.rand(1, 1, 224, 224).astype(np.float32).reshape(1, 1, 224, 224) * 2 - 1)
    bufferH.append( np.empty(context.get_binding_shape(1),dtype=trt.nptype(engine.get_binding_dtype(1))))
    # print(bufferH)
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print("check result:")
    temp1 = bufferH[-1]
    temp3 = layerNormCPU(bufferH[:1])
    _1 = bufferH[:1]
    t1 = torch.tensor(_1).reshape(1,1,50176)
    layer = nn.LayerNorm(50176)
    temp2  = layer(t1).detach()
    temp2  = temp2.numpy().reshape(1,1,224,224)
    print(temp1[0,0,0,2])
    print(temp2[0,0,0,2])
    print(temp3[0,0,0,2])
    print(check(temp3,temp2,True), "max diff=%f"%(np.max(np.abs(temp3 - temp2))))
    
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == '__main__':
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()