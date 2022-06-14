import os
import sys
import ctypes
import numpy as np
from glob import glob 
from time import time_ns
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt
import h5py

planFilePath = "../output/"
planFile = planFilePath + "swin-unet.plan"
testDataPath = "../data/"
testData = testDataPath + "case0005_slice000.npz"
saveFile = "../output/output.npy"
# h5testDataPath = "/root/workspace/pick/data/Synapse/test_vol_h5/"
# h5testData = h5testDataPath + "case0008.npy.h5"
soFileList = glob(planFilePath + "*.so")

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)

if os.path.isfile(planFile):
    with open(planFile, 'rb') as planf:
        engine = trt.Runtime(logger).deserialize_cuda_engine(planf.read())
    if engine is None:
        print("Failed loading %s"%planFile)
        exit()
    print("Succeeded loading %s"%planFile)
else:
    print("Failed finding %s"%planFile)
    exit()

context = engine.create_execution_context()

input_image = np.load(testData)['image'].reshape(1, 1, 512, 512)
# input_image = np.expand_dims(h5py.File(h5testData)['image'][:], axis=1)

context.set_binding_shape(0, input_image.shape)

bufferH = []
bufferH.append(input_image.astype(np.float32).reshape(-1))
bufferH.append(np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1))))

bufferD = []
bufferD.append(cudart.cudaMalloc(bufferH[0].nbytes)[1])
bufferD.append(cudart.cudaMalloc(bufferH[1].nbytes)[1])

cudart.cudaMemcpy(bufferD[0], bufferH[0].ctypes.data, bufferH[0].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_v2(bufferD)

cudart.cudaMemcpy(bufferH[1].ctypes.data, bufferD[1], bufferH[1].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(10):
    context.execute_v2(bufferD)

t0 = time_ns()
for i in range(30):
    context.execute_v2(bufferD)
t1 = time_ns()
timePerInference = (t1-t0)/1000/1000/30

indexOutput = engine.get_binding_index('output')

np.save(saveFile, bufferH[indexOutput])
print("Output shape:", bufferH[indexOutput].shape)
print("Inference time:", timePerInference)
print("Succeeded save {}".format(saveFile))
# print(bufferH[indexOutput])

for i in range(2):                
    cudart.cudaFree(bufferD[i])