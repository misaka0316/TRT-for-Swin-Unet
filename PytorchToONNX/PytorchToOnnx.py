import onnx
import torch
import torch.nn as nn
import numpy as np

import os

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

ModelPath = "../Swin-Unet-main/model_out/620-224-whole-model.pth"
StateDictPath = "../Swin-Unet-main/model_out/best.pth"
OnnxPath = "../model/model.onnx"

model = torch.load(ModelPath)
model.load_state_dict(torch.load(StateDictPath))
model.eval()

inputs = torch.randn(224, 224)
#inputs = np.load("./InputData/case0005_slice000.npz")['image']
inputs = np.expand_dims(inputs, axis=0)
inputs = np.expand_dims(inputs, axis=0)
inputs = torch.from_numpy(inputs)
device = torch.device("cuda:0")
inputs = inputs.to(device)
input_name = 'input'
output_name = 'output'
torch.onnx.export(model,               # model being run
                  inputs,                         # model input
                  OnnxPath,   # where to save the model (can be a file or file-like object)
                  opset_version=12,          # the ONNX version to export the model to
                  input_names=[input_name],   # the model's input names
                  output_names=[output_name],
                  dynamic_axes= {
                    input_name: {0: 'B'},
                    output_name: {0: 'B'}}
                  )


# os.system("onnxsim ./model/model.onnx ./model/model.onnx 1 --dynamic-input-shape --input-shape 1,1,512,512")

