import os
import sys
import tensorrt as trt
import ctypes
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--flag', type=str,
                    default='FP32', help='FP32/TF32/FP16 for builder')
args = parser.parse_args()

onnxFile = "../model/model.onnx"
trtFile = f"./output/model-{args.flag}.plan"
# soFile = "./LayerNorm.so"

logger = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(logger, '')
# ctypes.cdll.LoadLibrary(soFile)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile1 = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = 24 << 30
if args.flag == "TF32":
    config.flags = 1 << int(trt.BuilderFlag.TF32)
elif args.flag == "FP16":
    config.flags = 1 << int(trt.BuilderFlag.FP16)
else:
    config.flags = config.flags & ~ (1 << int(trt.BuilderFlag.TF32))
parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFile):
    print("Failed finding model.onnx file!")
    exit()
print("Succeeded finding model.onnx file!")
with open(onnxFile, 'rb') as model:
    if not parser.parse(model.read()):
        print("Failed parsing model.onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing model.onnx file!")

inputs = network.get_input(0)
inputs.shape = [-1, 1, 224, 224]
profile1.set_shape(inputs.name, (1, 1, 224, 224), (1, 1, 224, 224), (1, 1, 224, 224))
config.add_optimization_profile(profile1)
  
engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine and saving swin-unet.plan!")
with open(trtFile, 'wb') as f:
    f.write(engineString)
