import os
import sys
import tensorrt as trt
import ctypes

onnxFile = "../model/model_sim.onnx"
trtFile = "../output/swin-unet.plan"
# soFile = "./LayerNorm.so"

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
# ctypes.cdll.LoadLibrary(soFile)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile1 = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = 24 << 30
#config.flags = 1 << int(trt.BuilderFlag.FP16)
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

inputImage = network.get_input(0)
profile1.set_shape(inputImage.name, (1, 1, 512, 512), (16, 1, 512, 512), (32, 1, 512, 512))
config.add_optimization_profile(profile1)
  
engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine and saving swin-unet.plan!")
with open(trtFile, 'wb') as f:
    f.write(engineString)