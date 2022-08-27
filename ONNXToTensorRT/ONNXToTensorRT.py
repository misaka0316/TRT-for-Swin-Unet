import os
import sys
import argparse
import tensorrt as trt
import onnx
import ctypes
import onnx_graphsurgeon as gs
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--flag', type=str,
                    default='FP32', help='FP32/TF32/FP16 for builder')
args = parser.parse_args()

onnxFile = "../model/model-tmp.onnx"
trtFile = f"./output/model-{args.flag}.plan"
PluginFile = "./plugin/libpyt_swintransformer.so"
PluginFile2 = "./plugin/libswinTransformer_plugin.so"
tmp = "./plugin/LayerNorm.so"
inputbase = "../Swin-Unet-main/model_out/best.pth"

sys.path.insert(0, "./plugin")
import LoadSwinTransformerWeightTransposeQKVWeight

#替换节点
graph = gs.import_onnx(onnx.load("../model/model.onnx"))

Num = 0

#第一个SwimBlock x 2
Expand = [node for node in graph.nodes if node.name == "ReduceMean_25"][0]
Add = [node for node in graph.nodes if node.name == "Add_201"][0]

inputTensor = Expand.inputs[0]
outputTensor = Add.outputs[0]
#Add2.o(0).inputs[0] = inputTensor
#Add2.o(1).inputs[0] = inputTensor

layerNorm = gs.Node("LayerNorm", "CustomSwinTransformerPlugin-0", inputs=[inputTensor], outputs=[outputTensor])
graph.nodes.append(layerNorm)

Add.outputs = []
graph.cleanup().toposort()
Num = Num + 1

#第二个SwimBlock x 2
Expand = [node for node in graph.nodes if node.name == "ReduceMean_257"][0]
Add = [node for node in graph.nodes if node.name == "Add_433"][0]

inputTensor = Expand.inputs[0]
outputTensor = Add.outputs[0]

layerNorm = gs.Node("LayerNorm", "CustomSwinTransformerPlugin-1", inputs=[inputTensor], outputs=[outputTensor])
graph.nodes.append(layerNorm)

Add.outputs = []
graph.cleanup().toposort()
Num = Num + 1

#第三个SwimBlock x2
Expand = [node for node in graph.nodes if node.name == "ReduceMean_489"][0]
Add = [node for node in graph.nodes if node.name == "Add_665"][0]

inputTensor = Expand.inputs[0]
outputTensor = Add.outputs[0]

layerNorm = gs.Node("LayerNorm", "CustomSwinTransformerPlugin-2", inputs=[inputTensor], outputs=[outputTensor])
graph.nodes.append(layerNorm)

Add.outputs = []
graph.cleanup().toposort()
Num = Num + 1

#第4个Swim Block x 2
Expand = [node for node in graph.nodes if node.name == "ReduceMean_721"][0]
Add = [node for node in graph.nodes if node.name == "Add_858"][0]

inputTensor = Expand.inputs[0]
outputTensor = Add.outputs[0]

layerNorm = gs.Node("LayerNorm", "CustomSwinTransformerPlugin-3", inputs=[inputTensor], outputs=[outputTensor])
graph.nodes.append(layerNorm)

Add.outputs = []
graph.cleanup().toposort()
Num = Num + 1

onnx.save(gs.export_onnx(graph), onnxFile) #保存

#构建
logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

#加载plugin
trt.init_libnvinfer_plugins(logger, '')
ctypes.CDLL(PluginFile2, mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(PluginFile, mode=ctypes.RTLD_GLOBAL)
ctypes.cdll.LoadLibrary(tmp)
profile = builder.create_optimization_profile()
profileC = builder.create_optimization_profile()
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


#print(swinTransformer_plg_creator)

for num_name in range(4):
    weights_dict = LoadSwinTransformerWeightTransposeQKVWeight.load_weights(inputbase)
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if layer.name == "CustomSwinTransformerPlugin-" + str(num_name):
            inputTensor = layer.get_input(0)
            outputTensor = layer.get_output(0)
            print("get output")
            PluginOutput = LoadSwinTransformerWeightTransposeQKVWeight.swin_transformer(network, inputTensor, weights_dict, PluginFile, num_name).get_output(0)

    for index in range(network.num_layers):
        layer = network.get_layer(index)
        for i in range(layer.num_inputs):
            if layer.get_input(i) == outputTensor:
                print(i)
                print("find same input layer")
                layer.set_input(i, PluginOutput)

inputs = network.get_input(0)

inputs.shape = [1, 1, 224, 224]

profile.set_shape(inputs.name, [1, 1, 224, 224], [1, 1, 224, 224], [1, 1, 224, 224])

config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine and saving model.plan!")
# context = engineString.create_execution_context()
# print("succeed create context")
with open(trtFile, 'wb') as f:
    f.write(engineString)

