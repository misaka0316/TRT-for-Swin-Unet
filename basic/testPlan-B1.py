import os
import sys
import ctypes
import numpy as np
from glob import glob 
from time import time_ns
from datetime import datetime as dt
from tqdm import tqdm
from torch.utils.data import DataLoader
from cuda import cudart
from medpy import metric
import torch
import tensorrt as trt
from scipy.ndimage import zoom
import argparse
import h5py
import sys
sys.path.append("../Swin-Unet-main")
from datasets.dataset_synapse import Synapse_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--flag', type=str,
                    default='FP32', help='FP32/TF32/FP16 for builder')
args = parser.parse_args()

planFile = f"./output/model-{args.flag}.plan"
h5testDataPath = "../data/Synapse/test_vol_h5/"
soFileList = glob(planFile + "*.so")

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

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

db_test = Synapse_dataset(base_dir=h5testDataPath, split="test_vol", list_dir="../Swin-Unet-main/lists/lists_Synapse")
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

metric_list_mean = 0.0
for i_batch, sampled_batch in tqdm(enumerate(testloader)):
    h, w = sampled_batch["image"].size()[2:]
    image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0] #(1, B, H, W)
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() #(B, H, W)
    # inputs = np.zeros((image.shape[0], 224, 224)) #inputs for engine
    prediction = np.zeros_like(label) #outputs for calculate metric (B, H, W)
    if len(image.shape) == 3:
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != 224 or y != 224:
                slice = zoom(slice, (224 / x, 224 / y), order=3)

            slice = slice.reshape(1, 1, 224, 224)

            context.set_binding_shape(0, slice.shape)

            bufferH = []
            bufferH.append(slice.astype(np.float32).reshape(-1))
            bufferH.append(np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1))))

            bufferD = []
            bufferD.append(cudart.cudaMalloc(bufferH[0].nbytes)[1])
            bufferD.append(cudart.cudaMalloc(bufferH[1].nbytes)[1])

            cudart.cudaMemcpy(bufferD[0], bufferH[0].ctypes.data, bufferH[0].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            context.execute_v2(bufferD)

            cudart.cudaMemcpy(bufferH[1].ctypes.data, bufferD[1], bufferH[1].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

            for i in range(10):
                context.execute_v2(bufferD)

            torch.cuda.synchronize()
            t0 = time_ns()
            for i in range(30):
                context.execute_v2(bufferD)
            torch.cuda.synchronize()
            t1 = time_ns()
            timePerInference = (t1-t0)/1000/1000/30

            indexOutput = engine.get_binding_index('output')

            outputs = torch.argmax(torch.softmax(torch.from_numpy(bufferH[indexOutput]), dim=1), dim=1).squeeze(0).cpu().detach().numpy() #(H, W)
            
            x, y = outputs.shape[0], outputs.shape[1]
            if x !=512  or y != 512:
                outputs = zoom(outputs, (512 / x, 512 / y), order=0)
            prediction[ind] = outputs

            for i in range(2):                
                cudart.cudaFree(bufferD[i])
            # break
            # '''
    metric_list = []
    for i in range(1, 9):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    
    metric_i = np.array(metric_list)
    metric_list_mean += metric_i
    print(f"idx: {i_batch} case: {case_name} mean_dice: {np.mean(metric_i, axis=0)[0]} mean_hd95: {np.mean(metric_i, axis=0)[1]}")
    # break # test for only one case
    print(f"idx {i_batch} Inference time:", timePerInference)
    # np.save(saveFile, bufferH[indexOutput])
    # print(f"idx {i_batch} Output shape:", bufferH[indexOutput].shape)
    # print('\n')

metric_list_mean = metric_list_mean / len(db_test)
for i in range(1, 9):
    print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list_mean[i-1][0], metric_list_mean[i-1][1]))
    performance = np.mean(metric_list_mean, axis=0)[0]
    mean_hd95 = np.mean(metric_list_mean, axis=0)[1]
print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
print("Testing Finished!")

# print("Succeeded save {}".format(saveFile))
# print(bufferH[indexOutput])
# '''
