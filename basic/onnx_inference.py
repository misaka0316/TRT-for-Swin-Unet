import numpy as np
import onnxruntime as rt
from time import time_ns


def onnx_runtime():
    imgdata = (np.random.randn(1, 1, 224, 224)*10).astype(np.float32)
    timePerInference = 0
    for i in range(30):
        sess = rt.InferenceSession('../model/model.onnx', providers=['CUDAExecutionProvider'])
        input_name = sess.get_inputs()[0].name  
        output_name = sess.get_outputs()[0].name
        t0 = time_ns()
        pred_onnx = sess.run([output_name], {input_name: imgdata})
        t1 = time_ns()
        timePerInference += (t1-t0)
    timePerInference = timePerInference/1000/1000/30
    print(f"Onnx Inference time:", timePerInference)
 
    # print("outputs:")
    # print(np.array(pred_onnx))
 
onnx_runtime()
