cd ../model
onnxsim model.onnx model_sim.onnx 1 --dynamic-input-shape --input-shape 1,1,512,512
cd ../scripts
python3 Parser.py