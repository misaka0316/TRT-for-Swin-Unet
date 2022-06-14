# Swin-Unet - TensorRT Implementation
运行环境：TensorRT:8.4.1.4 + A10(24G)
模型训练以及推理使用的图像尺寸为(512, 512)

# Preparation
运行代码前，需要将 Swin-Unet Onnx 模型存放在 ``model`` 目录下[(模型下载)](https://pan.baidu.com/s/1fIlvpvnwB75Q60-IygoF5g?pwd=a7ye)
用于推理的图像存放在 ``data`` 目录下，以下将使用原论文作者给出的数据为例(已存放于 ``data`` 目录下)

# To Start

## Dependencies
安装Python dependencies:
```
pip3 install -r requirements.txt
```

## build and serialize
进入 ``scripts`` 执行接下来的操作:
```
cd scripts
```

运行 ``build.sh``
```
./build.sh
```

序列化后的文件将存放为 ``output/swin-unet.plan`` 

## infer
运行 ``infer.sh`` 对 ``data/case0005_slice000.npz`` 内的Image进行推理，推理结果将存放为 ``output/output.npy`` 
```
./infer.sh
```

# Citation
```
@INPROCEEDINGS{chien2021investigating,
  author={Chien, Chung-Ming and Lin, Jheng-Hao and Huang, Chien-yu and Hsu, Po-chun and Lee, Hung-yi},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Investigating on Incorporating Pretrained and Learnable Speaker Representations for Multi-Speaker Multi-Style Text-to-Speech}, 
  year={2021},
  volume={},
  number={},
  pages={8588-8592},
  doi={10.1109/ICASSP39728.2021.9413880}}
```