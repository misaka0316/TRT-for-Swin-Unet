#**Swin-Unet 用于 TRT**
___
- 项目链接[Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)

##**原始模型**

###**模型简介**
- 此模型适用于图像分割,在对医学影像中的多器官和心脏分割任务中从分割速度和准确率超越了全卷积,变换器和卷积的组合取得了更加优异的效果.
- 这是一种用于医学图像分割的类似Unet的纯Transformer.使用带有移位窗口的分层Swin Transformer作为编码器来提取上下文特征,并设计了具有补丁扩展层的基于对称Swin Transformer的解码器来执行采样操作来恢复特征图的空间分辨率.
很好的学习了全局和远程语义信息交互
![模型结构简图](/Swin-Unet-main/moxing_1.jpg)

###**模型优化的难点**
