# paper-reading
## 6.18
### UrbanIR: Large-Scale Urban Scene Inverse Rendering from a Single Video
- from: arxiv
- paper: https://arxiv.org/abs/2306.09349
- website: https://urbaninverserendering.github.io/
<br>可以对车载摄像头视角的城市景观视频进行重新渲染，支持光照调整、3d物体插入、黑夜模拟这些操作，感觉模型主要是学习到了对于阴影的知识。训练是图片维度的训练，使用了结合归一化图像、语义分割、去阴影图像、可见度图像、最终渲染图像的联合损失函数。感觉写文章要先有一个完整的故事，然后把这个故事叙述清楚，这篇文章感觉是直接上手做的，然后拼凑出了一些应用。

### Seeing the World through Your Eyes
- from: arxiv
- paper: https://arxiv.org/abs/2306.09348
- website: https://world-from-eyes.github.io/
<br>论文标题的意思是使用人眼的视频构建人实际上看到的3d世界。使用3d辐射场、角膜位置姿势、虹膜纹理信息进行训练，人类虹膜纹理信息具有通用特点，所以模型会针对虹膜纹理信息预训练，可以提高渲染的质量。文章中还涉及一些人眼相关的生物知识。这篇文章的故事感觉非常清楚，比上面这篇要好很多。

## 6.19
### All in One: Exploring Unified Video-Language Pre-training
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_All_in_One_Exploring_Unified_Video-Language_Pre-Training_CVPR_2023_paper.pdf
- code: https://github.com/showlab/all-in-one
- citation:
```bash
@article{wang2022allinone,
  title={All in One: Exploring Unified Video-Language Pre-training},
  author={Wang, Alex Jinpeng and Ge, Yixiao and Yan, Rui and Ge Yuying and Lin, Xudong and Cai, Guanyu  and Wu, Jianping and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
提出了一个VLP（Video Language Pretraining）领域的第一个端到端的模型，all-in-one Transformer。经过微调后可以解决文本-视频检索、视频-问题回答、视频-多项选择问题和视频标题任务，在10个数据集上取得了SOTA的结果。模型结构上从视频编码器+文本编码器+视频文本特征融合器的结构变为端到端的统一化模型。同时改进了时序自注意力机制，改进的方法是把多帧的图像特征的一部分特征向下一帧特征循环，最后一帧的特征向第一帧循环，从而在做自注意力计算的时候可以自然的包括帧与帧之前的关联。有了这种方法之后就不需要将多帧的特征并在一起进行自注意力计算，能够降低时间复杂度而且准确率也能得到提高。

## 6.20
### Towards Fast Adaptation of Pretrained Contrastive Models for Multi-channel Video-Language Retrieval
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Towards_Fast_Adaptation_of_Pretrained_Contrastive_Models_for_Multi-Channel_Video-Language_CVPR_2023_paper.pdf
- code(coming): https://github.com/XudongLinthu/upgradable-multimodal-intelligence
<br>通过对已有方法进行总结，视频表征方式有连续特征向量和离散文本特征两种方式，融合方法有多模态转换器和预训练对比学习模型两种方式，通过组合和实验发现，离散文本特征和预训练对比学习模型的组合的结果最好，甚至超过了之前的SOTA结果。解释为离散文本特征可以提取出视频中的关键视觉信息，然后可以自然的和预训练对比学习模型对齐。
