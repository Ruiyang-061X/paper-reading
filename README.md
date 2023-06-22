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

## 6.21
### DOAD: Decoupled One Stage Action Detection Network
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023W/LSHVU/papers/Chang_DOAD_Decoupled_One_Stage_Action_Detection_Network_CVPRW_2023_paper.pdf
<br>视频动作识别以往的方法通常是两阶段的，第一个阶段是actor-detection，第二个阶段是action-recognition，这个论文的方法是单阶段的，模型是个two-branch model，一个分支做actor-detection，另一个分支做action-recognition，做action-recognition的分支用到了transformer，但进行了改进，把自注意力机制的向量叉乘换成了矩阵的乘积，这样计算可以更多的表示actor和context的关联，实验结果和两阶段的SOTA持平，但性能更好。在我看来这篇论文主要就是串行改并行，然后提升了性能。

### Affordance Grounding from Demonstration Video to Target Image
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Affordance_Grounding_From_Demonstration_Video_To_Target_Image_CVPR_2023_paper.pdf
- code: https://github.com/showlab/afformer
<br>任务感觉比较小众，应用应该也是在AR领域。是给定一个图片和对这个图片中物体的操作教程视频，输出如何操作物体的热力图。模型结构是encoder-decoder模式，encoder是常见的backbone，decoder中使用多个空间和时间尺度，空间特征会使用自注意力，时间特征会使用交叉注意力，实验结果上提点明显。

## 6.22
### Making Vision Transformers Efficient from A Token Sparsification View
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Chang_Making_Vision_Transformers_Efficient_From_a_Token_Sparsification_View_CVPR_2023_paper.pdf
<br>ViT（Vison Transformer）存在计算量过大，性能过低的问题。已有的优化方法存在准确率下降、无法处理局部特征、无法适用于下游任务等问题。这个论文提出了STViT（Semantic Token ViT），Semantic Token是特征点集合的中心，能代表图片中的关键语义信息，从而较少数量的Semantic Token有和大量的图片Token相同的效果。实验显示在物体识别和实例分割这些任务上，在保持相似结果的前提下，减少了30%的计算量。实现方法是先对图片做spatial-downpooling，然后用attention做spatial-uppooling，就得到了Semantic Token。

### MIST : Multi-modal Iterative Spatial-Temporal Transformer for Long-form Video Question Answering
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_MIST_Multi-Modal_Iterative_Spatial-Temporal_Transformer_for_Long-Form_Video_Question_Answering_CVPR_2023_paper.pdf
- code: https://github.com/showlab/mist
- citation:
```bash
@inproceedings{gao2023mist,
  title={MIST: Multi-modal Iterative Spatial-Temporal Transformer for Long-form Video Question Answering}, 
  author={Difei Gao and Luowei Zhou and Lei Ji and Linchao Zhu and Yi Yang and Mike Zheng Shou},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14773--14783},
  year={2023}
}
```
解决的问题是视频问题回答，包括自然语言回答和多项选择问题。已有的方法在图像和短视频上已经有了很好的结果，但在长视频上，存在密集特征导致计算量无法接受、稀疏特征无法很好的处理长视频，长视频有一些特点，包括多个事件、多种粒度的事件和推理性。这个文章的方法是根据问题去挑选视频中相关的帧和帧中的相关图片区域，然后挑选出的特征会经过注意力，还有模型的多层中会循环挑选-注意力这个操作，从而能获得对长视频的较好表征。实验结果显示在较多数据集上有SOTA结果，并且性能更好。
