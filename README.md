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

## 6.23
### Position-guided Text Prompt for Vision-Language Pre-training
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Chang_Making_Vision_Transformers_Efficient_From_a_Token_Sparsification_View_CVPR_2023_paper.pdf
- code: https://github.com/sail-sg/ptp
- citation:
```bash
@article{wang2022ptp,
  title={Position-guided Text Prompt for Vision Language Pre-training},
  author={Wang, Alex Jinpeng and Zhou, Pan and Shou, Mike Zheng and Yan, Shui Cheng},
  journal={https://arxiv.org/abs/2212.09737},
  year={2022}
}
```
<br>方向是VLP（Vision Language Pretraining），在视觉和语言上进行预训练。但是目前已有的方法都缺少图像的位置信息，导致在某些任务上的表现不好，比如视觉推理。这个文章主要是让模型学习到视觉位置信息，具体方法是先把图片划分成N*N的方块，然后用常见的目标检测模型检测出每个方块中的目标，然后训练模型回答一个填空问题，例如，The block [P] has a object [O]，给P问O或者给O问P。在一些数据集上和VLP模型相比，提点明显，和Object Detection模型相比，准确率类似，但速度更快。


## 6.28
### DepGraph: Towards Any Structural Pruning
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.pdf
- code: https://github.com/VainF/Torch-Pruning
- citation:
```bash
@inproceedings{fang2023depgraph,
  title={Depgraph: Towards any structural pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16091--16101},
  year={2023}
}
```
方向是网络剪枝。可以在保持网络准确性的同时提升网络的速度。方法是依赖图，通过明确网络各节点之间的依赖关系，当用户指定某个要剪枝的网络层的时候，可以识别出需要同时剪枝的关联剪枝组，从而能够在保持网络结构完整性和准确性的同时实现剪枝。代码应该19年就开始写了，今年发了cvpr，已经封装成了python包，在github上也开源了，目前1.3k star。

### Diffusion Probabilistic Model Made Slim
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Diffusion_Probabilistic_Model_Made_Slim_CVPR_2023_paper.pdf
<br>是对LDM（Latent Diffusion Model）的性能优化。LDM生成图片的效果好，但是计算量大。作者先试着训练了一个小参数量的LDM，但是生成的图片效果很差。通过分析发现LDM有忽略高频信息的偏好，所以在LDM里面加入了wavelet gating（小波门控）这个信号处理领域的算法，帮助了LDM捕捉高频信息。最终实验显示，在保持相同保真度的同时，实现了8-18倍的性能提升。

## 6.29
### Partial Network Cloning
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Ye_Partial_Network_Cloning_CVPR_2023_paper.pdf
- code: https://github.com/JngwenYe/PNCloning
方向是网络克隆。目前已有的方法是知识蒸馏，通过教师模型训练学生模型。网络克隆是直接把一个网络的模块插入到另一个网络中。这个文章提出了一个方法来查确定插入模块和被插入位置，能够获取最好的准确性。实验显示网络克隆相比于知识蒸馏，能够提升准确性和局部性。

### Slimmable Dataset Condensation
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Slimmable_Dataset_Condensation_CVPR_2023_paper.pdf
<br>方向是数据集压缩。现有的数据集压缩方法在一定存储空间或者带宽下使用，当预设条件改变时，必须重新使用原数据集进行压缩，但有时候已经无法获取到原数据集了。通过对这个问题的分析发现了两个关键的因素：不同压缩时间训练出的网络的不一致性、无法确定压缩数据集的解空间。这个文章提出了针对这两个问题的解决方法，从而能够在压缩后的数据集上做进一步的压缩。实验结果显示在多个数据集上取得了较好的结果。

## 7.4
### Task Residual for Tuning Vision-Language Models
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Task_Residual_for_Tuning_Vision-Language_Models_CVPR_2023_paper.pdf
- code: https://github.com/geekyutao/TaskRes
- citation:
```bash
@inproceedings{yu2023task,
  title={Task Residual for Tuning Vision-Language Models},
  author={Yu, Tao and Lu, Zhihe and Jin, Xin and Chen, Zhibo and Wang, Xinchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10899--10909},
  year={2023}
}
```
方向是vision language model（VLM）预训练模型的迁移学习。现有的方法要么会抛弃预训练学习到的知识，要么会过度依赖预训练学习到的知识。这篇文章在训练时会冻结预训练模型的参数，同时新增一个分类器，用来学习下游任务的知识，从而解决了现有方法的问题。实验显示在多个数据集上超过了之前方法的结果。

### Deep Graph Reprogramming
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Jing_Deep_Graph_Reprogramming_CVPR_2023_paper.pdf
<br>方向是图的重编码，意思是重组一个gnn，不添加训练特征也不添加模型参数，使得这个gnn能用于不同的下游任务。方法是数据重编码和模型重编码，数据重编码是重组训练特征，使得这些特征能够代表不同任务的训练数据，模型重编码是重组模型，使得模型能够处理不同的下游任务。实验在14个数据集上进行了验证，发现这种方法的结果和每个任务分别训练模型的结果相似。

### Distribution Shift Inversion for Out-of-Distribution Prediction
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Distribution_Shift_Inversion_for_Out-of-Distribution_Prediction_CVPR_2023_paper.pdf
- code: https://github.com/yu-rp/Distribution-Shift-Iverson
- citation:
```bash
@article{yu2023distribution,
    author    = {Runpeng Yu, Songhua Liu, Xingyi Yang, Xinchao Wang},
    title     = {Distribution Shift Inversion for Out-of-Distribution Prediction},
    journal   = {The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
    year      = {2023},
}
```
方向是Out of Distribution(OoD) prediction。有时候训练数据和测试数据的分布是不一样的，导致在训练数据上训练的模型在测试数据上的效果不好。这篇文章的方法是在测试阶段会用一个在训练数据上训练的扩散模型，测试数据会先和高斯噪声线性组合，然后通过这个扩散模型，这个扩散模型会把测试数据的分布转化成训练数据的分布，这时候再通过预测模型时的效果就会好很多。实验显示，当这个方法被运用在现有的OoD方法上时，会有普遍的准确率提升。

## 7.5
### Master: Meta Style Transformer for Controllable Zero-Shot and Few-Shot Artistic Style Transfer
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_Master_Meta_Style_Transformer_for_Controllable_Zero-Shot_and_Few-Shot_Artistic_CVPR_2023_paper.pdf
<br>方向是图片风格迁移。使用Transformer做风格迁移，存在参数量大、内容失真的问题。这篇文章对vanilla Transformer进行了改进，不同的网络层共享同一组参数，使得参数量变少。同时对内容特征做了学习化的扩展，从而可以更好的保证内容的相似性。同时还加上了meta learning，使得模型能够应用于few shot learning和zero shot learning。同时还支持了text guided风格迁移。实验显示在few shot learning和zero shot learning场景下的结果更好。

### EqMotion: Equivariant Multi-agent Motion Predictionwith Invariant Interaction Reasoning
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.pdf
- code: https://github.com/MediaBrain-SJTU/EqMotion
- citation:
```bash
@inproceedings{xu2023eqmotion,
  title={EqMotion: Equivariant Multi-agent Motion Prediction with Invariant Interaction Reasoning},
  author={Xu, Chenxin and Tan, Robby T and Tan, Yuhong and Chen, Siheng and Wang, Yu Guang and Wang, Xinchao and Wang, Yanfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1410--1420},
  year={2023}
}
```
方向是动作预测。这个方向的一个基本原则是要保持在欧式空间下的动作的等变性和对象交互的不变性。但目前很多方法都忽略了这一点。这个文章会使用一个等变性空间特征学习模块学习保证动作的等变性。同时使用一个不变性交互推理模块保证对象交互的不变性。还是使用了不变性模式特征学习模块增强动作的表征。实验显示在粒子运动、分子运动、人类骨架运动、行人轨迹预测上取得了SOTA结果，而且提点明显。
