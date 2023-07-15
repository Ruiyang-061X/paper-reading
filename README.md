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
```
方向是动作预测。这个方向的一个基本原则是要保持在欧式空间下的动作的等变性和对象交互的不变性。但目前很多方法都忽略了这一点。这个文章会使用一个等变性空间特征学习模块学习保证动作的等变性。同时使用一个不变性交互推理模块保证对象交互的不变性。还是使用了不变性模式特征学习模块增强动作的表征。实验显示在粒子运动、分子运动、人类骨架运动、行人轨迹预测上取得了SOTA结果，而且提点明显。

## 7.6
### Overcoming the Trade-off Between Accuracy and Plausibility in 3D Hand Shape Reconstruction
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Overcoming_the_Trade-Off_Between_Accuracy_and_Plausibility_in_3D_Hand_CVPR_2023_paper.pdf
<br>方向是手的3D重建。非参数化的方法能获得很高的准确率，但是结果通常没有合理性。参数化的方法能获得合理性，但是通常准确率不高。这篇文章提出了一种弱监督的方法，把非参数化的网格拟合和参数化模型MANO结合了起来，而且可以进行端到端的训练。从而获得了准确率和合理性的平衡，在有挑战性的双手和手物体交互场景中取得了很好的结果。

### DBARF: Deep Bundle-Adjusting Generalizable Neural Radiance Fields
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_DBARF_Deep_Bundle-Adjusting_Generalizable_Neural_Radiance_Fields_CVPR_2023_paper.pdf
- code: https://github.com/AIBluefisher/dbarf
- website: https://aibluefisher.github.io/dbarf/
- citation:
```bash
@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Yu and Lee, Gim Hee},
    title     = {DBARF: Deep Bundle-Adjusting Generalizable Neural Radiance Fields},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {24-34}
}
```
方向是3D场景表示。目前的方法在NeRF（Neural Radiance Fields）上取得了很好的结果，这些方法是基于坐标MLP的。但无法在GeNeRF（Generalizable Neural Radiance Fields）上使用，因为GeNeRF需要使用3D CNN或者transformer进行特征提取。这篇文章先分析了GeNeRF上的难点，然后进行了解决。使用了损失特征图计算了一个显示损失函数，同时用自监督的方式进行训练，从而可以在GeNeRF上使用。实验显示在GeNeRF取得了很好的结果。

## 7.7
### NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_NeRF-DS_Neural_Radiance_Fields_for_Dynamic_Specular_Objects_CVPR_2023_paper.pdf
- code: https://github.com/JokerYan/NeRF-DS
- citation:
```bash
@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Yu and Lee, Gim Hee},
    title     = {DBARF: Deep Bundle-Adjusting Generalizable Neural Radiance Fields},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {24-34}
}
```
Dynamic NeRF算法无法使用于反光运动物体。这篇文章将Dynamic NeRF的公式做了调整，是它适用于表面的位置和旋转。同时加入了运动物体的mask，从而可以保持时序关联。在自建数据集上验证发现比现有模型的效果都要好。

### ScarceNet: Animal Pose Estimation with Scarce Annotations
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Li_ScarceNet_Animal_Pose_Estimation_With_Scarce_Annotations_CVPR_2023_paper.pdf
- code: https://github.com/chaneyddtt/ScarceNet
<br>
方向是动物姿态识别。动物姿态识别目前主要的困难在于数据量比较少。这篇文章首先在小数据集上训练一个模型，然后用这个模型给大量图片打标。使用small-loss trick挑选出一小部分好的标签。使用同意检查在剩下的那些图片中挑选出可以重复使用的图片，然后再进行打标。最后还会用学生-老师模型进行一致性限制，剔除掉挑选出的图片中的少量质量差的数据。从而得到了一个大数据集。实验显示在这个数据集上训练得到的模型，在AP-10K上，比半监督模型准确率高很多，在TigDog上，比领域迁移模型的准确率高。

### ContraNeRF: Generalizable Neural Radiance Fields for Synthetic-to-real Novel View Synthesis via Contrastive Learning
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_ContraNeRF_Generalizable_Neural_Radiance_Fields_for_Synthetic-to-Real_Novel_View_Synthesis_CVPR_2023_paper.pdf
<br>方向是3D重建。GeNeRF目前的模型在synthetic-to-real（合成数据上训练，真实数据上推理）场景上存在体积密度经常不准确的问题。这篇文章使用了几何意识对比学习来学习多视角下的特征，同时使用了多视角注意力来增强几何特征的感知。实验显示，在synthetic-to-real场景下，和现有方法相比可以生成质量更好的结果。同时在real-to-real场景下，也取得了SOTA的结果。

### Analyzing and Diagnosing Pose Estimation with Attributions
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/He_Analyzing_and_Diagnosing_Pose_Estimation_With_Attributions_CVPR_2023_paper.pdf
<br>方向是姿态识别。提出了Pose Integrated Gradient。能够根据挑选的关节生成相对应的热力图。同时提出了3个量化指标，并用这些指标对现有的模型进行了比较。发现了手部姿态识别中识别指关节的一条捷径，还有人体姿态识别中的一个反转错误。

## 7.8
### Cross-Domain 3D Hand Pose Estimation with Dual Modalities
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Cross-Domain_3D_Hand_Pose_Estimation_With_Dual_Modalities_CVPR_2023_paper.pdf
<br>方向是手部姿态估计。目前的方法有使用合成的数据去训练模型，但是因为存在领域区别，无法适用于真实数据。这篇文章提出了一个跨域半监督模型，在有标签的合成数据和无标签的真实数据上训练。模型是双模态的，在RGB数据和Depth数据上进行训练。与训练阶段使用了多模态对比学习和注意力融合监督。在精调阶段使用了一种新的自扩散技术，来减少伪标签噪声。实验显示在3D手部姿态估计和2D关键点检测上提点明显。

### DSFNet: Dual Space Fusion Network for Occlusion-Robust 3D Dense Face Alignment
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Li_DSFNet_Dual_Space_Fusion_Network_for_Occlusion-Robust_3D_Dense_Face_CVPR_2023_paper.pdf
- code: https://github.com/lhyfst/DSFNet
<br>方向是脸部对齐。目前的3D密集脸部对齐方法在严重遮挡和大角度视角时难以使用。目前的3D方法直接回归模型中的参数，导致对于2D空间和语义信息使用的较少，但这些信息对于脸部形状和朝向是有用的。这篇文章解决了遮挡和大角度视角问题。先使用可见的脸部特征做回归。然后使用关于脸部模型的先验知识去补全不可见的脸部。最终模型是个融合网络，同时使用图片信息和脸部模型信息做脸部对齐。实验显示在AFLW2000-3D数据集上，取得了3.80% NME，比SOTA提升了5.5%。

### Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Seeing_What_You_Said_Talking_Face_Generation_Guided_by_a_CVPR_2023_paper.pdf
<br>方向是说话脸部生成。输入是连续语音，输出是脸部视频尤其是唇部相关的运动。目前的方法都没有考虑到唇部运动的实际发音。这篇文章使用一个读唇专家阅读唇部运动的内容，惩罚错误的内容。因为数据稀少，使用的是音频-视觉自监督的方式训练读唇专家。基于训练好的读唇专家，使用对比学习来强化唇部和语音的同步，还是用了transformer同步编码语音和视频。验证使用的是两个不同的读唇专家。实验显示超过了之前的SOTA模型。

### 2PCNet: Two-Phase Consistency Training for Day-to-Night Unsupervised Domain Adaptive Object Detection
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Kennerley_2PCNet_Two-Phase_Consistency_Training_for_Day-to-Night_Unsupervised_Domain_Adaptive_Object_CVPR_2023_paper.pdf
- code: https://github.com/mecarill/2pcnet
- citation:
```
@inproceedings{kennerley2023tpcnet,
  title={2PCNet: Two-Phase Consistency Training for Day-to-Night Unsupervised Domain Adaptive Object Detection},
  author={Mikhail Kennerley, Jian-Gang Wang, Bharadwaj Veeravalli, Robby T. Tan},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
}
```
方向是夜间目标检测。目前主要的问题是数据少。目前的方法准确率都不高。小的物体和暗的物体很难检测出来。这篇文章提出了一个一致性两阶段无监督领域适应模型。第一阶段中的教师模型的高分预测结果，会被用于第二阶段学生模型的预测提议，然后重新使用教师模型进行评估。在输入学生模型之前，夜间图片和标签都会做scale-down，从而增强对于小的物体的检测。为了增强对于暗的物体的识别，提出了NightAug这个数据增强方式，通过对白天数据做一些处理，使得它们像夜晚数据。实验显示超过了之前的SOTA模型。

## 7.10
### Panoptic Video Scene Graph Generation
- from: cvpr2023
- papar: https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Panoptic_Video_Scene_Graph_Generation_CVPR_2023_paper.pdf
- code: https://github.com/Jingkang50/OpenPVSG
<br>方向是视频场景图网络（Video Scene Graph, VSG）。属于挖坑之作。目前的VSG是用bounding box表示人和物体，然后描述他们之间的交互。但bounding box的粒度比较粗，会丢失一些细节。这篇文章提出了PVSG（Panoptic Video Scene Graph），其实就是把bounding box换成了segmentation mask，这样粒度就比较细了，并且提出了一个这个任务上的数据集。还给出了一些baseline和一些可行的设计方案。

### The Nuts and Bolts of Adopting Transformer in GANs
- from: cvpr2023
- paper: https://arxiv.org/pdf/2110.13107.pdf
<br>方向是GAN。这篇文章把transformer用在了GAN里面。明确了特征局部性在图像合成中的重要性。发现了自注意力层中的残差连接对于GAN的图片生成有害。通过对transformer的结构进行调整使得能够适用于GAN。提出了STrans-G（generator），与GAN取得了相似的结果。提出了STrans-D（discriminator），与GAN的差距有变小。

### Generating Aligned Pseudo-Supervision from Non-Aligned Data for Image Restoration in Under-Display Camera
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_Generating_Aligned_Pseudo-Supervision_From_Non-Aligned_Data_for_Image_Restoration_in_CVPR_2023_paper.pdf
- code:  https://github.com/jnjaby/AlignFormer
- citation：
```
@InProceedings{Feng_2023_Generating,
   author    = {Feng, Ruicheng and Li, Chongyi and Chen, Huaijin and Li, Shuai and Gu, Jinwei and Loy, Chen Change},
   title     = {Generating Aligned Pseudo-Supervision from Non-Aligned Data for Image Restoration in Under-Display Camera},
   booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   month     = {June},
   year      = {2023},
}
```
方向是UDC（Under Display Camera）Image Restoration。主要问题还是数据集少。目前主流方法是把高清图片的一些细节粘贴到模糊图片的对应地方进行训练。导致了空间不对齐和领域不对齐。这篇文章提出了Domain Alignment Module(DAM)和Geometric Alignment Module(GAM)来解决这些问题。实验显示能够解决上述问题。

### Flexible Piecewise Curves Estimation for Photo Enhancement
- from: cvpr2023
- paper: https://arxiv.org/pdf/2010.13412.pdf
<br>方向是图片增强。这篇文章提出的方法是FlexiCurve，是使用图片全局曲线来强化图片的。调整曲线能够处理物体之间的映射。设计了多任务模型来处理照明问题。这种方法的训练数据可以不使用成对的数据。模型的性能很好。实验结果显示模型的增强能力和性能都达到了SOTA。

### BeautyREC: Robust, Efficient, and Component-Specific Makeup Transfer
- from: cvpr2023
- paper: https://arxiv.org/pdf/2212.05855.pdf
- code: https://github.com/learningyan/BeautyREC
- citation:
```
@inproceedings{BeautyREC,
          author = {Yan, Qixin and Guo, Chunle and Zhao, Jixin and Dai, Yuekun and Loy, Chen Change and Li, Chongyi},
          title = {BeautyREC: Robust, Efficient, and Component-Specific Makeup Transfer},
          booktitle = {Arixv},
          year = {2022}
}
```
方向是化妆迁移。使用了组件特定相关性，支持指定组件（皮肤、嘴唇、眼睛）的化妆迁移。使用了transformer进行全局化妆迁移。放弃了transformer中的循环结构，使用了内容一致性损失和内容编码器，从而实现了单向化妆迁移。提出了BeautyFace这个数据集。实验显示和SOTA结果相比具有有效性，同时模型的参数量只有1M，比SOTA模型少。

### Siamese DETR
- from: cvpr2023
- paper: https://arxiv.org/pdf/2303.18144.pdf
- code: https://github.com/Zx55/SiameseDETR
- citation:
```
@article{chen2023siamese,
  title={Siamese DETR},
  author={Chen, Zeren and Huang, Gengshi and Li, Wei and Teng, Jianing and Wang, Kun and Shao, Jing and Loy, Chen Change and Sheng, Lu},
  journal={arXiv preprint arXiv:2303.18144},
  year={2023}
}
```
方向是目标检测。在DETR上加入了孪生结构和自监督模块。把多视角检测和多视角分割联合起来训练，取得了较好效果。在COCO和VOC上取得了SOTA结果。

### CelebV-Text: A Large-Scale Facial Text-Video Dataset
- from: cvpe2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_CelebV-Text_A_Large-Scale_Facial_Text-Video_Dataset_CVPR_2023_paper.pdf
- website: https://celebv-text.github.io/
- code: https://github.com/CelebV-Text/CelebV-Text
- citation:
```
@inproceedings{yu2022celebvtext,
  title={{CelebV-Text}: A Large-Scale Facial Text-Video Dataset},
  author={Yu, Jianhui and Zhu, Hao and Jiang, Liming and Loy, Chen Change and Cai, Weidong and Wu, Wayne},
  booktitle={CVPR},
  year={2023}
}
```
提出了一个人脸文字视频数据集。有70000个视频和1400000个文字。文字描述了静态和动态信息。通过统计分析发现比其他数据集好。通过评估显示了数据集的有效和潜力。使用表征方法建立了一个benchmark。

### Learning Generative Structure Prior for Blind Text Image Super-resolution
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Learning_Generative_Structure_Prior_for_Blind_Text_Image_Super-Resolution_CVPR_2023_paper.pdf
- code: https://github.com/csxmli2016/MARCONet
- citation:
```
@InProceedings{li2023marconet,
author = {Li, Xiaoming and Zuo, Wangmeng and Loy, Chen Change},
title = {Learning Generative Structure Prior for Blind Text Image Super-resolution},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year = {2023}
}
```
方向是图像超分。使用StyleGAN学习到了大量文字的结构。同时使用codebook存储所有文字的特征，来限制StyleGAN的输出。实验显示取得了SOTA结果。

### Nighttime Smartphone Reflective Flare Removal Using Optical Center Symmetry Prior
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Dai_Nighttime_Smartphone_Reflective_Flare_Removal_Using_Optical_Center_Symmetry_Prior_CVPR_2023_paper.pdf
- website: https://ykdai.github.io/projects/BracketFlare
- code: https://github.com/ykdai/BracketFlare
- citation:
```
@inproceedings{dai2023nighttime,
  title={Nighttime Smartphone Reflective Flare Removal using Optical Center Symmetry Prior},
  author={Dai, Yuekun and Luo, Yihang and Zhou, Shangchen and Li, Chongyi and Loy, Chen Change},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
 }
```
方向是手机反射光去除。应该是开坑之作。提出了光学中心对称先验，就是光和光的反射一定关于光学中心对称。构建了第一个反射光去除数据集BracketFlare。训练了一个神经网络学习反射光的去除。实验显示了模型的有效性。

### Correlational Image Modeling for Self-Supervised Visual Pre-Training
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Correlational_Image_Modeling_for_Self-Supervised_Visual_Pre-Training_CVPR_2023_paper.pdf
- code:  https://github.com/weivision/Correlational-Image-Modeling
<br>方向是自监督视觉预训练。提出了Correlational Image Modeling(CIM)。就是把图片的某一部分截取出来，然后让模型预测原图和截图的相关图。会使用各种方式去截取不同的图。使用了引导学习框架，使用两个编码器，分别编码截图和原图。模型就是一个交叉注意力模块。实验显示结果与SOTA相似或者超过。

### Aligning Bag of Regions for Open-Vocabulary Object Detection
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Aligning_Bag_of_Regions_for_Open-Vocabulary_Object_Detection_CVPR_2023_paper.pdf
- code: https://github.com/wusize/ovdet
- citation:
```
@inproceedings{wu2023baron,
    title={Aligning Bag of Regions for Open-Vocabulary Object Detection},
    author={Size Wu and Wenwei Zhang and Sheng Jin and Wentao Liu and Chen Change Loy},
    year={2023},
    booktitle={CVPR},
}
```
方向是开放词汇目标检测。VLM（Vision Language Model）可以对齐图像和文本。现有方法是把单物体输入Image Encoder。这篇论文的方法是把多个物体作为一个整体的包围框输入Image Encoder。这样可以包含更多的结构语义信息。实验显示结果超过了Faster-RCNN。

### Self-Supervised Geometry-Aware Encoder for Style-Based 3D GAN Inversion
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Lan_Self-Supervised_Geometry-Aware_Encoder_for_Style-Based_3D_GAN_Inversion_CVPR_2023_paper.pdf
<br>
方向是3D生成。StyleGAN是2D维度的一个统一模型。目前3D维度缺少同一个模型。使用自监督的方式训练模型，只需要使用3D数据。在Generation Network里面加入了一个分支，加入了像素粒度的特征。还提出了一个视角不变的3D编辑的方法。实验显示超过了SOTA。

## 7.15
### StyleSync: High-Fidelity Generalized and Personalized Lip Sync in Style-based Generator
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Guan_StyleSync_High-Fidelity_Generalized_and_Personalized_Lip_Sync_in_Style-Based_Generator_CVPR_2023_paper.pdf
- code: https://github.com/guanjz20/StyleSync
- website: https://hangz-nju-cuhk.github.io/projects/StyleSync
- citation:
```
@inproceedings{guan2023stylesync,
  title = {StyleSync: High-Fidelity Generalized and Personalized Lip Sync in Style-based Generator},
  author = {Guan, Jiazhi and Zhang, Zhanwang and Zhou, Hang and HU, Tianshu and Wang, Kaisiyuan and He, Dongliang and Feng, Haocheng and Liu, Jingtuo and Ding, Errui and Liu, Ziwei and Wang, Jingdong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
}
```
方向是嘴唇合成。使用了脸部的mask来保持脸部的细节。嘴唇的形状使用模块化卷积修改。使用了样式空间和生成器限制，支持根据一个人的语音和视频，把另一个人的视频中的嘴唇换成前面那个人的。实验显示取得了有高保真度的结果。

### Collaborative Diffusion for Multi-Modal Face Generation and Editing
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Collaborative_Diffusion_for_Multi-Modal_Face_Generation_and_Editing_CVPR_2023_paper.pdf
- code: https://github.com/ziqihuangg/Collaborative-Diffusion
- website: https://ziqihuangg.github.io/projects/collaborative-diffusion.html
- citation:
```
@InProceedings{huang2023collaborative,
      author = {Huang, Ziqi and Chan, Kelvin C.K. and Jiang, Yuming and Liu, Ziwei},
      title = {Collaborative Diffusion for Multi-Modal Face Generation and Editing},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year = {2023},
}
```
方向是脸部生成和脸部编辑。目前的扩散模型都是单模态的，这个文章提出了多模态的扩散模型。支持根据文本条件和mask条件生成和编辑人脸。使用多个单模态的扩散模型，然后使用一个元网络预测每个模型的影响函数，从而组合成了一个多模态的扩散模型。实验显示在质量和数据上都取得了很好的结果。

### Detecting and Grounding Multi-Modal Media Manipulation
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Shao_Detecting_and_Grounding_Multi-Modal_Media_Manipulation_CVPR_2023_paper.pdf
- code: https://github.com/rshaojimmy/MultiModal-DeepFake
- citation:
```
@inproceedings{shao2023dgm4,
    title={Detecting and Grounding Multi-Modal Media Manipulation},
    author={Shao, Rui and Wu, Tianxing and Liu, Ziwei},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}
```
方向是造假检测。目前的任务都是单模态的，图片或者文本，然后也只支持二分类。提出了一个新任务，在图片和文本的多模态数据上，进行虚假检测，检测的结果用图片中的包围框和文本中的短语显示出来。提出了这个任务上的一个数据集。提出了HAMMER（HierArchical Multi-modal Manipulation rEasoning tRansformer）这个模型去解决这个问题。使用了两个单模态encoder的对比学习进行浅层推理。使用了交叉注意力进行深层推理。加入了判断和检测头。构建了一个benchmark。提出了丰富的评价指标。实验显示了模型的有效性。提出了多个有价值的观察，以便未来的研究。

### F2-NeRF: Fast Neural Radiance Field Training with Free Camera Trajectories
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_F2-NeRF_Fast_Neural_Radiance_Field_Training_With_Free_Camera_Trajectories_CVPR_2023_paper.pdf
- code: https://github.com/totoro97/f2-nerf
- website: totoro97.github.io/projects/f2-nerf
- citation:
```
@article{wang2023f2nerf,
  title={F2-NeRF: Fast Neural Radiance Field Training with Free Camera Trajectories},
  author={Wang, Peng and Liu, Yuan and Chen, Zhaoxi and Liu, Lingjie and Liu, Ziwei and Komura, Taku and Theobalt, Christian and Wang, Wenping},
  journal={CVPR},
  year={2023}
}
```
方向是3D重建。提出了一个快速、自由轨迹的grid-based NeRF。提出了一个全新的视角扭曲，从而实现了自由轨迹。实验显示结果的质量很好。

### Taming Diffusion Models for Audio-Driven Co-Speech Gesture Generation
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Taming_Diffusion_Models_for_Audio-Driven_Co-Speech_Gesture_Generation_CVPR_2023_paper.pdf
- code: https://github.com/Advocate99/DiffGesture
- citation:
```
@InProceedings{Zhu_2023_CVPR,
    author    = {Zhu, Lingting and Liu, Xian and Liu, Xuanyu and Qian, Rui and Liu, Ziwei and Yu, Lequan},
    title     = {Taming Diffusion Models for Audio-Driven Co-Speech Gesture Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {10544-10553}
}
```
方向是手势生成。给定音频，生成音频对应的手势。使用扩散模型处理语音片段。使用改造过的transformer处理长期时序依赖。使用退火噪声采样算法来保证时序一致性。使用了隐式无分类器知道来平衡多样性和质量。实验显示模型取得了SOTA结果。

### OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_OmniObject3D_Large-Vocabulary_3D_Object_Dataset_for_Realistic_Perception_Reconstruction_and_CVPR_2023_paper.pdf
- code: https://github.com/omniobject3d/OmniObject3D
- website: https://omniobject3d.github.io/
- citation:
```
@inproceedings{wu2023omniobject3d,
    author = {Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, 
    Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, Ziwei Liu},
    title = {OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation},
    journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}
```
方向是3D数据集。提出了一个真实的3D数据集。有6000个物体的数据。每个物体有丰富的标签。使用专业扫描器扫描。建立了4个赛道，3D感知、新视角合成、表面重建、3D物体生成。广泛的研究发现了一些未来研究的观察、挑战和机会。

### LaserMix for Semi-Supervised LiDAR Semantic Segmentation
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_LaserMix_for_Semi-Supervised_LiDAR_Semantic_Segmentation_CVPR_2023_paper.pdf
- code: https://github.com/ldkong1205/LaserMix
- website: https://ldkong.com/LaserMix
- citation:
```
@article{kong2022lasermix,
    title={LaserMix for Semi-Supervised LiDAR Semantic Segmentation},
    author={Kong, Lingdong and Ren, Jiawei and Pan, Liang and Liu, Ziwei},
    journal={arXiv preprint arXiv:2207.00026},
    year={2022}
}
```
方向是LiDAR语义分割。提出了一个半监督的方法。使用多个LiDAR扫描器的扫描结果，在混合这些数据之前和混合这些数据之后进行训练，让模型预测出一致的结果。模型与LiDAR的表示方式无关。理论分析显示模型的有效性。实验显示可以和全监督的模型取得相似的结果。而且这个方法运用在全监督模型上时，可以提点。
