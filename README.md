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

## 7.16
### Unbiased Multiple Instance Learning for Weakly Supervised Video Anomaly Detection
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Lv_Unbiased_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2023_paper.pdf
- code: https://github.com/ktr-hubrt/UMIL
- citation:
```
@inproceedings{Lv2023unbiased,
title={Unbiased Multiple Instance Learning for Weakly Supervised Video Anomaly Detection},
author={Hui Lv and Zhongqi Yue and Qianru Sun and Bin Luo and Zhen Cui and Hanwang Zhang},
booktitle={CVPR},
year={2023}
}
```
方向是视频异常检测。Weakly Supervised Video Anomaly Detection (WSVAD)的训练数据是视频是否异常的标签，但要预测出视频中的异常片段。目前使用的方法主要是Multiple Instance Learning (MIL)，但这个方法会倾向于异常。这篇文章提出了Unbiased MIL (UMIL)。每次训练时，都会用这个检测器把当前视频片段分为两类：最确信的异常片段和正常片段、其他的不确定片段。在这两类视频片段中寻找一致性特征，可以剔除变化的背景造成的偏差。实验显示方法的有效性。

### Weakly Supervised Class-agnostic Motion Prediction for Autonomous Driving
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Weakly_Supervised_Class-Agnostic_Motion_Prediction_for_Autonomous_Driving_CVPR_2023_paper.pdf
<br>方向是运动检测。3D点云的运动数据的标记很昂贵，但是3D点云的场景解析数据则比较便宜，因为只表示是否运动，不表示运动方向、运动速度、物体种类这些信息。这篇文章使用场景解析数据先训练了一个分割模型，然后使用这个分割模型预测的运动前景训练运动预测模型。还提出了一个Consistency-aware Chamfer Distance loss。实验显示比自监督模型的效果好，和某些全监督模型的效果差不多。

### TAPS3D: Text-Guided 3D Textured Shape Generation from Pseudo Supervision
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_TAPS3D_Text-Guided_3D_Textured_Shape_Generation_From_Pseudo_Supervision_CVPR_2023_paper.pdf
- code: https://github.com/plusmultiply/TAPS3D
- citation:
```
@inproceedings{wei2023taps3d,
  title={TAPS3D: Text-Guided 3D Textured Shape Generation from Pseudo Supervision},
  author={Wei, Jiacheng and Wang, Hao and Feng, Jiashi and Lin, Guosheng and Yap, Kim-Hui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16805--16815},
  year={2023}
}
```
方向是3D生成。是根据一个文本描述生成3D的有纹理的物体。训练数据是使用一个文本模板加上2D图片中的关键信息生成的。使用的是低维度的图片监督。实验显示了方法的有效性。

### Neural Vector Fields: Implicit Representation by Explicit Learning
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Neural_Vector_Fields_Implicit_Representation_by_Explicit_Learning_CVPR_2023_paper.pdf
- code: https://github.com/Wi-sc/NVF
- citation:
```
@misc{yang2023neural,
      title={Neural Vector Fields: Implicit Representation by Explicit Learning}, 
      author={Xianghui Yang and Guosheng Lin and Zhenghao Chen and Luping Zhou},
      year={2023},
      eprint={2303.04341},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
方向是3D表面重建。3D的表示方法目前有两种：显式的使用网格去包裹物体、隐式的使用unsigned distance functions (UDFs)去表示3D表面。这篇文章提出了Neural Vector Fields (NVF)，显式的进行学习，同时获得的是隐式的距离函数表征。实验显示超过了之前的SOTA结果。

### 3D Cinemagraphy from a Single Image
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Li_3D_Cinemagraphy_From_a_Single_Image_CVPR_2023_paper.pdf
- code: https://github.com/xingyi-li/3d-cinemagraphy
- website: https://xingyi-li.github.io/3d-cinemagraphy/
- citation:
```
@InProceedings{li2023_3dcinemagraphy,
    author    = {Li, Xingyi and Cao, Zhiguo and Sun, Huiqiang and Zhang, Jianming and Xian, Ke and Lin, Guosheng},
    title     = {3D Cinemagraphy From a Single Image},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {4595-4605}
}
```

## 7.17
### Face Transformer: Towards High Fidelity and Accurate Face Swapping
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Cui_Face_Transformer_Towards_High_Fidelity_and_Accurate_Face_Swapping_CVPRW_2023_paper.pdf
<br>方向是换脸。换脸指的是把源脸的身份信息和目标脸的属性信息融合起来。GAN用于换脸会丢失一些细节信息。这篇文章使用transformer进行换脸。能保留这些细节信息。同时使用了多尺度变换机制来保留细粒度的脸部信息。实验显示模型取得了较好的结果。

### 3D Semantic Segmentation in the Wild: Learning Generalized Models for Adverse-Condition Point Clouds
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Xiao_3D_Semantic_Segmentation_in_the_Wild_Learning_Generalized_Models_for_CVPR_2023_paper.pdf
- code: https://github.com/xiaoaoran/SemanticSTF
- citation:
```
@article{xiao20233d,
  title={3D Semantic Segmentation in the Wild: Learning Generalized Models for Adverse-Condition Point Clouds},
  author={Xiao, Aoran and Huang, Jiaxing and Xuan, Weihao and Ren, Ruijie and Liu, Kangcheng and Guan, Dayan and Saddik, Abdulmotaleb El and Lu, Shijian and Xing, Eric},
  journal={arXiv preprint arXiv:2304.00690},
  year={2023}
}
```
方向是3D语义分割。提出了一个恶劣天气下的3D点云数据集。进行了两步研究：从正常天气到恶劣天气的领域迁移、从正常天气到全天气的领域泛化。提出了领域随机化技术，通过随机化点云的几何形状，学习到了全天气3D语义分割模型。

### KD-DLGAN: Data Limited Image Generation via Knowledge Distillation
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Cui_KD-DLGAN_Data_Limited_Image_Generation_via_Knowledge_Distillation_CVPR_2023_paper.pdf
<br>方向是图像生成。GAN的训练往往需要大量的训练数据。这篇文章使用VLM对GAN进行知识蒸馏。训练Discriminator完成VLM中的更复杂的任务。训练Generator学习VLM中的丰富的文本图像关联。从而在有限训练数据上取得了很好的效果。同时这个方法运用在SOTA上时提点明显。

### StyleRF: Zero-shot 3D Style Transfer of Neural Radiance Fields
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_StyleRF_Zero-Shot_3D_Style_Transfer_of_Neural_Radiance_Fields_CVPR_2023_paper.pdf
- code: https://github.com/Kunhao-Liu/StyleRF
- website: https://kunhao-liu.github.io/StyleRF/
- citation:
```
@inproceedings{liu2023stylerf,
  title={StyleRF: Zero-shot 3D Style Transfer of Neural Radiance Fields},
  author={Liu, Kunhao and Zhan, Fangneng and Chen, Yiwen and Zhang, Jiahui and Yu, Yingchen and El Saddik, Abdulmotaleb and Lu, Shijian and Xing, Eric P},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8338--8348},
  year={2023}
}
```
方向是3D风格迁移。通过在辐射场特征空间中做风格迁移。使用了采样无关的内容转化保证了多视角的一致性。使用2D特征的推迟风格变化减少了内存消耗。实验显示取得了较好的结果。

### FAC: 3D Representation Learning via Foreground Aware Feature Contrast
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_FAC_3D_Representation_Learning_via_Foreground_Aware_Feature_Contrast_CVPR_2023_paper.pdf
<br>方向3D。对比学习在3D表征中有很大潜力。但目前对比学习是随机选取锚点进行比较，因为大部分场景中背景占多数，导致选取的锚点偏向于背景。这篇文章会区分出前景和背景。然后在前景之间和前景和背景之间进行对比，取得了较好的学习效果。实验显示在3D语义分割和目标检测上取得了较好的结果。

### Regularized Vector Quantization for Tokenized Image Synthesis
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Regularized_Vector_Quantization_for_Tokenized_Image_Synthesis_CVPR_2023_paper.pdf
<br>方向是图片量化。目前的方法有决策方法和随机方法，目前的决策方法有编码书坍塌和与推理阶段不对齐的问题，随机方法有编码书低使用率和困扰的重建目标的问题。这篇文章提出了基于监督的量化框架。使用了一个先验的分布监督，来测量预测分布和先验分布的差异，解决了编码书坍塌和编码书低使用率。使用了随机遮盖监督，平衡推理阶段不对齐和困扰的重建目标。提出了概率对比损失解决困扰的重建目标。实验显示取得了较好的结果。

### Backdoor Attacks Against Deep Image Compression via Adaptive Frequency Trigger
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Tan_Backdoor_Attacks_Against_Deep_Image_Compression_via_Adaptive_Frequency_Trigger_CVPR_2023_paper.pdf
<br>方向是模型攻击。提出了一个基于频率的触发注入模型，可以在DCT（Discrete Cosine Transfrom）域中加入触发。可以攻击图片压缩质量和任务模型的准确率。设计了一个简单的动态误差。实验显示取得了较好的结果。

### Towards Efficient Use of Multi-Scale Features in Transformer-Based Object Detectors
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Towards_Efficient_Use_of_Multi-Scale_Features_in_Transformer-Based_Object_Detectors_CVPR_2023_paper.pdf
- code: https://github.com/ZhangGongjie/IMFA
<br>方向是目标检测。提出了Iterative Multi-scale Feature Aggregation (IMFA)。改造了transformer的结构，使得编码特征可以根据预测结果循环更新。根据之前的检测结果，选取一小部分关键位置的多尺度特征进行使用。实验显示可以提点，而且计算量不大。

### DA-DETR: Domain Adaptive Detection Transformer with Information Fusion
- from: cvpr2023
- paper: https://arxiv.org/pdf/2103.17084.pdf
<br>方向是目标检测。detection transformer (DETR)在目标检测中得到了应用，但在领域适应目标检测中还没有使用。这篇文章提出了CNN-Transformer Blender (CTBlender)，把CNN的特征和Transformer的特征混合在一起，把高维语义信息和低维空间信息混合在一起。实验显示取得了较好的结果。

### UniDAformer: Unified Domain Adaptive Panoptic Segmentation Transformer via Hierarchical Mask Calibration
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_UniDAformer_Unified_Domain_Adaptive_Panoptic_Segmentation_Transformer_via_Hierarchical_Mask_CVPR_2023_paper.pdf
<br>方向是全景分割。是领域适应全景分割。已有方法是两支的网络，一支做实例分割，一支做语义分割，网络的参数比较多，训练和推理过程都比较慢。这篇文章提出了一个统一的网络。提出了Hierarchical Mask Calibration（HMC），在区域、超像素、像素这些尺度上，用来修正错误预测。支持统一的领域适应全景分割。有效的消除错误预测。支持端到端训练。实验显示取得了较好的结果。

### ABLE-NeRF: Attention-Based Rendering with Learnable Embeddings for Neural Radiance Field
- from: cvpr2023
- paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_LaserMix_for_Semi-Supervised_LiDAR_Semantic_Segmentation_CVPR_2023_paper.pdf
<br>方向是3D。NeRF在处理光滑或者反射表面时会变得模糊。这篇文章提出了一个基于自注意力的框架。提出了Learnable Embeddings来捕捉不同视角下的效果。从而可以解决光滑或者反射表面的模糊问题。实验显示取得了较好的结果。

## 8.1
### From Images to Textual Prompts: Zero-shot Visual Question Answering with Frozen Large Language Models
- from: cvpr2023
- paper: https://arxiv.org/pdf/2212.10846.pdf
- code: https://github.com/salesforce/LAVIS/tree/main/projects/img2llm-vqa
- citation:
```
@misc{guo2023from,
  title={From Images to Textual Prompts: Zero-shot {VQA} with Frozen Large Language Models},
  author={Jiaxian Guo and Junnan Li and Dongxu Li and Anthony Tiong and Boyang Li and Dacheng Tao and Steven HOI},
  year={2023},
  url={https://openreview.net/forum?id=Ck1UtnVukP8}
}
```
方向是VAQ。通过cv领域的一些现有模型，把图片的内容转化为question-answer pair，输入到大模型中，让大模型回答最初的问题。

## 8.2
### HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face
- from: arxiv 2023.3
- paper: https://arxiv.org/pdf/2303.17580.pdf
- code: https://github.com/microsoft/JARVIS
- citation:
```
@article{shen2023hugginggpt,
    title   = {HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace},
    author  = {Shen, Yongliang and Song, Kaitao and Tan, Xu and Li, Dongsheng and Lu, Weiming and Zhuang, Yueting},
    journal = {arXiv preprint arXiv:2303.17580},
    year    = {2023}
}
```
方向是通用人工智能。使用chatgpt控制huggingface中的模型完成任务。

### MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models
- from: arxiv 2023.4
- paper: https://arxiv.org/pdf/2304.10592.pdf
- website: https://minigpt-4.github.io/
- code: https://github.com/Vision-CAIR/MiniGPT-4
- citation:
```
@article{zhu2023minigpt,
  title={MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models},
  author={Zhu, Deyao and Chen, Jun and Shen, Xiaoqian and Li, Xiang and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2304.10592},
  year={2023}
}
```
方向是多模态大模型。用一个映射层把一个冻结视觉编码器和一个冻结大模型对齐起来。使用了一个对话式的数据集微调模型。训练数据只有5百万。在image question answering上取得了很好的结果。是对gpt4实现方案猜想。gpt4的多模态能力开放之后image question answering这个领域就不存在了。

### Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching
- from: arxiv 2023.5
- paper: https://arxiv.org/pdf/2305.13310.pdf
- code: https://github.com/aim-uofa/Matcher
- citation:
```
@article{liu2023matcher,
  title={Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching},
  author={Liu, Yang and Zhu, Muzhi and Li, Hengtao and Chen, Hao and Wang, Xinlong and Shen, Chunhua},
  journal={arXiv preprint arXiv:2305.13310},
  year={2023}
}
```
方向是目标分割。模型是一个全目标特征提取模型和一个类别无关的分割模型组合而成。直接组合这两个模型会产生问题。提出了双向匹配策略、鲁棒提示采样器、实例级匹配策略。实验显示结果比较好。

### IdealGPT: Iteratively Decomposing Vision and Language Reasoning via Large Language Models
- from: arxiv 2023.5
- paper: https://arxiv.org/abs/2305.14985
- code: https://github.com/Hxyou/IdealGPT
- citation:
```
@misc{you2023idealgpt,
      title={IdealGPT: Iteratively Decomposing Vision and Language Reasoning via Large Language Models}, 
      author={Haoxuan You and Rui Sun and Zhecan Wang and Long Chen and Gengyu Wang and Hammad A. Ayyubi and Kai-Wei Chang and Shih-Fu Chang},
      year={2023},
      eprint={2305.14985},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
方向是VAQ。通过大模型和cv领域的一些现有模型，把图片的内容转化为question-answer pair，输入到大模型中，让大模型回答最初的问题。

### Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.00973.pdf
- website: https://haoningwu3639.github.io/StoryGen_Webpage/
- code: https://github.com/haoningwu3639/StoryGen
- citation:
```
@article{liu2023intelligent,
  title={Intelligent Grimm -- Open-ended Visual Storytelling via Latent Diffusion Models}, 
  author={Chang Liu and Haoning Wu and Yujie Zhong and Xiaoyun Zhang and Weidi Xie},
  year={2023},
  journal={arXiv preprint arXiv:2306.00973},
}
```
方向是图片生成。基于一个童话故事生成一个配图的序列。在stable diffusion上加了两个模块。还加了一个自动回归的图像生成器。使用一种数据收集手段，创建了一个新的数据集StorySalon。使用三阶段的训练策略，支持风格迁移、视觉背景条件、人类反馈对齐。实验显示效果比较好。

### A Survey on Multimodal Large Language Models
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.13549.pdf
- code: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models
- citation:
```
@article{yin2023survey,
  title={A Survey on Multimodal Large Language Models},
  author={Yin, Shukang and Fu, Chaoyou and Zhao, Sirui and Li, Ke and Sun, Xing and Xu, Tong and Chen, Enhong},
  journal={arXiv preprint arXiv:2306.13549},
  year={2023}
}
```
方向是多模态大模型。是调研。给出了多模态大模型的定义和相关概念。总结了关键技术和应用。分析了现存挑战和有希望的研究方向。

### Mindstorms in Natural Language-Based Societies of Mind
- from: arxiv 2023.5
- paper: https://arxiv.org/pdf/2305.17066.pdf
方向是大模型。从社会学的角度理解大模型。大模型社会的社会结构应该是什么样的？君主专制和民主制度的优缺点是什么？整个社会的经济发展目标能不能用强化学习来实现？

### Foundational Models Defining a New Era in Vision: A Survey and Outlook
- from: arxiv 2023.7
- paper: https://arxiv.org/pdf/2307.13721.pdf
- code: https://github.com/awaisrauf/Awesome-CV-Foundational-Models
- citation:
```
@article{awais2023foundational,
  title={Foundational Models Defining a New Era in Vision: A Survey and Outlook},
  author={Awais, Muhammad and Naseer, Muzammal and Khan, Salman and Anwer, Rao Muhammad and Cholakkal, Hisham and Shah, Mubarak and Yang, Ming-Hsuan and Khan, Fahad Shahbaz},
  journal={arXiv preprint arXiv:2307.13721},
  year={2023}
}
```
方向是基础模型。是调研。

### Chatting Makes Perfect - Chat-based Image Retrieval
- from: arxiv 2023.5
- paper: https://arxiv.org/pdf/2305.20062.pdf
方向是图像检索。通过大模型，把图片的内容转化为question-answer pair。从而提升了图片检索的准确率。

## 8.3
### MovieChat: From Dense Token to Sparse Memory for Long Video Understanding
- from: arxiv 2023.7
- paper: https://arxiv.org/pdf/2307.16449.pdf
- code: https://github.com/rese1f/MovieChat
- website: https://rese1f.github.io/MovieChat/
方向是视频理解。XMem+MiniGPT4。gpt3是text->text，gpt4是text+image->text，这些都没必要做了，换一下的还可以做的。image->text、video->text、text->image、text->video、image+image->text、video+image->text、text+video->text、text+image->image、text+image->video。

### 👍3D-LLM: Injecting the 3D World into Large Language Models
- from: arxiv 2023.7
- paper: https://arxiv.org/pdf/2307.12981.pdf
- code: https://github.com/UMass-Foundation-Model/3D-LLM
- website: https://vis-www.cs.umass.edu/3dllm/
- citation:
```
@article{3dllm,
 author = {Hong, Yining and Zhen, Haoyu and Chen, Peihao and Zheng, Shuhong and Du, Yilun and Chen, Zhenfang and Gan, Chuang},
 title = {3D-LLM: Injecting the 3D World into Large Language Models},
 journal = {arXiv},
 year = {2023},
} 
```
方向是多模态大模型。支持3D标题、3D密集标题、3D问题回答、3D任务分解、3D grounding、3D辅助对话、3D导航等任务。设计了3种提示机制，收集了一个数据集。使用渲染多视角图片提取3D特征。使用2D VLM作为backbone。提出了一种3D定位机制。可以更好的捕捉位置信息。实验显示取得了较好的结果。

### ChatSpot: Bootstrapping Multimodal LLMs via Precise Referring Instruction Tuning
- from: arxiv 2023.7
- paper: https://arxiv.org/pdf/2307.09474.pdf
- website: https://chatspot.streamlit.app/
方向是多模态大模型。支持细粒度的交互，在图片上进行点和框。构建了一个数据集。提出了一些评价任务。实验显示取得了较好的结果。

### 👍BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs
- from: arxiv 2023.7
- paper: https://arxiv.org/pdf/2307.08581.pdf
- code: https://github.com/magic-research/bubogpt
- website: https://bubo-gpt.github.io/
- citation:
```
@article{zhao2023bubogpt,
  author      = {Yang Zhao and Zhijie Lin and Daquan Zhou and Zilong Huang and Jiashi Feng and Bingyi Kang},
  title       = {BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs},
  publisher   = {arXiv:2307.08581},
  year        = {2023}
}
```
方向是多模态大模型。支持text、image、audio三个模态。可以在图片中定位物体。提出了一个模块，基于SAM，可以提取出文本中的entities然后再图片中进行定位。两阶段的训练过程和一个数据集。实验显示取得了较好的结果。

### 👍Generative Pretraining in Multimodality
- from: arxiv 2023.7
- paper: https://arxiv.org/pdf/2307.05222.pdf
- code: https://github.com/baaivision/Emu
- citation:
```
@article{Emu,
  title={Generative Pretraining in Multimodality},
  author={Sun, Quan and Yu, Qiying and Cui, Yufeng and Zhang, Fan and Zhang, Xiaosong and Wang, Yueze and Gao, Hongcheng and Liu, Jingjing and Huang, Tiejun and Wang, Xinlong},
  publisher={arXiv preprint arXiv:2307.05222},
  year={2023},
}
```
方向是多模态大模型。支持text、image、video三个模态。EVA-CLIP+LLaMA+Stable Diffusion。

### 👍GPT4RoI: Instruction Tuning Large Language Model on Region-of-Interest
- from: arxiv 2023.7
- paper: https://arxiv.org/pdf/2307.03601.pdf
- code: https://github.com/jshilong/GPT4RoI
- citation:
```
@misc{zhang2023gpt4roi,
      title={GPT4RoI: Instruction Tuning Large Language Model on Region-of-Interest}, 
      author={Shilong Zhang and Peize Sun and Shoufa Chen and Min Xiao and Wenqi Shao and Wenwei Zhang and Kai Chen and Ping Luo},
      year={2023},
      eprint={2307.03601},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
方向是多模态大模型。使用region-text的数据集微调了大模型。支持根据region回答问题。支持单region的问题和多region的推理。任何object detector都可以使用。

### What Matters in Training a GPT4-Style Language Model with Multimodal Inputs?
- from: arxiv 2023.7
- paper: https://arxiv.org/pdf/2307.02469.pdf
- code: https://github.com/bytedance/lynx-llm
- website: https://lynx-llm.github.io/
- citation:
```
@article{zeng2023matters,
  title={What Matters in Training a GPT4-Style Language Model with Multimodal Inputs?},
  author={Zeng, Yan and Zhang, Hanbo and Zheng, Jiani and Xia, Jiangnan and Wei, Guoqiang and Wei, Yang and Zhang, Yuchen and Kong, Tao},
  journal={arXiv preprint arXiv:2307.02469},
  year={2023}
}
```
方向是多模态大模型。做了一个全面的实验。实现了20多种结构。网络结构上，比较了不同的大模型backbone和模型设计。训练数据上，比较了不同的数据和采样策略。指令上，比较了不同的提示对于模型能力的影响。Benchmark上，提出了第一个全面的验证集，包含图片和视频。基于以上的实验，提出了一种最好的模型。

### mPLUG-DocOwl : Modularized Multimodal Large Language Model for Document Understanding
- from: arxiv 2023.7
- paper: https://arxiv.org/pdf/2307.02499.pdf
- code: https://github.com/X-PLUG/mPLUG-DocOwl
- citation:
```
@misc{ye2023mplugdocowl,
      title={mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding}, 
      author={Jiabo Ye and Anwen Hu and Haiyang Xu and Qinghao Ye and Ming Yan and Yuhao Dan and Chenlin Zhao and Guohai Xu and Chenliang Li and Junfeng Tian and Qian Qi and Ji Zhang and Fei Huang},
      year={2023},
      eprint={2307.02499},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
方向是多模态大模型。支持文档理解。构建了一个指令微调数据集。在只有文本、通用文本-视觉数据、文本指令微调数据集上进行训练。构建了一个文档理解评价数据集。实验显示取得了较好的结果。

### 👍Visual Instruction Tuning with Polite Flamingo
- from: arxiv 2023.7
- paper: https://arxiv.org/pdf/2307.02499.pdf
- code: https://github.com/ChenDelong1999/polite_flamingo
- citation:
```
@article{chen2023visual,
  title={Visual Instruction Tuning with Polite Flamingo},
  author={Chen, Delong and Liu, Jianfeng and Dai, Wenliang and Wang, Baoyuan},
  journal={arXiv preprint arXiv:2307.01003},
  year={2023}
}
```
方向是多模态大模型。先训练一个polite flamingo，能把不礼貌的回答重写成礼貌的回答。使用polite flamingo把数据集中的不礼貌回答重写成礼貌回答。用这个礼貌的数据集训练出clever flamingo。实验显示效果比较好。

### LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.17107.pdf
- code: https://github.com/SALT-NLP/LLaVAR
- website: https://llavar.github.io/
- citation:
```
@misc{zhang2023llavar,
    title={LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding},
    author={Yanzhe Zhang and Ruiyi Zhang and Jiuxiang Gu and Yufan Zhou and Nedim Lipka and Diyi Yang and Tong Sun},
    year={2023},
    eprint={2306.17107},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

### Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.15195.pdf
- code: https://github.com/shikras/shikra
- citation:
```
@article{chen2023shikra,
  title={Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic},
  author={Chen, Keqin and Zhang, Zhao and Zeng, Weili and Zhang, Richong and Zhu, Feng and Zhao, Rui},
  journal={arXiv preprint arXiv:2306.15195},
  year={2023}
}
```

### Aligning Large Multi-Modal Model with Robust Instruction Tuning
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.14565.pdf
- code: https://github.com/FuxiaoLiu/LRV-Instruction
- website: https://fuxiaoliu.github.io/LRV/
- citation:
```
@article{liu2023aligning,
  title={Aligning Large Multi-Modal Model with Robust Instruction Tuning},
  author={Liu, Fuxiao and Lin, Kevin and Li, Linjie and Wang, Jianfeng and Yacoob, Yaser and Wang, Lijuan},
  journal={arXiv preprint arXiv:2306.14565},
  year={2023}
}
```

### MACAW-LLM: MULTI-MODAL LANGUAGE MODELING WITH IMAGE, AUDIO, VIDEO, AND TEXT INTEGRATION
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.09093.pdf
- code: https://github.com/lyuchenyang/Macaw-LLM
- citation:
```
@article{lyu2023macaw,
  title={Macaw-LLM: Multi-Modal Language Modeling with Image, Audio, Video, and Text Integration},
  author={Lyu, Chenyang and Wu, Minghao and Wang, Longyue and Huang, Xinting and Liu, Bingshuai and Du, Zefeng and Shi, Shuming and Tu, Zhaopeng},
  journal={arXiv preprint arXiv:2306.09093},
  year={2023}
}
```

### LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.06687.pdf
- code: https://github.com/OpenLAMM/LAMM
- citation:
```
@article{yin2023lamm,
        title={LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark},
        author={Yin, Zhenfei and Wang, Jiong and Cao, Jianjian and Shi, Zhelun and Liu, Dingning and Li, Mukai and Sheng, Lu and Bai, Lei and Huang, Xiaoshui and Wang, Zhiyong and others},
        journal={arXiv preprint arXiv:2306.06687},
        year={2023}
}
```

### Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.05424.pdf
- code: https://github.com/mbzuai-oryx/Video-ChatGPT
- website: https://www.ival-mbzuai.com/video-chatgpt
- citation:
```
@article{Maaz2023VideoChatGPT,
    title={Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models},
    author={Muhammad Maaz, Hanoona Rasheed, Salman Khan and Fahad Khan},
    journal={ArXiv 2306.05424},
    year={2023}
}
```

### MIMIC-IT: Multi-Modal In-Context Instruction Tuning
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.05425.pdf
- code: https://github.com/Luodian/Otter
- website: https://otter.cliangyu.com/
- citation:
```
@article{li2023mimicit,
    title={MIMIC-IT: Multi-Modal In-Context Instruction Tuning},
    author={Bo Li and Yuanhan Zhang and Liangyu Chen and Jinghao Wang and Fanyi Pu and Jingkang Yang and Chunyuan Li and Ziwei Liu},
    year={2023},
    eprint={2306.05425},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

### M3IT: A Large-Scale Dataset towards Multi-Modal Multilingual Instruction Tuning
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.04387.pdf
- website: https://huggingface.co/datasets/MMInstruction/M3IT
- citation:
```
@article{li2023m3it,
  title={M$^3$IT: A Large-Scale Dataset towards Multi-Modal Multilingual Instruction Tuning},
  author={Lei Li and Yuwei Yin and Shicheng Li and Liang Chen and Peiyi Wang and Shuhuai Ren and Mukai Li and Yazheng Yang and Jingjing Xu and Xu Sun and Lingpeng Kong and Qi Liu},
  journal={arXiv preprint arXiv:2306.04387},
  year={2023}
}
```

### Video-LLaMA An Instruction-tuned Audio-Visual Language Model for Video Understanding
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.02858.pdf
- code: https://github.com/DAMO-NLP-SG/Video-LLaMA
- citation:
```
@article{damonlpsg2023videollama,
  author = {Zhang, Hang and Li, Xin and Bing, Lidong},
  title = {Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding},
  year = 2023,
  journal = {arXiv preprint arXiv:2306.02858},
  url = {https://arxiv.org/abs/2306.02858}
}
```

### LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2306.00890.pdf
- code: https://github.com/microsoft/LLaVA-Med

### 👍GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2305.18752.pdf
- code: https://github.com/StevenGrove/GPT4Tools
- website: https://gpt4tools.github.io/
- citation:
```
@misc{gpt4tools,
  title = {GPT4Tools: Teaching LLM to Use Tools via Self-instruction},
  author={Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, Ying Shan},
  journal={arXiv preprint arXiv:2305.18752},
  year={2023}
}
```

### ImageBind-LLM: Multi-Modality Instruction Tuning
- from: arxiv 2023.6
- code: https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/imagebind_LLM
- website: http://imagebind-llm.opengvlab.com/

### PandaGPT: One Model To Instruction-Follow Them All
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2305.16355.pdf
- code: https://github.com/yxuansu/PandaGPT
- website: https://panda-gpt.github.io/
- citation:
```
@article{su2023pandagpt,
  title={PandaGPT: One Model To Instruction-Follow Them All},
  author={Su, Yixuan and Lan, Tian and Li, Huayang and Xu, Jialu and Wang, Yan and Cai, Deng},
  journal={arXiv preprint arXiv:2305.16355},
  year={2023}
}
```

### ChatBridge: Bridging Modalities with Large Language Model as a Language Catalyst
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2305.16103.pdf
- code: https://github.com/joez17/ChatBridge
- website: https://iva-chatbridge.github.io/
- citation:
```
@article{zhao2023chatbridge,
  title={ChatBridge: Bridging Modalities with Large Language Model as a Language Catalyst},
  author={Zhao, Zijia and Guo, Longteng and Yue, Tongtian and Chen, Sihan and Shao, Shuai and Zhu, Xinxin and Yuan, Zehuan and Liu, Jing},
  journal={arXiv preprint arXiv:2305.16103},
  year={2023}
}
```

### Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models
- from: arxiv 2023.6
- paper: https://arxiv.org/pdf/2305.15023.pdf
- code: https://github.com/luogen1996/LaVIN
- website: https://luogen1996.github.io/lavin/
- citation:
```
@article{luo2023cheap,
  title={Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models},
  author={Luo, Gen and  Zhou, Yiyi and Ren, Tianhe and Chen, Shengxin abd Sun, Xiaoshuai and Ji, Rongrong},
  journal={arXiv preprint arXiv:2305.15023},
  year={2023}
}
```

### ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding
- from: cvpr2023
- paper: https://arxiv.org/pdf/2212.05171.pdf
- code: https://github.com/salesforce/ULIP
- website: https://tycho-xue.github.io/ULIP/
- citation:
```
@article{xue2022ulip,
  title={ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding},
  author={Xue, Le and Gao, Mingfei and Xing, Chen and Mart{\'\i}n-Mart{\'\i}n, Roberto and Wu, Jiajun and Xiong, Caiming and Xu, Ran and Niebles, Juan Carlos and Savarese, Silvio},
  journal={arXiv preprint arXiv:2212.05171},
  year={2022}
}
```
方向是3D理解。学习3D点云、图片、文本3种模态的统一表示，3D-Vision-Language Model。使用预训练过的VLM。用少量自动合成的object triplet（3D、image、text）学习和image-text空间对齐的3D空间。ULIP和具体的3D backbone结构无关，可以容易的结合到现有的3D backbone中。实验显示可以提升多个最近的3D backbone的准确性。在ModelNet40和ScanObjectNN这两个数据集进行了实验。

### ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding
- from: arxiv 2023.5
- paper: https://arxiv.org/pdf/2305.08275.pdf
- code: https://github.com/salesforce/ULIP
- website: https://tycho-xue.github.io/ULIP/
- citation:
```
@misc{xue2023ulip2,
  title={ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding}, 
  author={Le Xue and Ning Yu and Shu Zhang and Junnan Li and Roberto Martín-Martín and Jiajun Wu and Caiming Xiong and Ran Xu and Juan Carlos Niebles and Silvio Savarese},
  year={2023},
  eprint={2305.08275},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
方向是3D理解。主要解决的是数据问题。对3D object进行多视角渲染生成image。使用gpt4-like model对image进行描述生成text。从而获取到了大量的数据。使用大量的数据训练ULIP。实验显示准确率得到了明显的提升。

### Learning Transferable Visual Models From Natural Language Supervision
- from: icml2021
- paper: https://arxiv.org/pdf/2305.08275.pdf
- code: https://github.com/openai/CLIP
- website: https://tycho-xue.github.io/ULIP/
- citation:
```
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
方向是VLM。使用image-text数据进行训练，得到Vision-Language Model。

## 8.10
### LayoutLLM-T2I: Eliciting Layout Guidance from LLM for Text-to-Image Generation
- from: arxiv 2023.8
- paper: https://arxiv.org/pdf/2308.05095.pdf
- code: https://github.com/LayoutLLM-T2I/LayoutLLM-T2I
- website: https://layoutllm-t2i.github.io/
- citation:
```
@article{qu2023layoutllm,
  title={LayoutLLM-T2I: Eliciting Layout Guidance from LLM for Text-to-Image Generation},
  author={Leigang Qu, Shengqiong Wu, Hao Fei, Liqiang Nie, Tat-Seng Chua},
  journal={Proceedings of the {ACM} International Conference on Multimedia},
  year={2023}
}
```
方向是图片生成。根据text，使用llm生成粗粒度的layout。根据text和layout，生成最终的图片。

### Seeing in Flowing: Adapting CLIP for Action Recognition with Motion Prompts Learning
- from: mm2023
- paper: https://arxiv.org/pdf/2308.04828.pdf
方向是动作识别。用clip做动作识别。两支动作建模模块。动态prompt学习模块生成动作相关的prompt。多模态交流模块。实验显示取得了较好的结果。

### TextPainter: Multimodal Text Image Generation with Visual-harmony and Text-comprehension for Poster Design
- from: mm2023
- paper: https://arxiv.org/pdf/2308.04733.pdf
方向是文字图片生成。根据全局-局部背景图片指导风格，来生成视觉和谐的图片。使用语言模型和一个文本理解模块来形成句子和单词的风格变换。提出了PosterT80K数据集。实验显示取得了较好的结果。

### Rendering Humans from Object-Occluded Monocular Videos
- from: iccv2023
- paper: https://arxiv.org/pdf/2308.04622.pdf
- website: https://cs.stanford.edu/~xtiange/projects/occnerf/
方向是人体渲染。遮挡场景的人体渲染有两个问题，可见和遮挡区域的不匹配，没有使用先验信息。提出了基于表面的渲染方法，使用几何和可见性先验。

### PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
- from: cvpr2017
- paper: https://arxiv.org/pdf/1612.00593.pdf
- code: https://github.com/charlesq34/pointnet
- website: http://stanford.edu/~rqi/pointnet/
- citation:
```
@article{qi2016pointnet,
  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:1612.00593},
  year={2016}
}
```

### PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
- from: nips2017
- paper: https://arxiv.org/pdf/1706.02413.pdf
- code: https://github.com/charlesq34/pointnet2
- website: http://stanford.edu/~rqi/pointnet2/
- citation:
```
@article{qi2017pointnetplusplus,
  title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:1706.02413},
  year={2017}
}
```

### Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling
- from: cvpr2022
- paper: https://arxiv.org/pdf/2111.14819.pdf
- code: https://github.com/lulutang0608/Point-BERT
- website: https://point-bert.ivg-research.xyz/
- citation:
```
@inproceedings{yu2021pointbert,
  title={Point-BERT: Pre-Training 3D Point Cloud Transformers with Masked Point Modeling},
  author={Yu, Xumin and Tang, Lulu and Rao, Yongming and Huang, Tiejun and Zhou, Jie and Lu, Jiwen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

### PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
- from: nips2022
- paper: https://arxiv.org/pdf/2206.04670.pdf
- code: https://github.com/guochengqian/PointNeXt
- website: https://guochengqian.github.io/PointNeXt/
- citation:
```
@InProceedings{qian2022pointnext,
  title   = {PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies},
  author  = {Qian, Guocheng and Li, Yuchen and Peng, Houwen and Mai, Jinjie and Hammoud, Hasan and Elhoseiny, Mohamed and Ghanem, Bernard},
  booktitle=Advances in Neural Information Processing Systems (NeurIPS),
  year    = {2022},
}
```
