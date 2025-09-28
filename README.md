_**This paper has been accepted to NeurIPS 2025 as a Spotlight.**_

<h2 align="center"> <a href="https://nips.cc/virtual/2025/poster/115223">Evolutionary Multi-View Classification via Eliminating Individual Fitness Bias</a></h2>

<div align="center">

**[Xinyan Liang<sup>1,2</sup>](https://xinyanliang.github.io/), [_Shuai Li_<sup>1</sup>](https://github.com/LiShuailzn), _Qian Guo_<sup>3</sup>, [_Yuhua Qian_<sup>1*</sup>](http://dig.sxu.edu.cn/qyh/),  _Bingbing Jiang_<sup>4</sup>, _Tingjin Luo_<sup>5</sup>, _Liang Du_<sup>1</sup>**

<sup>1</sup>Institute of Big Data Science and Industry, Shanxi University<br>
<sup>2</sup>State Key Laboratory of AI Safety, Beijing, 100086<br>
<sup>3</sup>School of Computer Science and Technology, Taiyuan University of Science and Technology<br>
<sup>4</sup>School of Information Science and Technology, Hangzhou Normal University<br>
<sup>5</sup>College of Science, National University of Defense Technology<br>


<a href='https://nips.cc/virtual/2025/poster/115223'><img src='https://img.shields.io/badge/NIPS%202025-Poster-blue'></a>&nbsp;

</div>


## Abstract
Evolutionary multi-view classification (EMVC) methods have gained wide recognition due to their adaptive mechanisms. Fitness evaluation (FE), which aims to calculate the classification performance of each individual in the population and provide reliable performance ranking for subsequent operations, is a core step in such methods. Its accuracy directly determines the correctness of the evolutionary direction.
However, when FE fails to correctly reflect the superiority-inferiority relationship among individuals, it will lead to confusion in individual performance ranking, which in turn misleads the evolutionary direction and results in trapping into local optima. 

This paper is the first to identify the aforementioned issue in the field of EMVC and call it as fitness evaluation bias (FEB).
FEB may be caused by a variety of factors, and this paper approaches the issue from the perspective of view information content: existing methods generally adopt joint training strategies, which restrict the exploration of key information in views with low information content. This makes it difficult for multi-view model (MVM) to achieve optimal performance during convergence, which in turn leads to FE failing to accurately reflect individual performance rankings and ultimately triggering FEB.
To address this issue, we propose an evolutionary multi-view classification via eliminating individual fitness bias (EFB-EMVC) method, which alleviates the FEB issue by introducing evolutionary navigators for each MVM, thereby providing more accurate individual ranking.


Experimental results fully verify the effectiveness of the proposed method in alleviating the FEB problem, and the EMVC method equipped with this strategy exhibits more superior performance compared with the original EMVC method.

## 🏗️Model
<div align="center">
  <img src="model.svg" />
</div>

## 🚀 Environment
In our experiments, all methods are implemented using TensorFlow 2.10.0.
The computing environment includes Ubuntu 24.04.2 LTS as the operating system, equipped with an AMD EPYC processor with 160 physical cores (320 logical threads), 566 GB of DDR4 memory, and 8 NVIDIA GeForce RTX 5090 GPUs, each with 32 GB of VRAM. The experimental setup is based on Python 3.9.23 and CUDA 11.2.

## 🎞️Experiment
In this experiment, we aim to investigate the effectiveness of EN in alleviating the FEB problem.
### Data
We used nine multi-view datasets in this experiment:

| Datasets            | Dataset URL                                            |    Password      | 
|---------------------|--------------------------------------------------------|------------------|
| MVoxCeleb           | [link](https://pan.baidu.com/s/1k6DN1m64bnrRfLK8RiFmqQ)|     ls12         |
| YoutubeFace         | [link](https://pan.baidu.com/s/1SVTWfHpAUdFWwiU5o-kD7Q)|     ls34         | 
| NUS-WIDE-128 (NUS)  | [link](https://pan.baidu.com/s/1udO5jvolHIbd8lOV3w4SYA)|     ls56         | 
| Reuters5            | [link](https://pan.baidu.com/s/1j8pmo88vXsO9pBWQiHVmYA)|     ls78         | 
| Reuters3            | [link](https://pan.baidu.com/s/1ti4OWqXTVnPDhsZ7VjahGQ)|     ls10         | 
| CB                  | [link](https://pan.baidu.com/s/1CqnQFkPkiT-e8ETh2iYcsw)|     lss1         |  
| MM-IMDB             | [link](https://pan.baidu.com/s/1FuiJHU8Xqjt5e_xCvnZwfw)|     lss2         |               
| NTU RGB-D           | [link](https://pan.baidu.com/s/1eam19lCIsXxfzyX6CaOgPw)|     lss3         |                
| EgoGesture          | [link](https://pan.baidu.com/s/1eobwPKqCRe6RereGEcwQWA)|     lss4         |                


For the nine multi-view datasets, the CB, MM-IMDB, NTU RGB-D, and EgoGesture datasets already come with predefined train-test splits provided by the original authors, so we only repeated the experiments five times on these datasets. The remaining datasets were evaluated using five-fold cross-validation.<br>
To facilitate code reproducibility, for the datasets where experiments were repeated five times, we provide not only the original split data but also the teacher model logits and soft labels, as well as the kernel and mutual information matrices required for the experiments. For the datasets requiring five-fold cross-validation, we likewise provide the original experimental data along with the five-fold split data, teacher model logits and soft labels, and the corresponding kernel and mutual information matrices. Readers can directly download and use these resources, enabling them to flexibly select and utilize the datasets according to their needs.

### Experiment Workflow
To mitigate the FEB problem, we introduce EN into each view branch of MVM. EN is designed as a pre-trained network with the same architecture as the teacher model. Accordingly, the experimental workflow consists of: (1) pre-training EN and extracting logits; (2) constructing the mutual information matrix; (3) the EMVC method driven by unbiased fitness evaluation.

#### Training
1. **Pre-training EN and extracting logits**
```bash
python code/train_T.py
python code/gain_T_logits.py
```
2. **Constructing the mutual information matrix**
```bash
python code/HSIC/kernel_matrix.py
python code/HSIC/HSIC.py
```
3. **The EMVC method driven by unbiased fitness evaluation**
```bash
python code/train_tree_youtube.py
```

## 📑Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{
liang2025EFB-EMVC,
title={Evolutionary Multi-View Classification via Eliminating Individual Fitness Bias},
author={Xinyan Liang, Shuai Li, Qian Guo, Yuhua Qian, Bingbing Jiang, Tingjin Luo, Liang Du},
booktitle={Proceedings of the Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS-25)},
year={2025},
}
```

## 🔬 Related Work
We list below the works most relevant to this paper, including but not limited to the following:<br>
**_Our research group [[link]](https://xinyanliang.github.io/publications/)_**
- Evolutionary deep fusion method and its application in chemical structure recognition, _IEEE TEVC21_, [[paper]](https://ieeexplore.ieee.org/document/9373673)
- Evolutionary multi-view classification via eliminating individual fitness bias, NeurIPS25, [[paper]](https://github.com/LiShuailzn/Neurips-2025-EFB-EMVC)
- Trusted multi-view classification via evolutionary multi-view fusion, _ICLR25_, [[paper]](https://openreview.net/pdf?id=M3kBtqpys5)
- DC-NAS: Divide-and-conquer neural architecture search for multi-modal classification, _AAAI24_, [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29281)
- Core-structures-guided multi-modal classification neural architecture search, _IJCAI24_, [[paper]](https://www.ijcai.org/proceedings/2024/0440.pdf)
- CoMO-NAS: Core-structures-guided multi-objective neural architecture search for multi-modal classification, _ACM MM24_, [[paper]](https://dl.acm.org/doi/10.1145/3664647.3681351)
- A fast neural architecture search method for multi-modal classification via Knowledge Sharing, _IJCAI25_, [[paper]](https://www.ijcai.org/proceedings/2025/557)
- Multi-scale features are effective for multi-modal classification: An architecture search viewpoint, _IEEE TCSVT25_, [[paper]](https://ieeexplore.ieee.org/document/10700772)
- Core structure-guided multi-modal classification via monte carlo tree search, _IJMLC25_, [[paper]](https://link.springer.com/article/10.1007/s13042-025-02606-z)


**_Others_**
- Enhancing multimodal learning via hierarchical fusion architecture search with inconsistency mitigation, _IEEE TIP25_, [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11134693)
- Enhanced protein secondary structure prediction through multi-view multi-feature evolutionary deep fusion method, _IEEE TETCI25_, [[paper]](https://ieeexplore.ieee.org/abstract/document/10839444)
- MFAS: Multimodal fusion architecture search, _CVPR19_, [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Perez-Rua_MFAS_Multimodal_Fusion_Architecture_Search_CVPR_2019_paper.pdf)
- BM-NAS: Bilevel multimodal neural architecture search, _AAAI22_, [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20872)
- Deep multimodal neural architecture search, _ACMMM20_, [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11134693)
- Hierarchical multi-modal fusion architecture search for stock market forecasting, _Applied Soft Computing25_, [[paper]](https://www.sciencedirect.com/science/article/pii/S1568494625008920?casa_token=TZOWE_icAokAAAAA:YH8vB-WqZC03tYf8DV6WaVqMH78aoprjybDSwEDQlF6nSJ0SrQrf1lFh-OHwzHDiYu-iFHz38U8)
- Harmonic-NAS: Hardware-aware multimodal neural architecture search on resource-constrained devices, _ACML23_, [[paper]](https://proceedings.mlr.press/v222/ghebriout24a/ghebriout24a.pdf)
- Multi-view information fusion based on federated multi-objective neural architecture search for MRI semantic segmentation, _IF25_, [[paper]](https://arxiv.org/pdf/2007.06002)
- Automatic fused multimodal deep learning for plant identification, _Frontiers in Plant Science25_, [[paper]](https://arxiv.org/pdf/2406.01455?)

<!-- ## 🙏 Acknowledgement -->




## 📬Contact
If you have any detailed questions or suggestions, you can email us: [lishuai_liuzhaona@163.com](mailto:lishuai_liuzhaona@163.com)
