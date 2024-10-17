# <p align="center">Eliminating Warping Shakes for Unsupervised Online Video Stitching

## 🚩Recommendation
We have released the complete code of [StabStitch++](https://github.com/nie-lang/StabStitch2) (an extension of StabStitch) with better alignment, fewer distortions, and higher stability.

## Introduction
This is the official implementation for [StabStitch](https://arxiv.org/abs/2403.06378) (ECCV2024).

[Lang Nie](https://nie-lang.github.io/)<sup>1</sup>, [Chunyu Lin](https://faculty.bjtu.edu.cn/8549/)<sup>1</sup>, [Kang Liao](https://kangliao929.github.io/)<sup>2</sup>, [Yun Zhang](http://zhangyunnet.cn/academic/index.html)<sup>3</sup>, [Shuaicheng Liu](http://www.liushuaicheng.org/)<sup>4</sup>, Rui Ai<sup>5</sup>, [Yao Zhao](https://faculty.bjtu.edu.cn/5900/)<sup>1</sup>

<sup>1</sup> Beijing Jiaotong University  {nielang, cylin, yzhao}@bjtu.edu.cn

<sup>2</sup> Nanyang Technological University

<sup>3</sup> Communication University of Zhejiang 

<sup>4</sup> University of Electronic Science and Technology of China

<sup>5</sup> HAMO.AI

> ### Feature
> Nowadays, the videos captured from hand-held cameras are typically stable due to the advancements and widespread adoption of video stabilization in both hardware and software. Under such circumstances, we retarget video stitching to an emerging issue, warping shake, which describes the undesired content instability in non-overlapping regions especially when image stitching technology is directly applied to videos. To address it, we propose the first unsupervised online video stitching framework, named StabStitch, by generating stitching trajectories and smoothing them. 
![image](https://github.com/nie-lang/StabStitch/blob/main/fig.png)
The above figure shows the occurrence and elimination of warping shakes.
> 
## Video
Here, we provide a [video](https://www.youtube.com/watch?v=03kGEZJHxzI&t) (released on YouTube) to show the stitched results from StabStitch and other solutions.

## 📝 Changelog

- [x] 2024.03.11: The paper of the arXiv version is online.
- [x] 2024.07.11: We have replaced the original arXiv version with the final camera-ready version.
- [x] 2024.07.11: The StabStitch-D dataset is available.
- [x] 2024.07.11: The inference code and pre-trained models are available.
- [x] 2024.07.12: We add a simple analysis of the limitations and prospects.

## Dataset (StabStitch-D)
The details of the dataset can be found in our paper. ([arXiv](https://arxiv.org/abs/2403.06378))

The dataset can be available at [Google Drive](https://drive.google.com/drive/folders/16EDGrKOLLwcMseOjpI7bCrv_aP1MYVcz?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1TKQAQ9zryUuU4uzTiswfHg)(Extraction code: 1234).

## Code
#### Requirement
We implement StabStitch with one GPU of RTX4090Ti. Refer to [environment.yml](https://github.com/nie-lang/StabStitch/blob/main/environment.yml) for more details.

#### Pre-trained model
The pre-trained models (spatial_warp.pth, temporal_warp.pth, and smooth_warp.pth) are available at [Google Drive](https://drive.google.com/drive/folders/1TuhQgD945MMnhmvnOwBS1LoLkYR1eetj?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1TTSbR4UYFL8f-nP3aGME7g) (extraction code: 1234). Please download them and put them in the 'model' folder.

#### Test on the StabStitch-D dataset
Modify the test_path in Codes/test_online.py and run:
```
python test_online.py
```
Then, a folder named 'result' will be created automatically to store the stitched videos.

About the TPS warping function, we set two modes to warp frames as follows:
* 'FAST' mode: It uses F.grid_sample to implement interpolation. It's fast but may produce thin black boundaries.
* 'NORMAL' mode: It uses our implemented interpolation function. It's a bit slower but avoid the black boundaries.

You can change the mode [here](https://github.com/nie-lang/StabStitch/blob/0c3665377e8bb76e062d5276cda72a7c7f0fab5b/Codes/test_online.py#L127).


#### Calculate the metrics on the StabStitch-D dataset
Modify the test_path in Codes/test_metric.py and run:
```
python test_metric.py
```

## Limitation and Future Prospect 

### Generalization
To test the model generalization, we adopt the pre-trained model (on the StabStitch-D dataset) to conduct some tests on traditional video stitching datasets. Surprisingly, it severely degrades and produces obvious distortions and artifacts, as illustrated in Figure (a) below. To further validate the generalization, we collect other video pairs from traditional video stitching datasets (over 30 video pairs) and retrain our model in the new dataset. As shown in Figure (b) below, it works well in the new dataset but fails to produce natural stitched videos on the StabStitch-D dataset.
![image](https://github.com/nie-lang/StabStitch/blob/main/limitation.png)

### Prospect
We found that performance degradation mainly occurs in the spatial warp model. Without corrected spatial warps, the subsequent smoothing process will amplify the distortion.

It then throws a question about how to ensure the model generalization in learning-based stitching models. A simple and intuitive idea is to establish a large-scale real-world stitching benchmark dataset with various complex scenes. It should benefit various stitching networks in the generalization. Another idea is to apply continuous learning to the field of stitching, enabling the network to work robustly across various datasets with different distributions

These are just a few simple proposals. We hope you, the intelligent minds in this field, can help to solve this problem and contribute to the advancement of this field. If you have some ideas and want to discuss them with me, please feel free to drop me an email. I’m open to any kinds of collaboration. 

## Meta
If you have any questions about this project, please feel free to drop me an email.

NIE Lang -- nielang@bjtu.edu.cn
```
@article{nie2024eliminating,
  title={Eliminating Warping Shakes for Unsupervised Online Video Stitching},
  author={Nie, Lang and Lin, Chunyu and Liao, Kang and Zhang, Yun and Liu, Shuaicheng and Zhao, Yao},
  journal={arXiv preprint arXiv:2403.06378},
  year={2024}
}
```


## References
[1] S. Liu, P. Tan, L. Yuan, J. Sun, and B. Zeng. Meshflow: Minimum latency online video stabilization. ECCV, 2016.  
[2] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images. TIP, 2021.   
[3] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Parallax-Tolerant Unsupervised Deep Image Stitching. ICCV, 2023.   
