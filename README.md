# <p align="center">Eliminating Warping Shakes for Unsupervised Online Video Stitching

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
- [x] Release the StabStitch-D dataset.
- [ ] Release the inference code and pre-trained model.
- [ ] Release a limitation analysis about generalization.

## Dataset (StabStitch-D)
The details of the dataset can be found in our paper. ([arXiv](https://arxiv.org/abs/2403.06378))

The dataset can be available at [Google Drive](https://drive.google.com/drive/folders/16EDGrKOLLwcMseOjpI7bCrv_aP1MYVcz?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1TKQAQ9zryUuU4uzTiswfHg)(Extraction code: 1234).

## Code
We will release the code within this week.

## Limitation and Future Prospect 
.

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
[1] S Liu, P Tan, L Yuan, J Sun, B Zeng. Meshflow: Minimum latency online video stabilization. ECCV, 2016.  
[2] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images. TIP, 2021.   
[3] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Parallax-Tolerant Unsupervised Deep Image Stitching. ICCV, 2023.   
