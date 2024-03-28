# (주)노타 - 군중 계수(Crowd Counting) 모델의 계산 효율성을 위한 경량 모델링 기업 연계 프로젝트


<a href="https://youtu.be/LCf51tyPkoY"><img src="https://img.shields.io/badge/Presentation video-FFFFFF?style=for-the-badge&logo=youtube&logoColor=ff0000"/></a> <a href="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-04/files/14772165/CV_4._Nota_.pdf"><img src="https://img.shields.io/badge/Presentation-FFFFFF?style=for-the-badge&logo=microsoftpowerpoint&logoColor=B7472A"/></a> <a href="https://synonymous-ton-89f.notion.site/Wrap-up-Reports-00e278b91e4c435e98433cff09d40f8a?pvs=4"><img src="https://img.shields.io/badge/Wrapup report-FFFFFF?style=for-the-badge&logo=googlesheets&logoColor=34A853"/></a> <a href="https://boostcampait.notion.site/CV-04-4f73bd9cbb4d483a9f424b82db49bcfa"><img src="https://img.shields.io/badge/Project summary-FFFFFF?style=for-the-badge&logo=notion&logoColor=000000"/></a>

## Overview

![image](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-04/assets/46666296/deade5da-6185-47ca-afde-d00425a2e676)

## [**PET(Point quEry Transformer)**](https://arxiv.org/abs/2308.13814)
![](https://imgur.com/WmgUquE.jpg)
Crowd Counting을 decomposable point querying process로 정의합니다. sparse input points는 필요시 4개의 new points로 split될 수 있습니다. 이러한 formulation은 많은 appealing properties를 보여줍니다.
- Intuitive: Input과 Output 모두 interpretable하고 steerable합니다.
- Generic: PET는 Input format을 간단히 조정함으로써, 다양한 crowd-related tasks에 적용할 수 있습니다.
- Effective: PET는 Crowd counting과 localization에서 State-of-the-art를 보여줍니다.

## Project Goal
- Transformer 기반의 군중 계수 SOTA 모델인 PET 모델을 구성하는 레이어/블록을 재설계하여 모델의 정확도를 최대한 유지하면서도, CPU/GPU에서의 추론속도를 개선
- 성능 지표는 MAE(Mean Absolute Error)로 BaseModel Mae에서 <span class="plusminus">&plusmn;3</span>의 오차범위 내에서 Inference Time 최대한 감소

## PET-CRT (Point quEry Transformer with Contextual cRowd counTing)
최종 결과로 다음과 같이 경량화된 PET구조를 제안합니다.

![](https://imgur.com/kD7V5rV.jpg)

- Only Encoder
- Multihead Self-Attention → HiLo with Pooling Attention
- Additional Residual Connection & Normalization

### Result
|Method|MAE|Inference Time(ms)|Std Dev|Parameters(M)|
|---|---|---|---|---|
|BaseModel|52.32|69.02|19.01|20.91|
|PET-CRT|51.34|**47.2**|14.02|**18.08**|
- MAE **2.12% 향상**
- 추론 속도 **31.61% 개선**
  
## Members
|<img src='https://imgur.com/ozd1yor.jpg' height=100 width=100px></img>|<img src='https://imgur.com/GXteBDS.jpg' height=100 width=100px></img>|<img src='https://imgur.com/aMVcwCF.jpg' height=100 width=100px></img>|<img src='https://imgur.com/F6ZfcEl.jpg' height=100 width=100px></img>|<img src='https://imgur.com/ZSVCV82.jpg' height=100 width=100px></img>|<img src='https://imgur.com/GBdY0k4.jpg' height=100 width=100px></img>|
|:---:|:---:|:---:|:---:|:---:|:---:|
| [강승환](https://github.com/kangshwan) | [김승민](https://github.com/viitamin) | [설훈](https://github.com/tjfgns6043) | [이도형](https://github.com/leedohyeong) | [전병관](https://github.com/wjsqudrhks) | [조성혁](https://github.com/seonghyeokcho) |

### Contribution
- 강승환 : Swin-T, Encoder 개수 줄이기, Pooling, Encoder-Decoder 통합 실험 
- 김승민 : VGG16_bn, Efficient_b0등 Backbone교체 실험
- 설훈 : Hydra Attention, Pooling ,Encoder-Decoder Pooling 실험
- 이도형 : Pooling, Service Pipeline, Demo 제작
- 전병관 : Pooling, HiLo Attention 실험
- 조성혁 : Swin-T, Encoder Window change, Pooling, Encoder-Decoder 통합 실험 

## Dataset
- SHA, Shanghai Tech Dataset(CVPR 2016 paper, Single Image Crowd Counting via Multi Column Convolutional Neural Network에서 제공)
- Total Images : 483장 (train : 300, test : 182)
- Image Size : Train (256, 256), Test (1024, 1024)

## Demo
<img width="80%" src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-04/assets/82288357/df833f79-de46-440b-b2a0-2af97489cab8.gif"/>

## Detail
- [발표 자료](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-04/files/14772165/CV_4._Nota_.pdf)
- [프로젝트 랩업 리포트](https://synonymous-ton-89f.notion.site/Wrap-up-Reports-00e278b91e4c435e98433cff09d40f8a?pvs=4)
- [프로젝트 소개 노션 페이지](https://boostcampait.notion.site/CV-04-4f73bd9cbb4d483a9f424b82db49bcfa)
