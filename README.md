# MADGNet
The reproduction code of MADGNet which is accepted in CVPR 2024

## Abstract
Generalizability in deep neural networks plays a pivotal role in medical image segmentation. However, deep learning-based medical image analyses tend to overlook the importance of frequency variance, which is critical element for achieving a model that is both modality-agnostic and domain-generalizable.  Additionally, various models fail to account for the potential information loss that can arise from multi-task learning under deep supervision, a factor that can impair the model’s representation ability. To address these challenges, we propose a Modality-agnostic Domain Generalizable Network (MADGNet) for medical image segmentation, which comprises two key components: a Multi-Frequency in Multi-Scale Attention (MFMSA) block and Ensemble Sub-Decoding Module (E-SDM). The MFMSA block refines the process of spatial feature extraction, particularly in capturing boundary features, by incorporating multi-frequency and multi-scale features, thereby offering informative cues for tissue outline and anatomical structures. Moreover, we propose E-SDM to mitigate information loss in multi-task learning with deep supervision, especially during substantial upsampling from low resolution. We evaluate the segmentation performance of MADGNet across six modalities and fifteen datasets. Through extensive experiments, we demonstrate that MADGNet consistently outperforms  state-of-the-art models across various modalities, showcasing superior segmentation performance.  This affirms MADGNet as a robust solution for medical image segmentation that excels in diverse imaging scenarios. 

## Overall Architecture of MADGNet
![MFMSNet](https://github.com/BlindReview922/MADGNet/assets/142275582/8c1d54c5-b03d-4c71-b7f1-81e8c91e0d36)
![CascadedDecoder](https://github.com/BlindReview922/MADGNet/assets/142275582/8c057fd3-e681-4b52-b630-591f4bc5a8f5)

## Experiment Results

### Seen Clinical Settings Results
![Screenshot from 2023-11-26 16-16-13](https://github.com/BlindReview922/MADGNet/assets/142275582/30767364-13a7-43b1-8b00-dff7aa531e7d)

### Unseen Clinical Settings Results
![Screenshot from 2023-11-26 16-15-44](https://github.com/BlindReview922/MADGNet/assets/142275582/cef29e7d-5c41-4c82-9f9a-45c45de46cb9)

# Code Usage

STEP1. Download Github Code

STEP2. Download medical image segmentation dataset in following links
  1) ISIC2018: https://challenge.isic-archive.com/data/
  2) COVID19-1 & BUSI & Polyp Segmentation Dataset: https://github.com/Xiaoqi-Zhao-DLUT/MSNet-M2SNet
  3) DSB2018: https://www.kaggle.com/c/data-science-bowl-2018
  4) PH2 test dataset: https://www.fc.up.pt/addi/ph2%20database.html
  5) COVID19-2 test dataset: https://www.kaggle.com/datasets/piyushsamant11/pidata-new-names
  6) STU: https://drive.google.com/file/d/1k3OvEnYZaPWrng74aP4hAhgPXNHjpPj3/view?usp=drive_link
  7) MonuSeg2018: https://www.kaggle.com/datasets/tuanledinh/monuseg2018
  
STEP3. Move dataset into folder 'dataset/BioMedicalDataset/[dataset_type]'

STEP4. Download model pre-trained weights in following [link](https://drive.google.com/file/d/1y1DG5yh4zO9m6OhN4WHNqFqcV4mArYRx/view?usp=drive_link)

STEP5. Move pre-trained weights into folder 'model_weights/[dataset_type]'

STEP6. Enter following command

```
CUDA_VISIBLE_DEVICES=[GPU Number] python3 IS2D_main.py --num_workers 4 --data_path dataset/BioMedicalDataset --save_path model_weights --train_data_type [Train Dataset Type] --test_data_type [Test Dataset Type] --final_epoch [Final Epochs]
```

# Bibitex
```
@InProceedings{Nam_2024_CVPR,
    author    = {Nam, Ju-Hyeon and Syazwany, Nur Suriza and Kim, Su Jung and Lee, Sang-Chul},
    title     = {Modality-agnostic Domain Generalizable Medical Image Segmentation by Multi-Frequency in Multi-Scale Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {11480-11491}
}
```
