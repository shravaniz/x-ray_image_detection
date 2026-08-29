# Ensemble deep‐learning networks for automated osteoarthritis grading in knee X‐ray images

## Overview
This repository contains the code and resources for the paper ["Ensemble deep‐learning networks for automated osteoarthritis grading in knee X‐ray images"](https://www.nature.com/articles/s41598-023-50210-4). The aim of this project is to develop an ensemble network that predicts the Kellgren-Lawrence (KL) grade for knee osteoarthritis (OA) severity using a deep learning approach.

![alt text](./image/kl_grade.png)

## Introduction
Osteoarthritis (OA) is a common joint disease that affects millions of people worldwide. The Kellgren-Lawrence (KL) grading system is the standard for classifying the severity of knee OA using X-ray images. However, the grading depends on the clinician’s subjective assessment, leading to significant variability. This project proposes an ensemble deep learning model that provides consistent and accurate KL grade predictions.

## Dataset
The dataset used in this study is from the Osteoarthritis Initiative (OAI), which consists of 8260 knee X-ray images. The dataset includes images for KL grades ranging from 0 to 4. It is publicly available and can be accessed [here](https://data.mendeley.com/datasets/56rmx5bjcr/1).

## Model architecture
The ensemble network consists of several deep learning models spanning both Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), including:
- DenseNet-161
- EfficientNet-b5
- EfficientNet-V2-s
- RegNet-Y-8GF
- ResNet-101
- ResNext-50-32x4d
- Wide-ResNet-50-2
- ShuffleNet-V2-x2-0
- Vision Transformer (ViT-B/16)
- Swin Transformer (Swin-S)
- Swin Transformer V2 (Swin-V2-S)

Each model is trained with optimal image sizes to enhance performance. The models leverage pre-trained weights from [`torchvision.models`](https://pytorch.org/vision/stable/models.html) and [`timm`](https://github.com/huggingface/pytorch-image-models). Combining the local feature extraction of CNNs with the global self-attention mechanisms of Vision Transformers allows the ensemble to effectively capture both fine-grained knee joint details and overall anatomical structure. The final prediction is made using a mix voting method, which combines hard and soft voting strategies.

```mermaid
graph TD
    Input[Knee X-Ray Image] --> CNNs[Convolutional Neural Networks]
    Input --> ViT[Vision Transformer - ViT-B/16]
    Input --> Swin[Swin Transformers - Swin-S & Swin-V2-S]

    subgraph CNN_Branch [CNN Models]
        CNNs --> C1[DenseNet-161 / ResNet-101 / Wide-ResNet-50]
        CNNs --> C2[EfficientNet-b5 / EfficientNet-V2-s]
        CNNs --> C3[RegNet-Y-8GF / ResNeXt-50 / ShuffleNet-V2]
    end

    subgraph ViT_Branch [Vision Transformer Layers]
        ViT --> V1[16x16 Non-Overlapping Patch Projection]
        V1 --> V2[Positional Encoding]
        V2 --> V3[Multi-Head Self-Attention Encoders]
    end

    subgraph Swin_Branch [Swin Transformer Layers]
        Swin --> S1[Patch Partitioning & Linear Embedding]
        S1 --> S2[Shifted Window Self-Attention - W-MSA / SW-MSA]
        S2 --> S3[Patch Merging & Hierarchical Feature Maps]
    end

    C1 --> Logits[Model Classifier Heads]
    C2 --> Logits
    C3 --> Logits
    V3 --> Logits
    S3 --> Logits

    Logits --> Ensemble[Mix Voting Strategy]
    Ensemble --> Soft[Soft Voting - Avg Probabilities]
    Ensemble --> Hard[Hard Voting - Majority Class]
    Soft --> Output[Final KL Grade Prediction 0-4]
    Hard --> Output
```



## Training strategy
The training process involves the following steps:
1. Models are trained using different image sizes.
2. The initial layers are frozen, and only the fully connected layers are trained with a learning rate of 0.01.
3. Subsequently, all layers are unfrozen, and the learning rate is reduced progressively to stabilize training.
4. Stratified five-fold cross-validation is used to handle class imbalance and improve generalization.

## License
Ensemble deep‐learning networks for automated osteoarthritis grading in knee X‐ray image is released under the [MIT License](LICENSE).

## Citation
```
Chen, Pingjun (2018), “Knee Osteoarthritis Severity Grading Dataset”, Mendeley Data, V1, doi: 10.17632/56rmx5bjcr.1
```
