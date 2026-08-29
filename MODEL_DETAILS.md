# Model Details — Ensemble KL Grade Classification

## Overview

All 8 models come from `torchvision.models` with ImageNet pretrained weights (`weights='DEFAULT'`). Each follows the same recipe:

1. Load pretrained backbone
2. Replace final classification layer with `nn.Linear(in_features, 5)` for KL grades 0–4
3. Freeze backbone → train head only (epoch 1, LR 0.01)
4. Unfreeze all → fine-tune (epochs 2–30, LR 0.001 → 0.0001)

---

## 1. DenseNet-161

| Property | Value |
|----------|-------|
| **Paper model #** | 1 |
| **Command** | `python main.py -m densenet_161 -i 456` |
| **Optimal input size** | 456×456 |
| **Parameters** | ~28.7M |
| **Architecture family** | Densely Connected Convolutional Networks |
| **Key feature** | Every layer connected to every other layer — feature reuse, strong gradient flow |
| **Head replacement** | `model_ft.classifier` — `nn.Linear(2208, 5)` |
| **Target layer (Grad-CAM)** | `model_ft.features[-1]` |
| **Published year** | 2017 (CVPR Best Paper) |

**How it works:**
Each layer receives feature maps from ALL previous layers: `x_l = H_l([x_0, x_1, ..., x_{l-1}])`. This dense connectivity means the classifier sees features from every depth simultaneously — shallow edges, mid-level textures, and deep semantic features all feed into the final decision. This is especially useful for KL grading where both fine details (tiny osteophytes) and global structure (joint space width) matter.

**Why it's in the ensemble:**
DenseNets excel at preserving fine-grained details through the network. For KL grading, small osteophytes (grade 1→2 distinction) and subtle joint space narrowing are critical — DenseNet's architecture is naturally suited to this.

---

## 2. EfficientNet-b5

| Property | Value |
|----------|-------|
| **Paper model #** | 2 |
| **Command** | `python main.py -m efficientnet_b5 -i 456` |
| **Optimal input size** | 456×456 |
| **Parameters** | ~30M |
| **Architecture family** | EfficientNet (compound scaling) |
| **Key feature** | Balanced scaling of depth, width, and resolution |
| **Head replacement** | `model_ft.classifier[1]` — `nn.Linear(2048, 5)` |
| **Target layer (Grad-CAM)** | `model_ft.features[-1]` |
| **Published year** | 2019 (ICML) |

**How it works:**
Uses a compound coefficient φ to uniformly scale network depth (d = α^φ), width (w = β^φ), and resolution (r = γ^φ). The base EfficientNet-B0 was found via neural architecture search (NAS), then scaled up. B5 is the sweet spot — large enough to capture medical image detail, small enough to train efficiently.

**Why it's in the ensemble:**
Provides a different inductive bias than DenseNet/ResNet. NAS-designed architectures often discover patterns humans wouldn't — complementary to hand-designed networks.

---

## 3. EfficientNet-V2-s

| Property | Value |
|----------|-------|
| **Paper model #** | 3 |
| **Command** | `python main.py -m efficientnet_v2_s -i 384` |
| **Optimal input size** | 384×384 |
| **Parameters** | ~21M |
| **Architecture family** | EfficientNet-V2 |
| **Key feature** | Faster training, better than EfficientNet-V1, uses MBConv + Fused-MBConv |
| **Head replacement** | `model_ft.classifier[1]` — `nn.Linear(1280, 5)` |
| **Target layer (Grad-CAM)** | `model_ft.features[-1]` |
| **Published year** | 2021 (ICML) |

**How it works:**
Improves on EfficientNet-V1 by: (1) using Fused-MBConv in early layers (faster than regular MBConv), (2) non-uniform scaling (more aggressive in later stages), (3) progressive learning (starts small, grows during training). The "s" (small) variant is used here for efficiency — still powerful but ~30% fewer params than B5.

**Why it's in the ensemble:**
Different variant of the EfficientNet family — captures different feature representations than B5. The Fused-MBConv blocks in early layers process spatial information differently.

---

## 4. RegNet-Y-8GF

| Property | Value |
|----------|-------|
| **Paper model #** | 4 |
| **Command** | `python main.py -m regnet_y_8gf -i 448` |
| **Optimal input size** | 448×448 |
| **Parameters** | ~39M |
| **Architecture family** | RegNet (Regular Network) |
| **Key feature** | Network design space with quantized linear parameterization |
| **Head replacement** | `model_ft.fc` — `nn.Linear(2016, 5)` |
| **Target layer (Grad-CAM)** | `model_ft.trunk_output[-1]` |
| **Published year** | 2020 (CVPR) |

**How it works:**
Instead of designing one architecture, the authors designed a "design space" — a parameterized family of networks. RegNet-Y-8GF means ~8 GigaFLOPs compute budget. The key innovation: within each stage (block group), width and depth follow a simple linear rule determined by a slope parameter — no irregular jumps.

**Why it's in the ensemble:**
Completely different design philosophy — not hand-crafted (like ResNet/DenseNet) and not NAS-searched (like EfficientNet). It's from a parameterized design space, providing yet another distinct architectural prior. The 8GF variant is the largest model in the ensemble.

---

## 5. ResNet-101

| Property | Value |
|----------|-------|
| **Paper model #** | 5 |
| **Command** | `python main.py -m resnet_101 -i 456` |
| **Optimal input size** | 456×456 |
| **Parameters** | ~44.5M |
| **Architecture family** | Residual Networks |
| **Key feature** | Skip connections — solves vanishing gradients in deep networks |
| **Head replacement** | `model_ft.fc` — `nn.Linear(2048, 5)` |
| **Target layer (Grad-CAM)** | `model_ft.layer4[-1]` |
| **Published year** | 2015 (CVPR Best Paper) |

**How it works:**
Each "residual block" learns `F(x) + x` — the identity shortcut `+ x` means the network only needs to learn the residual (difference) from the input. This made training 100+ layer networks possible. ResNet-101 uses "bottleneck" blocks: 1×1 conv (reduce) → 3×3 conv (process) → 1×1 conv (expand), keeping computation manageable.

**Why it's in the ensemble:**
The foundational deep architecture. Even though newer designs exist, ResNet-101's straightforward approach provides a reliable baseline. Its residual connections produce very different feature representations than dense connections.

---

## 6. ResNext-50-32x4d

| Property | Value |
|----------|-------|
| **Paper model #** | 6 |
| **Command** | `python main.py -m resnext_50_32x4d -i 512` |
| **Optimal input size** | 512×512 |
| **Parameters** | ~25M |
| **Architecture family** | ResNeXt (Aggregated Residual Transformations) |
| **Key feature** | Grouped convolutions — split channels into 32 groups of 4 |
| **Head replacement** | `model_ft.fc` — `nn.Linear(2048, 5)` |
| **Target layer (Grad-CAM)** | `model_ft.layer4[-1]` |
| **Published year** | 2017 (CVPR) |

**How it works:**
Introduces "cardinality" — instead of making networks deeper (ResNet) or wider (Wide-ResNet), ResNeXt splits computation into parallel paths (32 groups, each with 4 channels: 32×4d). This is the same principle as Inception's "split-transform-merge" but simplified — all paths share the same topology. The 32×4d configuration means 32 parallel groups, each bottleneck-dimension 4.

**Why it's in the ensemble:**
The grouped convolution approach creates diverse feature detectors — each group specializes in different patterns. With 32 groups at 512×512 (the largest input size), it captures fine osteophyte details that other models might miss.

---

## 7. Wide-ResNet-50-2

| Property | Value |
|----------|-------|
| **Paper model #** | 7 |
| **Command** | `python main.py -m wide_resnet_50_2 -i 456` |
| **Optimal input size** | 456×456 |
| **Parameters** | ~68.9M |
| **Architecture family** | Wide Residual Networks |
| **Key feature** | Decreased depth, increased width — wider = more features per layer |
| **Head replacement** | `model_ft.fc` — `nn.Linear(2048, 5)` |
| **Target layer (Grad-CAM)** | `model_ft.layer4[-1]` |
| **Published year** | 2016 (BMVC) |

**How it works:**
The original ResNet paper went very deep (152, 1001 layers). This paper showed that going wider (more channels per layer) is often more effective than going deeper. Wide-ResNet-50-2 means 50 layers with 2× the channel width of standard ResNet-50. Each layer has more capacity to learn diverse features — dropout between convolutions prevents overfitting from the increased parameters.

**Why it's in the ensemble:**
The widest model (most features per layer). Where ResNeXt splits channels into many groups, Wide-ResNet makes every group wider — the opposite design choice applied to the same problem. This architectural diversity is key to ensemble performance.

---

## 8. ShuffleNet-V2-x2-0

| Property | Value |
|----------|-------|
| **Paper model #** | 8 |
| **Command** | `python main.py -m shufflenet_v2_x2_0 -i 512` |
| **Optimal input size** | 512×512 |
| **Parameters** | ~7.4M |
| **Architecture family** | ShuffleNet-V2 |
| **Key feature** | Channel shuffle — efficient communication between groups |
| **Head replacement** | `model_ft.fc` — `nn.Linear(2048, 5)` |
| **Target layer (Grad-CAM)** | `model_ft.conv5` |
| **Published year** | 2018 (ECCV) |

**How it works:**
Channel shuffle randomly permutes channels between groups after each grouped convolution — it's a "free" operation (no parameters, no FLOPs) that ensures information flows between all groups. The V2 version incorporates practical efficiency metrics (memory access cost, degree of parallelism) into the design, not just theoretical FLOPs. The x2-0 scale means 2× channel multiplier across all stages.

**Why it's in the ensemble:**
The lightweight option — only 7.4M parameters vs. 69M for Wide-ResNet. Lightweight architectures learn different features because they're forced to be efficient. Its channel shuffle provides yet another distinct computational pattern. At 512×512 (max input size), it compensates for fewer parameters with higher resolution.

---

---

## 9. Vision Transformer (ViT-B/16)

| Property | Value |
|----------|-------|
| **Command** | `python main.py -m vit_b_16 -i 224` |
| **Optimal input size** | 224×224 |
| **Parameters** | ~86M |
| **Architecture family** | Vision Transformer |
| **Key feature** | Self-attention mechanism across non-overlapping 16x16 image patches |
| **Head replacement** | `model_ft.heads.head` — `nn.Linear(768, 5)` |
| **Target layer (Grad-CAM)** | `model_ft.encoder.layers[-1].ln_1` |
| **Published year** | 2020 (ICLR 2021) |

**How it works:**
Splits images into sequences of 16x16 flattened patches, projects them into embeddings with positional encodings, and processes them through standard Transformer encoder blocks. Captures global interactions across the entire image from early stages.

**Why it's in the ensemble:**
Provides global self-attention feature representation, complementary to local inductive biases of CNNs.

---

## 10. Swin Transformer (Swin-S)

| Property | Value |
|----------|-------|
| **Command** | `python main.py -m swin_s -i 224` |
| **Optimal input size** | 224×224 |
| **Parameters** | ~50M |
| **Architecture family** | Hierarchical Vision Transformer |
| **Key feature** | Shifted window self-attention for hierarchical multi-scale feature maps |
| **Head replacement** | `model_ft.head` — `nn.Linear(768, 5)` |
| **Target layer (Grad-CAM)** | `model_ft.features[-1]` |
| **Published year** | 2021 (ICCV Best Paper) |

**How it works:**
Computes self-attention within local non-overlapping windows and shifts windows between layers to allow cross-window connections, generating hierarchical representations like CNNs.

**Why it's in the ensemble:**
Combines the multiscale spatial feature hierarchy of CNNs with the flexible modeling power of self-attention.

---

## 11. Swin Transformer V2 (Swin-V2-S)

| Property | Value |
|----------|-------|
| **Command** | `python main.py -m swin_v2_s -i 224` |
| **Optimal input size** | 224×224 |
| **Parameters** | ~50M |
| **Architecture family** | Hierarchical Vision Transformer |
| **Key feature** | Post-normalization, log-CPB (continuous position bias), and residual-post-norm for stable scaling |
| **Head replacement** | `model_ft.head` — `nn.Linear(768, 5)` |
| **Target layer (Grad-CAM)** | `model_ft.features[-1]` |
| **Published year** | 2022 (CVPR) |

**How it works:**
Enhances Swin Transformer with residual-post-norm and scaled cosine attention to improve stability and transferability across varied resolutions.

**Why it's in the ensemble:**
Improves fine-grained representation stability and joint detail extraction.

---

## Architecture Comparison

| # | Model | Params | Input | Family | Key Innovation |
|---|-------|--------|-------|--------|----------------|
| 1 | DenseNet-161 | 28.7M | 456 | DenseNet | Dense connectivity — every layer sees all previous features |
| 2 | EfficientNet-b5 | 30M | 456 | EfficientNet | NAS + compound scaling |
| 3 | EfficientNet-V2-s | 21M | 384 | EfficientNet-V2 | Fused-MBConv + progressive learning |
| 4 | RegNet-Y-8GF | 39M | 448 | RegNet | Design space parameterization |
| 5 | ResNet-101 | 44.5M | 456 | ResNet | Residual (skip) connections |
| 6 | ResNext-50-32x4d | 25M | 512 | ResNeXt | Grouped convolutions (cardinality) |
| 7 | Wide-ResNet-50-2 | 68.9M | 512 | Wide-ResNet | Width over depth |
| 8 | ShuffleNet-V2-x2-0 | 7.4M | 512 | ShuffleNet-V2 | Channel shuffle + practical efficiency |
| 9 | ViT-B/16 | 86M | 224 | Vision Transformer | Patch-based global self-attention |
| 10 | Swin-S | 50M | 224 | Swin Transformer | Shifted window hierarchical self-attention |
| 11 | Swin-V2-S | 50M | 224 | Swin Transformer V2 | Log-CPB & scaled cosine attention |

## Input Size Rationale

The paper found that different architectures benefit from different input resolutions:

| 384×384 | 448×448 | 456×456 | 512×512 |
|---------|---------|---------|---------|
| EfficientNet-V2-s | RegNet-Y-8GF | DenseNet-161 | ResNext-50-32x4d |
| | | EfficientNet-b5 | Wide-ResNet-50-2 |
| | | ResNet-101 | ShuffleNet-V2-x2-0 |

**Why not one size for all?** Larger images capture more fine detail (osteophytes, joint space) but require more memory and compute. Each architecture's internal downsampling ratio and receptive field determine its optimal input size.

## Training Hyperparameters (shared across all models)

| Parameter | Value |
|-----------|-------|
| Batch size | 8 (reduced from 16 for 4GB VRAM) |
| Epochs | 30 (max, early stopping applies) |
| Cross-validation | 5-fold stratified |
| Loss | CrossEntropyLoss with label smoothing = 0.1 |
| **Phase 1 (Epoch 1)** | |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Trainable params | Classifier head only (~11K) |
| **Phase 2 (Epochs 2-30)** | |
| Optimizer | Adam |
| Learning rate | 0.001 → 0.0001 (MultiStepLR, gamma=0.1 at epoch 2) |
| Weight decay | 0.0001 |
| Trainable params | All (~7M–69M depending on model) |
| **Early stopping** | patience=7, delta=0.1 |
| **Augmentation** | HorizontalFlip (p=0.5), Rotate (±20°), ImageNet normalization |
| **Seed** | 42 |

## Ensemble Combination

After all 8 models are trained, 5 checkpoints per model = 40 total checkpoints producing predictions. These are combined via:

1. **Hard Voting:** Each model votes → majority class wins (ties → minimum class selected)
2. **Soft Voting:** Average probability distributions → argmax
3. **Mix Voting (used in paper):** Hard voting, but falls back to soft voting on ties

The paper exhaustively tested all combinations from 8C2 (2-model ensembles) through 8C8 (all 8 models) and found the optimal configuration.
