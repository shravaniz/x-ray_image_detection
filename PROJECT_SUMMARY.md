# Project Summary: Knee KL Grade Classification

## Overview
Ensemble deep-learning networks for automated osteoarthritis grading in knee X-ray images.

- **Paper:** [Nature Scientific Reports (2023)](https://www.nature.com/articles/s41598-023-50210-4)
- **Author:** Sunwoo Pi
- **License:** MIT

## Problem
Classify knee osteoarthritis severity using the **Kellgren-Lawrence (KL) grading system** (grades 0–4) from X-ray images. Manual grading is subjective with high variability — this project automates it with deep learning.

## Dataset
- **Source:** [Osteoarthritis Initiative (OAI)](https://data.mendeley.com/datasets/56rmx5bjcr/1) — 8,260 knee X-ray images
- **Location in repo:** `KneeXrayData/` (7.3 GB, untracked)
- **Class distribution (train/val/test = 5,778/826/1,656):**

  | KL Grade | Train | Val | Test | Total |
  |----------|-------|-----|------|-------|
  | 0 (none) | 2,286 | 328 | 639 | 3,253 |
  | 1 (doubtful) | 1,046 | 153 | 296 | 1,495 |
  | 2 (mild) | 1,516 | 212 | 447 | 2,175 |
  | 3 (moderate) | 757 | 106 | 223 | 1,086 |
  | 4 (severe) | 173 | 27 | 51 | 251 |

- **auto_test:** 1,526 additional unlabeled images for inference
- **Resolutions:** 224×224 and 299×299 grayscale PNGs
- **Naming:** `9003175L.png` / `9003175R.png` (left/right knee), `9003175_1.png` / `9003175_2.png` (auto-cropped)

## Models (Hybrid Ensemble: CNNs + Transformers)
All from `torchvision.models` with pretrained ImageNet weights, final layer replaced with 5-class head:

| # | Model | Type | Optimal Input Size |
|---|-------|------|-------------------|
| 1 | DenseNet-161 | CNN | 456×456 |
| 2 | EfficientNet-b5 | CNN | 456×456 |
| 3 | EfficientNet-V2-s | CNN | 384×384 |
| 4 | RegNet-Y-8GF | CNN | 448×448 |
| 5 | ResNet-101 | CNN | 456×456 |
| 6 | ResNext-50-32x4d | CNN | 512×512 |
| 7 | Wide-ResNet-50-2 | CNN | 456×456 |
| 8 | ShuffleNet-V2-x2-0 | CNN | 512×512 |
| 9 | ViT-B/16 | Transformer | 224×224 |
| 10 | Swin-S | Transformer | 224×224 |
| 11 | Swin-V2-S | Transformer | 224×224 |

## Architecture (16 Python files in `OAI-KL/`)

### Core Training Pipeline
| File | Purpose |
|------|---------|
| `model.py` | Model factory — loads 8 pretrained architectures, replaces FC head with 5-class Linear |
| `dataset.py` | PyTorch Dataset — grayscale → RGB, albumentations transforms |
| `main.py` | Training entry point — 5-fold stratified CV, 2-phase training, label-smoothing CE loss |
| `main_optuna.py` | Hyperparameter tuning via Optuna (optimizer, LR, batch size) |
| `early_stop.py` | Early stopping + model checkpointing to `./models/{model}/{size}/` |

### Inference & Ensemble
| File | Purpose |
|------|---------|
| `test_auto.py` | Per-model inference with TTA (horizontal flip), outputs CSV submissions |
| `test_ensemble.py` | Manual ensemble — combines CSVs with equal or custom weights |
| `ensemble_combinations.py` | Exhaustive search — all 8C2…8C8 combinations with hard/soft/mix voting |

### Evaluation & Visualization
| File | Purpose |
|------|---------|
| `score.py` | Confusion matrix, ROC curves, PR curves, classification report |
| `score_auto.py` | Batch scoring across all submissions, optional below-average pruning (`-r` flag) |
| `write_performance.py` | Logs per-model accuracy/F1 to `performance.txt` |
| `box_plot.py` | Box plot of accuracy vs ensemble size (1 to 8 models) |
| `cam.py` | Single-model Grad-CAM / ScoreCAM visualization |
| `cam_ensemble.py` | Averaged Grad-CAM across all 8 models |

### Utilities
| File | Purpose |
|------|---------|
| `make_csv.py` | Generate CSV from class-foldered image directories |
| `my_custom_loss.py` | Custom CE, MSE, CE+MSE losses (unused in final pipeline) |

## Training Strategy
1. **Phase 1 (Epoch 1):** Only the new FC head trained (backbone frozen), LR = 0.01
2. **Phase 2 (Epochs 2–30):** All layers unfrozen, LR = 0.001, weight decay = 1e-4, MultiStepLR (gamma=0.1 at epoch 2)
3. **Loss:** CrossEntropyLoss with label smoothing = 0.1
4. **Optimizer:** Adam
5. **Batch size:** 16
6. **Validation:** 5-fold stratified cross-validation (seed=42)
7. **Early stopping:** patience=7, delta=0.1
8. **Augmentations:** HorizontalFlip (p=0.5), Rotate (±20°), ImageNet normalization, INTER_CUBIC resize

## Ensemble Voting Methods
- **Hard Voting:** Majority class wins; ties → minimum KL grade selected
- **Soft Voting:** Average probability distributions across models, argmax
- **Mix Voting (final):** Soft voting on ties, hard voting otherwise

## Reported Results
- **Accuracy:** 76.93%
- **F1 Score:** 0.7665

## Dependencies
```
torch==2.0.0+cu117, torchvision==0.15.1+cu117
albumentations, pytorch_grad_cam, ttach (TTA)
opencv-python, Pillow, pandas, numpy
scikit-learn, matplotlib, natsort, tqdm
optuna (hyperparameter search only)
h5py (data loading)
```

## Data Directory Structure
```
KneeXrayData/KneeXrayData/
├── ClsKLData/
│   ├── kneeKL224/          (224×224 grayscale PNGs)
│   │   ├── train/0..4/     (5,778 images)
│   │   ├── val/0..4/       (826 images)
│   │   ├── test/0..4/      (1,656 images — ground truth labels)
│   │   └── auto_test/0..4/ (1,526 images — no labels)
│   ├── kneeKL299/          (299×299 — same images, same counts)
│   └── models/
│       └── model_best/     (26 baseline checkpoints — different architectures)
└── DetKneeData/            (Knee detection from full X-rays)
    ├── H5/                 (HDF5 files with full 256×320 X-rays + bounding boxes)
    └── best_models/        (Detection model checkpoints)
```

## CLI Usage Examples
```bash
# Train a single model
python main.py -m densenet_161 -i 456

# Hyperparameter search
python main_optuna.py -m resnet_101 -i 224

# Run inference with TTA
python test_auto.py -m densenet_161 -i 456

# Score all submissions
python score_auto.py -m densenet_161 -i 456 -t 0.65

# Remove below-average checkpoints
python score_auto.py -m densenet_161 -i 456 -r

# Generate Grad-CAM
python cam.py -m densenet_161

# Generate ensemble CAM
python cam_ensemble.py

# Exhaustive ensemble combination search
python ensemble_combinations.py
```

## Key Notes
- **No trained ensemble weights in repo** — all `.pt` files are gitignored. Must train from scratch or obtain author's weights.
- **No CSVs provided** — `train.csv` and `test_correct.csv` must be regenerated via `make_csv.py` logic.
- **GPU required** — all scripts use `.cuda()` throughout.
- **`score_auto.py -r` is destructive** — deletes below-average checkpoints.
- **No Docker, CI/CD, or deployment files** — pure research codebase.
- **No Jupyter notebooks** — all CLI-based scripts.
