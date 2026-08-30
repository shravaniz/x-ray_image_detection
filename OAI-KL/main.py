# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn, optim
# from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from dataset import ImageDataset
from early_stop import EarlyStopping
from model import model_return
# from my_custom_loss import my_ce_mse_loss

def train_for_kfold(model, dataloader, criterion, optimizer, scheduler, device, fold, epoch):
    train_loss = 0.0
    model.train() # Model을 Train Mode로 변환 >> Dropout Layer 같은 경우 Train시 동작 해야 함
    with torch.set_grad_enabled(True): # with문 : 자원의 효율적 사용, 객체의 life cycle을 설계 가능, 항상(True) gradient 연산 기록을 추적
        for batch in tqdm(dataloader, desc=f'Fold {fold} Epoch {epoch} Train', unit='Batch'):
            optimizer.zero_grad() # 반복 시 gradient(기울기)를 0으로 초기화, gradient는 += 되기 때문
            image = batch['image'].to(device)
            labels = batch['target'].to(device)

            # labels = F.one_hot(labels, num_classes=5).float() # nn.MSELoss() 사용 시 필요
            output = model(image) # image(data)를 model에 넣어서 hypothesis(가설) 값을 획득

            loss = criterion(output, labels) # Error, Prediction Loss 계산
            train_loss += loss.item() # loss.item()을 통해 Loss의 스칼라 값을 가져온다.

            loss.backward() # Prediction Loss를 Back Propagation으로 계산
            optimizer.step() # optimizer를 이용해 Loss를 효율적으로 최소화 할 수 있게 Parameter 수정

        if scheduler is not None:
            scheduler.step()

    return train_loss

def val_for_kfold(model, dataloader, criterion, device, fold, epoch):
    val_loss = 0.0
    targets = []
    predictions = []
    model.eval() # Model을 Eval Mode로 전환 >> Dropout Layer 같은 경우 Eval시 동작 하지 않아야 함
    with torch.no_grad(): # gradient 연산 기록 추적 off
        for batch in tqdm(dataloader, desc=f'Fold {fold} Epoch {epoch} Valid', unit='Batch'):
            image = batch['image'].to(device)
            labels = batch['target'].to(device)

            # labels = F.one_hot(labels, num_classes=5).float() # nn.MSELoss() 사용 시 필요
            output = model(image)

            loss = criterion(output, labels)
            val_loss += loss.item()
            targets.extend(labels.cpu().tolist())
            predictions.extend(output.argmax(dim=1).cpu().tolist())

    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision_macro': precision_score(targets, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(targets, predictions, average='macro', zero_division=0),
        'f1_macro': f1_score(targets, predictions, average='macro', zero_division=0),
    }
    metrics['dice_macro'] = metrics['f1_macro']
    return val_loss, metrics

def train(train_dataset, val_dataset, args, batch_size, epochs, k, splits, labels, foldperf, device):
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_dataset)), labels), start=1):
        print(f"\n{'=' * 16} Fold {fold}/{k} {'=' * 16}")
        print(f"Training samples: {len(train_idx)} | Validation samples: {len(val_idx)}")
        # Data Load에 사용되는 index, key의 순서를 지정하는데 사용, Sequential , Random, SubsetRandom, Batch 등 + Sampler
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        # Data Load
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=args.workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=args.workers)

        model_ft = model_return(args)

        class_weights = None
        if args.class_weighting == 'balanced':
            train_labels = labels[train_idx]
            class_counts = np.bincount(train_labels, minlength=5)
            class_weights = torch.tensor(len(train_labels) / (5 * class_counts), dtype=torch.float32, device=device)
            print(f"Class weights: {class_weights.cpu().numpy().round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1) # Loss Function
        # criterion = nn.MSELoss()
        # criterion = my_ce_mse_loss

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.01) # Optimizer
        scheduler = None

        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            model_ft = nn.DataParallel(model_ft) # model이 여러 대의 gpu에 할당되도록 병렬 처리
        model_ft.to(device)

        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

        patience = 7
        early_stopping = EarlyStopping(args, patience=args.patience, verbose=True, delta=args.early_stopping_delta)

        for epoch in range(1, epochs + 1):
            if epoch == 2:
                for param in model_ft.parameters():
                    param.requires_grad=True

                optimizer = optim.Adam(filter(lambda p: p.requires_grad,model_ft.parameters()), weight_decay=0.0001, lr=0.001)
                # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
                scheduler = MultiStepLR(optimizer, milestones=[2], gamma=0.1)

            print(f"Learning Rate : {optimizer.param_groups[0]['lr']}")

            train_loss = train_for_kfold(model_ft, train_loader, criterion, optimizer, scheduler, device, fold, epoch)
            val_loss, val_metrics = val_for_kfold(model_ft, val_loader, criterion, device, fold, epoch)

            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)

            print(
                f"Epoch: {epoch}/{epochs} | Train Loss: {train_loss:.3f} | Valid Loss: {val_loss:.3f} | "
                f"Accuracy: {val_metrics['accuracy']:.3f} | Precision (macro): {val_metrics['precision_macro']:.3f} | "
                f"Recall (macro): {val_metrics['recall_macro']:.3f} | F1 (macro): {val_metrics['f1_macro']:.3f} | "
                f"Dice (macro): {val_metrics['dice_macro']:.3f}"
            )

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)

            early_stopping(val_loss, model_ft, args, fold, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        foldperf[f"fold{fold}"] = history

    tl_f, vall_f = [], []
    metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'dice_macro']
    final_metrics = {metric_name: [] for metric_name in metric_names}

    for f in range(1, k+1):
        tl_f.append(np.mean(foldperf[f'fold{f}']['train_loss']))
        vall_f.append(np.mean(foldperf[f'fold{f}']['val_loss']))
        for metric_name in metric_names:
            final_metrics[metric_name].append(foldperf[f'fold{f}']['val_metrics'][-1][metric_name])

    print()
    print(f"Performance of {k} Fold Cross Validation")
    print(f"Avg Train Loss: {np.mean(tl_f):.3f} \t Avg Valid Loss: {np.mean(vall_f):.3f}")
    print(
        f"Final Validation Accuracy: {np.mean(final_metrics['accuracy']):.3f} | "
        f"Precision (macro): {np.mean(final_metrics['precision_macro']):.3f} | "
        f"Recall (macro): {np.mean(final_metrics['recall_macro']):.3f} | "
        f"F1 (macro): {np.mean(final_metrics['f1_macro']):.3f} | "
        f"Dice (macro): {np.mean(final_metrics['dice_macro']):.3f}"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', required=True, choices=[
        'resnet_101', 'resnext_50_32x4d', 'wide_resnet_50_2', 'densenet_161',
        'efficientnet_b5', 'efficientnet_v2_s', 'regnet_y_8gf', 'shufflenet_v2_x2_0',
        'vit_b_16', 'swin_s', 'swin_v2_s'
    ])
    parser.add_argument('-i', '--image_size', type=int, default=224, dest='image_size', action='store')
    parser.add_argument('--data-dir', type=Path, default=Path('../KneeXrayData/ClsKLData/kneeKL224/train'))
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--class-weighting', choices=['balanced', 'none'], default='balanced')
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--early-stopping-delta', type=float, default=0.0)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available. Use --device cpu or install a CUDA-enabled PyTorch build.')
    device = torch.device(args.device)
    image_paths = []
    labels = []
    for label in range(5):
        class_dir = args.data_dir / str(label)
        if not class_dir.is_dir():
            raise FileNotFoundError(f'Missing class directory: {class_dir}')
        class_images = sorted(class_dir.glob('*.png'))
        image_paths.extend(str(path) for path in class_images)
        labels.extend([label] * len(class_images))
    if not image_paths:
        raise RuntimeError(f'No PNG images found under {args.data_dir}')
    train_csv = pd.DataFrame({'data': image_paths, 'label': labels})

    image_size_tuple = (args.image_size, args.image_size)

    print(f"Model Type : {args.model_type}")
    print(f"Image Size : {image_size_tuple}")
    print(f"Data directory : {args.data_dir.resolve()}")
    print(f"Device : {device}")
    print(f"Training images : {len(train_csv)}")
    print(f"Cross-validation : {args.folds} folds x up to {args.epochs} epochs")

    train_transform = A.Compose([
                    A.Resize(args.image_size, args.image_size, interpolation=cv2.INTER_CUBIC, p=1),
                    # A.RandomCrop(height=int(384*0.8), width=int(384*0.8), p=1),
                    # A.GridDistortion(p=0.5),
                    # A.ElasticTransform(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=20, p=1),
                    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # -1 ~ 1의 범위를 가지도록 정규화
                    ToTensorV2() # 0 ~ 1의 범위를 가지도록 정규화
                    ])
    val_transform = A.Compose([
                    A.Resize(args.image_size, args.image_size, interpolation=cv2.INTER_CUBIC, p=1),
                    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # -1 ~ 1의 범위를 가지도록 정규화
                    ToTensorV2() # 0 ~ 1의 범위를 가지도록 정규화
                    ])
    train_dataset = ImageDataset(train_csv, transforms=train_transform)
    val_dataset = ImageDataset(train_csv, transforms=val_transform)

    batch_size = args.batch_size
    epochs = args.epochs
    k = args.folds
    torch.manual_seed(42)
    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labels = train_dataset.get_labels()
    foldperf = {}

    train(train_dataset, val_dataset, args, batch_size, epochs, k, splits, labels, foldperf, device)