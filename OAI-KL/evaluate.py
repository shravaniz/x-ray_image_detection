import argparse
import json
from pathlib import Path

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader

from dataset import ImageDataset
from model import model_return


def load_labeled_images(data_dir):
    image_paths = []
    labels = []
    for label in range(5):
        class_dir = data_dir / str(label)
        if not class_dir.is_dir():
            raise FileNotFoundError(f'Missing class directory: {class_dir}')
        class_images = sorted(class_dir.glob('*.png'))
        image_paths.extend(str(path) for path in class_images)
        labels.extend([label] * len(class_images))
    if not image_paths:
        raise RuntimeError(f'No PNG images found under {data_dir}')
    return pd.DataFrame({'data': image_paths, 'label': labels})


def main():
    parser = argparse.ArgumentParser(description='Evaluate best cross-validation checkpoints on held-out test images.')
    parser.add_argument('-m', '--model-type', required=True)
    parser.add_argument('-i', '--image-size', type=int, default=224)
    parser.add_argument('--data-dir', type=Path, default=Path('../KneeXrayData/ClsKLData/kneeKL224/test'))
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available. Use --device cpu or install a CUDA-enabled PyTorch build.')
    device = torch.device(args.device)
    image_size = (args.image_size, args.image_size)
    checkpoint_dir = Path('./models') / args.model_type / str(image_size)
    checkpoint_paths = sorted(checkpoint_dir.glob('*fold_best.pt'))
    if not checkpoint_paths:
        raise FileNotFoundError(f'No best checkpoints found in {checkpoint_dir}')

    test_csv = load_labeled_images(args.data_dir)
    transform = A.Compose([
        A.Resize(args.image_size, args.image_size, interpolation=cv2.INTER_CUBIC, p=1),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    test_loader = DataLoader(
        ImageDataset(test_csv, transforms=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    print(f'Device: {device}')
    print(f'Test images: {len(test_csv)}')
    print(f'Checkpoints: {[path.name for path in checkpoint_paths]}')
    ensemble_probabilities = np.zeros((len(test_csv), 5), dtype=np.float32)

    for checkpoint_path in checkpoint_paths:
        model = model_return(args).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        probabilities = []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch['image'].to(device))
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        ensemble_probabilities += np.concatenate(probabilities) / len(checkpoint_paths)
        print(f'Evaluated {checkpoint_path.name}')

    targets = test_csv['label'].to_numpy()
    predictions = ensemble_probabilities.argmax(axis=1)
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision_macro': precision_score(targets, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(targets, predictions, average='macro', zero_division=0),
        'f1_macro': f1_score(targets, predictions, average='macro', zero_division=0),
        'qwk': cohen_kappa_score(targets, predictions, weights='quadratic'),
    }
    metrics['dice_macro'] = metrics['f1_macro']
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"Macro F1 / Dice: {metrics['f1_macro']:.4f}")
    print(f"Quadratic weighted kappa: {metrics['qwk']:.4f}")
    report = classification_report(targets, predictions, labels=range(5), digits=4, zero_division=0, output_dict=True)
    print(classification_report(targets, predictions, labels=range(5), digits=4, zero_division=0))

    results_dir = Path('./results') / args.model_type / str(image_size)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / 'test_metrics.json').open('w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    pd.DataFrame(report).transpose().to_csv(results_dir / 'classification_report.csv')
    pd.DataFrame(confusion_matrix(targets, predictions, labels=range(5)), index=range(5), columns=range(5)).to_csv(results_dir / 'confusion_matrix.csv')
    pd.DataFrame({
        'data': test_csv['data'],
        'target': targets,
        'prediction': predictions,
        **{f'prob_{label}': ensemble_probabilities[:, label] for label in range(5)},
    }).to_csv(results_dir / 'predictions.csv', index=False)
    print(f'Results saved to {results_dir}')


if __name__ == '__main__':
    main()
