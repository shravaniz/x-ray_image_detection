import argparse
from pathlib import Path
import subprocess
import sys


TRANSFORMER_MODELS = [
    ('vit_b_16', 224),
    ('swin_s', 224),
    ('swin_v2_s', 224),
]


def main():
    parser = argparse.ArgumentParser(description='Train three transformer models and evaluate their soft-voting ensemble.')
    parser.add_argument('--data-dir', type=Path, default=Path('../KneeXrayData/ClsKLData/kneeKL299'))
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    train_dir = args.data_dir / 'train'
    test_dir = args.data_dir / 'test'
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f'Expected train and test folders under {args.data_dir}')

    for model_type, image_size in TRANSFORMER_MODELS:
        print(f'\n{"=" * 16} Training {model_type} at {image_size}x{image_size} {"=" * 16}')
        command = [
            sys.executable,
            str(Path(__file__).with_name('main.py')),
            '--model_type', model_type,
            '--image_size', str(image_size),
            '--data-dir', str(train_dir),
            '--device', args.device,
            '--batch-size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--folds', str(args.folds),
            '--workers', str(args.workers),
            '--class-weighting', 'none',
            '--skip-test-evaluation',
        ]
        subprocess.run(command, check=True)

    members = [f'{model_type}:{image_size}' for model_type, image_size in TRANSFORMER_MODELS]
    print(f'\n{"=" * 16} Evaluating {len(TRANSFORMER_MODELS)}-transformer ensemble {"=" * 16}')
    evaluate_command = [
        sys.executable,
        str(Path(__file__).with_name('evaluate_ensemble.py')),
        '--members', *members,
        '--data-dir', str(test_dir),
        '--device', args.device,
        '--batch-size', str(args.eval_batch_size),
        '--workers', str(args.workers),
    ]
    subprocess.run(evaluate_command, check=True)


if __name__ == '__main__':
    main()
