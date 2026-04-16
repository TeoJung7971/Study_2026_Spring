'''
CIFAR-10 Ablation Study

Data Augmentation / Gradient Clipping / Scheduler 세 독립변수의
ON/OFF 조합(2^3 = 8)에 대해 ResNet18 학습을 순차 실행하고,
완료 후 Best Test Accuracy 결과 테이블을 출력

*Contribution: Claude Code
'''

import subprocess
import sys
import os

COMMON = [
    sys.executable, 'main.py',
    '--dataset', 'cifar10',
    '--model', 'resnet18',
    '--optimizer', 'adam',
    '--lr', '1e-3',
    '--epochs', '20',
    '--batch-size', '64',
    '--seed', '42',
]

EXPERIMENTS = [
    {'tag': 'abl_000', 'augment': False, 'grad_clip': False, 'scheduler': False},
    {'tag': 'abl_001', 'augment': False, 'grad_clip': False, 'scheduler': True},
    {'tag': 'abl_010', 'augment': False, 'grad_clip': True,  'scheduler': False},
    {'tag': 'abl_011', 'augment': False, 'grad_clip': True,  'scheduler': True},
    {'tag': 'abl_100', 'augment': True,  'grad_clip': False, 'scheduler': False},
    {'tag': 'abl_101', 'augment': True,  'grad_clip': False, 'scheduler': True},
    {'tag': 'abl_110', 'augment': True,  'grad_clip': True,  'scheduler': False},
    {'tag': 'abl_111', 'augment': True,  'grad_clip': True,  'scheduler': True},
]


def build_command(exp):
    '''
    Build the CLI command list for a single experiment.

    Parameters
    ----------
    exp : dict
        Experiment config with keys ``tag``, ``augment``, ``grad_clip``, ``scheduler``.

    Returns
    -------
    list of str
    '''
    cmd = COMMON + ['--tag', exp['tag']]
    cmd += ['--scheduler', 'cosine' if exp['scheduler'] else 'none']

    if exp['augment']:
        cmd.append('--augment')
    if exp['grad_clip']:
        cmd += ['--grad-clip', '1.0']

    return cmd


def print_results():
    '''
    Load best checkpoints and print a comparison table.
    '''
    try:
        import torch
    except ImportError:
        print('[WARN] torch not available — skipping result summary')
        return

    print('\n' + '=' * 60)
    print(' Ablation Study Results')
    print('=' * 60)
    print(f'{"Tag":<12} {"Augment":<9} {"GradClip":<10} {"Scheduler":<11} {"Best Acc":>8}')
    print('-' * 60)

    for exp in EXPERIMENTS:
        path = os.path.join('result', f'{exp["tag"]}_best.pth')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            acc = ckpt.get('best_acc', 0.0) * 100
            acc_str = f'{acc:.2f}%'
        else:
            acc_str = 'N/A'

        aug = 'O' if exp['augment'] else 'X'
        gc  = 'O' if exp['grad_clip'] else 'X'
        sch = 'O' if exp['scheduler'] else 'X'
        print(f'{exp["tag"]:<12} {aug:<9} {gc:<10} {sch:<11} {acc_str:>8}')

    print('=' * 60)


def main():
    for i, exp in enumerate(EXPERIMENTS):
        cmd = build_command(exp)
        label = f'[{i + 1}/{len(EXPERIMENTS)}] {exp["tag"]}'

        aug = 'O' if exp['augment'] else 'X'
        gc  = 'O' if exp['grad_clip'] else 'X'
        sch = 'O' if exp['scheduler'] else 'X'

        print(f'\n{"=" * 60}')
        print(f' {label}  (Aug={aug}, GC={gc}, Sch={sch})')
        print(f'{"=" * 60}\n')

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f'[ERROR] {exp["tag"]} failed with return code {result.returncode}')

    print_results()


if __name__ == '__main__':
    main()
