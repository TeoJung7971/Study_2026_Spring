import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from dataset.dataset import IMDBDataset
from model import build_model
from test import test_loop
from train import train_loop
from util import (
    GRUCollator,
    HFTextCollator,
    Vocabulary,
    build_split_indices,
    build_vocab,
    get_tokenizer,
    resolve_device,
    resolve_path,
    set_seed,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Week3 IMDB sentiment classification')

    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--model', choices=['gru', 'transformer_encoder', 'roberta'])

    parser.add_argument('--data-dir', type=str, default='dataset/imdb')
    parser.add_argument('--save-dir', type=str, default='result')
    parser.add_argument('--log-dir', type=str, default='run')
    parser.add_argument('--tag', type=str, default='imdb') # test case 분류
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--eval-batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--grad-clip', type=float, default=None)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--max-len', type=int, default=256)

    parser.add_argument('--vocab-size', type=int, default=25000)
    parser.add_argument('--min-freq', type=int, default=2)
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--encoder-tokenizer-name', type=str, default='bert-base-uncased')
    parser.add_argument('--encoder-hidden-size', type=int, default=256)
    parser.add_argument('--encoder-layers', type=int, default=4)
    parser.add_argument('--encoder-heads', type=int, default=4)
    parser.add_argument('--encoder-intermediate-size', type=int, default=512)

    parser.add_argument('--roberta-name', type=str, default='FacebookAI/roberta-base')

    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)

    return parser


def apply_default_hparams(args):
    # Arg 전달하지 않을 경우 대체값
    if args.batch_size is None:
        args.batch_size = 8

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    if args.epochs is None:
        args.epochs = 5

    if args.lr is None:
        args.lr = 1e-4

    if args.weight_decay is None:
        args.weight_decay = 0.01

    if args.grad_clip is None:
        args.grad_clip = 1.0

    return args


def restore_args_from_checkpoint(args, saved_args):
    '''
    - Resume에 필요한 Argument Checkpoint에서 복구
    - Optimizer는 제외 (필요시 따로 지정)
    '''
    fields = [
        'model',
        'max_len',
        'valid_ratio',
        'vocab_size',
        'min_freq',
        'embed_dim',
        'hidden_dim',
        'num_layers',
        'dropout',
        'encoder_tokenizer_name',
        'encoder_hidden_size',
        'encoder_layers',
        'encoder_heads',
        'encoder_intermediate_size',
        'roberta_name',
    ]

    for field in fields:
        if field in saved_args:
            setattr(args, field, saved_args[field])

    return args


def prepare_dataloaders(args, extra, training=True):
    '''
    Model 종류에 따른 Text input pipeline 구성

    - GRU: Training subset -> Vocabulary
    - Transformers: Pretrained Tokenizer -> Encoding
    '''
    train_dataset = IMDBDataset(args.data_dir, split='train')
    test_dataset = IMDBDataset(args.data_dir, split='test')

    train_indices, valid_indices = build_split_indices(
        len(train_dataset),
        args.valid_ratio,
        args.seed,
        extra.get('train_indices'),
        extra.get('valid_indices'),
    )

    extra['train_indices'] = train_indices
    extra['valid_indices'] = valid_indices

    vocab = None
    tokenizer = None

    if args.model == 'gru':
        # GRU는 Discrete token id를 직접 다루므로 Training subset 기준 Vocabulary 필요
        if extra.get('vocab') is None:
            train_texts = [train_dataset.texts[idx] for idx in train_indices]
            vocab = build_vocab(train_texts, args.vocab_size, args.min_freq)
            extra['vocab'] = vocab.state_dict()
        else:
            # Resume 또는 Test에서는 저장된 Vocabulary를 그대로 사용해야 함
            vocab = Vocabulary.from_state_dict(extra['vocab'])
        collate_fn = GRUCollator(vocab, args.max_len)
    else:
        tokenizer_name = args.encoder_tokenizer_name if args.model == 'transformer_encoder' else args.roberta_name
        tokenizer = get_tokenizer(tokenizer_name)
        collate_fn = HFTextCollator(tokenizer, args.max_len)

    # CUDA를 사용할 경우, Memory 효율성 증가*?
    pin_memory = args.device != 'cpu' and torch.cuda.is_available()

    train_loader = None
    valid_loader = None

    if training:
        # Train -> Shuffle
        train_subset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        if valid_indices:
            # Validation -> Fix
            valid_subset = Subset(train_dataset, valid_indices)
            valid_loader = DataLoader(
                valid_subset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, test_loader, tokenizer, vocab, extra


def build_optimizer(args, model):
    # Text classification에서 비교적 안정적으로 쓰이는 AdamW Optimizer를 공통으로 사용한다
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def get_checkpoint_paths(args):
    # Latest와 Best를 분리 저장하여 Resume 편의성과 최종 성능 확인을 함께 보장한다
    latest_path = os.path.join(args.save_dir, f'{args.tag}_{args.model}_latest.pth')
    best_path = os.path.join(args.save_dir, f'{args.tag}_{args.model}_best.pth')
    return latest_path, best_path


def train_main(args):
    '''
    Training Flow

    - Prepare DataLoader 
    - Create Model and Optimizer
    - Repeat through # of Epoch
    - Save Checkpoint
    - Final Test Evaluation 
    '''
    extra = {}
    if args.resume is not None:
        resume_checkpoint = read_checkpoint(args.resume, device=resolve_device(args.device))
        args = restore_args_from_checkpoint(args, resume_checkpoint.get('args', {}))
        extra = resume_checkpoint.get('extra', {})

    args = apply_default_hparams(args)
    device = resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.log_dir, f'{args.tag}_{args.model}'))

    train_loader, valid_loader, test_loader, tokenizer, vocab, extra = prepare_dataloaders(
        args,
        extra,
        training=True,
    )

    model = build_model(
        args,
        vocab=vocab,
        tokenizer=tokenizer,
        device=device,
        load_pretrained=args.resume is None,
    )
    optimizer = build_optimizer(args, model)

    start_epoch = 0
    best_score = 0.0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model, optimizer=optimizer, device=device)
        start_epoch = checkpoint.get('epoch', 0)
        best_score = checkpoint.get('best_score', 0.0)

    latest_path, best_path = get_checkpoint_paths(args)
    monitor_name = 'valid' if valid_loader is not None else 'test'

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_loop(train_loader, model, optimizer, device, args.grad_clip)

        # Validation split이 있으면 Validation accuracy를 기준으로 Best model 선정
        # Validation split이 없으면 Test split을 기준으로 선정
        eval_loader = valid_loader if valid_loader is not None else test_loader
        eval_loss, eval_acc = test_loop(eval_loader, model, device)

        print(
            f'Epoch {epoch + 1}/{args.epochs} | '
            f'train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | '
            f'{monitor_name}_loss={eval_loss:.4f} | {monitor_name}_acc={eval_acc:.4f}'
        )

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar(f'Loss/{monitor_name}', eval_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar(f'Accuracy/{monitor_name}', eval_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Checkpoint <- Weight + Arguments + Data pipeline
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': max(best_score, eval_acc),
            'args': vars(args),
            'extra': extra,
        }
        save_checkpoint(state, latest_path)

        if eval_acc >= best_score:
            best_score = eval_acc
            state['best_score'] = best_score
            save_checkpoint(state, best_path)

    best_checkpoint = load_checkpoint(best_path, model, device=device)
    final_test_loss, final_test_acc = test_loop(test_loader, model, device)
    writer.add_scalar('Loss/test', final_test_loss, args.epochs)
    writer.add_scalar('Accuracy/test', final_test_acc, args.epochs)
    writer.close()
    print(
        f'Best {monitor_name} accuracy: {best_checkpoint.get("best_score", 0.0):.4f} | '
        f'test_loss={final_test_loss:.4f} | test_acc={final_test_acc:.4f}'
    )


def test_main(args):
    '''
    Testing Flow
    '''
    if args.checkpoint is None:
        raise ValueError('Checkpoint path is required in test mode.')

    device = resolve_device(args.device)
    checkpoint = read_checkpoint(args.checkpoint, device=device)
    args = restore_args_from_checkpoint(args, checkpoint.get('args', {}))
    args = apply_default_hparams(args)

    extra = checkpoint.get('extra', {})
    _, _, test_loader, tokenizer, vocab, _ = prepare_dataloaders(args, extra, training=False)
    model = build_model(
        args,
        vocab=vocab,
        tokenizer=tokenizer,
        device=device,
        load_pretrained=False,
    )
    # Test 모드에서는 Pretrained weight를 새로 받지 않고 저장된 Checkpoint weight만 사용
    load_checkpoint(args.checkpoint, model, device=device)

    test_loss, test_acc = test_loop(test_loader, model, device)
    print(f'test_loss={test_loss:.4f} | test_acc={test_acc:.4f}')


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == 'train' and args.model is None:
        parser.error('--model is required in train mode')

    base_dir = os.path.dirname(os.path.abspath(__file__))

    args.data_dir = resolve_path(base_dir, args.data_dir)
    args.save_dir = resolve_path(base_dir, args.save_dir)
    args.log_dir = resolve_path(base_dir, args.log_dir)
    args.resume = resolve_path(base_dir, args.resume)
    args.checkpoint = resolve_path(base_dir, args.checkpoint)

    set_seed(args.seed)

    if args.mode == 'train':
        train_main(args)
    else:
        test_main(args)


if __name__ == '__main__':
    main()
