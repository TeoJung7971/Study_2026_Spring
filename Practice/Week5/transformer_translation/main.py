import argparse
import os
import random
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from data import (
    DATASET_NAME,
    SimpleVocab,
    create_dataloaders,
    load_tokenizers,
    load_vocab,
)
from model import make_model
from test import check_outputs, evaluate_bleu
from train import (
    Batch,
    DummyOptimizer,
    DummyScheduler,
    LabelSmoothing,
    SimpleLossCompute,
    TrainState,
    rate,
    run_epoch,
)


# Reference: Week3/main.py SummaryWriter and checkpoint flow.
def build_arg_parser():
    """
    학습/평가 실행에 필요한 CLI argument를 정의한다.

    - mode: train 또는 test 실행 모드
    - dataset-name: Hugging Face dataset repository name
    - tag: checkpoint 및 TensorBoard run name
    - device: cuda/cpu 선택
    """
    parser = argparse.ArgumentParser(description="Transformer Multi30k translation")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME)
    parser.add_argument("--data-dir", type=str, default=".data")
    parser.add_argument("--save-dir", type=str, default="result")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--tag", type=str, default="multi30k_transformer")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--accum-iter", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=72)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--src-lang", type=str, default="de")
    parser.add_argument("--tgt-lang", type=str, default="en")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bleu-max-batches", type=int, default=None)
    parser.add_argument("--show-examples", type=int, default=3)
    return parser


def resolve_path(base_dir, path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_checkpoint_paths(args):
    latest_path = os.path.join(args.save_dir, f"{args.tag}_latest.pt")
    best_path = os.path.join(args.save_dir, f"{args.tag}_best.pt")
    return latest_path, best_path


def save_checkpoint(state, filename, retries=5, delay=1.0):
    """
    Windows file lock을 피하기 위해 임시 파일 저장 후 checkpoint를 교체한다.

    - state: 저장할 checkpoint dictionary
    - filename: 최종 checkpoint path
    - retries: 저장 실패 시 재시도 횟수
    - delay: 재시도 간 기본 대기 시간
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    last_error = None

    for attempt in range(retries):
        tmp_filename = f"{filename}.tmp.{os.getpid()}.{time.time_ns()}"
        try:
            torch.save(state, tmp_filename)
            os.replace(tmp_filename, filename)
            return
        except (OSError, RuntimeError) as exc:
            last_error = exc
            if os.path.exists(tmp_filename):
                try:
                    os.remove(tmp_filename)
                except OSError:
                    pass
            if attempt + 1 < retries:
                time.sleep(delay * (attempt + 1))

    raise RuntimeError(
        "Checkpoint save failed. Close any program that is reading this "
        f"checkpoint, then retry: {filename}"
    ) from last_error


def read_checkpoint(filename, device):
    return torch.load(filename, map_location=device, weights_only=False)


def restore_args_from_checkpoint(args, saved_args):
    """
    Checkpoint 평가/재개 시 model과 data pipeline 관련 argument를 복원한다.

    - args: 현재 CLI arguments
    - saved_args: checkpoint에 저장된 arguments
    """
    fields = [
        "dataset_name",
        "src_lang",
        "tgt_lang",
        "max_len",
        "min_freq",
        "label_smoothing",
        "n_layers",
        "d_model",
        "d_ff",
        "heads",
        "dropout",
    ]
    for field in fields:
        if field in saved_args:
            setattr(args, field, saved_args[field])
    return args


def prepare_data(args, device, vocab_state=None):
    """
    Tokenizer, vocabulary, DataLoader를 학습/평가 모드에 맞게 준비한다.

    - args: CLI arguments
    - device: tensor device
    - vocab_state: checkpoint에 저장된 vocabulary state
    """
    tokenizer_src, tokenizer_tgt = load_tokenizers(args.src_lang, args.tgt_lang)
    if vocab_state is None:
        vocab_path = os.path.join(args.save_dir, f"{args.tag}_vocab.pt")
        vocab_src, vocab_tgt = load_vocab(
            tokenizer_src,
            tokenizer_tgt,
            vocab_path=vocab_path,
            cache_dir=args.data_dir,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            dataset_name=args.dataset_name,
            min_freq=args.min_freq,
        )
    else:
        vocab_src = SimpleVocab.from_state(vocab_state["src"])
        vocab_tgt = SimpleVocab.from_state(vocab_state["tgt"])

    train_loader, valid_loader, test_loader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        tokenizer_src,
        tokenizer_tgt,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_padding=args.max_len,
        is_distributed=False,
        num_workers=args.num_workers,
        cache_dir=args.data_dir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        dataset_name=args.dataset_name,
        include_test=True,
    )
    return (
        train_loader,
        valid_loader,
        test_loader,
        vocab_src,
        vocab_tgt,
        tokenizer_src,
        tokenizer_tgt,
    )


def build_translation_model(args, vocab_src, vocab_tgt, device):
    """
    CLI hyperparameter와 vocabulary size를 바탕으로 Transformer 모델을 생성한다.

    - args: model hyperparameter를 포함한 CLI arguments
    - vocab_src: source vocabulary
    - vocab_tgt: target vocabulary
    - device: model device
    """
    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        N=args.n_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        h=args.heads,
        dropout=args.dropout,
    )
    return model.to(device)


def evaluate_loss_acc(model, dataloader, criterion, pad_idx):
    """
    Teacher forcing 기반 validation/test loss와 token accuracy를 계산한다.

    - model: translation model
    - dataloader: evaluation DataLoader
    - criterion: LabelSmoothing loss
    - pad_idx: padding token id
    """
    model.eval()
    metrics, _ = run_epoch(
        (Batch(src, tgt, pad_idx) for src, tgt in dataloader),
        model,
        SimpleLossCompute(model.generator, criterion, padding_idx=pad_idx),
        DummyOptimizer(),
        DummyScheduler(),
        mode="eval",
    )
    return metrics


def train_main(args):
    """
    Multi30k translation 모델의 전체 학습, 검증, checkpoint, TensorBoard 기록을 수행한다.

    - args: training CLI arguments
    """
    device = torch.device(resolve_device(args.device))
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    resume_checkpoint = None
    vocab_state = None
    if args.resume is not None:
        resume_checkpoint = read_checkpoint(args.resume, device)
        args = restore_args_from_checkpoint(args, resume_checkpoint.get("args", {}))
        vocab_state = resume_checkpoint.get("vocab")

    (
        train_loader,
        valid_loader,
        test_loader,
        vocab_src,
        vocab_tgt,
        _,
        _,
    ) = prepare_data(args, device, vocab_state)

    pad_idx = vocab_tgt["<blank>"]
    bos_idx = vocab_tgt["<s>"]
    eos_idx = vocab_tgt["</s>"]
    model = build_translation_model(args, vocab_src, vocab_tgt, device)
    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=args.label_smoothing
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, args.d_model, factor=1.0, warmup=args.warmup
        ),
    )

    start_epoch = 0
    best_bleu = 0.0
    train_state = TrainState()
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if resume_checkpoint.get("scheduler_state_dict") is not None:
            lr_scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        start_epoch = resume_checkpoint.get("epoch", 0)
        best_bleu = resume_checkpoint.get("best_bleu", 0.0)
        train_state = TrainState(**resume_checkpoint.get("train_state", {}))

    writer = SummaryWriter(os.path.join(args.log_dir, args.tag))
    latest_path, best_path = get_checkpoint_paths(args)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{args.epochs} Training ====")
        train_metrics, train_state = run_epoch(
            (Batch(src, tgt, pad_idx) for src, tgt in train_loader),
            model,
            SimpleLossCompute(model.generator, criterion, padding_idx=pad_idx),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=args.accum_iter,
            train_state=train_state,
            grad_clip=args.grad_clip,
        )

        valid_metrics = evaluate_loss_acc(model, valid_loader, criterion, pad_idx)
        valid_bleu, _, _ = evaluate_bleu(
            model,
            valid_loader,
            vocab_tgt,
            max_len=args.max_len,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            max_batches=args.bleu_max_batches,
        )

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['acc']:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | "
            f"valid_acc={valid_metrics['acc']:.4f} | "
            f"valid_bleu4={valid_bleu:.4f} | lr={lr:.6e}"
        )

        writer.add_scalar("Loss/train", train_metrics["loss"], epoch + 1)
        writer.add_scalar("Loss/valid", valid_metrics["loss"], epoch + 1)
        writer.add_scalar("Accuracy/train", train_metrics["acc"], epoch + 1)
        writer.add_scalar("Accuracy/valid", valid_metrics["acc"], epoch + 1)
        writer.add_scalar("BLEU4/valid", valid_bleu, epoch + 1)
        writer.add_scalar("LR", lr, epoch + 1)

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_bleu": max(best_bleu, valid_bleu),
            "train_state": vars(train_state),
            "args": vars(args),
            "vocab": {
                "src": vocab_src.to_state(),
                "tgt": vocab_tgt.to_state(),
            },
        }
        save_checkpoint(state, latest_path)
        if valid_bleu >= best_bleu:
            best_bleu = valid_bleu
            state["best_bleu"] = best_bleu
            save_checkpoint(state, best_path)

    best_checkpoint = read_checkpoint(best_path, device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate_loss_acc(model, test_loader, criterion, pad_idx)
    test_bleu, _, _ = evaluate_bleu(
        model,
        test_loader,
        vocab_tgt,
        max_len=args.max_len,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        max_batches=args.bleu_max_batches,
    )
    writer.add_scalar("Loss/test", test_metrics["loss"], args.epochs)
    writer.add_scalar("Accuracy/test", test_metrics["acc"], args.epochs)
    writer.add_scalar("BLEU4/test", test_bleu, args.epochs)
    writer.close()

    print(
        f"Best valid BLEU-4: {best_checkpoint.get('best_bleu', 0.0):.4f} | "
        f"test_loss={test_metrics['loss']:.4f} | "
        f"test_acc={test_metrics['acc']:.4f} | test_bleu4={test_bleu:.4f}"
    )
    if args.show_examples > 0:
        check_outputs(
            test_loader,
            model,
            vocab_src,
            vocab_tgt,
            n_examples=args.show_examples,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )


def test_main(args):
    """
    저장된 checkpoint를 load하여 test loss, token accuracy, BLEU4를 계산한다.

    - args: test CLI arguments
    """
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in test mode.")

    device = torch.device(resolve_device(args.device))
    checkpoint = read_checkpoint(args.checkpoint, device)
    args = restore_args_from_checkpoint(args, checkpoint.get("args", {}))
    (
        _,
        _,
        test_loader,
        vocab_src,
        vocab_tgt,
        _,
        _,
    ) = prepare_data(args, device, checkpoint.get("vocab"))

    pad_idx = vocab_tgt["<blank>"]
    bos_idx = vocab_tgt["<s>"]
    eos_idx = vocab_tgt["</s>"]
    model = build_translation_model(args, vocab_src, vocab_tgt, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=args.label_smoothing
    ).to(device)

    test_metrics = evaluate_loss_acc(model, test_loader, criterion, pad_idx)
    test_bleu, _, _ = evaluate_bleu(
        model,
        test_loader,
        vocab_tgt,
        max_len=args.max_len,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        max_batches=args.bleu_max_batches,
    )
    print(
        f"test_loss={test_metrics['loss']:.4f} | "
        f"test_acc={test_metrics['acc']:.4f} | test_bleu4={test_bleu:.4f}"
    )
    if args.show_examples > 0:
        check_outputs(
            test_loader,
            model,
            vocab_src,
            vocab_tgt,
            n_examples=args.show_examples,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )


def main():
    """
    CLI argument를 해석하고 train/test entrypoint로 분기한다.
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    args.data_dir = resolve_path(base_dir, args.data_dir)
    args.save_dir = resolve_path(base_dir, args.save_dir)
    args.log_dir = resolve_path(base_dir, args.log_dir)
    args.resume = resolve_path(base_dir, args.resume)
    args.checkpoint = resolve_path(base_dir, args.checkpoint)

    set_seed(args.seed)

    if args.mode == "train":
        train_main(args)
    else:
        test_main(args)


if __name__ == "__main__":
    main()
