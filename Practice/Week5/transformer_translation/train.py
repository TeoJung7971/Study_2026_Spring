import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from model import subsequent_mask

# Dummy는 Evaluation 중 사용

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class DummyScheduler:
    def step(self):
        pass

# ----


class Batch:
    """
    변형: Annotated Transformer

    Encoder/decoder 입력과 padding/future-position mask를 함께 보관한다.

    - src: source token id tensor
    - tgt: target token id tensor
    - pad: padding token id
    """

    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Hide padding tokens and future target positions."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


@dataclass
class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=None,
    grad_clip=None,
):
    """
    변형: Annotated Transformer

    Teacher forcing으로 한 epoch을 수행하고 loss/token accuracy를 반환한다.

    - data_iter: Batch iterator
    - model: translation model
    - loss_compute: loss 계산 객체
    - optimizer: optimizer
    - scheduler: learning rate scheduler
    - mode: train/eval mode
    - accum_iter: gradient accumulation interval
    - grad_clip: gradient clipping threshold
    """
    if train_state is None:
        train_state = TrainState()

    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct = 0
    tokens = 0
    n_accum = 0
    pending_batches = 0

    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node, correct = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += int(batch.ntokens.item())
            pending_batches += 1
            if (i + 1) % accum_iter == 0:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
                pending_batches = 0
                scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        total_correct += correct
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node

    if (mode == "train" or mode == "train+log") and pending_batches > 0:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_state.accum_step += 1
        scheduler.step()

    metrics = {
        "loss": (total_loss / total_tokens).item(),
        "acc": (total_correct / total_tokens).item(),
    }
    return metrics, train_state


def rate(step, model_size, factor, warmup):
    """
    추가: Annotated Transformer

    Transformer warmup learning rate schedule 값을 계산한다.

    - step: current optimization step
    - model_size: Transformer hidden dimension
    - factor: learning rate scale factor
    - warmup: warmup step count
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    """
    추가: Annotated Transformer

    정답 분포를 smoothing하여 과도하게 확신하는 출력을 완화한다.

    - size: target vocabulary size
    - padding_idx: padding token id
    - smoothing: smoothing ratio
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    """
    변형: Annotated Transformer

    Generator와 criterion을 적용하고 loss/token accuracy를 함께 계산한다.

    - generator: model output을 vocabulary log-probability로 변환하는 module
    - criterion: loss function
    - padding_idx: accuracy 계산에서 제외할 padding token id
    """

    def __init__(self, generator, criterion, padding_idx=None):
        self.generator = generator
        self.criterion = criterion
        self.padding_idx = (
            padding_idx
            if padding_idx is not None
            else getattr(criterion, "padding_idx", None)
        )

    def __call__(self, x, y, norm):
        x = self.generator(x)
        y_flat = y.contiguous().view(-1)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y_flat
            )
            / norm
        )
        pred = x.argmax(dim=-1).contiguous().view(-1)
        if self.padding_idx is None:
            correct = (pred == y_flat).sum()
        else:
            non_pad = y_flat != self.padding_idx
            correct = ((pred == y_flat) & non_pad).sum()
        return sloss.data * norm, sloss, correct


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol=None):
    """
    변형: Annotated Transformer

    이전 예측 token을 다시 decoder 입력으로 넣어 문장을 순차적으로 생성한다.

    - model: translation model
    - src: source token id tensor
    - src_mask: source padding mask
    - max_len: maximum decoding length
    - start_symbol: BOS token id
    - end_symbol: EOS token id
    """
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        if end_symbol is not None and next_word.item() == end_symbol:
            break
    return ys
