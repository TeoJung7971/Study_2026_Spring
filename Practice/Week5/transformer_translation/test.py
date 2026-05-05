import math
from collections import Counter

import torch

from data import create_dataloaders
from model import make_model
from train import Batch, greedy_decode


def strip_special_tokens(token_ids, vocab, pad_idx, bos_idx, eos_idx):
    """
    Generated token id sequence에서 special token을 제거하고 token string으로 복원한다.

    - token_ids: generated token id sequence
    - vocab: target vocabulary
    - pad_idx: padding token id
    - bos_idx: begin-of-sentence token id
    - eos_idx: end-of-sentence token id
    """
    tokens = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == eos_idx:
            break
        if token_id in (pad_idx, bos_idx):
            continue
        tokens.append(vocab.get_itos()[token_id])
    return tokens


def ngram_counter(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(predictions, references, max_n=4):
    """
    BLEU4 Score 계산을 목적으로 하는 함수

    - predictions: 예측 Token Sequences
    - references: 정답 Token Sequences
    - max_n: 1-gram부터 n-gram까지 비교
    """
    clipped = [0] * max_n
    total = [0] * max_n
    pred_len = 0
    ref_len = 0

    for pred, ref in zip(predictions, references):
        pred_len += len(pred)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            pred_counts = ngram_counter(pred, n)
            ref_counts = ngram_counter(ref, n)
            clipped[n - 1] += sum(
                min(count, ref_counts[ngram])
                for ngram, count in pred_counts.items()
            )
            total[n - 1] += max(len(pred) - n + 1, 0)

    if pred_len == 0 or ref_len == 0:
        return 0.0

    precisions = []
    for match_count, total_count in zip(clipped, total):
        if total_count == 0:
            precisions.append(0.0)
        elif match_count == 0:
            precisions.append(1.0 / (2 * total_count))
        else:
            precisions.append(match_count / total_count)

    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    return brevity_penalty * math.exp(
        sum(math.log(max(precision, 1e-12)) for precision in precisions) / max_n
    )


@torch.no_grad()
def evaluate_bleu(
    model,
    dataloader,
    vocab_tgt,
    max_len=72,
    pad_idx=2,
    bos_idx=0,
    eos_idx=1,
    max_batches=None,
):
    """
    Test/validation DataLoader에 대해 greedy decoding 후 corpus BLEU4를 계산한다.

    - model: translation model
    - dataloader: evaluation DataLoader
    - vocab_tgt: target vocabulary
    - max_len: maximum decoding length
    - pad_idx: padding token id
    - bos_idx: begin-of-sentence token id
    - eos_idx: end-of-sentence token id
    - max_batches: BLEU 계산에 사용할 최대 batch 수
    """
    model.eval()
    predictions = []
    references = []

    for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        for row in range(src_batch.size(0)):
            src = src_batch[row : row + 1]
            src_mask = (src != pad_idx).unsqueeze(-2)
            model_out = greedy_decode(
                model,
                src,
                src_mask,
                max_len=max_len,
                start_symbol=bos_idx,
                end_symbol=eos_idx,
            )[0]
            predictions.append(
                strip_special_tokens(model_out, vocab_tgt, pad_idx, bos_idx, eos_idx)
            )
            references.append(
                strip_special_tokens(tgt_batch[row], vocab_tgt, pad_idx, bos_idx, eos_idx)
            )

    return corpus_bleu(predictions, references), predictions, references


def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=5,
    pad_idx=2,
    bos_idx=0,
    eos_idx=1,
):
    """
    변형: Annotated Transformer

    Source, reference, model output 예시를 출력하여 inference 결과를 확인한다.

    - valid_dataloader: example을 가져올 DataLoader
    - model: translation model
    - vocab_src: source vocabulary
    - vocab_tgt: target vocabulary
    - n_examples: 출력할 예시 수
    - pad_idx: padding token id
    """
    results = []
    for batch in valid_dataloader:
        rb = Batch(batch[0], batch[1], pad_idx)
        for row in range(rb.src.size(0)):
            if len(results) >= n_examples:
                return results
            src = rb.src[row : row + 1]
            src_mask = rb.src_mask[row : row + 1]
            model_out = greedy_decode(
                model,
                src,
                src_mask,
                max_len=rb.src.size(1),
                start_symbol=bos_idx,
                end_symbol=eos_idx,
            )[0]
            src_tokens = strip_special_tokens(
                rb.src[row], vocab_src, pad_idx, bos_idx, eos_idx
            )
            tgt_tokens = strip_special_tokens(
                batch[1][row], vocab_tgt, pad_idx, bos_idx, eos_idx
            )
            out_tokens = strip_special_tokens(
                model_out, vocab_tgt, pad_idx, bos_idx, eos_idx
            )
            print("\nExample %d ========" % len(results))
            print("Source Text (Input)        : " + " ".join(src_tokens))
            print("Target Text (Ground Truth) : " + " ".join(tgt_tokens))
            print("Model Output               : " + " ".join(out_tokens))
            results.append((src_tokens, tgt_tokens, out_tokens))
    return results


def run_model_example(
    vocab_src,
    vocab_tgt,
    spacy_src,
    spacy_tgt,
    n_examples=5,
    checkpoint_path="multi30k_model_final.pt",
):
    """
    변형: Annotated Transformer

    저장된 checkpoint를 load한 뒤 validation 예시 번역을 출력한다.

    - vocab_src: source vocabulary
    - vocab_tgt: target vocabulary
    - spacy_src: source tokenizer
    - spacy_tgt: target tokenizer
    - n_examples: 출력할 예시 수
    - checkpoint_path: model checkpoint path
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Preparing data ...")
    _, valid_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_src,
        spacy_tgt,
        batch_size=1,
        eval_batch_size=1,
        is_distributed=False,
    )

    print("Loading trained model ...")
    model = make_model(len(vocab_src), len(vocab_tgt), N=6).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)
    )

    print("Checking model outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data
