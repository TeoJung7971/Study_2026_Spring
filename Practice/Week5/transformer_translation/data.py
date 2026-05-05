import re
from collections import Counter
from pathlib import Path

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


DATASET_NAME = "bentrevett/multi30k"
HF_SPLIT_NAMES = {"train": "train", "valid": "validation", "test": "test"}
SPECIALS = ["<s>", "</s>", "<blank>", "<unk>"]


class Multi30kTranslationDataset(Dataset):
    """
    Hugging Face Multi30k split을 PyTorch Dataset 형태로 감싸는 클래스

    - hf_split: datasets.load_dataset으로 읽은 단일 split
    - src_lang: source language key
    - tgt_lang: target language key

    Reference: Week3/dataset/dataset.py custom Dataset pattern.
    """

    def __init__(self, hf_split, src_lang="de", tgt_lang="en"):
        self.data = hf_split
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return self._get_text(example, self.src_lang), self._get_text(
            example, self.tgt_lang
        )

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    @staticmethod
    def _get_text(example, lang):
        if lang in example:
            return example[lang]
        if "translation" in example and lang in example["translation"]:
            return example["translation"][lang]
        raise KeyError(f"Language '{lang}' is not in this Multi30k example.")


class RegexTokenizer:
    def __call__(self, text):
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def load_tokenizers(src_lang="de", tgt_lang="en"):
    """
    변형: Annotated Transformer

    Source/target 문장을 token 단위로 나누기 위한 tokenizer를 생성한다.

    - src_lang: source language code
    - tgt_lang: target language code
    """
    def load_one(lang):
        try:
            import spacy
        except ImportError:
            return RegexTokenizer()

        model_name = {"de": "de_core_news_sm", "en": "en_core_web_sm"}.get(lang)
        if model_name is not None:
            try:
                return spacy.load(model_name)
            except OSError:
                pass
        return spacy.blank(lang)

    return load_one(src_lang), load_one(tgt_lang)


def tokenize(text, tokenizer):
    if hasattr(tokenizer, "tokenizer"):
        return [tok.text for tok in tokenizer.tokenizer(text)]
    return tokenizer(text)


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


class SimpleVocab:
    """
    Token과 index 사이의 mapping을 관리하는 단순 vocabulary 클래스

    - tokens: vocabulary에 등록할 token sequence
    - default_token: OOV token을 대체할 기본 token
    """

    def __init__(self, tokens, default_token="<unk>"):
        self.itos = list(tokens)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.default_index = self.stoi[default_token]

    def __call__(self, tokens):
        return [self.stoi.get(token, self.default_index) for token in tokens]

    def __getitem__(self, token):
        return self.stoi[token]

    def __len__(self):
        return len(self.itos)

    def get_stoi(self):
        return self.stoi

    def get_itos(self):
        return self.itos

    def set_default_index(self, index):
        self.default_index = index

    def to_state(self):
        return {"itos": self.itos, "default_index": self.default_index}

    @classmethod
    def from_state(cls, state):
        vocab = cls(state["itos"])
        vocab.default_index = state["default_index"]
        return vocab


def load_raw_multi30k_splits(dataset_name=DATASET_NAME, cache_dir=".data"):
    """
    Hugging Face Hub에서 Multi30k raw split을 load한다.

    - dataset_name: Hugging Face dataset repository name
    - cache_dir: dataset cache directory
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the Hugging Face 'datasets' package to load "
            f"{dataset_name}."
        ) from exc

    return load_dataset(dataset_name, cache_dir=cache_dir)


def read_multi30k_split(
    split,
    cache_dir=".data",
    src_lang="de",
    tgt_lang="en",
    dataset_name=DATASET_NAME,
):
    raw_splits = load_raw_multi30k_splits(dataset_name, cache_dir)
    return make_multi30k_dataset(raw_splits, split, src_lang, tgt_lang)


def make_multi30k_dataset(raw_splits, split, src_lang="de", tgt_lang="en"):
    hf_split_name = HF_SPLIT_NAMES.get(split, split)
    return Multi30kTranslationDataset(raw_splits[hf_split_name], src_lang, tgt_lang)


def multi30k_splits(
    cache_dir=".data",
    src_lang="de",
    tgt_lang="en",
    dataset_name=DATASET_NAME,
):
    """
    Hugging Face Multi30k의 train/valid/test split을 Custom Dataset으로 반환한다.

    - cache_dir: dataset cache directory
    - src_lang: source language key
    - tgt_lang: target language key
    - dataset_name: Hugging Face dataset repository name
    """
    raw_splits = load_raw_multi30k_splits(dataset_name, cache_dir)
    return (
        make_multi30k_dataset(raw_splits, "train", src_lang, tgt_lang),
        make_multi30k_dataset(raw_splits, "valid", src_lang, tgt_lang),
        make_multi30k_dataset(raw_splits, "test", src_lang, tgt_lang),
    )


def build_simple_vocab(token_iter, specials, min_freq=2):
    """
    Token iterator로부터 frequency 기반 vocabulary를 생성한다.

    - token_iter: tokenized sentence iterator
    - specials: special token list
    - min_freq: vocabulary에 포함할 최소 등장 빈도
    """
    counter = Counter()
    ordered_tokens = []

    for tokens in token_iter:
        for token in tokens:
            if token not in counter:
                ordered_tokens.append(token)
            counter[token] += 1

    vocab_tokens = list(specials)
    vocab_tokens.extend(
        token
        for token in ordered_tokens
        if counter[token] >= min_freq and token not in specials
    )
    return SimpleVocab(vocab_tokens)


def build_vocabulary(
    spacy_src,
    spacy_tgt,
    cache_dir=".data",
    src_lang="de",
    tgt_lang="en",
    dataset_name=DATASET_NAME,
    min_freq=2,
):
    """
    변형: Annotated Transformer

    Multi30k train split으로 source/target vocabulary를 구성한다.

    - spacy_src: source tokenizer
    - spacy_tgt: target tokenizer
    - cache_dir: dataset cache directory
    - src_lang: source language key
    - tgt_lang: target language key
    - min_freq: vocabulary에 포함할 최소 등장 빈도
    """
    def tokenize_src(text):
        return tokenize(text, spacy_src)

    def tokenize_tgt(text):
        return tokenize(text, spacy_tgt)

    train, _, _ = multi30k_splits(cache_dir, src_lang, tgt_lang, dataset_name)

    print(f"Building {src_lang} vocabulary ...")
    vocab_src = build_simple_vocab(
        yield_tokens(train, tokenize_src, index=0),
        min_freq=min_freq,
        specials=SPECIALS,
    )

    print(f"Building {tgt_lang} vocabulary ...")
    vocab_tgt = build_simple_vocab(
        yield_tokens(train, tokenize_tgt, index=1),
        min_freq=min_freq,
        specials=SPECIALS,
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])
    return vocab_src, vocab_tgt


def load_vocab(
    spacy_src,
    spacy_tgt,
    vocab_path="vocab_multi30k_simple.pt",
    cache_dir=".data",
    src_lang="de",
    tgt_lang="en",
    dataset_name=DATASET_NAME,
    min_freq=2,
):
    """
    변형: Annotated Transformer

    저장된 vocabulary를 load하거나 없으면 새로 build하여 저장한다.

    - spacy_src: source tokenizer
    - spacy_tgt: target tokenizer
    - vocab_path: vocabulary checkpoint path
    - cache_dir: dataset cache directory
    - min_freq: vocabulary에 포함할 최소 등장 빈도
    """
    vocab_path = Path(vocab_path)
    if not vocab_path.exists():
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_src, vocab_tgt = build_vocabulary(
            spacy_src,
            spacy_tgt,
            cache_dir,
            src_lang,
            tgt_lang,
            dataset_name,
            min_freq,
        )
        torch.save(
            {
                "src": vocab_src.to_state(),
                "tgt": vocab_tgt.to_state(),
            },
            vocab_path,
        )
    else:
        state = torch.load(vocab_path, map_location="cpu", weights_only=False)
        vocab_src = SimpleVocab.from_state(state["src"])
        vocab_tgt = SimpleVocab.from_state(state["tgt"])

    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def numericalize(text, pipeline, vocab, max_padding, bos_id, eos_id, device):
    """
    문장을 token id sequence로 변환하고 BOS/EOS token을 추가한다.

    - text: raw sentence
    - pipeline: tokenizer function
    - vocab: token-to-index vocabulary
    - max_padding: maximum sequence length
    - bos_id: begin-of-sentence token id
    - eos_id: end-of-sentence token id
    """
    token_ids = vocab(pipeline(text))[: max_padding - 2]
    return torch.tensor(
        [bos_id, *token_ids, eos_id],
        dtype=torch.int64,
        device=device,
    )


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
):
    """
    변형: Annotated Transformer

    Translation mini-batch를 padding된 source/target tensor로 변환한다.

    - batch: sentence pair batch
    - src_pipeline: source tokenizer
    - tgt_pipeline: target tokenizer
    - src_vocab: source vocabulary
    - tgt_vocab: target vocabulary
    - device: tensor device
    - max_padding: maximum sequence length
    """
    bos_id = src_vocab["<s>"]
    eos_id = src_vocab["</s>"]
    src_pad_id = src_vocab["<blank>"]
    tgt_pad_id = tgt_vocab["<blank>"]
    src_list, tgt_list = [], []

    for _src, _tgt in batch:
        processed_src = numericalize(
            _src, src_pipeline, src_vocab, max_padding, bos_id, eos_id, device
        )
        processed_tgt = numericalize(
            _tgt,
            tgt_pipeline,
            tgt_vocab,
            max_padding,
            tgt_vocab["<s>"],
            tgt_vocab["</s>"],
            device,
        )
        src_list.append(
            pad(
                processed_src,
                (0, max_padding - len(processed_src)),
                value=src_pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=tgt_pad_id,
            )
        )

    return torch.stack(src_list), torch.stack(tgt_list)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_src,
    spacy_tgt,
    batch_size=32,
    eval_batch_size=None,
    max_padding=128,
    is_distributed=False,
    num_workers=0,
    cache_dir=".data",
    src_lang="de",
    tgt_lang="en",
    dataset_name=DATASET_NAME,
    include_test=False,
):
    """
    변형: Annotated Transformer

    Multi30k Custom Dataset을 train/valid/test DataLoader로 구성한다.

    - device: tensor device
    - vocab_src: source vocabulary
    - vocab_tgt: target vocabulary
    - spacy_src: source tokenizer
    - spacy_tgt: target tokenizer
    - batch_size: training batch size
    - eval_batch_size: validation/test batch size
    - max_padding: maximum sequence length
    - include_test: test DataLoader 반환 여부
    """
    def tokenize_src(text):
        return tokenize(text, spacy_src)

    def tokenize_tgt(text):
        return tokenize(text, spacy_tgt)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_src,
            tokenize_tgt,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
        )

    if eval_batch_size is None:
        eval_batch_size = batch_size

    train_data, valid_data, test_data = multi30k_splits(
        cache_dir, src_lang, tgt_lang, dataset_name
    )
    train_sampler = DistributedSampler(train_data) if is_distributed else None
    valid_sampler = DistributedSampler(valid_data) if is_distributed else None

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=valid_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    if not include_test:
        return train_dataloader, valid_dataloader

    test_dataloader = DataLoader(
        test_data,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return train_dataloader, valid_dataloader, test_dataloader
