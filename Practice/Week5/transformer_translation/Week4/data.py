import shutil
import subprocess
import sys
import tarfile
import urllib.request
from collections import Counter
from itertools import chain
from os.path import exists
from pathlib import Path

import spacy
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


MULTI30K_URLS = {
    "train": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
    "valid": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
    "test": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz",
}

MULTI30K_PREFIX = {"train": "train", "valid": "val", "test": "test"}


def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "de_core_news_sm"],
            check=True,
        )
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
        )
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


class SimpleVocab:
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


def download_multi30k(root=".data"):
    root = Path(root) / "Multi30k"
    root.mkdir(parents=True, exist_ok=True)

    for split, url in MULTI30K_URLS.items():
        prefix = MULTI30K_PREFIX[split]
        expected = [root / f"{prefix}.de", root / f"{prefix}.en"]
        if all(path.exists() for path in expected):
            continue

        archive_path = root / Path(url).name
        extract_dir = root / f"extract_{split}"

        if not archive_path.exists():
            urllib.request.urlretrieve(url, archive_path)

        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)

        for lang in ("de", "en"):
            matches = list(extract_dir.rglob(f"{prefix}.{lang}"))
            if not matches:
                raise FileNotFoundError(
                    f"Could not find {prefix}.{lang} in {archive_path}"
                )
            shutil.copyfile(matches[0], root / f"{prefix}.{lang}")

    return root


def read_multi30k_split(split, root=".data"):
    root = download_multi30k(root)
    prefix = MULTI30K_PREFIX[split]

    with open(root / f"{prefix}.de", encoding="utf-8") as f:
        src = [line.strip() for line in f]
    with open(root / f"{prefix}.en", encoding="utf-8") as f:
        tgt = [line.strip() for line in f]

    return list(zip(src, tgt))


def multi30k_splits(root=".data"):
    return (
        read_multi30k_split("train", root),
        read_multi30k_split("valid", root),
        read_multi30k_split("test", root),
    )


def build_simple_vocab(token_iter, specials, min_freq=2):
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


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    train, val, test = multi30k_splits()

    print("Building German Vocabulary ...")
    vocab_src = build_simple_vocab(
        yield_tokens(chain(train, val, test), tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    vocab_tgt = build_simple_vocab(
        yield_tokens(chain(train, val, test), tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en, vocab_path="vocab_multi30k_simple.pt"):
    if not exists(vocab_path):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save(
            {
                "src": vocab_src.to_state(),
                "tgt": vocab_tgt.to_state(),
            },
            vocab_path,
        )
    else:
        state = torch.load(vocab_path, map_location="cpu")
        vocab_src = SimpleVocab.from_state(state["src"])
        vocab_tgt = SimpleVocab.from_state(state["tgt"])

    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)
    eos_id = torch.tensor([1], device=device)
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )[:max_padding]
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )[:max_padding]
        src_list.append(
            pad(
                processed_src,
                (0, max_padding - len(processed_src)),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter_map, valid_iter_map, _ = multi30k_splits()

    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader
