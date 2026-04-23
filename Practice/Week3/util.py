import os
import random
import re
from collections import Counter

import numpy as np
import torch


def resolve_path(base_dir, path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def resolve_device(device_arg):
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def set_seed(seed):
    # Default Seed Val = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text):
    '''
    IMDB raw text를 간단한 정규화 형태로 수정

    - HTML line break -> ' '(blank space)
    - Decapitalization
    - Delete Redundant ' ' 
    '''
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_text(text):
    '''
    정규화된 Text를 Token 단위로 분리
    '''
    return re.findall(r"[a-z0-9']+|[^\w\s]", clean_text(text))


class Vocabulary:
    '''
    Token과 index 사이의 매핑을 관리

    - itos: index to string
    - stoi: string to index 
    - Save PAD(Padding) UNK(Unknown) special token
    '''
    def __init__(self, tokens):
        self.itos = list(tokens)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.pad_idx = self.stoi[self.pad_token]
        self.unk_idx = self.stoi[self.unk_token]

    def __len__(self):
        # Vocabulary Size -> Embedding layer Dimension
        return len(self.itos)

    def encode(self, text, max_len):
        # OOV Token -> UNK index
        # Sequence < max_len
        token_ids = [self.stoi.get(token, self.unk_idx) for token in tokenize_text(text)]
        token_ids = token_ids[:max_len]

        if not token_ids:
            # 빈 Sequence는 RNN 입력이 불가능, 최소 한 개의 UNK token 존재해야 함
            token_ids = [self.unk_idx]
        return token_ids

    def state_dict(self):
        return {'itos': self.itos}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state['itos'])


def build_vocab(texts, vocab_size, min_freq):
    '''
   Training text 집합으로부터 Word level Vocabulary를 구축

    - Count Token frequency
    - Remove Token | Token_freq < min_freq
    - Control Vocabulary Size
    '''
    counter = Counter()
    for text in texts:
        counter.update(tokenize_text(text))

    sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    tokens = ['<pad>', '<unk>']
    for token, freq in sorted_tokens:
        if freq < min_freq:
            continue
        tokens.append(token)
        if len(tokens) >= vocab_size:
            break
    return Vocabulary(tokens)


class GRUCollator:
    '''
    Variable length text batch -> GRU 입력 형식

    - Sentence -> Vocabulary index sequence
    - Padding to match max_len
    - Return: lengths & attention_mask
    '''
    def __init__(self, vocab, max_len):
        self.vocab = vocab
        self.max_len = max_len

    def __call__(self, batch):
        texts, labels = zip(*batch)
        sequences = [self.vocab.encode(text, self.max_len) for text in texts]
        lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
        max_batch_len = int(lengths.max().item())

        input_ids = torch.full(
            (len(sequences), max_batch_len),
            self.vocab.pad_idx,
            dtype=torch.long,
        )

        attention_mask = torch.zeros((len(sequences), max_batch_len), dtype=torch.long)

        for idx, sequence in enumerate(sequences):
            seq_len = len(sequence)
            input_ids[idx, :seq_len] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[idx, :seq_len] = 1

        # 반환 키 이름을 Transformer 인터페이스와 유사하게 맞춤: train loop 재사용성
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'lengths': lengths,
            'labels': torch.tensor(labels, dtype=torch.long),
        }


class HFTextCollator:
    '''
    Hugging Face Tokenizer를 이용, Batch encoding 수행

    - Padding
    - Truncation
    - Tensor Transformation
    - Add Label tensor
    '''
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        texts, labels = zip(*batch)
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
        )

        encoded['labels'] = torch.tensor(labels, dtype=torch.long)
        return encoded


def build_split_indices(size, valid_ratio, seed, train_indices=None, valid_indices=None):
    if train_indices is not None:
        return list(train_indices), list(valid_indices or [])

    if valid_ratio <= 0:
        return list(range(size)), []

    valid_size = int(size * valid_ratio)
    if valid_size <= 0 or valid_size >= size:
        return list(range(size)), []

    # 별도 Generator를 사용: 전역 Random state와 분리된 재현 가능한 split 생성
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(size, generator=generator).tolist()
    valid_indices = permutation[:valid_size]
    train_indices = permutation[valid_size:]
    return train_indices, valid_indices


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if hasattr(value, 'to') else value
        for key, value in batch.items()
    }


def get_tokenizer(name):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, use_fast=True)
