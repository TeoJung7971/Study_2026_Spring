import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class GRUClassifier(nn.Module):
    '''
    Bidirectional GRU 기반 Sentiment classifier 구현

    - Token -> Embedding
    - Last Hidden state를 이어 Sentence representation으로 사용
    '''
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, input_ids, attention_mask=None, lengths=None, labels=None):
        embedded = self.embedding(input_ids)

        # Padding은 읽지 않도록 함
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Output으로 반환되는 각각의 Hidden State는 사용하지 않음
        # Last Hidden State만 받아 사용
        _, hidden = self.gru(packed) 

        # PyTorch GRU의 Hidden shape는 num_layers * num_directions 축이 합쳐져 있으므로 다시 분리한다
        # Left(Forward) Path / Right(Backward) Path
        hidden = hidden.view(self.gru.num_layers, 2, input_ids.size(0), self.gru.hidden_size)

        # 마지막 Layer의 Forward와 Backward Hidden state를 이어 붙여 분류용 Feature를 만든다
        features = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)
        logits = self.classifier(self.dropout(features))

        # Labels가 주어질 때만 CrossEntropy loss를 계산
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {'loss': loss, 'logits': logits}


def build_transformer_encoder(args, tokenizer):
    '''
    BERT style Transformer Encoder을 Config만으로 새로 생성
    '''
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.encoder_hidden_size,
        num_hidden_layers=args.encoder_layers,
        num_attention_heads=args.encoder_heads,
        intermediate_size=args.encoder_intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=max(args.max_len + 2, 512),
        pad_token_id=tokenizer.pad_token_id,
        num_labels=2,
    )
    return BertForSequenceClassification(config)


def build_roberta(args, load_pretrained=True):
    '''
    RoBERTa based Sequence classification model을 Pretrained Weight를 이용하여 생성
    '''
    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.from_pretrained(
        args.roberta_name,
        num_labels=2,
    )

    if load_pretrained:
        return AutoModelForSequenceClassification.from_pretrained(
            args.roberta_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

    return AutoModelForSequenceClassification.from_config(config)


def build_model(args, vocab=None, tokenizer=None, device='cpu', load_pretrained=True):
    if args.model == 'gru':
        model = GRUClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pad_idx=vocab.pad_idx,
        )
    elif args.model == 'transformer_encoder':
        model = build_transformer_encoder(args, tokenizer)
    elif args.model == 'roberta':
        model = build_roberta(args, load_pretrained=load_pretrained)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    
    return model.to(device)
