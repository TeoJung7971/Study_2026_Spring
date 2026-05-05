import altair as alt
import pandas as pd
import torch

from model import PositionalEncoding, subsequent_mask
from test import run_model_example
from train import LabelSmoothing

def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    """
    추가: Annotated Transformer

    Label smoothing이 confidence가 높은 예측에 주는 penalty 변화를 시각화한다.
    """
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(x="Steps", y="Loss")
    )


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    """
    추가: Annotated Transformer

    Attention matrix를 Altair heatmap 입력용 DataFrame으로 변환한다.

    - m: attention matrix
    - max_row: 시각화할 최대 row 수
    - max_col: 시각화할 최대 column 수
    - row_tokens: row token labels
    - col_tokens: column token labels
    """
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(max_row)
            for c in range(max_col)
        ],
        columns=["row", "column", "value", "row_token", "column_token"],
    )


def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    """
    추가: Annotated Transformer

    단일 layer/head의 attention weight를 heatmap으로 시각화한다.

    - attn: attention tensor
    - layer: layer index
    - head: attention head index
    - row_tokens: query token labels
    - col_tokens: key token labels
    - max_dim: 시각화할 최대 token 수
    """
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("column_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "column_token"],
        )
        .properties(height=400, width=400, title=f"Layer {layer + 1} Head {head}")
        .interactive()
    )


def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    """
    변형: Annotated Transformer

    지정한 Transformer layer의 여러 head attention map을 묶어 표시한다.

    - model: translation model
    - layer: layer index
    - getter_fn: attention tensor getter function
    - ntokens: 시각화할 token 수
    - row_tokens: query token labels
    - col_tokens: key token labels
    """
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            layer,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0] | charts[2] | charts[4] | charts[6]
    ).properties(title="Layer %d" % (layer + 1))


def viz_encoder_self(vocab_src, vocab_tgt, spacy_de, spacy_en):
    """
    변형: Annotated Transformer

    학습된 모델의 encoder self-attention을 예시 문장 기준으로 시각화한다.

    - vocab_src: source vocabulary
    - vocab_tgt: target vocabulary
    - spacy_de: source tokenizer
    - spacy_en: target tokenizer
    """
    model, example_data = run_model_example(
        vocab_src, vocab_tgt, spacy_de, spacy_en, n_examples=1
    )
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_encoder,
            len(example[1]),
            example[1],
            example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(layer_viz[0], layer_viz[1], layer_viz[2])


def viz_decoder_self(vocab_src, vocab_tgt, spacy_de, spacy_en):
    """
    변형: Annotated Transformer

    학습된 모델의 decoder masked self-attention을 예시 문장 기준으로 시각화한다.

    - vocab_src: source vocabulary
    - vocab_tgt: target vocabulary
    - spacy_de: source tokenizer
    - spacy_en: target tokenizer
    """
    model, example_data = run_model_example(
        vocab_src, vocab_tgt, spacy_de, spacy_en, n_examples=1
    )
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[2]),
            example[2],
            example[2],
        )
        for layer in range(6)
    ]
    return alt.hconcat(layer_viz[0], layer_viz[1], layer_viz[2])


def viz_decoder_src(vocab_src, vocab_tgt, spacy_de, spacy_en):
    """
    변형: Annotated Transformer

    학습된 모델의 encoder-decoder attention을 예시 문장 기준으로 시각화한다.

    - vocab_src: source vocabulary
    - vocab_tgt: target vocabulary
    - spacy_de: source tokenizer
    - spacy_en: target tokenizer
    """
    model, example_data = run_model_example(
        vocab_src, vocab_tgt, spacy_de, spacy_en, n_examples=1
    )
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            len(example[1]),
            example[2],
            example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(layer_viz[0], layer_viz[1], layer_viz[2])
