import altair as alt
import pandas as pd
import torch

from model import PositionalEncoding, subsequent_mask
from test import run_model_example
from train import LabelSmoothing


def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


def example_learning_schedule():
    from train import rate

    opts = [
        [512, 1, 4000],
        [512, 1, 8000],
        [256, 1, 4000],
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    for idx, example in enumerate(opts):
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step, example=example: rate(step, *example),
        )
        tmp = []
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(
            pd.DataFrame(
                {
                    "learning_rate": tmp,
                    "model_size:warmup": [f"{example[0]}:{example[2]}"]
                    * 20000,
                    "step": range(20000),
                }
            )
        )

    return (
        alt.Chart(pd.concat(learning_rates))
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="learning_rate", color="model_size:warmup:N")
        .interactive()
    )


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(predict.log(), torch.LongTensor([2, 1, 0]))

    return (
        alt.Chart(
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[0, :],
                    "columns": list(range(5)),
                }
            )
        )
        .mark_bar()
        .properties(height=200)
        .encode(x="columns", y="target distribution")
    )


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
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
    "convert a dense matrix to a data frame with row and column indices"
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
