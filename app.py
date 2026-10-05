"""
Attention & Embedding Explorer
An interactive Gradio app for looking inside a pretrained transformer:
what each attention head attends to, and how token/sentence embeddings
organize across layers.
"""

from functools import lru_cache

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model_utils import (
    AVAILABLE_MODELS,
    MODEL_SHAPES,
    get_attention_data,
    get_embedding_data,
    load_model,
    reduce_embeddings,
)

DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_ATTENTION_SENTENCE = "The cat sat on the mat because it was tired."
DEFAULT_EMBEDDING_SENTENCES = """The bank raised its interest rates.
She sat on the river bank to fish.
The cat sat on the mat.
The dog sat on the rug.
He deposited a check at the bank."""


@lru_cache(maxsize=4)
def cached_load_model(model_name: str):
    """In-process cache so switching back to a previously used model is
    instant rather than re-downloading / re-initializing it."""
    return load_model(model_name)


def on_model_change(model_name: str):
    """Loads the newly selected model and resizes the layer/head sliders
    to match its actual architecture."""
    _, model = cached_load_model(model_name)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    info_md = (
        f"**{num_layers} layers × {num_heads} heads**, "
        f"hidden size {model.config.hidden_size}"
    )
    new_layer_slider = gr.Slider(minimum=0, maximum=num_layers - 1, value=0, step=1, label="Layer")
    new_head_slider = gr.Slider(minimum=0, maximum=num_heads - 1, value=0, step=1, label="Head")
    new_emb_layer_slider = gr.Slider(
        minimum=0, maximum=num_layers, value=num_layers, step=1,
        label="Embedding layer (0 = input embeddings)",
    )
    return info_md, new_layer_slider, new_head_slider, new_emb_layer_slider


def render_attention(model_name, sentence, layer, head, average_heads):
    tokenizer, model = cached_load_model(model_name)
    empty = go.Figure()
    if not sentence or not sentence.strip():
        return empty, empty

    result = get_attention_data(tokenizer, model, sentence)
    layer = min(int(layer), result.num_layers - 1)
    head = min(int(head), result.num_heads - 1)

    if average_heads:
        matrix = result.attentions[layer].mean(axis=0)
        title = f"Layer {layer} — average over all {result.num_heads} heads"
    else:
        matrix = result.attentions[layer, head]
        title = f"Layer {layer}, Head {head}"

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=result.tokens,
            y=result.tokens,
            colorscale="Blues",
            hovertemplate="attends to <b>%{x}</b><br>from <b>%{y}</b><br>weight: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Attended-to token",
        yaxis_title="Attending token",
        yaxis_autorange="reversed",
        height=500,
    )

    grid_cols = 4
    grid_rows = int(np.ceil(result.num_heads / grid_cols))
    grid_fig = make_subplots(
        rows=grid_rows,
        cols=grid_cols,
        subplot_titles=[f"Head {h}" for h in range(result.num_heads)],
    )
    for h in range(result.num_heads):
        r, c = divmod(h, grid_cols)
        grid_fig.add_trace(
            go.Heatmap(
                z=result.attentions[layer, h],
                colorscale="Blues",
                showscale=False,
                hoverinfo="skip",
            ),
            row=r + 1,
            col=c + 1,
        )
    grid_fig.update_xaxes(showticklabels=False)
    grid_fig.update_yaxes(showticklabels=False, autorange="reversed")
    grid_fig.update_layout(height=180 * grid_rows, title=f"All heads — layer {layer}")

    return fig, grid_fig


def render_embeddings(model_name, text, layer, granularity, method, dims):
    tokenizer, model = cached_load_model(model_name)
    sentences = [s.strip() for s in (text or "").split("\n") if s.strip()]
    if not sentences:
        return go.Figure()

    layer = min(int(layer), model.config.num_hidden_layers)
    n_components = 3 if dims == "3D" else 2
    emb_data = get_embedding_data(tokenizer, model, sentences, layer=layer)

    if granularity == "Token":
        vectors = np.stack([t.vector for t in emb_data.token_embeddings])
        labels = [t.token for t in emb_data.token_embeddings]
        sentence_idx = [t.sentence_idx for t in emb_data.token_embeddings]
        hover = [
            f"token: {t.token}<br>sentence: {t.sentence_text}"
            for t in emb_data.token_embeddings
        ]
    else:
        vectors = np.stack(emb_data.sentence_embeddings)
        labels = [f"S{i}" for i in range(len(emb_data.sentence_texts))]
        sentence_idx = list(range(len(emb_data.sentence_texts)))
        hover = emb_data.sentence_texts

    coords = reduce_embeddings(vectors, method=method.lower(), n_components=n_components)

    if coords.shape[1] >= 3:
        fig = go.Figure(
            data=go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode="markers+text", text=labels, textposition="top center",
                marker=dict(size=5, color=sentence_idx, colorscale="Viridis"),
                hovertext=hover, hoverinfo="text",
            )
        )
    else:
        fig = go.Figure(
            data=go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(coords)),
                mode="markers+text", text=labels, textposition="top center",
                marker=dict(size=9, color=sentence_idx, colorscale="Viridis"),
                hovertext=hover, hoverinfo="text",
            )
        )

    fig.update_layout(title=f"{granularity} embeddings — layer {layer}, {method}", height=600)
    return fig


# ---------- Build the UI ----------
default_num_layers, default_num_heads = MODEL_SHAPES[DEFAULT_MODEL]

with gr.Blocks(title="Attention & Embedding Explorer") as demo:
    gr.Markdown(
        "# Attention & Embedding Explorer\n"
        "Loads a pretrained transformer with `output_attentions=True` and "
        "`output_hidden_states=True`, then visualizes both."
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=DEFAULT_MODEL,
                label="Model",
            )
            model_info = gr.Markdown(
                f"**{default_num_layers} layers × {default_num_heads} heads**"
            )
            attention_sentence = gr.Textbox(
                value=DEFAULT_ATTENTION_SENTENCE, label="Sentence for attention view"
            )
            embedding_text = gr.Textbox(
                value=DEFAULT_EMBEDDING_SENTENCES, lines=6,
                label="Sentences for embedding view (one per line)",
            )
            gr.Markdown(
                "_Try the bank / river-bank example: the same word gets a "
                "different embedding depending on context -- something "
                "static word vectors (word2vec/GloVe) can't do._"
            )

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("🔍 Attention Heatmap"):
                    gr.Markdown(
                        "Each cell (row, col) is how much the token in that "
                        "row attends to the token in that column, "
                        "softmax-normalized per row."
                    )
                    with gr.Row():
                        layer_slider = gr.Slider(
                            0, default_num_layers - 1, value=0, step=1, label="Layer"
                        )
                        average_heads = gr.Checkbox(label="Average all heads", value=False)
                        head_slider = gr.Slider(
                            0, default_num_heads - 1, value=0, step=1, label="Head"
                        )
                    attn_plot = gr.Plot(label="Attention heatmap")
                    with gr.Accordion("See all heads in this layer", open=False):
                        grid_plot = gr.Plot(label="All heads grid")

                with gr.TabItem("🌐 Embedding Space"):
                    gr.Markdown("Contextual embeddings from a chosen layer, reduced to 2D/3D.")
                    with gr.Row():
                        emb_layer_slider = gr.Slider(
                            0, default_num_layers, value=default_num_layers, step=1,
                            label="Embedding layer (0 = input embeddings)",
                        )
                        granularity_radio = gr.Radio(
                            ["Token", "Sentence"], value="Token", label="Granularity"
                        )
                        method_radio = gr.Radio(["PCA", "UMAP"], value="PCA", label="Reduction")
                        dims_radio = gr.Radio(["2D", "3D"], value="2D", label="Dimensions")
                    emb_plot = gr.Plot(label="Embedding scatter")

    attn_inputs = [model_dd, attention_sentence, layer_slider, head_slider, average_heads]
    emb_inputs = [model_dd, embedding_text, emb_layer_slider, granularity_radio, method_radio, dims_radio]

    # Changing the model reloads it, resizes the sliders to match its
    # architecture, then recomputes both views with the (possibly
    # clamped) slider values.
    model_dd.change(
        fn=on_model_change,
        inputs=model_dd,
        outputs=[model_info, layer_slider, head_slider, emb_layer_slider],
    ).then(
        fn=render_attention, inputs=attn_inputs, outputs=[attn_plot, grid_plot]
    ).then(
        fn=render_embeddings, inputs=emb_inputs, outputs=[emb_plot]
    )

    for component in [attention_sentence, layer_slider, head_slider, average_heads]:
        component.change(fn=render_attention, inputs=attn_inputs, outputs=[attn_plot, grid_plot])

    for component in [embedding_text, emb_layer_slider, granularity_radio, method_radio, dims_radio]:
        component.change(fn=render_embeddings, inputs=emb_inputs, outputs=[emb_plot])

    demo.load(fn=render_attention, inputs=attn_inputs, outputs=[attn_plot, grid_plot])
    demo.load(fn=render_embeddings, inputs=emb_inputs, outputs=[emb_plot])


if __name__ == "__main__":
    demo.launch()
