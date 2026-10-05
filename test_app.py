"""
Tests for app.py's render functions (the Gradio UI layer). Uses the same
tiny local model approach as test_model_utils.py, and monkeypatches
app.load_model directly so no real model download is needed.
"""
import os
import tempfile

import pytest
from transformers import BertConfig, BertModel, BertTokenizer

import app

TOY_VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "the", "cat", "sat", "on", "mat", "bank", "river", "rates", "interest",
]
VALID_MODEL_NAME = "distilbert-base-uncased"  # must be one of app's Dropdown choices


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """Patches app.load_model (not model_utils.load_model -- app.py did a
    `from model_utils import load_model`, binding its own name at import
    time, so the patch has to target app's copy) to return a tiny local
    model regardless of which model name is requested."""
    tmpdir = tempfile.mkdtemp()
    vocab_path = os.path.join(tmpdir, "vocab.txt")
    with open(vocab_path, "w") as f:
        f.write("\n".join(TOY_VOCAB))

    tokenizer = BertTokenizer(vocab=vocab_path)
    config = BertConfig(
        vocab_size=len(TOY_VOCAB),
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        output_attentions=True,
        output_hidden_states=True,
        attn_implementation="eager",
    )
    model = BertModel(config)
    model.eval()

    monkeypatch.setattr(app, "load_model", lambda name: (tokenizer, model))
    app.cached_load_model.cache_clear()
    yield
    app.cached_load_model.cache_clear()


def test_render_attention_basic():
    fig, grid_fig = app.render_attention(VALID_MODEL_NAME, "the cat sat on the mat", 1, 2, False)
    assert len(fig.data) == 1
    assert len(grid_fig.data) == 4  # num_heads in the dummy config


def test_render_attention_average_heads():
    fig, _ = app.render_attention(VALID_MODEL_NAME, "the cat sat on the mat", 1, 0, True)
    assert len(fig.data) == 1


def test_render_attention_empty_sentence_does_not_crash():
    fig, grid_fig = app.render_attention(VALID_MODEL_NAME, "   ", 0, 0, False)
    assert len(fig.data) == 0
    assert len(grid_fig.data) == 0


def test_render_attention_clamps_out_of_range_slider_values():
    """Switching models can leave a stale slider value briefly out of
    range for the new model until on_model_change resizes it -- this
    must not crash."""
    fig, grid_fig = app.render_attention(VALID_MODEL_NAME, "the cat sat", 99, 99, False)
    assert len(fig.data) == 1


def test_on_model_change_returns_correct_shape():
    info_md, layer_slider, head_slider, emb_layer_slider = app.on_model_change(VALID_MODEL_NAME)
    assert "4 layers" in info_md
    assert "4 heads" in info_md


@pytest.mark.parametrize("granularity", ["Token", "Sentence"])
@pytest.mark.parametrize("method", ["PCA", "UMAP"])
@pytest.mark.parametrize("dims", ["2D", "3D"])
def test_render_embeddings_all_control_combinations(granularity, method, dims):
    fig = app.render_embeddings(
        VALID_MODEL_NAME, app.DEFAULT_EMBEDDING_SENTENCES, 2, granularity, method, dims
    )
    assert len(fig.data) == 1


def test_render_embeddings_empty_text_does_not_crash():
    fig = app.render_embeddings(VALID_MODEL_NAME, "   ", 2, "Token", "PCA", "2D")
    assert len(fig.data) == 0


def test_render_embeddings_clamps_out_of_range_layer():
    fig = app.render_embeddings(
        VALID_MODEL_NAME, app.DEFAULT_EMBEDDING_SENTENCES, 99, "Token", "PCA", "2D"
    )
    assert len(fig.data) == 1
