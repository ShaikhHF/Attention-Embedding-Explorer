"""
Tests for model_utils.py's core logic (attention extraction, embedding
extraction, dimensionality reduction).

Deliberately uses a tiny, locally-constructed BERT (random weights, small
vocab, from BertConfig — no download) rather than a real pretrained
checkpoint. That keeps the suite fast and runnable offline / in CI without
pulling gigabytes of weights, while still exercising the exact same
transformers APIs (output_attentions, output_hidden_states, attn
implementation) that the real app depends on.
"""
import os
import tempfile

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from model_utils import get_attention_data, get_embedding_data, reduce_embeddings

TOY_VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "the", "cat", "sat", "on", "mat", "bank", "river", "rates", "interest",
    "raised", "its", "she", "to", "fish", "dog", "rug", "he", "deposited",
    "a", "check", "at", "because", "it", "was", "tired", ".",
]

NUM_LAYERS = 4
NUM_HEADS = 4
HIDDEN_SIZE = 32


@pytest.fixture(scope="module")
def tiny_model():
    tmpdir = tempfile.mkdtemp()
    vocab_path = os.path.join(tmpdir, "vocab.txt")
    with open(vocab_path, "w") as f:
        f.write("\n".join(TOY_VOCAB))

    tokenizer = BertTokenizer(vocab=vocab_path)
    config = BertConfig(
        vocab_size=len(TOY_VOCAB),
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=64,
        max_position_embeddings=64,
        output_attentions=True,
        output_hidden_states=True,
        # See README: sdpa (the transformers default) can't return
        # attention weights, so the real app forces eager too.
        attn_implementation="eager",
    )
    model = BertModel(config)
    model.eval()
    return tokenizer, model


def test_attention_extraction_shape_and_softmax(tiny_model):
    tokenizer, model = tiny_model
    result = get_attention_data(tokenizer, model, "the cat sat on the mat")

    assert result.num_layers == NUM_LAYERS
    assert result.num_heads == NUM_HEADS
    seq_len = len(result.tokens)
    assert result.attentions.shape == (NUM_LAYERS, NUM_HEADS, seq_len, seq_len)

    # Each row of an attention matrix is a softmax distribution over keys.
    row_sums = result.attentions[0, 0].sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)


def test_embedding_extraction_across_layers(tiny_model):
    tokenizer, model = tiny_model
    sentences = [
        "the bank raised its interest rates",
        "she sat on the river bank to fish",
        "the cat sat on the mat",
    ]

    for layer in (0, NUM_LAYERS // 2, NUM_LAYERS):
        result = get_embedding_data(tokenizer, model, sentences, layer=layer)
        assert len(result.sentence_embeddings) == len(sentences)
        assert result.sentence_embeddings[0].shape == (HIDDEN_SIZE,)
        assert len(result.token_embeddings) > 0


def test_reduce_embeddings_pca(tiny_model):
    tokenizer, model = tiny_model
    emb = get_embedding_data(tokenizer, model, ["the cat sat on the mat"], layer=1)
    vectors = np.stack([t.vector for t in emb.token_embeddings])

    coords_2d = reduce_embeddings(vectors, method="pca", n_components=2)
    assert coords_2d.shape == (vectors.shape[0], 2)

    coords_3d = reduce_embeddings(vectors, method="pca", n_components=3)
    assert coords_3d.shape[1] <= 3


def test_reduce_embeddings_umap_small_n(tiny_model):
    """UMAP's default spectral init fails outright below a certain sample
    count -- this is the scenario a short demo sentence produces."""
    tokenizer, model = tiny_model
    emb = get_embedding_data(tokenizer, model, ["the cat sat"], layer=1)
    vectors = np.stack([t.vector for t in emb.token_embeddings])
    assert vectors.shape[0] < 10  # confirms this is the small-N case

    coords = reduce_embeddings(vectors, method="umap", n_components=2)
    assert coords.shape == (vectors.shape[0], 2)


def test_reduce_embeddings_single_sample_no_crash(tiny_model):
    tokenizer, model = tiny_model
    emb = get_embedding_data(tokenizer, model, ["the cat sat"], layer=1)
    vec = np.stack(emb.sentence_embeddings)  # shape (1, hidden_size)

    coords = reduce_embeddings(vec, method="pca", n_components=2)
    assert coords.shape == (1, 2)
