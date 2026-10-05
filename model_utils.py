"""
model_utils.py
Core logic for the Attention & Embedding Explorer: loading a pretrained
transformer, running a forward pass with attentions + hidden states,
and reducing embeddings for visualization. Kept free of any Streamlit /
Plotly code so it can be tested and reused independently of the UI layer.
"""

from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Models chosen for: (a) CPU-friendly size for a free HF Spaces instance,
# (b) architectural variety worth comparing (layer/head counts differ).
AVAILABLE_MODELS = {
    "distilbert-base-uncased": "DistilBERT (6 layers, 12 heads) — fastest",
    "bert-base-uncased": "BERT-base (12 layers, 12 heads)",
    "bert-base-cased": "BERT-base cased (12 layers, 12 heads)",
    "roberta-base": "RoBERTa-base (12 layers, 12 heads)",
}

# Known (num_layers, num_heads) per model, used only to size UI sliders
# before any network call has happened (e.g. building the Gradio layout at
# import time, which must not require hitting the Hub). The real values
# always come from the loaded model's config once it's actually fetched --
# this dict is just a UI bootstrap, not a source of truth.
MODEL_SHAPES = {
    "distilbert-base-uncased": (6, 12),
    "bert-base-uncased": (12, 12),
    "bert-base-cased": (12, 12),
    "roberta-base": (12, 12),
}


@dataclass
class AttentionResult:
    tokens: List[str]
    # attentions[layer] has shape (num_heads, seq_len, seq_len)
    attentions: np.ndarray  # shape: (num_layers, num_heads, seq_len, seq_len)
    num_layers: int
    num_heads: int


@dataclass
class TokenEmbedding:
    token: str
    sentence_idx: int
    sentence_text: str
    position: int
    vector: np.ndarray


@dataclass
class EmbeddingResult:
    token_embeddings: List[TokenEmbedding] = field(default_factory=list)
    sentence_embeddings: List[np.ndarray] = field(default_factory=list)
    sentence_texts: List[str] = field(default_factory=list)


def load_model(model_name: str):
    """Load tokenizer + model with attentions and hidden states enabled.
    Cache this at the call site (e.g. st.cache_resource) — it's the
    expensive step.

    Note: recent `transformers` versions default new models to the `sdpa`
    attention implementation, which does not support returning attention
    weights at all (silently yields an empty `attentions` tuple). We force
    `eager` attention so `output_attentions=True` actually works."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_attentions=True,
        output_hidden_states=True,
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


def get_attention_data(tokenizer, model, sentence: str) -> AttentionResult:
    """Run a forward pass and extract every layer's attention matrices for
    a single sentence."""
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # outputs.attentions: tuple of length num_layers, each
    # (batch=1, num_heads, seq_len, seq_len)
    attn_stack = torch.stack(outputs.attentions, dim=0)  # (layers, 1, heads, seq, seq)
    attn_stack = attn_stack.squeeze(1).numpy()  # (layers, heads, seq, seq)

    num_layers, num_heads = attn_stack.shape[0], attn_stack.shape[1]
    return AttentionResult(
        tokens=tokens,
        attentions=attn_stack,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def get_embedding_data(
    tokenizer,
    model,
    sentences: List[str],
    layer: int,
    skip_special_tokens: bool = True,
) -> EmbeddingResult:
    """Run each sentence through the model and collect token- and
    sentence-level embeddings at the requested hidden-state layer.
    Layer 0 is the (non-contextual) embedding layer output; layer i>0 is
    the output of transformer block i.
    Looped per-sentence rather than batched: corpora here are small
    (a handful of demo sentences), and looping avoids padding/masking
    bookkeeping, keeping the logic easy to audit."""
    result = EmbeddingResult()

    for sent_idx, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        # hidden_states: tuple of length num_layers + 1 (embeddings + each block)
        hidden = outputs.hidden_states[layer][0].numpy()  # (seq_len, hidden_dim)

        special_ids = set(tokenizer.all_special_ids)
        input_ids = inputs["input_ids"][0].tolist()

        kept_vectors = []
        for pos, (tok, tok_id) in enumerate(zip(tokens, input_ids)):
            if skip_special_tokens and tok_id in special_ids:
                continue
            vec = hidden[pos]
            kept_vectors.append(vec)
            result.token_embeddings.append(
                TokenEmbedding(
                    token=tok,
                    sentence_idx=sent_idx,
                    sentence_text=sentence,
                    position=pos,
                    vector=vec,
                )
            )

        # Sentence embedding = mean pool over non-padding tokens actually
        # used above (mirrors what's shown at the token level).
        pool_source = np.stack(kept_vectors) if kept_vectors else hidden
        sentence_vec = pool_source.mean(axis=0)
        result.sentence_embeddings.append(sentence_vec)
        result.sentence_texts.append(sentence)

    return result


def reduce_embeddings(
    vectors: np.ndarray,
    method: Literal["pca", "umap"],
    n_components: int,
):
    """Reduce a (n_samples, hidden_dim) matrix to n_components dimensions.
    Guards against n_samples too small for the requested method/params,
    which otherwise raises inside sklearn/umap for small demo corpora."""
    n_samples = vectors.shape[0]
    if n_samples < 2:
        # Nothing meaningful to project — pad with zeros so the caller
        # still gets a well-shaped array back instead of a crash.
        return np.zeros((n_samples, n_components))

    n_components = min(n_components, n_samples - 1, vectors.shape[1])
    n_components = max(n_components, 1)

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components)
        return reducer.fit_transform(vectors)

    elif method == "umap":
        import umap

        n_neighbors = max(2, min(15, n_samples - 1))
        # init="spectral" (UMAP's default) does an eigendecomposition that
        # fails outright on small sample counts typical of a demo corpus
        # (a handful of sentences => a few dozen tokens at most). Random
        # init is more robust here and the difference is negligible at
        # this scale.
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=42,
            init="random",
        )
        return reducer.fit_transform(vectors)

    raise ValueError(f"Unknown reduction method: {method}")
