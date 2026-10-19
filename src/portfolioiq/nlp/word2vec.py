"""Word2Vec trained per walk-forward fold on documents available BEFORE the fold start."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from portfolioiq.config import Word2VecConfig

_TOKEN = re.compile(r"[a-z][a-z\-]{1,24}")


def tokenize(text: str, max_tokens: int | None = None) -> list[str]:
    """Lower-case alphabetic tokens (length >= 2)."""
    toks = _TOKEN.findall((text or "").lower())
    return toks[:max_tokens] if max_tokens else toks


def corpus_before(docs: pd.DataFrame, cutoff: pd.Timestamp, date_col: str = "feature_available_date") -> pd.DataFrame:
    """Documents strictly available before ``cutoff`` — the only ones allowed for training."""
    return docs[pd.to_datetime(docs[date_col]) < pd.Timestamp(cutoff)]


def train_word2vec(token_lists: list[list[str]], cfg: Word2VecConfig, seed: int):
    """Train a gensim Word2Vec model; returns its ``KeyedVectors``."""
    from gensim.models import Word2Vec

    # gensim caps sentences at 10k tokens; chunk long documents
    sents = [t[i:i + 1000] for t in token_lists for i in range(0, len(t), 1000) if t[i:i + 1000]]
    model = Word2Vec(
        sentences=sents, vector_size=cfg.vector_size, window=cfg.window, min_count=cfg.min_count,
        workers=cfg.workers, epochs=cfg.epochs, seed=seed, sg=0,
    )
    return model.wv


def idf_weights(token_lists: list[list[str]]) -> dict[str, float]:
    """Smoothed IDF computed on the training corpus only."""
    n = len(token_lists)
    df = Counter()
    for toks in token_lists:
        df.update(set(toks))
    return {w: float(np.log((1 + n) / (1 + c)) + 1.0) for w, c in df.items()}


def doc_embedding(tokens: list[str], kv, idf: dict[str, float]) -> np.ndarray | None:
    """TF-IDF weighted mean of word vectors (``None`` if no known word)."""
    counts = Counter(t for t in tokens if t in kv.key_to_index)
    if not counts:
        return None
    words = list(counts)
    weights = np.array([counts[w] * idf.get(w, 1.0) for w in words])
    vecs = np.stack([kv[w] for w in words])
    return (weights[:, None] * vecs).sum(0) / weights.sum()


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return np.nan
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else np.nan
