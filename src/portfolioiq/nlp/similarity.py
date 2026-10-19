"""Year-over-year 10-K change measures ("Lazy Prices", Cohen-Malloy-Nguyen 2020)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_cosine(current: str, previous: str, max_features: int = 20000) -> float:
    """Cosine similarity of TF-IDF vectors fitted on the two documents only (no other document leaks in)."""
    if not current or not previous:
        return np.nan
    vec = TfidfVectorizer(max_features=max_features, sublinear_tf=True, stop_words="english",
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,24}\b")
    try:
        m = vec.fit_transform([current, previous])
    except ValueError:  # empty vocabulary
        return np.nan
    return float(cosine_similarity(m[0], m[1])[0, 0])


def log_length_change(current: str, previous: str) -> float:
    """log(len(current) / len(previous)) in characters."""
    if not current or not previous:
        return np.nan
    return float(np.log(len(current) / len(previous)))
