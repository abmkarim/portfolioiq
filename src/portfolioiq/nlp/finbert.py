"""Sentence-level FinBERT (``ProsusAI/finbert``) tone scores with batching and a disk cache."""

from __future__ import annotations

import os
import re
import zlib

import numpy as np
import pandas as pd

from portfolioiq.config import Config
from portfolioiq.utils import get_logger, write_parquet

log = get_logger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?;])\s+(?=[A-Z\(\"“])|\n+")


def split_sentences(text: str, min_chars: int = 40, max_chars: int = 1500) -> list[str]:
    """Split text into sentences, dropping fragments that look like tables or headers."""
    out = []
    for s in _SENT_SPLIT.split(text or ""):
        s = s.strip()
        if not (min_chars <= len(s) <= max_chars):
            continue
        letters = sum(ch.isalpha() for ch in s)
        if letters / len(s) < 0.6 or len(s.split()) < 6:
            continue
        out.append(s)
    return out


def sample_sentences(sentences: list[str], cap: int, key: str) -> list[str]:
    """Deterministic uniform sample of at most ``cap`` sentences (order preserved)."""
    if cap <= 0 or len(sentences) <= cap:
        return sentences
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    idx = np.sort(rng.choice(len(sentences), cap, replace=False))
    return [sentences[i] for i in idx]


def aggregate_probs(probs: np.ndarray) -> dict[str, float]:
    """Document scores from sentence probabilities ``[p_pos, p_neg, p_neu]``.

    * ``tone`` = mean(p_pos - p_neg)
    * ``neg_share`` = share of sentences whose argmax is negative
    * ``uncertainty`` = mean normalised entropy of the predictive distribution
    """
    if probs.size == 0:
        return {"tone": np.nan, "neg_share": np.nan, "uncertainty": np.nan}
    p = np.clip(probs, 1e-12, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])
    return {
        "tone": float((probs[:, 0] - probs[:, 1]).mean()),
        "neg_share": float((probs.argmax(axis=1) == 1).mean()),
        "uncertainty": float(entropy.mean()),
    }


class FinbertScorer:
    """Thin wrapper around the HF model returning probabilities in ``[pos, neg, neu]`` order."""

    def __init__(self, model_name: str, max_length: int = 128, batch_size: int = 32,
                 quantize: bool = True, num_threads: int = 0, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if num_threads:
            torch.set_num_threads(num_threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        if quantize and device == "cpu":
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = model.to(device)
        self.device = device
        labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        self.order = [labels["positive"], labels["negative"], labels["neutral"]]
        self.max_length, self.batch_size = max_length, batch_size

    def score(self, sentences: list[str]) -> np.ndarray:
        """Return an ``(n, 3)`` array of probabilities."""
        if not sentences:
            return np.zeros((0, 3))
        out = []
        # sort by length for efficient padding, then restore order
        order = np.argsort([len(s) for s in sentences])
        with self.torch.inference_mode():
            for i in range(0, len(sentences), self.batch_size):
                batch = [sentences[j] for j in order[i:i + self.batch_size]]
                enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_length,
                               return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                out.append(self.torch.softmax(logits, dim=-1).cpu().numpy()[:, self.order])
        probs = np.concatenate(out)
        res = np.empty_like(probs)
        res[order] = probs
        return res


def cache_path(cfg: Config):
    fb = cfg.nlp.finbert
    tag = f"{fb.model_name.replace('/', '_')}_cap{fb.max_sentences_per_item}_len{fb.max_length}_q{int(fb.quantize)}"
    return cfg.interim_dir / "finbert" / f"{tag}.parquet"


def score_filings(cfg: Config, texts: pd.DataFrame, device: str = "cpu", flush_every: int = 25) -> pd.DataFrame:
    """Score Item 1A and Item 7 of every filing (cached by accession)."""
    path = cache_path(cfg)
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["accession"])
    todo = texts[~texts["accession"].isin(set(cached["accession"]))]
    log.info("finbert to score", n=len(todo), cached=len(cached))
    if len(todo):
        fb = cfg.nlp.finbert
        scorer = FinbertScorer(fb.model_name, fb.max_length, fb.batch_size, fb.quantize,
                               fb.num_threads or (os.cpu_count() or 4), device)
        rows = []
        for i, r in enumerate(todo.itertuples(index=False)):
            row = {"accession": r.accession}
            for item in ("item1a", "item7"):
                sents = split_sentences(getattr(r, item))
                n_total = len(sents)
                sents = sample_sentences(sents, fb.max_sentences_per_item, f"{r.accession}_{item}")
                agg = aggregate_probs(scorer.score(sents))
                suffix = "1a" if item == "item1a" else "7"
                row.update({f"fb_tone_{suffix}": agg["tone"], f"fb_neg_{suffix}": agg["neg_share"],
                            f"fb_unc_{suffix}": agg["uncertainty"], f"n_sent_{suffix}": n_total})
            rows.append(row)
            if (i + 1) % flush_every == 0 or i + 1 == len(todo):
                cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
                write_parquet(cached, path)
                rows = []
                log.info("finbert progress", done=i + 1, total=len(todo))
    return cached[cached["accession"].isin(set(texts["accession"]))].reset_index(drop=True)
