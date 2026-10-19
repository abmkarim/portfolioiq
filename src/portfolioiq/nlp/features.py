"""Feature construction: filing-level NLP features, fold-specific Word2Vec features, monthly panel."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from portfolioiq.config import Config
from portfolioiq.ingestion.parsing import load_texts
from portfolioiq.nlp import finbert, similarity, word2vec
from portfolioiq.utils import get_logger, write_parquet

log = get_logger(__name__)

FINBERT_FEATURES = ["fb_tone_1a", "fb_neg_1a", "fb_unc_1a", "fb_tone_7", "fb_neg_7", "fb_unc_7",
                    "d_tone_1a", "d_tone_7"]
TEXT_FEATURES = ["tfidf_sim_1a", "tfidf_sim_7", "len_chg_1a", "len_chg_7", "log_len_1a", "log_len_7"]
W2V_FEATURES = ["w2v_sim_1a", "w2v_sim_7"]
CONTROL_FEATURES = ["log_mcap", "mom_12_1", "vol_3m", "rev_1m"]
ALL_FEATURES = FINBERT_FEATURES + TEXT_FEATURES + W2V_FEATURES + CONTROL_FEATURES

FEATURE_GROUPS = {
    **{f: "finbert" for f in FINBERT_FEATURES},
    **{f: "tfidf_length" for f in TEXT_FEATURES},
    **{f: "word2vec" for f in W2V_FEATURES},
    **{f: "controls" for f in CONTROL_FEATURES},
}


def query(cfg: Config, sql: str) -> pd.DataFrame:
    """Read-only query against the dbt warehouse."""
    con = duckdb.connect(str(cfg.warehouse_path), read_only=True)
    try:
        return con.sql(sql).df()
    finally:
        con.close()


def previous_filing(filings: pd.DataFrame, min_gap_days: int = 270, max_gap_days: int = 730) -> pd.Series:
    """Accession of the same company's previous 10-K (9 to 24 months earlier), else NaN."""
    f = filings.sort_values(["cik", "acceptance_datetime"])
    prev_acc = f.groupby("cik")["accession"].shift(1)
    prev_dt = f.groupby("cik")["acceptance_datetime"].shift(1)
    gap = (f["acceptance_datetime"] - prev_dt).dt.days
    ok = gap.between(min_gap_days, max_gap_days)
    return prev_acc.where(ok).reindex(filings.index)


def cs_rank(df: pd.DataFrame, cols: list[str], by: str = "month_end") -> pd.DataFrame:
    """Cross-sectional percentile rank per date, centred on 0 (range [-0.5, 0.5]); NaN preserved."""
    out = df.copy()
    for c in cols:
        out[c] = df.groupby(by)[c].rank(pct=True) - 0.5
    return out


def make_target(df: pd.DataFrame) -> pd.Series:
    """Next-month return ranked cross-sectionally, then demeaned within FF12 sector."""
    r = df.groupby("month_end")["ret_fwd_1m"].rank(pct=True) - 0.5
    return r - r.groupby([df["month_end"], df["ff12"]]).transform("mean")


# --------------------------------------------------------------------------- filing level
def build_filing_features(cfg: Config) -> pd.DataFrame:
    """FinBERT, TF-IDF similarity and length changes for every filing; writes ``filing_features.parquet``."""
    filings = query(cfg, "select cik, accession, acceptance_datetime, feature_available_date "
                         "from int_filings_available where feature_available_date is not null")
    filings["prev_accession"] = previous_filing(filings)
    texts = load_texts(cfg, filings["cik"].unique().tolist())
    texts = texts[texts["accession"].isin(set(filings["accession"]))]
    scores = finbert.score_filings(cfg, texts, device="cuda" if _cuda_requested(cfg) else "cpu")

    text_map = texts.set_index("accession")[["item1a", "item7"]]
    rows = []
    for r in filings.itertuples(index=False):
        cur = text_map.loc[r.accession] if r.accession in text_map.index else None
        prev = text_map.loc[r.prev_accession] if isinstance(r.prev_accession, str) and \
            r.prev_accession in text_map.index else None
        row = {"accession": r.accession}
        for item, sfx in (("item1a", "1a"), ("item7", "7")):
            c = cur[item] if cur is not None else ""
            p = prev[item] if prev is not None else ""
            row[f"tfidf_sim_{sfx}"] = similarity.tfidf_cosine(c, p, cfg.nlp.tfidf_max_features)
            row[f"len_chg_{sfx}"] = similarity.log_length_change(c, p)
            row[f"log_len_{sfx}"] = float(np.log(len(c))) if c else np.nan
        rows.append(row)
    feats = filings.merge(pd.DataFrame(rows), on="accession").merge(scores, on="accession", how="left")
    prev = feats.set_index("accession")[["fb_tone_1a", "fb_tone_7"]]
    for sfx in ("1a", "7"):
        prev_tone = feats["prev_accession"].map(prev[f"fb_tone_{sfx}"])
        feats[f"d_tone_{sfx}"] = feats[f"fb_tone_{sfx}"] - prev_tone
    write_parquet(feats, cfg.processed_dir / "filing_features.parquet")
    log.info("filing features", n=len(feats), with_prev=int(feats["prev_accession"].notna().sum()))
    return feats


def _cuda_requested(cfg: Config) -> bool:
    import os

    if os.environ.get("PIQ_COMPUTE", "cpu") != "cuda":
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# --------------------------------------------------------------------------- word2vec per fold
def fold_cutoffs(month_ends: pd.Series, years: list[int]) -> dict[int, pd.Timestamp]:
    """First formation date of each fold = last month end of year Y-1 (holding month = January Y)."""
    me = pd.to_datetime(pd.Series(month_ends).drop_duplicates()).sort_values()
    out = {}
    for y in years:
        prior = me[me.dt.year == y - 1]
        out[y] = prior.max() if len(prior) else pd.Timestamp(f"{y - 1}-12-31")
    return out


def build_w2v_features(cfg: Config, filing_feats: pd.DataFrame, cutoffs: dict[int, pd.Timestamp]) -> pd.DataFrame:
    """For each fold: train Word2Vec on docs available before the fold cutoff, then compute YoY similarity."""
    texts = load_texts(cfg, filing_feats["cik"].unique().tolist()).set_index("accession")
    wc = cfg.nlp.word2vec
    meta = filing_feats[["accession", "prev_accession", "feature_available_date", "acceptance_datetime"]].copy()
    tokens = {
        (acc, item): word2vec.tokenize(texts.at[acc, item], wc.max_tokens_per_doc)
        for acc in meta["accession"] if acc in texts.index for item in ("item1a", "item7")
    }
    out = []
    for year, cutoff in sorted(cutoffs.items()):
        train_docs = word2vec.corpus_before(meta, cutoff)
        corpus = [tokens[(a, it)] for a in train_docs["accession"] for it in ("item1a", "item7")
                  if (a, it) in tokens and tokens[(a, it)]]
        if len(corpus) < 10:
            log.warning("w2v corpus too small", fold=year, n_docs=len(corpus))
            continue
        kv = word2vec.train_word2vec(corpus, wc, cfg.seed + year)
        idf = word2vec.idf_weights(corpus)
        horizon = meta[pd.to_datetime(meta["acceptance_datetime"]) <= pd.Timestamp(f"{year}-12-31")]
        cache: dict = {}

        def emb(acc: str, item: str, kv=kv, idf=idf, cache=cache):
            key = (acc, item)
            if key not in cache:
                toks = tokens.get(key)
                cache[key] = word2vec.doc_embedding(toks, kv, idf) if toks else None
            return cache[key]

        for r in horizon.itertuples(index=False):
            row = {"fold_year": year, "accession": r.accession}
            for item, sfx in (("item1a", "1a"), ("item7", "7")):
                row[f"w2v_sim_{sfx}"] = word2vec.cosine(emb(r.accession, item), emb(r.prev_accession, item)) \
                    if isinstance(r.prev_accession, str) else np.nan
            out.append(row)
        log.info("w2v fold", fold=year, cutoff=str(cutoff.date()), n_train_docs=len(train_docs),
                 vocab=len(kv.key_to_index))
    res = pd.DataFrame(out, columns=["fold_year", "accession", "w2v_sim_1a", "w2v_sim_7"])
    write_parquet(res, cfg.processed_dir / "w2v_features.parquet")
    return res


# --------------------------------------------------------------------------- panel
def build_features(cfg: Config) -> pd.DataFrame:
    """Build the modelling panel (``panel_features.parquet``) and fold-specific W2V features."""
    filing_feats = build_filing_features(cfg)
    panel = query(cfg, "select * from fct_monthly_panel where is_investable")
    panel["month_end"] = pd.to_datetime(panel["month_end"])
    years = list(range(cfg.model.oos_start_year, cfg.model.oos_end_year + 1))
    build_w2v_features(cfg, filing_feats, fold_cutoffs(panel["month_end"], years))

    cols = ["accession"] + FINBERT_FEATURES + TEXT_FEATURES
    df = panel.merge(filing_feats[cols], on="accession", how="left")
    df = cs_rank(df, FINBERT_FEATURES + TEXT_FEATURES + CONTROL_FEATURES)
    df["log_mcap"] = df["log_mcap"].fillna(0.0)  # documented fallback: missing size -> cross-sectional median
    df["target"] = make_target(df)
    write_parquet(df, cfg.processed_dir / "panel_features.parquet")
    log.info("panel features", rows=len(df), months=df["month_end"].nunique(),
             with_filing=float(df["accession"].notna().mean()))
    return df
