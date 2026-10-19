import numpy as np
import pandas as pd

from portfolioiq.ingestion.parsing import extract_section, html_to_text, parse_document
from portfolioiq.nlp import word2vec
from portfolioiq.nlp.features import cs_rank, make_target
from portfolioiq.nlp.finbert import aggregate_probs, sample_sentences, split_sentences
from portfolioiq.nlp.similarity import log_length_change, tfidf_cosine

FILLER = "The company faces competition and regulatory risk in its markets every year. " * 60


def _html():
    return f"""<html><body>
    <p>TABLE OF CONTENTS</p><p>Item 1A. Risk Factors 12</p><p>Item 1B. Unresolved Staff Comments 20</p>
    <p>Item 7. Management's Discussion and Analysis 30</p><p>Item 7A. Quantitative 40</p>
    <p>PART I</p><p>Item 1. Business</p><p>We make widgets.</p>
    <p>Item 1A.&nbsp;Risk Factors</p><p>{FILLER}</p>
    <p>Item 1B. Unresolved Staff Comments</p><p>None.</p>
    <p>Item 7. Management&#8217;s Discussion and Analysis of Financial Condition</p><p>{FILLER}</p>
    <p>Item 7A. Quantitative and Qualitative Disclosures</p><p>Rates.</p>
    <p>Item 8. Financial Statements</p></body></html>""".encode()


def test_parse_items_skips_table_of_contents():
    secs = parse_document(_html(), min_chars=1000)
    assert secs["item1a"].lower().startswith("item 1a")
    assert "competition" in secs["item1a"] and "Unresolved" not in secs["item1a"]
    assert len(secs["item1a"]) > 3000
    assert "competition" in secs["item7"] and "Rates." not in secs["item7"]


def test_extract_missing_section_returns_empty():
    text = html_to_text(b"<html><p>No items here</p></html>")
    assert extract_section(text, r"item\s*1a", r"item\s*2") == ""


def test_sentence_split_and_sampling():
    sents = split_sentences("Revenue increased due to strong demand in all segments. Total 12 34 56 78 90 11 22 33 44 55. "
                            "We expect margins to remain under pressure next year.")
    assert len(sents) == 2
    many = [f"Sentence number {i} is long enough to be kept here." for i in range(100)]
    a = sample_sentences(many, 10, "acc_item1a")
    assert a == sample_sentences(many, 10, "acc_item1a") and len(a) == 10
    assert sample_sentences(many[:5], 10, "k") == many[:5]


def test_aggregate_probs():
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    agg = aggregate_probs(p)
    assert agg["tone"] == np.mean([0.7, -0.7, 0.0])
    assert agg["neg_share"] == 1 / 3
    assert 0 < agg["uncertainty"] < 1
    assert np.isnan(aggregate_probs(np.zeros((0, 3)))["tone"])


def test_similarity_measures():
    assert tfidf_cosine("risk of loss from competition", "risk of loss from competition") > 0.99
    assert tfidf_cosine("risk of loss from competition", "weather in the sahara desert") < 0.2
    assert np.isnan(tfidf_cosine("", "x"))
    assert log_length_change("a" * 200, "a" * 100) == np.log(2)


def test_word2vec_trained_without_future_documents():
    docs = pd.DataFrame({
        "accession": ["old1", "old2", "future"],
        "feature_available_date": pd.to_datetime(["2016-03-01", "2017-03-01", "2018-02-01"]),
        "text": ["alpha beta gamma delta " * 50, "beta gamma delta epsilon " * 50, "zeta eta theta iota " * 50],
    })
    train = word2vec.corpus_before(docs, pd.Timestamp("2017-12-29"))
    assert set(train["accession"]) == {"old1", "old2"}
    from portfolioiq.config import Word2VecConfig

    kv = word2vec.train_word2vec([word2vec.tokenize(t) for t in train["text"]],
                                 Word2VecConfig(vector_size=10, min_count=1, epochs=2, workers=1), seed=1)
    assert "beta" in kv.key_to_index
    for future_word in ("zeta", "eta", "theta", "iota"):
        assert future_word not in kv.key_to_index      # nothing from documents after the fold start
    idf = word2vec.idf_weights([word2vec.tokenize(t) for t in train["text"]])
    e1 = word2vec.doc_embedding(word2vec.tokenize(docs.text[0]), kv, idf)
    assert word2vec.cosine(e1, e1) > 0.999
    assert word2vec.doc_embedding(word2vec.tokenize(docs.text[2]), kv, idf) is None


def test_cs_rank_and_target():
    df = pd.DataFrame({"month_end": ["m1"] * 4 + ["m2"] * 4, "x": [1, 2, 3, np.nan, 10, 20, 30, 40],
                       "ret_fwd_1m": [0.1, 0.2, -0.1, 0.0, 0.3, 0.1, 0.2, 0.0], "ff12": [1, 1, 2, 2] * 2})
    r = cs_rank(df, ["x"])
    assert r["x"].between(-0.5, 0.5).all() or r["x"].isna().any()
    assert np.isnan(r.loc[3, "x"])
    t = make_target(df)
    assert abs(t.groupby([df["month_end"], df["ff12"]]).sum()).max() < 1e-12
