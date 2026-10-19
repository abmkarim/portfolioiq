"""SHAP analysis: per-fold TreeExplainer, global importance, beeswarm, stability across folds."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def shap_values(model, X: pd.DataFrame, n: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """TreeExplainer SHAP values on a deterministic sample of ``X``."""
    import shap

    Xs = X.sample(min(n, len(X)), random_state=seed) if len(X) > n else X
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(Xs), Xs


def importance_table(values: np.ndarray, features: list[str], fold_year: int, groups: dict[str, str]) -> pd.DataFrame:
    imp = np.abs(values).mean(axis=0)
    df = pd.DataFrame({"fold_year": fold_year, "feature": features, "mean_abs_shap": imp})
    df["feature_group"] = df["feature"].map(groups).fillna("other")
    df["rank"] = df["mean_abs_shap"].rank(ascending=False, method="first").astype(int)
    return df


def beeswarm(values: np.ndarray, X: pd.DataFrame, path: Path, title: str) -> Path:
    import shap

    plt.figure()
    shap.summary_plot(values, X, show=False, max_display=15)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return path


def stability_figure(imp: pd.DataFrame, path: Path) -> tuple[Path, float]:
    """Heatmap of normalised importances across folds + mean Spearman corr between consecutive folds."""
    piv = imp.pivot(index="feature", columns="fold_year", values="mean_abs_shap").fillna(0.0)
    piv = piv / piv.sum(axis=0).replace(0, np.nan)
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    cols = list(piv.columns)
    corrs = [piv[a].rank().corr(piv[b].rank()) for a, b in zip(cols[:-1], cols[1:], strict=True)]
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 3, 0.32 * len(piv) + 1.5))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols)), [str(c) for c in cols])
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_title(f"Part de l'importance SHAP par fold (corr. rang moy. entre folds = {np.nanmean(corrs):.2f})")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, float(np.nanmean(corrs)) if corrs else float("nan")


def global_importance_figure(imp: pd.DataFrame, path: Path) -> Path:
    g = imp.groupby(["feature", "feature_group"], as_index=False)["mean_abs_shap"].mean()
    g = g.sort_values("mean_abs_shap")
    colors = {"finbert": "#4C72B0", "word2vec": "#DD8452", "tfidf_length": "#55A868", "controls": "#8C8C8C"}
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(g) + 1.2))
    ax.barh(g["feature"], g["mean_abs_shap"], color=[colors.get(x, "#999") for x in g["feature_group"]])
    ax.set_xlabel("moyenne |SHAP| (moyenne sur les folds)")
    ax.set_title("Importance globale des features")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, list(colors), loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
