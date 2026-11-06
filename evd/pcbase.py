"""PC-only baseline classifier (item 6), nested CV with everything fitted inside the training split.

Inside each training split: low-expression filter -> top-N genes by training-split variance of
log-CPM -> PCA fitted on the training samples -> L2 logistic regression on the first n_pc scores.
n_pc and C are tuned by the same inner scheme as the primary pipeline.
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .splits import inner_splits, outer_splits

warnings.filterwarnings("ignore")


def _pc_space(ds, idx, y_idx, top_n, max_pc):
    min_group = int(np.bincount(y_idx, minlength=2).min())
    keep = np.flatnonzero(ds.GE[idx].sum(0) >= min_group)
    X = ds.LC[np.ix_(idx, keep)]
    top = keep[np.argsort(-X.var(0, ddof=1), kind="stable")[:top_n]]
    Xt = ds.LC[np.ix_(idx, top)]
    mu = Xt.mean(0)
    _, _, vt = np.linalg.svd(Xt - mu, full_matrices=False)
    return top, mu, vt[:max_pc]


def _fit_lr(S, y, C, seed):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed).fit(S, y)


def pc_baseline_cv(ds, y, seed, cfg, n_pcs=None):
    """n_pcs: override of the PC grid (S7 uses up to 20 PCs)."""
    c6, mcfg = cfg["item6"], cfg["model"]
    n_pcs = list(n_pcs) if n_pcs is not None else c6["n_pc_grid"]
    Cs, top_n = c6["c_grid"], c6["top_var_genes"]
    oof = np.full(len(y), np.nan)
    fold_id = np.zeros(len(y), int)
    folds = []
    for f, (tr, te) in enumerate(outer_splits(y, seed, mcfg["outer_splits"]), 1):
        y_tr = y[tr]
        scores = np.zeros((len(n_pcs), len(Cs), mcfg["inner_splits"]))
        for j, (a, b) in enumerate(inner_splits(y_tr, seed, mcfg["inner_splits"])):
            itr, iva = tr[a], tr[b]
            top, mu, vt = _pc_space(ds, itr, y[itr], top_n, max(n_pcs))
            Str = (ds.LC[np.ix_(itr, top)] - mu) @ vt.T
            Sva = (ds.LC[np.ix_(iva, top)] - mu) @ vt.T
            for p, npc in enumerate(n_pcs):
                for q, C in enumerate(Cs):
                    m = _fit_lr(Str[:, :npc], y[itr], C, seed)
                    scores[p, q, j] = roc_auc_score(y[iva], m.predict_proba(Sva[:, :npc])[:, 1])
        mean = scores.mean(2)
        best = min(((-round(mean[p, q], 12), n_pcs[p], Cs[q]) for p in range(len(n_pcs)) for q in range(len(Cs))))
        _, npc, C = best
        top, mu, vt = _pc_space(ds, tr, y_tr, top_n, max(n_pcs))
        Str = (ds.LC[np.ix_(tr, top)] - mu) @ vt.T
        Ste = (ds.LC[np.ix_(te, top)] - mu) @ vt.T
        m = _fit_lr(Str[:, :npc], y_tr, C, seed)
        p_te = m.predict_proba(Ste[:, :npc])[:, 1]
        oof[te] = p_te
        fold_id[te] = f
        folds.append({"seed": seed, "fold": f, "n_pc": npc, "C": C, "inner_auc": float(-best[0]),
                      "auc": float(roc_auc_score(y[te], p_te)), "ap": float(average_precision_score(y[te], p_te))})
    return {"seed": seed, "oof": oof, "fold_id": fold_id, "folds": folds,
            "pooled_auc": float(roc_auc_score(y, oof)), "pooled_ap": float(average_precision_score(y, oof)),
            "mean_fold_auc": float(np.mean([r["auc"] for r in folds]))}
