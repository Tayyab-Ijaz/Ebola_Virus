"""Nested CV for a FIXED feature matrix (no supervised gene selection): sex-only and composition-only
baselines. Standardisation and the L2 logistic regression are fitted on the training samples of each split;
C is tuned on the same inner scheme as the primary pipeline."""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .splits import inner_splits, outer_splits

warnings.filterwarnings("ignore")


def _fit(F, y, C, seed):
    sc = StandardScaler().fit(F)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed).fit(sc.transform(F), y)
    return sc, lr


def _auc_or_half(y, s):
    return roc_auc_score(y, s) if len(set(y)) == 2 else 0.5


def fixed_features_cv(F, y, seed, c_grid, n_outer=5, n_inner=3):
    F, y = np.asarray(F, float), np.asarray(y)
    oof = np.full(len(y), np.nan)
    fold_id = np.zeros(len(y), int)
    folds = []
    for f, (tr, te) in enumerate(outer_splits(y, seed, n_outer), 1):
        sc_tab = np.zeros((len(c_grid), n_inner))
        for j, (a, b) in enumerate(inner_splits(y[tr], seed, n_inner)):
            itr, iva = tr[a], tr[b]
            for q, C in enumerate(c_grid):
                sc, lr = _fit(F[itr], y[itr], C, seed)
                sc_tab[q, j] = _auc_or_half(y[iva], lr.predict_proba(sc.transform(F[iva]))[:, 1])
        mean = sc_tab.mean(1)
        q = min(range(len(c_grid)), key=lambda i: (-round(float(mean[i]), 12), c_grid[i]))
        sc, lr = _fit(F[tr], y[tr], c_grid[q], seed)
        p = lr.predict_proba(sc.transform(F[te]))[:, 1]
        oof[te], fold_id[te] = p, f
        folds.append({"seed": seed, "fold": f, "C": c_grid[q], "inner_auc": float(mean[q]),
                      "auc": float(roc_auc_score(y[te], p)), "ap": float(average_precision_score(y[te], p))})
    return {"seed": seed, "oof": oof, "fold_id": fold_id, "folds": folds,
            "pooled_auc": float(roc_auc_score(y, oof)), "pooled_ap": float(average_precision_score(y, oof)),
            "fold_auc": [r["auc"] for r in folds], "mean_fold_auc": float(np.mean([r["auc"] for r in folds]))}
