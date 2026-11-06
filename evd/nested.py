"""Nested CV with the supervised selection step refitted inside every inner AND outer training split.

Unit of work = run_outer_fold(): it receives the full label vector but reads only y[train];
test labels are used afterwards (in run_nested_cv) purely for scoring.
"""
import warnings
from dataclasses import dataclass, replace

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .selection import welch_rank
from .splits import inner_splits, outer_splits

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class ModelCfg:
    k_grid: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)
    c_grid: tuple = (0.05, 0.1, 0.3, 1, 3, 10)
    l1_grid: tuple = (0.2, 0.5, 0.8)
    max_iter: int = 20000
    outer_splits: int = 5
    inner_splits: int = 3
    mode: str = "rank"                 # "rank" top-K by |t| | "gate" FDR+LFC gate (S1) | "effect" |LFC|>=1 then rank (S5)
    selection_level: str = "inner"     # "inner" = refit in every inner split ; "outer" = legacy structure (S4)
    exclude_sex_chrom: bool = False    # S3
    use_cov: bool = False              # append ds.cov (inferred sex) to the selected genes
    ranker: object = None              # None = Welch; PrecomputedRanker for S2
    gate_fdr: float = 0.05
    gate_lfc: float = 1.0

    @classmethod
    def from_config(cls, cfg, **over):
        m = cfg["model"]
        base = cls(tuple(m["k_grid"]), tuple(m["c_grid"]), tuple(m["l1_grid"]), m["max_iter"], m["outer_splits"],
                   m["inner_splits"], gate_fdr=cfg["selection"]["gate_fdr"], gate_lfc=cfg["selection"]["gate_lfc"])
        return replace(base, **over) if over else base


class ConstantModel:
    coef_ = None

    def predict_proba1(self, X):
        return np.full(len(X), 0.5)


class FittedModel:
    def __init__(self, scaler, lr):
        self.scaler, self.lr = scaler, lr
        self.coef_ = lr.coef_.ravel()           # coefficients on standardised features

    def predict_proba1(self, X):
        return self.lr.predict_proba(self.scaler.transform(X))[:, 1]


def fit_model(X, y, C, l1, seed, max_iter):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1, max_iter=max_iter,
                            class_weight="balanced", random_state=seed).fit(sc.transform(X), y)
    return FittedModel(sc, lr)


def _X(ds, rows, cols, mc):
    X = ds.LC[np.ix_(rows, cols)]
    if mc.use_cov:
        X = np.hstack([X, ds.cov[rows]])
    return X


def _best(mean, mc):
    """Highest mean inner AUC; ties (|diff| < 1e-12) -> smaller K, then smaller C, then smaller l1."""
    cands = [(-round(float(mean[a, b, c]), 12), mc.k_grid[a], mc.c_grid[b], mc.l1_grid[c])
             for a in range(len(mc.k_grid)) for b in range(len(mc.c_grid)) for c in range(len(mc.l1_grid))]
    s, K, C, l1 = min(cands)
    return K, C, l1, -s


def _rank(mc, ds, idx, y_idx, allowed, audit, tag):
    fn = mc.ranker or welch_rank
    return fn(ds, idx, y_idx, allowed, mc.gate_fdr, mc.gate_lfc, audit, tag)


def tune_inner(ds, y, tr, seed, mc, audit=None, tag=0, rk_outer=None, inner=None):
    """Inner-CV tuning on the samples `tr`. `inner` = optional list of (train_pos, val_pos) positions into `tr`
    (default: StratifiedKFold(inner_splits, shuffle, seed)). Returns the score table and chosen params."""
    y_tr = y[tr]
    allowed = ~ds.sex_chrom if mc.exclude_sex_chrom else None
    ks = list(mc.k_grid)
    kmax = max(ks)
    inner = inner if inner is not None else inner_splits(y_tr, seed, mc.inner_splits)
    scores = np.full((len(ks), len(mc.c_grid), len(mc.l1_grid), len(inner)), np.nan)
    info = []
    for j, (a, b) in enumerate(inner):
        itr, iva = tr[a], tr[b]
        ctx = ("inner", tag, j)
        if mc.selection_level == "inner":
            rk = _rank(mc, ds, itr, y[itr], allowed, audit, ctx)
        else:                                   # legacy structure: ranking from the whole outer-training fold
            rk = rk_outer
        if audit is not None:                   # which samples determined the gene list used in this inner split
            audit.record("select_source", ctx, itr if mc.selection_level == "inner" else tr)
        info.append({"outer_fold": tag, "inner_split": j + 1, "n_train": int(len(itr)), "n_kept": rk.n_kept,
                     "n_pass_gate": rk.n_pass, "n_effect": rk.n_effect})
        sel_max = rk.select(kmax, mc.mode)
        if audit is not None:
            audit.record("fit", ctx, itr)
            audit.record("eval", ctx, iva)
        Xtr = _X(ds, itr, sel_max, mc)
        Xva = _X(ds, iva, sel_max, mc)
        done = {}
        for iK, K in enumerate(ks):
            kk = min(K, len(sel_max))
            if kk not in done:
                res = np.full((len(mc.c_grid), len(mc.l1_grid)), 0.5)
                if kk > 0:
                    extra = Xtr.shape[1] - len(sel_max)          # covariate columns sit after the genes
                    cols = list(range(kk)) + list(range(len(sel_max), len(sel_max) + extra))
                    for iC, C in enumerate(mc.c_grid):
                        for iL, l1 in enumerate(mc.l1_grid):
                            m = fit_model(Xtr[:, cols], y[itr], C, l1, seed, mc.max_iter)
                            res[iC, iL] = roc_auc_score(y[iva], m.predict_proba1(Xva[:, cols]))
                done[kk] = res
            scores[iK, :, :, j] = done[kk]
    mean = scores.mean(3)
    K, C, l1, best = _best(mean, mc)
    return {"K": K, "C": C, "l1": l1, "inner_auc": best, "mean_table": mean, "scores": scores, "info": info}


def run_outer_fold(ds, y, tr, te, seed, mc, audit=None, fold=1, keep_top=200):
    """Selection + tuning + final fit on `tr`; predictions for `te`. Reads y[tr] only."""
    y_tr = y[tr]
    allowed = ~ds.sex_chrom if mc.exclude_sex_chrom else None
    rk = _rank(mc, ds, tr, y_tr, allowed, audit, ("outer", fold))
    tun = tune_inner(ds, y, tr, seed, mc, audit, tag=fold, rk_outer=rk)
    K, C, l1 = tun["K"], tun["C"], tun["l1"]
    sel = rk.select(K, mc.mode)
    if audit is not None:
        audit.record("fit", ("outer-final", fold), tr)
        audit.record("eval", ("outer-final", fold), te)
    if len(sel) == 0:
        model, coef = ConstantModel(), np.array([])
        proba = np.full(len(te), 0.5)
    else:
        model = fit_model(_X(ds, tr, sel, mc), y_tr, C, l1, seed, mc.max_iter)
        coef = model.coef_
        proba = model.predict_proba1(_X(ds, te, sel, mc))
    return {"fold": fold, "te": te, "proba": proba, "K": K, "C": C, "l1": l1, "inner_auc": tun["inner_auc"],
            "n_selected": int(len(sel)), "sel": sel, "coef": coef, "n_kept": rk.n_kept, "n_pass_gate": rk.n_pass,
            "n_effect": rk.n_effect, "inner_info": tun["info"], "top_order": rk.order[:keep_top],
            "top_t": rk.t[:keep_top], "mean_table": tun["mean_table"]}


def run_nested_cv(ds, y, seed, mc, n_jobs=1, audit=None, keep_top=200, splits=None):
    y = np.asarray(y)
    splits = splits if splits is not None else outer_splits(y, seed, mc.outer_splits)
    if audit is not None:
        n_jobs = 1
    if n_jobs == 1:
        folds = [run_outer_fold(ds, y, tr, te, seed, mc, audit, f, keep_top) for f, (tr, te) in enumerate(splits, 1)]
    else:
        folds = Parallel(n_jobs=n_jobs)(delayed(run_outer_fold)(ds, y, tr, te, seed, mc, None, f, keep_top)
                                        for f, (tr, te) in enumerate(splits, 1))
    oof = np.full(len(y), np.nan)
    fold_id = np.zeros(len(y), int)
    for r in folds:
        oof[r["te"]] = r["proba"]
        fold_id[r["te"]] = r["fold"]
        r["auc"] = float(roc_auc_score(y[r["te"]], r["proba"]))
        r["ap"] = float(average_precision_score(y[r["te"]], r["proba"]))
    aucs = [r["auc"] for r in folds]
    return {"seed": seed, "oof": oof, "fold_id": fold_id, "folds": folds,
            "pooled_auc": float(roc_auc_score(y, oof)), "pooled_ap": float(average_precision_score(y, oof)),
            "fold_auc": aucs, "mean_fold_auc": float(np.mean(aucs))}
