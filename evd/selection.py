"""Supervised gene selection (low-expression filter + Welch ranking), always fitted on an explicit
set of training samples.  Nothing in this module can see samples outside `idx`:
the label vector passed in is y[idx], and an optional Audit records exactly which samples each
selection and each model fit used, so tests can prove that held-out samples were never involved."""
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.multitest import multipletests

from .data import Dataset


class Audit:
    """Records which sample indices were used by every selection/fit/evaluation call."""

    def __init__(self):
        self.records = []

    def record(self, kind, tag, idx):
        self.records.append((kind, tag, frozenset(int(i) for i in np.asarray(idx))))


class Ranking:
    """Genes passing the low-expression filter, sorted by |Welch t| (descending, stable)."""

    def __init__(self, order, t, lfc, padj, n_kept, gate_fdr, gate_lfc):
        self.order, self.t, self.lfc, self.padj = order, t, lfc, padj
        self.n_kept = int(n_kept)
        self.gate = (padj < gate_fdr) & (np.abs(lfc) >= gate_lfc)
        self.n_pass = int(self.gate.sum())
        self.eff = np.abs(lfc) >= gate_lfc            # S5: effect-size filter only
        self.n_effect = int(self.eff.sum())

    def select(self, k, mode="rank"):
        """Global column indices of the selected genes (a prefix of one fixed ordering)."""
        if mode == "rank":
            return self.order[:k]
        if mode == "gate":
            return self.order[self.gate][:k]
        if mode == "effect":
            return self.order[self.eff][:k]
        raise ValueError(mode)


def welch_rank(ds, idx, y_idx, allowed=None, gate_fdr=0.05, gate_lfc=1.0, audit=None, tag=None):
    """Rank genes using ONLY the samples in idx (labels y_idx = y[idx])."""
    idx = np.asarray(idx)
    y_idx = np.asarray(y_idx)
    if audit is not None:
        audit.record("rank", tag, idx)
    min_group = int(np.bincount(y_idx, minlength=2).min())
    keep = ds.GE[idx].sum(0) >= min_group          # low-expression filter on the training samples
    if allowed is not None:
        keep &= allowed
    cols = np.flatnonzero(keep)
    X = ds.LC[np.ix_(idx, cols)]
    x1, x0 = X[y_idx == 1], X[y_idx == 0]
    t, p = stats.ttest_ind(x1, x0, axis=0, equal_var=False)
    t = np.nan_to_num(t, nan=0.0)
    p = np.where(np.isnan(p), 1.0, p)
    lfc = x1.mean(0) - x0.mean(0)
    padj = multipletests(p, method="fdr_bh")[1]
    o = np.argsort(-np.abs(t), kind="stable")
    return Ranking(cols[o], t[o], lfc[o], padj[o], len(cols), gate_fdr, gate_lfc)


class PrecomputedRanker:
    """Ranker whose per-split statistics were computed outside Python (limma-voom in R, S2) on exactly the
    splits the nested CV uses. Refuses to answer if the samples it is given differ from the ones the
    table was computed on. Orders genes by |statistic| (descending, stable), like the Welch ranking."""

    def __init__(self, tables, train_sets, gene_ids, gate_fdr=0.05, gate_lfc=1.0):
        pos = {g: i for i, g in enumerate(gene_ids)}
        self.rankings, self.train_sets = {}, {k: frozenset(v) for k, v in train_sets.items()}
        for key, df in tables.items():
            cols = np.array([pos[g] for g in df["GeneID"]])
            t = np.nan_to_num(df["stat"].to_numpy(float), nan=0.0)
            lfc = df["log2FC"].to_numpy(float)
            padj = np.where(np.isnan(df["padj"].to_numpy(float)), 1.0, df["padj"].to_numpy(float))
            o = np.argsort(-np.abs(t), kind="stable")
            self.rankings[key] = Ranking(cols[o], t[o], lfc[o], padj[o], len(cols), gate_fdr, gate_lfc)

    @staticmethod
    def key(tag):
        if tag[0] == "outer":
            return f"outer{tag[1]}"
        if tag[0] == "inner":
            return f"outer{tag[1]}_inner{tag[2] + 1}"
        raise KeyError(tag)

    def __call__(self, ds, idx, y_idx, allowed=None, gate_fdr=0.05, gate_lfc=1.0, audit=None, tag=None):
        key = self.key(tag)
        if frozenset(int(i) for i in np.asarray(idx)) != self.train_sets[key]:
            raise ValueError(f"{key}: samples differ from the ones the external ranking was computed on")
        if allowed is not None:
            raise NotImplementedError("sex-chromosome exclusion is not combined with the external ranker")
        if audit is not None:
            audit.record("rank", tag, np.asarray(idx))
        return self.rankings[key]


class WelchTopK(BaseEstimator, TransformerMixin):
    """sklearn-compatible selector operating on RAW counts (n x G). fit() uses only the rows it
    is given. Used to show that the hand-written inner loop equals a standard
    Pipeline + GridSearchCV in which selection is refitted inside every inner split."""

    def __init__(self, k=5, mode="rank", min_counts=10, gate_fdr=0.05, gate_lfc=1.0):
        self.k, self.mode, self.min_counts = k, mode, min_counts
        self.gate_fdr, self.gate_lfc = gate_fdr, gate_lfc

    def fit(self, X, y):
        X = np.asarray(X)
        ds = Dataset.from_arrays(X, y, min_counts=self.min_counts)
        rk = welch_rank(ds, np.arange(len(y)), np.asarray(y), None, self.gate_fdr, self.gate_lfc)
        self.cols_ = rk.select(self.k, self.mode)
        return self

    def transform(self, X):
        X = np.asarray(X)
        lib = X.sum(1, keepdims=True).astype(float)
        return np.log2(X / lib * 1e6 + 1.0)[:, self.cols_]
