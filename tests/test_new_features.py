"""Tests for the pieces added for the combined rerun: S5 effect-size mode, the external (limma-voom) ranker, the
covariate option, the tie-break and recurrence functions, intervals, permutations and the fixed-feature baseline."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from evd.fixedfeat import fixed_features_cv
from evd.metrics import threshold_metrics
from evd.nested import run_nested_cv
from evd.runs import permute_labels
from evd.selection import Audit, PrecomputedRanker, welch_rank
from evd.splits import inner_splits, outer_splits
from evd.stability import gene_stats, recurrence, tie_break_sort
from evd.stats import delong_ci, delong_paired, paired_bootstrap_diff, repeated_cv_bootstrap, wilson_ci
from conftest import SMALL, make_noise
from test_leakage import leakage_violations

SEED = 42


def _signal_ds(seed=0, n=40, G=400, n_signal=3):
    ds = make_noise(seed, n=n, G=G)
    counts = ds.counts.copy()
    counts[ds.y == 1, :n_signal] = (counts[ds.y == 1, :n_signal] * 4)
    from evd.data import Dataset
    return Dataset.from_arrays(counts, ds.y)


def test_effect_mode_keeps_only_genes_with_lfc_at_least_one_computed_on_the_split():
    ds = _signal_ds()
    idx = np.arange(0, 40, 2)
    rk = welch_rank(ds, idx, ds.y[idx], gate_lfc=1.0)
    sel = rk.select(50, "effect")
    assert 1 <= len(sel) <= 50
    X = ds.LC[idx]
    lfc = X[ds.y[idx] == 1].mean(0) - X[ds.y[idx] == 0].mean(0)
    assert np.all(np.abs(lfc[sel]) >= 1.0)
    assert np.array_equal(sel, rk.order[np.abs(rk.lfc) >= 1.0][:50])     # filter first, then the |t| order
    t = rk.t[np.isin(rk.order, sel)]
    assert np.all(np.diff(np.abs(t)) <= 1e-12)                       # ranked by |t|


def test_effect_mode_constant_half_when_nothing_passes_and_audit_clean():
    ds = make_noise(3)
    r = run_nested_cv(ds, ds.y, SEED, replace(SMALL, mode="effect", gate_lfc=100.0))
    assert r["pooled_auc"] == 0.5 and all(f["n_selected"] == 0 for f in r["folds"])
    audit = Audit()
    mc = replace(SMALL, mode="effect")
    run_nested_cv(ds, ds.y, SEED, mc, audit=audit)
    assert leakage_violations(audit, outer_splits(ds.y, SEED)) == []


def test_precomputed_ranker_reproduces_welch_and_refuses_other_samples():
    ds = make_noise(2, n=40, G=300)
    y = ds.y
    tables, train = {}, {}

    def tab(idx):
        rk = welch_rank(ds, idx, y[idx])
        return pd.DataFrame({"GeneID": ds.gene_ids[rk.order], "stat": rk.t, "log2FC": rk.lfc, "padj": rk.padj})
    for f, (tr, _) in enumerate(outer_splits(y, SEED), 1):
        tables[f"outer{f}"], train[f"outer{f}"] = tab(tr), tr
        for j, (a, b) in enumerate(inner_splits(y[tr], SEED, 3), 1):
            tables[f"outer{f}_inner{j}"], train[f"outer{f}_inner{j}"] = tab(tr[a]), tr[a]
    ranker = PrecomputedRanker(tables, train, ds.gene_ids)
    a = run_nested_cv(ds, y, SEED, SMALL)
    b = run_nested_cv(ds, y, SEED, replace(SMALL, ranker=ranker))
    assert np.array_equal(a["oof"], b["oof"])
    with pytest.raises(ValueError):
        ranker(ds, np.arange(10), y[:10], None, 0.05, 1.0, None, ("outer", 1))


def test_covariate_is_appended_after_the_genes():
    ds = make_noise(4, n=40, G=300)
    ds.cov = np.random.default_rng(0).normal(size=(40, 1))
    with_cov = run_nested_cv(ds, ds.y, SEED, replace(SMALL, use_cov=True))
    without = run_nested_cv(ds, ds.y, SEED, SMALL)
    for a, b in zip(with_cov["folds"], without["folds"]):
        assert len(a["coef"]) == len(a["sel"]) + 1 and len(b["coef"]) == len(b["sel"])


def test_tie_break_order_follows_the_approved_rule():
    def make_ds(symbols):
        class DS:
            pass
        d = DS()
        d.symbols, d.gene_ids, d.sex_chrom = np.array(symbols), np.array([str(i) for i in range(len(symbols))]), np.zeros(len(symbols), bool)
        return d
    fit = lambda sel, coef, top: {"sel": np.array(sel), "coef": np.array(coef, float), "top_order": np.array(top)}
    # frequencies: A, B, C, D = 2/3 ; E, F = 1/3.   mean |coef|: A .2, B .9, C = D = .5, E .1, F 5.0
    fits = [fit([0, 1, 2, 3], [0.2, 0.9, 0.5, 0.5], [2, 3, 1, 0, 4, 5]),
            fit([0, 1, 2, 3], [0.2, 0.9, 0.5, 0.5], [2, 3, 1, 0, 4, 5]),
            fit([4, 5], [0.1, 5.0], [4, 5, 0, 1, 2, 3])]
    order = tie_break_sort(gene_stats(fits, make_ds(list("ABCDEF")))).symbol.tolist()
    # rule 1: A-D before E-F even though F has the largest coefficient; rule 2: B (.9) > C = D (.5) > A (.2), F > E;
    # rule 3: C (median rank 1) before D (median rank 2)
    assert order == ["B", "C", "D", "A", "F", "E"]
    # rule 4: identical frequency, coefficient and median rank (orders swapped between fits) -> alphabetical
    tie = gene_stats([fit([0, 1], [0.5, 0.5], [0, 1]), fit([0, 1], [0.5, 0.5], [1, 0])], make_ds(["B", "A"]))
    assert tie_break_sort(tie).symbol.tolist() == ["A", "B"]


def test_recurrence_measures():
    fit = lambda sel, coef: {"sel": np.array(sel), "coef": np.array(coef, float), "top_order": np.array(sel)}
    fits = [fit([0, 1, 2], [1, 1, 1]), fit([0, 1], [1, 1]), fit([0, 1, 2], [1, 0, 1]), fit([3, 4], [1, 1])]
    rec, pre = recurrence(fits, np.array([0, 1, 2]), 3)
    assert rec["exact_equal_topK_set"] == 0.5 and rec["contains_panel_topK_set"] == 0.5
    assert rec["contains_panel_nonzero_set"] == 0.25
    assert pre.loc[pre.k == 1, "prefix_contained_in_topK_set"].iloc[0] == 0.75


def test_intervals_and_paired_tests_are_consistent():
    rng = np.random.default_rng(0)
    y = np.array([0] * 20 + [1] * 15)
    s = rng.normal(size=35) + y
    s2 = rng.normal(size=35) + 0.5 * y
    d = delong_paired(y, s, s)
    assert d["diff"] == 0 and d["p"] == 1.0
    assert abs(delong_paired(y, s, s2)["auc_a"] - delong_ci(y, s)["auc"]) < 1e-12
    folds = np.tile(np.arange(5).repeat(7), (3, 1))
    oofs = np.vstack([s, s, s])
    assert paired_bootstrap_diff(y, oofs, folds, oofs, folds, B=50)["estimate"] == 0
    lo, hi = wilson_ci(9, 10)
    assert 0.55 < lo < 0.62 and 0.97 < hi < 0.99
    b = repeated_cv_bootstrap(y, oofs, folds, B=100)
    assert b["boot_lo"] <= b["boot_hi"] and b["B"] == 100


def test_permutation_is_reproducible_and_keeps_strata():
    y = np.array([0] * 20 + [1] * 15)
    sex = np.array([0, 1] * 17 + [0])
    assert np.array_equal(permute_labels(y, 7), permute_labels(y, 7))
    assert not np.array_equal(permute_labels(y, 7), permute_labels(y, 8))
    p = permute_labels(y, 3, sex)
    for s in (0, 1):
        assert p[sex == s].sum() == y[sex == s].sum()          # class counts preserved inside each stratum
    assert p.sum() == y.sum()


def test_threshold_metrics_and_fixed_feature_baseline():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.array([0.1, 0.2, 0.6, 0.9, 0.8, 0.4, 0.7, 0.3])
    m = threshold_metrics(y, p)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (3, 1, 3, 1)
    assert m["sensitivity"] == 0.75 and m["specificity"] == 0.75 and abs(m["brier"] - np.mean((p - y) ** 2)) < 1e-12
    rng = np.random.default_rng(1)
    yy = np.array([0] * 30 + [1] * 30)
    F = (yy[:, None] * 2.0 + rng.normal(size=(60, 3)))
    assert fixed_features_cv(F, yy, SEED, [0.1, 1.0])["pooled_auc"] > 0.9
    noise = fixed_features_cv(rng.normal(size=(60, 3)), yy, SEED, [0.1, 1.0])["pooled_auc"]
    assert 0.25 < noise < 0.75
