"""Leakage tests for the nested CV (item 1). They must pass before any item 1 result is reported.

What is proved here
  * audit: no outer-test sample is ever used to rank, select, tune or fit; inside every inner split the
    gene list is derived from the inner-TRAINING samples only (Reviewer 3's point);
  * the audit has power: it flags the legacy structure (S4, ranking once per outer fold);
  * behaviour: changing the labels or the expression of the outer-test samples changes nothing except
    the predictions for those samples; changing the labels of an inner-validation fold does not change
    that inner split's ranking;
  * equivalence: the hand-written inner loop equals a standard sklearn Pipeline + GridSearchCV in which
    the selector is a pipeline step;
  * a noise-only dataset gives chance-level AUC with honest selection and a high AUC with leaky selection.
"""
from dataclasses import replace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import evd.nested as nested
from evd.data import Dataset
from evd.nested import _best, run_nested_cv, run_outer_fold, tune_inner
from evd.selection import Audit, WelchTopK, welch_rank
from evd.splits import inner_splits, outer_splits
from conftest import SMALL, make_noise

SEED = 42


def leakage_violations(audit, splits):
    te = {f: set(map(int, t)) for f, (_, t) in enumerate(splits, 1)}
    tr = {f: set(map(int, a)) for f, (a, _) in enumerate(splits, 1)}
    viol, inner = [], {}
    for kind, tag, s in audit.records:
        f = tag[1]
        if kind in ("rank", "fit", "select_source"):
            if s & te[f]:
                viol.append(f"{kind} {tag} uses outer-test samples")
            if not s <= tr[f]:
                viol.append(f"{kind} {tag} uses samples outside the outer-training fold")
        if tag[0] == "inner":
            inner.setdefault((f, tag[2]), {}).setdefault(kind, []).append(s)
    for (f, j), d in inner.items():
        ev = d["eval"][0]
        for kind in ("rank", "fit", "select_source"):
            for s in d.get(kind, []):
                if s & ev:
                    viol.append(f"inner split ({f},{j}): {kind} uses inner-validation samples")
        if d["select_source"][0] != d["fit"][0]:
            viol.append(f"inner split ({f},{j}): gene list not derived from exactly the inner-training samples")
    return viol


def test_audit_clean_for_corrected_pipeline(real):
    y = real.y
    audit = Audit()
    run_nested_cv(real, y, SEED, SMALL, audit=audit)
    kinds = {k for k, _, _ in audit.records}
    assert {"rank", "fit", "eval", "select_source"} <= kinds
    assert leakage_violations(audit, outer_splits(y, SEED)) == []
    n_inner = sum(1 for k, t, _ in audit.records if k == "select_source")
    assert n_inner == 5 * 3


def test_audit_has_power_it_flags_the_legacy_structure(real):
    y = real.y
    audit = Audit()
    run_nested_cv(real, y, SEED, replace(SMALL, selection_level="outer"), audit=audit)
    viol = leakage_violations(audit, outer_splits(y, SEED))
    assert any("inner-validation samples" in v for v in viol)
    assert not any("outer-test" in v for v in viol)       # the legacy structure only leaks into the inner scores


def _same_fit(a, b):
    assert np.array_equal(a["sel"], b["sel"])
    assert (a["K"], a["C"], a["l1"]) == (b["K"], b["C"], b["l1"])
    assert a["inner_auc"] == b["inner_auc"]
    assert np.array_equal(a["mean_table"], b["mean_table"])
    assert np.array_equal(a["coef"], b["coef"])
    assert np.array_equal(a["top_order"], b["top_order"])


def test_outer_test_labels_do_not_matter(real):
    y = real.y
    tr, te = outer_splits(y, SEED)[0]
    base = run_outer_fold(real, y, tr, te, SEED, SMALL)
    rng = np.random.default_rng(0)
    y2 = y.copy()
    y2[te] = rng.permutation(y2[te])
    y2[te] = 1 - y2[te]
    assert not np.array_equal(y, y2)
    alt = run_outer_fold(real, y2, tr, te, SEED, SMALL)
    _same_fit(base, alt)
    assert np.array_equal(base["proba"], alt["proba"])


def test_outer_test_expression_only_affects_test_predictions(real):
    y = real.y
    tr, te = outer_splits(y, SEED)[0]
    base = run_outer_fold(real, y, tr, te, SEED, SMALL)
    rng = np.random.default_rng(1)
    LC, GE, counts = real.LC.copy(), real.GE.copy(), real.counts.copy()
    for i in te:                                        # scramble every test sample across genes
        p = rng.permutation(real.G)
        LC[i], GE[i], counts[i] = LC[i][p], GE[i][p], counts[i][p]
    alt = run_outer_fold(replace(real, LC=LC, GE=GE, counts=counts), y, tr, te, SEED, SMALL)
    _same_fit(base, alt)
    assert not np.array_equal(base["proba"], alt["proba"])      # the test data do reach the predictions


def test_inner_ranking_ignores_inner_validation_labels(real, monkeypatch):
    """Flip the labels of the inner-validation samples of split 0 (splits held fixed): the ranking made in
    that inner split must not change, the rankings of the other inner splits (which train on them) must."""
    y = real.y
    tr, _ = outer_splits(y, SEED)[0]
    fixed = inner_splits(y[tr], SEED, 3)
    monkeypatch.setattr(nested, "inner_splits", lambda y_tr, seed, n: fixed)
    val0 = tr[fixed[0][1]]
    y2 = y.copy()
    y2[val0] = 1 - y2[val0]

    def rankings(yy):
        seen, orig = {}, nested.welch_rank

        def spy(ds, idx, y_idx, allowed, fdr, lfc, audit, tag):
            rk = orig(ds, idx, y_idx, allowed, fdr, lfc, audit, tag)
            seen[tag] = (rk.order[:200].copy(), rk.t[:200].copy())
            return rk
        monkeypatch.setattr(nested, "welch_rank", spy)
        tune_inner(real, yy, tr, SEED, SMALL, tag=1)
        monkeypatch.setattr(nested, "welch_rank", orig)
        return seen

    a, b = rankings(y), rankings(y2)
    assert np.array_equal(a[("inner", 1, 0)][0], b[("inner", 1, 0)][0])
    assert np.array_equal(a[("inner", 1, 0)][1], b[("inner", 1, 0)][1])
    for j in (1, 2):
        assert not np.array_equal(a[("inner", 1, j)][1], b[("inner", 1, j)][1])


def test_equals_sklearn_pipeline_with_selector_inside_gridsearch(real):
    y = real.y
    tr, _ = outer_splits(y, SEED)[0]
    mine = tune_inner(real, y, tr, SEED, SMALL, tag=1)["mean_table"][:, 0, 0]
    pipe = Pipeline([("sel", WelchTopK()), ("sc", StandardScaler()),
                     ("lr", LogisticRegression(penalty="elasticnet", solver="saga", C=0.3, l1_ratio=0.5,
                                               max_iter=5000, class_weight="balanced", random_state=SEED))])
    gs = GridSearchCV(pipe, {"sel__k": list(SMALL.k_grid)}, scoring="roc_auc", refit=False,
                      cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
    gs.fit(real.counts[tr], y[tr])
    theirs = gs.cv_results_["mean_test_score"]
    assert np.allclose(mine, theirs, atol=1e-9), (mine, theirs)


def test_low_expression_filter_is_fitted_on_the_training_samples_only():
    rng = np.random.default_rng(3)
    counts = rng.poisson(50, (20, 30))
    counts[:, 0] = 0
    counts[10:, 0] = 500                                 # gene 0 expressed (>=10) in exactly 10 samples = smaller class size
    y = np.array([0, 1] * 10)
    ds = Dataset.from_arrays(counts, y)
    assert 0 in welch_rank(ds, np.arange(20), y).order   # 10 >= 10 -> kept when all samples are available
    tr = np.arange(18)                                   # hold out samples 18, 19 (both expressing): classes 9/9, gene in 8 samples
    assert 0 not in welch_rank(ds, tr, y[tr]).order      # 8 < 9 -> filtered out using the training samples only


def test_ranking_is_welch_t_sorted_and_select_is_a_prefix(real):
    from scipy import stats
    idx = np.arange(0, 59, 2)
    rk = welch_rank(real, idx, real.y[idx])
    assert np.all(np.diff(np.abs(rk.t)) <= 1e-12)
    j = rk.order[:5]
    t_ref = stats.ttest_ind(real.LC[idx][real.y[idx] == 1][:, j], real.LC[idx][real.y[idx] == 0][:, j],
                            equal_var=False).statistic
    assert np.allclose(rk.t[:5], t_ref)
    assert np.array_equal(rk.select(3), rk.select(10)[:3])


def test_tie_break_prefers_smaller_k_then_c_then_l1():
    from evd.nested import ModelCfg
    mc = ModelCfg(k_grid=(1, 2, 3), c_grid=(0.1, 1.0), l1_grid=(0.2, 0.8))
    mean = np.full((3, 2, 2), 0.90)
    mean[2, 1, 1] = 0.95
    mean[1, 0, 1] = 0.95
    mean[1, 1, 0] = 0.95
    mean[1, 1, 1] = 0.95 + 1e-14                         # below the 1e-12 tolerance -> still a tie
    K, C, l1, best = _best(mean, mc)
    assert (K, C, l1) == (2, 0.1, 0.8)


def test_gate_mode_without_passing_genes_gives_constant_half_but_rank_mode_always_selects():
    ds = make_noise(5)
    y = ds.y
    gate = run_nested_cv(ds, y, SEED, replace(SMALL, mode="gate"))
    assert all(f["n_selected"] == 0 for f in gate["folds"])
    assert gate["pooled_auc"] == 0.5
    rank = run_nested_cv(ds, y, SEED, SMALL)
    assert all(f["n_selected"] >= 1 for f in rank["folds"])


def test_sex_chromosome_exclusion(real):
    y = real.y
    r = run_nested_cv(real, y, SEED, replace(SMALL, exclude_sex_chrom=True))
    for f in r["folds"]:
        assert not real.sex_chrom[f["sel"]].any()
        assert not real.sex_chrom[f["top_order"]].any()


def test_deterministic_and_independent_of_n_jobs(real):
    y = real.y
    a = run_nested_cv(real, y, SEED, SMALL, n_jobs=1)
    b = run_nested_cv(real, y, SEED, SMALL, n_jobs=1)
    c = run_nested_cv(real, y, SEED, SMALL, n_jobs=2)
    assert np.array_equal(a["oof"], b["oof"]) and np.array_equal(a["oof"], c["oof"])


def test_noise_gives_chance_with_honest_selection_and_high_auc_with_leaky_selection():
    """40 samples x 5000 genes of pure noise. 'Leaky' = a panel-sized set (5 genes, as in the real study) is
    chosen with the labels of ALL samples before CV. Calibrated beforehand on 10 datasets: honest pooled AUC
    0.54 (0.42-0.71), leaky 0.94 (0.92-0.99); the assertions below use the first 6 of those datasets."""
    honest, leaky = [], []
    for s in range(6):
        ds = make_noise(s, G=5000)
        y = ds.y
        honest.append(run_nested_cv(ds, y, SEED, SMALL)["pooled_auc"])
        rk = welch_rank(ds, np.arange(ds.n), y)             # selection on ALL samples, then CV
        keep = np.zeros(ds.G, bool)
        keep[rk.order[:5]] = True
        leaky.append(run_nested_cv(ds.subset_genes(keep), y, SEED, SMALL)["pooled_auc"])
    assert 0.40 < np.mean(honest) < 0.65, honest
    assert np.mean(leaky) > 0.85, leaky
    assert np.mean(leaky) - np.mean(honest) > 0.30
