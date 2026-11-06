"""Rank-agreement metrics, DeLong CI and subject-bootstrap CI for repeated-CV AUC."""
import numpy as np
from scipy import stats as sps


def spearman_abs(a, b):
    return float(sps.spearmanr(np.abs(a), np.abs(b)).statistic)


def topn_overlap(score_a, score_b, n):
    """Number of genes shared by the top-n by |score| of two aligned score vectors."""
    ta = set(np.argsort(-np.abs(score_a), kind="stable")[:n])
    tb = set(np.argsort(-np.abs(score_b), kind="stable")[:n])
    return len(ta & tb)


def delong_ci(y, s, alpha=0.05):
    """DeLong AUC, SE and normal-approximation CI (clipped to [0, 1])."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    v10 = np.array([(neg < p).mean() + 0.5 * (neg == p).mean() for p in pos])
    v01 = np.array([(pos > q).mean() + 0.5 * (pos == q).mean() for q in neg])
    auc = v10.mean()
    se = np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    z = sps.norm.ppf(1 - alpha / 2)
    return {"auc": float(auc), "se": float(se), "lo": float(max(0, auc - z * se)),
            "hi": float(min(1, auc + z * se))}


def _pair_matrix(sp, sn):
    return (sp[:, None] > sn[None, :]).astype(float) + 0.5 * (sp[:, None] == sn[None, :])


def repeated_cv_bootstrap(y, oofs, fold_ids, B=2000, seed=7, return_stats=False):
    """Stratified subject bootstrap of the repeated-CV statistic
    'mean over repeats of (mean over folds of fold AUC)'.

    oofs, fold_ids: arrays (R, n). Bootstrap multiplicities act as sample weights inside every
    (repeat, fold) test set. Pass R = 1 and fold_ids = 0 for the pooled OOF AUC of a single run.
    Returns point estimate (no resampling) and percentile CI; refit variance is NOT included.
    """
    y = np.asarray(y)
    oofs = np.atleast_2d(oofs)
    fold_ids = np.atleast_2d(fold_ids)
    R, n = oofs.shape
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    W = np.zeros((B + 1, n))
    W[0] = 1.0                                      # row 0 = observed data
    for b in range(1, B + 1):
        W[b] = (np.bincount(rng.choice(i0, len(i0)), minlength=n) +
                np.bincount(rng.choice(i1, len(i1)), minlength=n))
    per_repeat = np.zeros((B + 1, R))
    for r in range(R):
        tot, cnt = np.zeros(B + 1), np.zeros(B + 1)
        for f in np.unique(fold_ids[r]):
            T = np.flatnonzero(fold_ids[r] == f)
            pos, neg = T[y[T] == 1], T[y[T] == 0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            M = _pair_matrix(oofs[r, pos], oofs[r, neg])
            Wp, Wn = W[:, pos], W[:, neg]
            num = ((Wp @ M) * Wn).sum(1)
            den = Wp.sum(1) * Wn.sum(1)
            ok = den > 0
            tot[ok] += num[ok] / den[ok]
            cnt[ok] += 1
        per_repeat[:, r] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    stat = np.nanmean(per_repeat, axis=1)
    if return_stats:
        return stat                                  # row 0 = observed, rows 1..B = bootstrap replicates
    lo, hi = np.nanpercentile(stat[1:], [2.5, 97.5])
    return {"estimate": float(stat[0]), "boot_lo": float(lo), "boot_hi": float(hi), "B": B,
            "boot_sd": float(np.nanstd(stat[1:], ddof=1))}


def paired_bootstrap_diff(y, oofs_a, folds_a, oofs_b, folds_b, B=2000, seed=7):
    """Paired subject bootstrap of stat(A) - stat(B), stat = mean over repeats of mean-fold AUC.
    Both models see the SAME bootstrap multiplicities (the generator depends only on y and seed)."""
    a = repeated_cv_bootstrap(y, oofs_a, folds_a, B, seed, return_stats=True)
    b = repeated_cv_bootstrap(y, oofs_b, folds_b, B, seed, return_stats=True)
    d = a - b
    lo, hi = np.nanpercentile(d[1:], [2.5, 97.5])
    return {"estimate": float(d[0]), "boot_lo": float(lo), "boot_hi": float(hi), "B": B,
            "boot_sd": float(np.nanstd(d[1:], ddof=1)), "frac_boot_gt0": float(np.nanmean(d[1:] > 0))}


def delong_paired(y, s_a, s_b, alpha=0.05):
    """DeLong test for two correlated AUCs measured on the same subjects. Returns AUC difference A - B,
    its SE, normal-approximation CI and two-sided p."""
    y = np.asarray(y)
    sa, sb = np.asarray(s_a, float), np.asarray(s_b, float)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    comps = []
    for s in (sa, sb):
        M = _pair_matrix(s[pos], s[neg])                      # m x n
        comps.append((M.mean(1), M.mean(0)))                  # V10 (per positive), V01 (per negative)
    aucs = [c[0].mean() for c in comps]
    s10 = np.cov(np.vstack([comps[0][0], comps[1][0]]))
    s01 = np.cov(np.vstack([comps[0][1], comps[1][1]]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    d = aucs[0] - aucs[1]
    se = float(np.sqrt(max(var, 0.0)))
    z = sps.norm.ppf(1 - alpha / 2)
    p = float(2 * (1 - sps.norm.cdf(abs(d) / se))) if se > 0 else (1.0 if d == 0 else 0.0)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": float(d), "se": se,
            "lo": float(d - z * se), "hi": float(d + z * se), "p": p}


def wilson_ci(k, n, alpha=0.05):
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def repeat_summary(per_repeat_values):
    v = np.asarray(per_repeat_values, float)
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
            "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5)), "n": int(len(v))}
