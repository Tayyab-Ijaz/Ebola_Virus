"""Helpers shared by the stages: cached repeated-CV runs of a model variant, a generic statistics summary
(the same for every variant, so all of them can be compared side by side), and the permutation worker.

Interval vocabulary used everywhere (see ANALYSIS_PLAN_ADDENDUM_comments.md section 4):
  CI-A  stratified bootstrap over subjects (main interval; no refit variance)
  CI-B  DeLong on the seed-42 pooled OOF AUC (main interval)
  partition variability = 2.5-97.5 percentiles of the per-repeat statistic across the 100 repeat seeds (NOT a CI)
"""
import numpy as np
from joblib import Parallel, delayed

from .nested import run_nested_cv
from .stats import delong_ci, repeat_summary, repeated_cv_bootstrap


def get_variant(ctx, tag, mc, with_repeats=True):
    """Seed-42 run (+ 100 repeat seeds) of the nested CV with config `mc`; cached for the whole run."""
    cache = ctx.cache.setdefault("runs", {})
    if tag in cache and (cache[tag]["reps"] is not None or not with_repeats):
        return cache[tag]
    ds, y = ctx.ds, ctx.ds.y
    ref = run_nested_cv(ds, y, ctx.cfg["seeds"]["reference"], mc, n_jobs=ctx.n_jobs)
    reps = None
    if with_repeats:
        reps = Parallel(n_jobs=ctx.n_jobs)(delayed(run_nested_cv)(ds, y, s, mc, 1) for s in ctx.cfg.repeat_seeds)
    cache[tag] = {"ref": ref, "reps": reps, "mc": mc}
    return cache[tag]


def register(ctx, tag, ref, reps):
    """Register a run produced elsewhere (PC-only, composition-only, ...) so it can be compared."""
    ctx.cache.setdefault("runs", {})[tag] = {"ref": ref, "reps": reps, "mc": None}


def repeat_arrays(run):
    return (np.vstack([r["oof"] for r in run["reps"]]), np.vstack([r["fold_id"] for r in run["reps"]]))


def save_arrays(ctx, stage, tag, run):
    """Persist OOF predictions so every interval can be recomputed after the run."""
    kw = {"y": ctx.ds.y, "seed42_oof": run["ref"]["oof"], "seed42_fold": run["ref"]["fold_id"]}
    if run["reps"] is not None:
        oofs, folds = repeat_arrays(run)
        kw.update(repeat_seeds=np.array([r["seed"] for r in run["reps"]]), repeat_oof=oofs, repeat_fold=folds)
    np.savez_compressed(ctx.out(stage, f"{tag}_oof_arrays.npz"), **kw)


def variant_stats(y, run, B, seed):
    ref, reps = run["ref"], run["reps"]
    fa = [f["auc"] for f in ref["folds"]]
    out = {"seed42": {
        "pooled_oof_auc": ref["pooled_auc"], "pooled_oof_ap": ref["pooled_ap"],
        "pooled_ci_B_delong": delong_ci(y, ref["oof"]),
        "pooled_ci_A_subject_bootstrap": repeated_cv_bootstrap(y, ref["oof"], np.zeros(len(y), int), B=B, seed=seed),
        "per_fold_auc": fa, "per_fold_mean": float(np.mean(fa)), "per_fold_sd_ddof1": float(np.std(fa, ddof=1)),
        "K_chosen": [f["K"] for f in ref["folds"] if "K" in f],
        "mean_inner_auc": float(np.mean([f["inner_auc"] for f in ref["folds"]]))}}
    if reps:
        oofs, folds = repeat_arrays(run)
        boot = repeated_cv_bootstrap(y, oofs, folds, B=B, seed=seed)
        mf = [r["mean_fold_auc"] for r in reps]
        out["repeated_cv"] = {
            "statistic": "mean over repeats of mean-of-5-fold AUC", "n_repeats": len(reps),
            "mean_fold_auc": boot["estimate"],
            "ci_A_subject_bootstrap": {"lo": boot["boot_lo"], "hi": boot["boot_hi"], "B": boot["B"]},
            "partition_variability": repeat_summary(mf),
            "per_repeat_pooled_auc": repeat_summary([r["pooled_auc"] for r in reps]),
            "mean_fold_ap": float(np.mean([np.mean([f["ap"] for f in r["folds"]]) for r in reps])),
            "mean_inner_auc": float(np.mean([f["inner_auc"] for r in reps for f in r["folds"]])),
            "K_chosen_median": float(np.median([f["K"] for r in reps for f in r["folds"] if "K" in f]))
            if any("K" in f for f in reps[0]["folds"]) else None}
    return out


def permute_labels(y, i, strata=None):
    """Permutation i of the labels (all samples, or within `strata`); uses default_rng([42, i])."""
    rng = np.random.default_rng([42, i])
    y = np.asarray(y)
    if strata is None:
        return rng.permutation(y)
    yp = y.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        yp[idx] = rng.permutation(y[idx])
    return yp


def perm_once(ds, y, i, mc, seed, strata=None):
    """One permutation of the labels and the COMPLETE pipeline on it: selection, inner tuning, fitting."""
    yp = permute_labels(y, i, strata)
    r = run_nested_cv(ds, yp, seed, mc, 1, keep_top=1)
    return {"i": i, "pooled_auc": r["pooled_auc"], "mean_fold_auc": r["mean_fold_auc"],
            "empty_fits": int(sum(f["n_selected"] == 0 for f in r["folds"])),
            "K_chosen": ";".join(str(f["K"]) for f in r["folds"])}
