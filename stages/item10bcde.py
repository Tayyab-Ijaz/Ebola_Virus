"""Item 10(b)-(e): inferred sex as a classifier input and as a source of errors.

* sex-only baseline (inferred sex -> logistic regression), seed 42 + repeats
* primary pipeline with inferred sex appended to the selected genes (panel + sex), seed 42 + repeats; paired with primary
* error analysis by sex on the primary pipeline: AUC within sex, misclassification by sex x condition, the samples
  misclassified most often across repeats
(S3, sex-chromosome genes excluded, is run in stage item1; the sex-stratified permutation in item2_permutation;
fold composition by sex in item10a.)  Inferred sex = mean log2(CPM+1) of KDM5D, RPS4Y1, DDX3Y, UTY > 2; one sample is
discordant with XIST and is reported, not relabelled."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import roc_auc_score

from evd.fixedfeat import fixed_features_cv
from evd.nested import ModelCfg
from evd.runs import get_variant, register, repeat_arrays, save_arrays, variant_stats
from evd.stats import paired_bootstrap_diff, repeat_summary

STAGE = "item10bcde"


def run(ctx):
    cfg, ds, log, y = ctx.cfg, ctx.ds, ctx.log, ctx.ds.y
    B, rs = cfg["item1"]["bootstrap_B"], cfg["seeds"]["bootstrap"]
    seeds = [cfg["seeds"]["reference"]] + cfg.repeat_seeds
    male = ds.sex_male
    out = {}

    # ---- sex-only baseline ----------------------------------------------------------------------
    res = Parallel(n_jobs=ctx.n_jobs)(delayed(fixed_features_cv)(male[:, None].astype(float), y, s, cfg["item6"]["c_grid"])
                                      for s in seeds)
    register(ctx, "sex_only", res[0], res[1:])
    save_arrays(ctx, STAGE, "sex_only", {"ref": res[0], "reps": res[1:]})
    out["sex_only"] = variant_stats(y, {"ref": res[0], "reps": res[1:]}, B, rs)
    log.info("[sex_only] seed42 pooled %.4f, per-fold mean %.4f | repeated CV mean-fold AUC %.4f (CI-A %.4f-%.4f)",
             res[0]["pooled_auc"], res[0]["mean_fold_auc"], out["sex_only"]["repeated_cv"]["mean_fold_auc"],
             out["sex_only"]["repeated_cv"]["ci_A_subject_bootstrap"]["lo"], out["sex_only"]["repeated_cv"]["ci_A_subject_bootstrap"]["hi"])

    # ---- primary + inferred sex -----------------------------------------------------------------
    base = ModelCfg.from_config(cfg)
    prim = get_variant(ctx, "primary", base, True)
    ps = get_variant(ctx, "primary_plus_sex", replace(base, use_cov=True), True)
    save_arrays(ctx, STAGE, "primary_plus_sex", ps)
    out["primary_plus_sex"] = variant_stats(y, ps, B, rs)
    sexcoef = [f["coef"][len(f["sel"])] for r in ps["reps"] for f in r["folds"] if len(f["coef"]) > len(f["sel"])]
    out["primary_plus_sex"]["sex_coefficient"] = {"n_fits": len(sexcoef), "frac_nonzero": float(np.mean(np.array(sexcoef) != 0)),
                                                  "mean_std_coef": float(np.mean(sexcoef))}
    oa, fa = repeat_arrays(ps)
    ob, fb = repeat_arrays(prim)
    d = np.array([a["mean_fold_auc"] - b["mean_fold_auc"] for a, b in zip(ps["reps"], prim["reps"])])
    out["plus_sex_minus_primary"] = {"CI_A_paired_subject_bootstrap": paired_bootstrap_diff(y, oa, fa, ob, fb, B=B, seed=rs),
                                     "partition_variability_of_difference": repeat_summary(d)}
    log.info("[primary + sex] repeated CV mean-fold AUC %.4f | minus primary: %.4f (CI-A %.4f to %.4f) | sex coefficient non-zero in %.0f%% of fits",
             out["primary_plus_sex"]["repeated_cv"]["mean_fold_auc"], out["plus_sex_minus_primary"]["CI_A_paired_subject_bootstrap"]["estimate"],
             out["plus_sex_minus_primary"]["CI_A_paired_subject_bootstrap"]["boot_lo"], out["plus_sex_minus_primary"]["CI_A_paired_subject_bootstrap"]["boot_hi"],
             100 * out["primary_plus_sex"]["sex_coefficient"]["frac_nonzero"])

    # ---- error analysis by sex on the primary pipeline ------------------------------------------
    sexlab = np.where(male == 1, "male", "female")
    ref = prim["ref"]
    err42 = (ref["oof"] >= 0.5) != (y == 1)
    tab = pd.crosstab([pd.Series(np.where(y == 1, "EVD-S", "Healthy"), name="condition"), pd.Series(sexlab, name="sex")],
                      pd.Series(err42, name="misclassified"))
    ea = {}
    for s in ("male", "female"):
        m = sexlab == s
        ea[s] = {"n": int(m.sum()), "n_EVD_S": int(y[m].sum()),
                 "within_sex_auc_seed42_pooled": float(roc_auc_score(y[m], ref["oof"][m])) if len(set(y[m])) == 2 else None,
                 "misclassified_seed42": int(err42[m].sum())}
    f = stats.fisher_exact(pd.crosstab(sexlab, err42).reindex(index=["female", "male"], columns=[False, True], fill_value=0).to_numpy())
    # repeats: per-sample misclassification frequency
    errs = np.vstack([((r["oof"] >= 0.5) != (y == 1)) for r in prim["reps"]])
    freq = errs.mean(0)
    per = pd.DataFrame({"sample": ds.sample_ids, "condition": np.where(y == 1, "EVD-S", "Healthy"), "sex": sexlab,
                        "misclassified_seed42": err42, "misclassification_freq_100_repeats": freq,
                        "mean_oof_score_seed42": ref["oof"]}).sort_values("misclassification_freq_100_repeats", ascending=False)
    per.to_csv(ctx.out(STAGE, "per_sample_errors_primary.csv"), index=False)
    grp = per.groupby(["condition", "sex"]).misclassification_freq_100_repeats.agg(["mean", "count"]).reset_index()
    grp.to_csv(ctx.out(STAGE, "error_rate_by_condition_and_sex_100_repeats.csv"), index=False)
    out["error_analysis"] = {"by_sex": ea, "seed42_error_sex_fisher_p": float(f[1]),
                             "note": "Error rate by sex x condition over 100 repeats is in error_rate_by_condition_and_sex_100_repeats.csv"}
    ctx.out(STAGE, "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("seed-42 errors by condition and sex:\n%s", tab.to_string())
    log.info("mean misclassification frequency over 100 repeats:\n%s", grp.round(3).to_string(index=False))
    log.info("within-sex pooled OOF AUC (seed 42): %s | Fisher test errors~sex p = %.3f", {k: v["within_sex_auc_seed42_pooled"] for k, v in ea.items()}, f[1])
    log.info("most often misclassified samples:\n%s", per.head(6).round(3).to_string(index=False))
