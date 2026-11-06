"""Item 4: panel size.

* Primary evidence on K is the distribution of K chosen by inner CV (stage item3_stability / item1).
* Fixed-K sweep: K fixed, C and l1_ratio still tuned in the inner loop, same 100 repeats. DESCRIPTIVE ONLY - it is
  never used to select K (K is chosen by inner CV only). K = 75 and 100 belong to the grid-boundary sensitivity (S6).
* Ablation inside CV: K = 5 versus K = 4 (dropping the rank-5 gene of each fit), paired over the same repeats and
  outer splits: percentile interval of the per-repeat difference (partition variability) and paired subject bootstrap."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from evd import plotting
from evd.nested import ModelCfg
from evd.runs import get_variant, repeat_arrays, variant_stats
from evd.stats import paired_bootstrap_diff, repeat_summary

STAGE = "item4_panel_size"


def run(ctx):
    cfg, ds, log, y = ctx.cfg, ctx.ds, ctx.log, ctx.ds.y
    base = ModelCfg.from_config(cfg)
    B, rs = cfg["item1"]["bootstrap_B"], cfg["seeds"]["bootstrap"]
    rows, runs = [], {}
    for K in cfg["item4"]["k_sweep"]:
        run_ = get_variant(ctx, f"fixedK_{K}", replace(base, k_grid=(K,)), True)
        runs[K] = run_
        st = variant_stats(y, run_, B, rs)
        s, rc = st["seed42"], st["repeated_cv"]
        rows.append({"K": K, "descriptive_only": True, "sensitivity_S6_only": K > 50,
                     "seed42_pooled_oof_auc": s["pooled_oof_auc"], "seed42_pooled_ci_B_delong_lo": s["pooled_ci_B_delong"]["lo"],
                     "seed42_pooled_ci_B_delong_hi": s["pooled_ci_B_delong"]["hi"], "seed42_per_fold_mean": s["per_fold_mean"],
                     "repeated_cv_mean_fold_auc": rc["mean_fold_auc"], "ci_A_subject_bootstrap_lo": rc["ci_A_subject_bootstrap"]["lo"],
                     "ci_A_subject_bootstrap_hi": rc["ci_A_subject_bootstrap"]["hi"],
                     "partition_variability_p2.5": rc["partition_variability"]["p2.5"],
                     "partition_variability_p97.5": rc["partition_variability"]["p97.5"],
                     "mean_fold_ap": rc["mean_fold_ap"]})
        log.info("fixed K=%-3d repeated-CV mean-fold AUC %.4f (CI-A %.4f-%.4f)", K, rc["mean_fold_auc"],
                 rc["ci_A_subject_bootstrap"]["lo"], rc["ci_A_subject_bootstrap"]["hi"])
    sweep = pd.DataFrame(rows)
    sweep.to_csv(ctx.out(STAGE, "fixedK_sweep_DESCRIPTIVE.csv"), index=False)

    # ---- ablation K = 5 vs K = 4 -------------------------------------------------------------
    a, b = runs[5], runs[4]
    d = np.array([x["mean_fold_auc"] - z["mean_fold_auc"] for x, z in zip(a["reps"], b["reps"])])
    oa, fa = repeat_arrays(a)
    ob, fb = repeat_arrays(b)
    pb = paired_bootstrap_diff(y, oa, fa, ob, fb, B=B, seed=rs)
    abl = {"comparison": "K=5 minus K=4 (mean-fold AUC, same repeats and outer splits)",
           "paired_subject_bootstrap_CI_A": pb, "per_repeat_difference_partition_variability": repeat_summary(d),
           "seed42_pooled_diff": a["ref"]["pooled_auc"] - b["ref"]["pooled_auc"],
           "frac_repeats_K5_better": float((d > 0).mean())}
    ctx.out(STAGE, "ablation_K5_vs_K4.json").write_text(json.dumps(abl, indent=2), encoding="utf-8")
    log.info("ablation K=5 minus K=4: %.4f (paired subject bootstrap %.4f to %.4f; partition variability %.4f to %.4f); "
             "K=5 better in %.0f%% of repeats", pb["estimate"], pb["boot_lo"], pb["boot_hi"],
             abl["per_repeat_difference_partition_variability"]["p2.5"], abl["per_repeat_difference_partition_variability"]["p97.5"],
             100 * abl["frac_repeats_K5_better"])

    # ---- inner-CV choice (the evidence used for K) --------------------------------------------
    prim = get_variant(ctx, "primary", base, True)
    kc = pd.Series([f["K"] for r in prim["reps"] for f in r["folds"]]).value_counts().sort_index()
    pd.DataFrame({"K": kc.index, "n_fits_chosen": kc.values, "fraction": kc.values / kc.sum()}).to_csv(
        ctx.out(STAGE, "K_chosen_by_inner_cv_500_fits.csv"), index=False)

    # ---- figure ---------------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    plotting.style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(sweep.K, sweep.repeated_cv_mean_fold_auc, marker="o", color=plotting.EVD)
    ax.fill_between(sweep.K, sweep.ci_A_subject_bootstrap_lo, sweep.ci_A_subject_bootstrap_hi, color=plotting.EVD, alpha=0.15,
                    label="CI-A (subject bootstrap)")
    ax.set_xscale("log")
    ax.set_xticks(sweep.K)
    ax.set_xticklabels(sweep.K, fontsize=7)
    ax.set_xlabel("Fixed panel size K")
    ax.set_ylabel("Repeated-CV mean-fold AUC")
    ax.set_title("Descriptive sweep (not used to choose K)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plotting.save(fig, ctx.out(STAGE, "fixedK_sweep_DESCRIPTIVE.png"))
