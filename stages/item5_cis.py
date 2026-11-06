"""Item 5: threshold-based metrics (threshold 0.5 on the predicted probability, fixed in advance), Brier score,
calibration intercept/slope, Wilson 95% CIs for sensitivity and specificity.

Seed 42: metrics of the pooled out-of-fold predictions (one model per fold, so scores are not on one common scale).
Repeats: the same metrics per repeat on that repeat's pooled OOF predictions; mean over 100 repeats and the 2.5-97.5
percentiles across repeats (partition variability, not a CI)."""
import numpy as np
import pandas as pd

from evd.metrics import threshold_metrics

STAGE = "item5_cis"
KEYS = ["sensitivity", "specificity", "balanced_accuracy", "mcc", "brier", "calib_intercept", "calib_slope"]


def run(ctx):
    log, y = ctx.log, ctx.ds.y
    rows = []
    for tag, run_ in ctx.cache.get("runs", {}).items():
        if tag.startswith("fixedK_"):
            continue
        m = threshold_metrics(y, run_["ref"]["oof"])
        row = {"variant": tag, "scope": "seed42_pooled_OOF", "n_fits": 1,
               "sens_wilson_lo": m["sens_wilson"][0], "sens_wilson_hi": m["sens_wilson"][1],
               "spec_wilson_lo": m["spec_wilson"][0], "spec_wilson_hi": m["spec_wilson"][1],
               "tp": m["tp"], "fn": m["fn"], "tn": m["tn"], "fp": m["fp"]}
        row.update({k: m[k] for k in KEYS})
        rows.append(row)
        if run_["reps"]:
            ms = [threshold_metrics(y, r["oof"]) for r in run_["reps"]]
            row = {"variant": tag, "scope": "repeat_mean_pooled_OOF", "n_fits": len(ms)}
            for k in KEYS:
                v = np.array([x[k] for x in ms], float)
                row[k] = float(np.nanmean(v))
                row[f"{k}_partition_p2.5"] = float(np.nanpercentile(v, 2.5))
                row[f"{k}_partition_p97.5"] = float(np.nanpercentile(v, 97.5))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(ctx.out(STAGE, "threshold_metrics_by_variant.csv"), index=False)
    p = df[(df.variant == "primary")]
    for _, r in p.iterrows():
        log.info("[primary %s] sens %.3f spec %.3f bal.acc %.3f MCC %.3f Brier %.3f calib intercept %.2f slope %.2f", r.scope,
                 r.sensitivity, r.specificity, r.balanced_accuracy, r.mcc, r.brier, r.calib_intercept, r.calib_slope)
    s = p[p.scope == "seed42_pooled_OOF"].iloc[0]
    log.info("[primary seed 42] sensitivity %d/%d (Wilson %.3f-%.3f), specificity %d/%d (Wilson %.3f-%.3f)", s.tp, s.tp + s.fn,
             s.sens_wilson_lo, s.sens_wilson_hi, s.tn, s.tn + s.fp, s.spec_wilson_lo, s.spec_wilson_hi)
