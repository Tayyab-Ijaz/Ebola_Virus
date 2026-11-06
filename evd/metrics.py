"""Threshold-based and calibration metrics (item 5). Threshold 0.5 on the predicted probability, fixed in advance."""
import numpy as np
import statsmodels.api as sm

from .stats import wilson_ci


def calibration(y, p):
    """Logistic recalibration: y ~ a + b * logit(p). Returns (intercept, slope); NaN if it does not converge."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    try:
        r = sm.Logit(np.asarray(y), sm.add_constant(z)).fit(disp=0, maxiter=200)
        return float(r.params[0]), float(r.params[1])
    except Exception:
        return float("nan"), float("nan")


def threshold_metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    tp, tn = int((pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    fp, fn = int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    a, b = calibration(y, p)
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": sens, "specificity": spec,
            "balanced_accuracy": (sens + spec) / 2, "mcc": (tp * tn - fp * fn) / den if den > 0 else 0.0,
            "brier": float(np.mean((p - y) ** 2)), "calib_intercept": a, "calib_slope": b,
            "sens_wilson": wilson_ci(tp, tp + fn), "spec_wilson": wilson_ci(tn, tn + fp)}
