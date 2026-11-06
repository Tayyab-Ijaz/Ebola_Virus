"""Item 1: nested CV with the Welch ranking refitted inside every inner AND outer training split.

Gate: the leakage unit tests must pass first, otherwise nothing is reported.
Variants (seed 42 + 100 repeats each):
  primary                 D1: rank by Welch |t| on log2(CPM+1), top-K, K tuned in the inner loop
  S1_hard_gate            FDR < 0.05 and |LFC| >= 1 gate, then top-min(K, n_pass); constant 0.5 if nothing passes
  S3_no_sex_chrom         primary with X/Y-chromosome genes excluded from the candidates
  S4_legacy_structure     ranking computed once per outer fold (the structure Reviewer 3 found)
  S5_effect_size_filter   |LFC| >= 1 (training split), then rank by Welch |t|; no FDR criterion
  S6_K_to_100             primary with K grid extended to 100 (grid-boundary sensitivity)
(S2, limma-voom ranker, is in stage item1_s2 because it needs the item 7 tables.)
"""
import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from evd.config import V2_DIR
from evd.nested import ModelCfg
from evd.runs import get_variant, repeat_arrays, save_arrays, variant_stats
from evd.stats import repeat_summary

STAGE = "item1"


def _run_tests(ctx):
    cmd = [sys.executable, "-m", "pytest", str(V2_DIR / "tests"), "-q", "-p", "no:cacheprovider", "--tb=short",
           "--durations=5"]
    r = subprocess.run(cmd, cwd=V2_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONUTF8": "1"})
    txt = r.stdout + r.stderr
    ctx.out(STAGE, "leakage_tests.txt").write_text(txt, encoding="utf-8")
    for ln in txt.strip().splitlines()[-14:]:
        ctx.log.info("  [pytest] %s", ln)
    if r.returncode != 0:
        raise RuntimeError("leakage unit tests failed; no item 1 result is reported")


def variants(cfg):
    base = ModelCfg.from_config(cfg)
    return [("primary", base),
            ("S1_hard_gate", replace(base, mode="gate")),
            ("S3_no_sex_chrom", replace(base, exclude_sex_chrom=True)),
            ("S4_legacy_structure", replace(base, selection_level="outer")),
            ("S5_effect_size_filter", replace(base, mode="effect")),
            ("S6_K_to_100", replace(base, k_grid=tuple(cfg["sensitivity"]["k_grid_s6"])))]


def _fold_table(ds, res):
    rows = []
    for r in res["folds"]:
        genes = ds.symbols[r["sel"]]
        rows.append({"fold": r["fold"], "auc": r["auc"], "ap": r["ap"], "K": r["K"], "C": r["C"], "l1_ratio": r["l1"],
                     "inner_mean_auc": r["inner_auc"], "n_kept_outer_train": r["n_kept"],
                     "n_pass_gate_outer_train": r["n_pass_gate"], "n_effect_outer_train": r["n_effect"],
                     "n_selected": r["n_selected"], "selected_genes": ";".join(genes),
                     "nonzero_coef_genes": ";".join(f"{s}:{c:+.3f}" for s, c in zip(genes, r["coef"]) if c != 0)})
    return pd.DataFrame(rows)


def _inner_counts(res, seed):
    return pd.DataFrame([dict(seed=seed, **i) for r in res["folds"] for i in r["inner_info"]])


def gene_frequency(ds, fits):
    n = len(fits)
    sel, nz, coef = {}, {}, {}
    for r in fits:
        for g, c in zip(r["sel"], r["coef"]):
            sel[g] = sel.get(g, 0) + 1
            if c != 0:
                nz[g] = nz.get(g, 0) + 1
                coef.setdefault(g, []).append(abs(c))
    rows = [{"symbol": ds.symbols[g], "sex_chrom": bool(ds.sex_chrom[g]), "selected_freq": sel[g] / n,
             "nonzero_coef_freq": nz.get(g, 0) / n,
             "mean_abs_std_coef_when_nonzero": float(np.mean(coef[g])) if g in coef else 0.0} for g in sel]
    return pd.DataFrame(rows).sort_values(["nonzero_coef_freq", "selected_freq", "symbol"], ascending=[False, False, True])


def _dist(v):
    v = np.asarray(v)
    return {"min": int(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": int(v.max()), "frac_lt5": float((v < 5).mean()),
            "frac_eq0": float((v == 0).mean()), "n_splits": int(len(v))}


def _one(ctx, tag, mc):
    cfg, ds, log, y = ctx.cfg, ctx.ds, ctx.log, ctx.ds.y
    run = get_variant(ctx, tag, mc, True)
    ref, reps = run["ref"], run["reps"]
    seed = cfg["seeds"]["reference"]
    _fold_table(ds, ref).to_csv(ctx.out(STAGE, f"{tag}_seed42_folds.csv"), index=False)
    _inner_counts(ref, seed).to_csv(ctx.out(STAGE, f"{tag}_seed42_inner_gene_counts.csv"), index=False)
    pd.DataFrame({"sample": ds.sample_ids, "y": y, "fold": ref["fold_id"], "oof": ref["oof"]}).to_csv(
        ctx.out(STAGE, f"{tag}_seed42_oof.csv"), index=False)
    pd.DataFrame([{"seed": r["seed"], "mean_fold_auc": r["mean_fold_auc"], "pooled_auc": r["pooled_auc"],
                   "pooled_ap": r["pooled_ap"], "mean_fold_ap": float(np.mean([f["ap"] for f in r["folds"]])),
                   "mean_inner_auc": float(np.mean([f["inner_auc"] for f in r["folds"]])),
                   "K_chosen": ";".join(str(f["K"]) for f in r["folds"])} for r in reps]
                 ).to_csv(ctx.out(STAGE, f"{tag}_repeats.csv"), index=False)
    pd.concat([_inner_counts(r, r["seed"]) for r in reps]).to_csv(
        ctx.out(STAGE, f"{tag}_repeats_inner_gene_counts.csv"), index=False)
    save_arrays(ctx, STAGE, tag, run)
    fits = [f for r in reps for f in r["folds"]]
    freq = gene_frequency(ds, fits)
    freq.to_csv(ctx.out(STAGE, f"{tag}_gene_frequency_500_fits.csv"), index=False)
    out = variant_stats(y, run, cfg["item1"]["bootstrap_B"], cfg["seeds"]["bootstrap"])
    kc = pd.Series([f["K"] for f in fits]).value_counts().sort_index()
    out["tag"] = tag
    out["definition"] = {"mode": mc.mode, "selection_level": mc.selection_level, "k_grid": list(mc.k_grid),
                         "exclude_sex_chrom": mc.exclude_sex_chrom}
    out["seed42"]["inner_n_pass_gate"] = _dist([i["n_pass_gate"] for r in ref["folds"] for i in r["inner_info"]])
    out["seed42"]["inner_n_effect"] = _dist([i["n_effect"] for r in ref["folds"] for i in r["inner_info"]])
    out["seed42"]["outer_n_pass_gate"] = [r["n_pass_gate"] for r in ref["folds"]]
    rc = out["repeated_cv"]
    rc["K_chosen_counts"] = {int(k): int(v) for k, v in kc.items()}
    rc["inner_n_pass_gate"] = _dist([i["n_pass_gate"] for r in reps for f in r["folds"] for i in f["inner_info"]])
    rc["inner_n_effect"] = _dist([i["n_effect"] for r in reps for f in r["folds"] for i in f["inner_info"]])
    rc["n_selected_per_fit"] = {"median": float(np.median([f["n_selected"] for f in fits])),
                                "min": int(min(f["n_selected"] for f in fits))}
    rc["frac_fits_empty_selection"] = float(np.mean([f["n_selected"] == 0 for f in fits]))
    rc["frac_fits_with_a_sex_chrom_gene_selected"] = float(np.mean([bool(ds.sex_chrom[f["sel"]].any()) for f in fits]))
    s = out["seed42"]
    log.info("[%s] seed 42: pooled OOF AUC %.4f (CI-B DeLong %.3f-%.3f; CI-A bootstrap %.3f-%.3f) AP %.4f | per-fold %s "
             "mean %.4f | K %s", tag, s["pooled_oof_auc"], s["pooled_ci_B_delong"]["lo"], s["pooled_ci_B_delong"]["hi"],
             s["pooled_ci_A_subject_bootstrap"]["boot_lo"], s["pooled_ci_A_subject_bootstrap"]["boot_hi"],
             s["pooled_oof_ap"], np.round(s["per_fold_auc"], 4).tolist(), s["per_fold_mean"], s["K_chosen"])
    log.info("[%s] repeated CV (%d repeats): mean-fold AUC %.4f | CI-A subject bootstrap %.4f-%.4f | partition "
             "variability (2.5-97.5%%) %.4f-%.4f | median K %s | empty selections %.1f%%", tag, rc["n_repeats"],
             rc["mean_fold_auc"], rc["ci_A_subject_bootstrap"]["lo"], rc["ci_A_subject_bootstrap"]["hi"],
             rc["partition_variability"]["p2.5"], rc["partition_variability"]["p97.5"], rc["K_chosen_median"],
             100 * rc["frac_fits_empty_selection"])
    log.info("[%s] inner-split genes passing the hard gate (%d splits): median %.0f, range %d-%d, <5 genes in %.1f%%, "
             "0 genes in %.1f%%; |LFC|>=1 genes: median %.0f", tag, rc["inner_n_pass_gate"]["n_splits"],
             rc["inner_n_pass_gate"]["median"], rc["inner_n_pass_gate"]["min"], rc["inner_n_pass_gate"]["max"],
             100 * rc["inner_n_pass_gate"]["frac_lt5"], 100 * rc["inner_n_pass_gate"]["frac_eq0"],
             rc["inner_n_effect"]["median"])
    log.info("[%s] top genes by non-zero-coefficient frequency over %d fits:\n%s", tag, len(fits),
             freq.head(8).round(3).to_string(index=False))
    return out


def run(ctx):
    cfg, log = ctx.cfg, ctx.log
    _run_tests(ctx)
    results = {}
    for tag, mc in variants(cfg):
        results[tag] = _one(ctx, tag, mc)

    # paired effect of each variant against the primary (same seeds, same outer splits)
    prim = ctx.cache["runs"]["primary"]
    paired = {}
    for tag in results:
        if tag == "primary":
            continue
        other = ctx.cache["runs"][tag]
        d = np.array([a["mean_fold_auc"] - b["mean_fold_auc"] for a, b in zip(prim["reps"], other["reps"])])
        di = np.array([np.mean([f["inner_auc"] for f in a["folds"]]) - np.mean([f["inner_auc"] for f in b["folds"]])
                       for a, b in zip(prim["reps"], other["reps"])])
        paired[tag] = {"outer_mean_fold_auc_diff_primary_minus_variant": repeat_summary(d),
                       "inner_mean_auc_diff_primary_minus_variant": repeat_summary(di)}
    results["paired_primary_minus_variant"] = paired
    s4 = paired["S4_legacy_structure"]
    log.info("paired primary - S4 (leak): outer mean-fold AUC diff %.4f (SD across repeats %.4f); inner AUC diff %.4f",
             s4["outer_mean_fold_auc_diff_primary_minus_variant"]["mean"],
             s4["outer_mean_fold_auc_diff_primary_minus_variant"]["sd"],
             s4["inner_mean_auc_diff_primary_minus_variant"]["mean"])
    bl = cfg["baseline"]
    results["baseline_legacy_seed42"] = {"pooled_oof_auc": bl["pooled_oof_auc"], "per_fold_mean": bl["per_fold_mean_auc"],
                                         "pooled_oof_ap": bl["pooled_oof_ap"]}
    ctx.out(STAGE, "item1_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
