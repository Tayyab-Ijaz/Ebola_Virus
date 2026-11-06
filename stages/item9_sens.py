"""Item 9 sensitivity: deconvolution without the two samples for which the optimisation did not fully converge (R warning
in item 9: GSM4261710, GSM4261715; list in config item9.exclude_nonconverged).

Reports whether the fractions change: (a) deconvolution is re-run on the remaining samples and compared with the full run
(scores of the remaining samples; the methods score each sample on its own, so no change is expected), (b) the group
comparison without the two samples against the full one, (c) the PC1-5 correlations with the scores without them (PCs from
the 59-sample VST, as in item 6b), (d) the composition-only baseline (all scores) on the remaining samples, seed 42 + 100
repeats, against the 59-sample baseline (different samples, so not a paired comparison)."""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

from evd.fixedfeat import fixed_features_cv
from evd.runs import variant_stats
from stages.item6 import _pca

STAGE = "item9_sens"


def _scores(d, methods, samples):
    feats = {}
    for m in methods:
        p = d / f"deconv_{m}.tsv"
        if p.exists():
            df = pd.read_csv(p, sep="\t").set_index("cell_type")[samples]
            for ct, v in df.iterrows():
                v = v.to_numpy(float)
                if np.isfinite(v).all() and np.nanstd(v) > 0:
                    feats[f"{m}:{ct}"] = v
    return pd.DataFrame(feats, index=samples)


def _group_test(F, y):
    rows = []
    for c in F.columns:
        v = F[c].to_numpy(float)
        a, b = v[y == 1], v[y == 0]
        rows.append({"score": c, "diff": a.mean() - b.mean(), "p_mannwhitney": stats.mannwhitneyu(a, b, alternative="two-sided")[1]})
    df = pd.DataFrame(rows)
    df["q_mannwhitney_BH"] = multipletests(df.p_mannwhitney, method="fdr_bh")[1]
    return df.set_index("score")


def run(ctx):
    cfg, ds, log, y = ctx.cfg, ctx.ds, ctx.log, ctx.ds.y
    d9, d = ctx.run_dir / "item9", ctx.stage_dir(STAGE)
    if not (d9 / "deconvolution_scores_per_sample.csv").exists():
        raise RuntimeError("item9_sens needs stage item9 in the same run")
    ex = cfg["item9"]["exclude_nonconverged"]
    names = list(ds.sample_ids)
    missing = [s for s in ex if s not in names]
    if missing:
        raise RuntimeError(f"excluded samples not in the data: {missing}")
    keep = np.array([s not in ex for s in names])
    ks = [s for s in names if s not in ex]
    yk = y[keep]
    cond = {s: ("EVD-S" if y[names.index(s)] else "Healthy") for s in ex}
    log.info("excluding %s (%s); remaining %d samples (EVD-S %d, Healthy %d)", ex, cond, keep.sum(), yk.sum(), (1 - yk).sum())

    tpm = pd.read_csv(d9 / "tpm_symbols_input.tsv", sep="\t", index_col=0)
    tpm[ks].rename_axis("gene_symbol").to_csv(d / "tpm_symbols_input.tsv", sep="\t")
    n_before = len(ctx.r_warnings)
    ctx.run_r("deconv.R", d / "tpm_symbols_input.tsv", d, ",".join(cfg["item9"]["methods"]))
    new_warn = [w for w in ctx.r_warnings[n_before:] if "built under R version" not in w]
    log.info("R warnings of the reduced deconvolution (excluding package-load notices): %s", new_warn or "none")

    full = pd.read_csv(d9 / "deconvolution_scores_per_sample.csv", index_col=0).drop(columns="condition")
    red = _scores(d, cfg["item9"]["methods"], ks)
    common = [c for c in full.columns if c in red.columns]
    diff = red[common].to_numpy() - full.loc[ks, common].to_numpy()
    per_method = {m: float(np.abs(diff[:, [i for i, c in enumerate(common) if c.startswith(m + ":")]]).max())
                  for m in cfg["item9"]["methods"] if any(c.startswith(m + ":") for c in common)}
    out = {"excluded": cond, "n_remaining": int(keep.sum()), "new_R_warnings_in_reduced_run": new_warn,
           "scores_in_both_runs": len(common), "scores_only_in_one_run": sorted(set(full.columns) ^ set(red.columns)),
           "max_abs_change_of_remaining_samples_per_method": per_method}
    log.info("(a) scores of the remaining samples, reduced vs full deconvolution: max |change| per method %s", per_method)

    g = _group_test(full[common], y).join(_group_test(red[common], yk), lsuffix="_full", rsuffix="_excl")
    g.to_csv(ctx.out(STAGE, "group_comparison_full_vs_excluded.csv"))
    flips = g[(g.q_mannwhitney_BH_full < 0.05) != (g.q_mannwhitney_BH_excl < 0.05)]
    out["group_comparison"] = {
        "n_scores": len(g), "n_q<0.05_full": int((g.q_mannwhitney_BH_full < 0.05).sum()),
        "n_q<0.05_excluded": int((g.q_mannwhitney_BH_excl < 0.05).sum()),
        "spearman_rho_of_group_differences": float(stats.spearmanr(g.diff_full, g.diff_excl).statistic),
        "scores_changing_significance": {k: [float(r.q_mannwhitney_BH_full), float(r.q_mannwhitney_BH_excl)] for k, r in flips.iterrows()}}
    log.info("(b) group comparison: BH q<0.05 in %d (full) vs %d (excluded) of %d scores; rank correlation of group differences %.4f; "
             "significance flips: %s", out["group_comparison"]["n_q<0.05_full"], out["group_comparison"]["n_q<0.05_excluded"], len(g),
             out["group_comparison"]["spearman_rho_of_group_differences"], list(flips.index) or "none")

    _, sc, _ = _pca(ctx.run_dir / "item6" / "vst_blindTRUE.tsv", names)
    rows = []
    for k in range(5):
        for c in common:
            rf = stats.spearmanr(sc[:, k], full[c].to_numpy(float))
            re_ = stats.spearmanr(sc[keep, k], red[c].to_numpy(float))
            rows.append({"PC": k + 1, "score": c, "rho_full": rf.statistic, "p_full": rf.pvalue,
                         "rho_excl": re_.statistic, "p_excl": re_.pvalue})
    pc = pd.DataFrame(rows)
    pc["q_full"] = multipletests(pc.p_full, method="fdr_bh")[1]
    pc["q_excl"] = multipletests(pc.p_excl, method="fdr_bh")[1]
    pc.to_csv(ctx.out(STAGE, "pc_correlations_full_vs_excluded.csv"), index=False)
    pflip = pc[(pc.q_full < 0.05) != (pc.q_excl < 0.05)]
    out["pc_correlations"] = {"n_pairs": len(pc), "n_q<0.05_full": int((pc.q_full < 0.05).sum()),
                              "n_q<0.05_excluded": int((pc.q_excl < 0.05).sum()),
                              "max_abs_change_in_rho": float((pc.rho_excl - pc.rho_full).abs().max()),
                              "n_pairs_changing_significance": len(pflip)}
    log.info("(c) PC1-5 vs scores: BH q<0.05 in %d (full) vs %d (excluded) of %d pairs; max |change in rho| %.3f; %d pairs change significance",
             out["pc_correlations"]["n_q<0.05_full"], out["pc_correlations"]["n_q<0.05_excluded"], len(pc),
             out["pc_correlations"]["max_abs_change_in_rho"], len(pflip))

    seeds = [cfg["seeds"]["reference"]] + cfg.repeat_seeds
    F = red[common].to_numpy(float)
    res = Parallel(n_jobs=ctx.n_jobs)(delayed(fixed_features_cv)(F, yk, s, cfg["item6"]["c_grid"]) for s in seeds)
    B, rs = cfg["item1"]["bootstrap_B"], cfg["seeds"]["bootstrap"]
    st = variant_stats(yk, {"ref": res[0], "reps": res[1:]}, B, rs)
    full_run = ctx.cache.get("runs", {}).get("composition_all")
    st_full = variant_stats(y, full_run, B, rs) if full_run else None
    out["composition_only_baseline"] = {"excluded_57": st, "full_59": st_full}
    np.savez_compressed(ctx.out(STAGE, "composition_all_excluded_oof.npz"), y=yk, oof=res[0]["oof"], fold=res[0]["fold_id"])
    for lab, s_ in (("excluded 57", st), ("full 59", st_full)):
        if s_:
            log.info("(d) composition-only baseline, %s: seed42 pooled %.4f (DeLong %.4f-%.4f) | repeated mean-fold AUC %.4f (CI-A %.4f-%.4f)",
                     lab, s_["seed42"]["pooled_oof_auc"], s_["seed42"]["pooled_ci_B_delong"]["lo"],
                     s_["seed42"]["pooled_ci_B_delong"]["hi"], s_["repeated_cv"]["mean_fold_auc"],
                     s_["repeated_cv"]["ci_A_subject_bootstrap"]["lo"], s_["repeated_cv"]["ci_A_subject_bootstrap"]["hi"])
    ctx.out(STAGE, "deconvolution_sensitivity.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
