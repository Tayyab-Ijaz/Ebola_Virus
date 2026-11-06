"""Item 3: stability, tie-break ordering (D4), final K by inner CV on all 59 samples, final panel, SHAP, P5.

Fits used for stability (all with the primary pipeline, including inner tuning):
  * the 100 repeats x 5 outer folds of the primary run (500 fits)
  * 200 stratified 80% subsamples without replacement (rng seed 11), inner CV seed 42
Final K: tune_inner on all 59 samples with RepeatedStratifiedKFold(n_splits=3, n_repeats=20, random_state=42).
Final panel: first K genes of the tie-break ordering; model refitted on all 59 samples with the C / l1_ratio of
that tuning; SHAP LinearExplainer (interventional) on that model."""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from evd import plotting
from evd.nested import ModelCfg, fit_model, run_outer_fold, tune_inner
from evd.runs import get_variant
from evd.stability import gene_stats, recurrence, tie_break_sort

STAGE = "item3_stability"


def _subsample_splits(y, n, frac, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        tr = []
        for c in (0, 1):
            idx = np.flatnonzero(y == c)
            tr += rng.choice(idx, int(round(frac * len(idx))), replace=False).tolist()
        tr = np.sort(np.array(tr))
        out.append((tr, np.setdiff1d(np.arange(len(y)), tr)))
    return out


def run(ctx):
    cfg, ds, log, y = ctx.cfg, ctx.ds, ctx.log, ctx.ds.y
    c3 = cfg["item3"]
    mc = ModelCfg.from_config(cfg)
    prim = get_variant(ctx, "primary", mc, True)
    cv_fits = [f for r in prim["reps"] for f in r["folds"]]

    splits = _subsample_splits(y, c3["n_subsamples"], c3["subsample_frac"], cfg["seeds"]["subsample"])
    log.info("200 subsamples: training size %d (of 59)", len(splits[0][0]))
    sub_fits = Parallel(n_jobs=ctx.n_jobs)(
        delayed(run_outer_fold)(ds, y, tr, te, cfg["seeds"]["reference"], mc, None, i + 1, c3["keep_top"])
        for i, (tr, te) in enumerate(splits))
    all_fits = cv_fits + sub_fits
    log.info("stability fits: %d CV fits + %d subsample fits", len(cv_fits), len(sub_fits))

    # ---- per-gene statistics and the pre-specified ordering ------------------------------------
    tab = gene_stats(all_fits, ds)
    tab_cv, tab_sub = gene_stats(cv_fits, ds), gene_stats(sub_fits, ds)
    tab = tab.merge(tab_cv[["col", "freq_nonzero"]].rename(columns={"freq_nonzero": "freq_nonzero_cv500"}), on="col", how="left")
    tab = tab.merge(tab_sub[["col", "freq_nonzero"]].rename(columns={"freq_nonzero": "freq_nonzero_sub200"}), on="col", how="left")
    tab[["freq_nonzero_cv500", "freq_nonzero_sub200"]] = tab[["freq_nonzero_cv500", "freq_nonzero_sub200"]].fillna(0.0)
    order = tie_break_sort(tab)
    order.insert(0, "ordering_rank", np.arange(1, len(order) + 1))
    order["in_baseline_panel"] = order.symbol.isin(cfg["baseline"]["panel"])
    kc_cv = pd.Series([f["K"] for f in cv_fits]).value_counts().sort_index()
    kc_sub = pd.Series([f["K"] for f in sub_fits]).value_counts().sort_index()

    # ---- final K, C, l1 by inner CV on all samples ---------------------------------------------
    rs = RepeatedStratifiedKFold(n_splits=cfg["model"]["inner_splits"], n_repeats=c3["final_k_repeats"],
                                 random_state=cfg["seeds"]["final_k"])
    inner = [(a, b) for a, b in rs.split(np.zeros(len(y)), y)]
    tun = tune_inner(ds, y, np.arange(len(y)), cfg["seeds"]["final_k"], mc, tag=0, inner=inner)
    K, C, l1 = tun["K"], tun["C"], tun["l1"]
    cols = order["col"].to_numpy()[:K]
    panel = order.iloc[:K].copy()
    log.info("final K by inner CV on all 59 samples (%d inner splits): K=%d C=%s l1=%s (inner AUC %.4f)",
             len(inner), K, C, l1, tun["inner_auc"])
    kmean = pd.DataFrame(tun["mean_table"].max(axis=(1, 2)), index=list(mc.k_grid), columns=["best_inner_auc_for_K"])
    kmean.to_csv(ctx.out(STAGE, "final_K_inner_cv_auc_by_K.csv"), index_label="K")

    # ---- final model, SHAP ---------------------------------------------------------------------
    import shap
    X = ds.LC[:, cols]
    model = fit_model(X, y, C, l1, cfg["seeds"]["final_k"], mc.max_iter)
    Xs = model.scaler.transform(X)
    sv = shap.LinearExplainer(model.lr, Xs, feature_perturbation="interventional").shap_values(Xs)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    panel["std_coef_final_model"] = model.coef_
    panel["mean_abs_shap"] = np.abs(sv).mean(0)
    t, p = stats.ttest_ind(X[y == 1], X[y == 0], axis=0, equal_var=False)
    panel["log2FC_log2cpm_all59"] = X[y == 1].mean(0) - X[y == 0].mean(0)
    panel["welch_t_all59"] = t
    apparent = float(roc_auc_score(y, model.predict_proba1(X)))
    sex_p = None
    p10 = ctx.run_dir / "item10a" / "deseq_sex_effect.tsv"
    if p10.exists():
        sx = pd.read_csv(p10, sep="\t", dtype={"GeneID": str}).set_index("GeneID")
        sex_p = sx["padj"]
        order["sex_effect_padj"] = order.gene_id.map(sex_p)
        order["sex_effect_log2FC_male_vs_female"] = order.gene_id.map(sx["log2FC"])
        panel["sex_effect_padj"] = panel.gene_id.map(sex_p)
    panel.to_csv(ctx.out(STAGE, "final_panel.csv"), index=False)
    order.to_csv(ctx.out(STAGE, "stability_ordering_all_genes.csv"), index=False)

    # ---- D5 stop rule --------------------------------------------------------------------------
    top10 = order.head(10)
    def flags(df):
        f = df.sex_chrom.copy()
        if sex_p is not None:
            f = f | (df.gene_id.map(sex_p).fillna(1.0) < 0.05)
        return f
    flagged_panel = panel.symbol[flags(panel)].tolist()
    flagged_top10 = top10.symbol[flags(top10)].tolist()
    stop = bool(flagged_panel or flagged_top10)
    if stop:
        log.warning("D5 STOP RULE TRIGGERED: sex-linked gene(s) in the final panel %s / top-10 of the ordering %s. "
                    "The authors must be told before anything further is done.", flagged_panel, flagged_top10)
    else:
        log.info("D5 stop rule: no sex-chromosome gene and no gene with a significant sex effect (FDR < 0.05) in the "
                 "final panel or the top 10 of the ordering%s", "" if sex_p is not None else " (sex-effect table missing: chromosome check only)")

    # ---- P5: how often does the exact panel recur --------------------------------------------
    rec_all, pre_all = recurrence(all_fits, cols, K)
    rec_cv, _ = recurrence(cv_fits, cols, K)
    rec_sub, _ = recurrence(sub_fits, cols, K)
    pre_all.to_csv(ctx.out(STAGE, "P5_prefix_recurrence.csv"), index=False)
    pd.DataFrame([{"group": g, **r} for g, r in (("cv500", rec_cv), ("subsamples200", rec_sub), ("all700", rec_all))]
                 ).to_csv(ctx.out(STAGE, "P5_panel_recurrence.csv"), index=False)
    log.info("P5 final panel (K=%d) recurrence over 700 fits: exact top-K set %.1f%%, contained in top-K set %.1f%%, "
             "contained in non-zero set %.1f%%, mean Jaccard %.3f", K, 100 * rec_all["exact_equal_topK_set"],
             100 * rec_all["contains_panel_topK_set"], 100 * rec_all["contains_panel_nonzero_set"],
             rec_all["mean_jaccard_topK_set"])

    # ---- correlation structure of the panel -----------------------------------------------------
    rho = pd.DataFrame(X, columns=panel.symbol.to_numpy()).corr(method="spearman")
    rho.to_csv(ctx.out(STAGE, "final_panel_spearman_log2cpm.csv"))

    # ---- figure ---------------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    plotting.style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    top = order.head(20).iloc[::-1]
    axes[0].barh(top.symbol, top.freq_nonzero, color=[plotting.EVD if b else plotting.HEALTHY for b in top.in_baseline_panel])
    axes[0].set_xlabel("Fraction of 700 fits with non-zero coefficient")
    axes[0].set_title("Selection frequency, top 20 of the ordering")
    ks = sorted(set(kc_cv.index) | set(kc_sub.index))
    axes[1].bar(range(len(ks)), [kc_cv.get(k, 0) / len(cv_fits) for k in ks], color=plotting.HEALTHY, label="500 CV fits")
    axes[1].plot(range(len(ks)), [kc_sub.get(k, 0) / len(sub_fits) for k in ks], color=plotting.EVD, marker="o", label="200 subsamples")
    axes[1].set_xticks(range(len(ks)))
    axes[1].set_xticklabels(ks)
    axes[1].set_xlabel("K chosen by inner CV")
    axes[1].set_ylabel("Fraction of fits")
    axes[1].legend()
    axes[1].set_title("Panel size chosen in the inner loop")
    fig.tight_layout()
    plotting.save(fig, ctx.out(STAGE, "stability_and_K.png"))

    summ = {"final_K": K, "final_C": C, "final_l1": l1, "final_inner_auc": tun["inner_auc"],
            "n_inner_splits_final_tuning": len(inner), "final_panel": panel.symbol.tolist(),
            "final_panel_sex_chrom": panel.symbol[panel.sex_chrom].tolist(),
            "apparent_in_sample_auc_OPTIMISTIC": apparent,
            "K_chosen_counts_cv500": {int(k): int(v) for k, v in kc_cv.items()},
            "K_chosen_counts_sub200": {int(k): int(v) for k, v in kc_sub.items()},
            "K_chosen_median_cv500": float(np.median([f["K"] for f in cv_fits])),
            "K_chosen_median_sub200": float(np.median([f["K"] for f in sub_fits])),
            "n_genes_ever_selected": int(len(order)),
            "D5_stop_rule_triggered": stop, "D5_flagged_in_panel": flagged_panel, "D5_flagged_in_top10": flagged_top10,
            "P5_recurrence": {"cv500": rec_cv, "subsamples200": rec_sub, "all700": rec_all},
            "baseline_panel_in_ordering": order[order.in_baseline_panel][
                ["symbol", "ordering_rank", "freq_nonzero", "mean_abs_std_coef", "median_rank"]].to_dict("records"),
            "top20_ordering": order.head(20)[["ordering_rank", "symbol", "freq_nonzero", "freq_selected_topK",
                                              "mean_abs_std_coef", "median_rank", "sex_chrom"]].to_dict("records")}
    ctx.out(STAGE, "stability_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    ctx.cache["panel"] = {"symbols": panel.symbol.tolist(), "cols": cols, "K": K, "C": C, "l1": l1,
                          "order": order, "ordering_cols": order["col"].to_numpy()}
    log.info("FINAL PANEL (K=%d): %s", K, ", ".join(panel.symbol))
    log.info("top 12 of the ordering:\n%s", order.head(12)[["ordering_rank", "symbol", "freq_nonzero", "freq_nonzero_cv500",
             "freq_nonzero_sub200", "mean_abs_std_coef", "median_rank", "sex_chrom"]].round(3).to_string(index=False))
    log.info("baseline-panel genes in the ordering:\n%s", order[order.in_baseline_panel][
             ["ordering_rank", "symbol", "freq_nonzero", "median_rank"]].round(3).to_string(index=False)
             if order.in_baseline_panel.any() else "none of the five baseline genes was ever selected")
