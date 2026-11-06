"""Item 6b: what the leading PCs reflect, a composition-only baseline, and the P1 comparisons.

1. PC1-PC5 (VST, blind=TRUE primary; blind=FALSE as legacy) against inferred sex, log10 library size, globin
   fraction, every deconvolution score, and condition (for context). Spearman and Pearson, one BH over all tests.
2. Composition-only baseline classifier: standardised deconvolution scores into L2 logistic regression, C tuned in
   the same inner scheme, seed 42 + 100 repeats (each method alone and all combined).
3. Panel genes (final panel, baseline panel, top 5 of the ordering) against the deconvolution scores. Exploratory.
4. P1: paired difference primary minus {PC-only, PC-only up to 20 PCs, composition-only}, same repeats and outer splits:
   CI-A = paired subject bootstrap; partition variability = percentiles of the per-repeat difference;
   seed 42 = pooled-AUC difference with the DeLong test for correlated AUCs."""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

from evd import plotting
from evd.fixedfeat import fixed_features_cv
from evd.runs import get_variant, register, repeat_arrays, save_arrays, variant_stats
from evd.nested import ModelCfg
from evd.stats import delong_paired, paired_bootstrap_diff, repeat_summary
from stages.item6 import _pca

STAGE = "item6b"


def _deconv(ctx):
    d = ctx.cache.get("deconv")
    if d is not None:
        return d
    p = ctx.run_dir / "item9" / "deconvolution_scores_per_sample.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0).drop(columns="condition")
    return None


def _covariates(ctx, dec):
    ds, cfg = ctx.ds, ctx.cfg
    tot = ds.counts.sum(1).astype(float)
    gl = [np.flatnonzero(ds.symbols == g) for g in cfg["globin_genes"]]
    gl = np.concatenate([g for g in gl if len(g)])
    cov = pd.DataFrame({"sex_male": ds.sex_male.astype(float), "log10_library_size": np.log10(tot),
                        "globin_fraction": ds.counts[:, gl].sum(1) / tot, "condition_EVD_S": ds.y.astype(float)},
                       index=ds.sample_ids)
    grp = {c: "technical/demographic" for c in cov.columns}
    if dec is not None:
        for c in dec.columns:
            cov[c] = dec[c].to_numpy()
            grp[c] = "deconvolution:" + c.split(":")[0]
    return cov, grp


def _pc_correlations(ctx, cov, grp):
    ds, d6 = ctx.ds, ctx.run_dir / "item6"
    out = {}
    for label, f in (("blindTRUE", "vst_blindTRUE.tsv"), ("blindFALSE", "vst_blindFALSE.tsv")):
        ev, sc, _ = _pca(d6 / f, list(ds.sample_ids))
        rows = []
        for k in range(5):
            for c in cov.columns:
                v = cov[c].to_numpy(float)
                rs, ps = stats.spearmanr(sc[:, k], v)
                rp, pp = stats.pearsonr(sc[:, k], v)
                rows.append({"vst": label, "PC": k + 1, "pct_variance": ev[k], "covariate": c, "group": grp[c],
                             "spearman_rho": rs, "spearman_p": ps, "pearson_r": rp, "pearson_p": pp})
        df = pd.DataFrame(rows)
        df["q_BH_spearman"] = multipletests(df.spearman_p, method="fdr_bh")[1]
        out[label] = df
    allc = pd.concat(out.values())
    allc.to_csv(ctx.out(STAGE, "pc1_5_correlations.csv"), index=False)
    return out


def _heatmap(ctx, df):
    import matplotlib.pyplot as plt
    plotting.style()
    piv = df.pivot(index="covariate", columns="PC", values="spearman_rho")
    qv = df.pivot(index="covariate", columns="PC", values="q_BH_spearman")
    order = list(df.drop_duplicates("covariate").covariate)
    fig, ax = plt.subplots(figsize=(4.6, 0.26 * len(order) + 1.4))
    im = ax.imshow(piv.loc[order].to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"PC{k}" for k in range(1, 6)])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    for i, c in enumerate(order):
        for j in range(5):
            if qv.loc[c].iloc[j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title("PC1-5 (VST blind=TRUE); * = BH q < 0.05", fontsize=8)
    fig.tight_layout()
    plotting.save(fig, ctx.out(STAGE, "pc_correlations_heatmap.png"))


def _composition_baselines(ctx, dec):
    cfg, y = ctx.cfg, ctx.ds.y
    seeds = [cfg["seeds"]["reference"]] + cfg.repeat_seeds
    cg = cfg["item6"]["c_grid"]
    sets = {m: [c for c in dec.columns if c.startswith(m + ":")] for m in cfg["item9"]["methods"]}
    sets["all"] = list(dec.columns)
    out = {}
    for name, cols in sets.items():
        if not cols:
            continue
        F = dec[cols].to_numpy(float)
        res = Parallel(n_jobs=ctx.n_jobs)(delayed(fixed_features_cv)(F, y, s, cg) for s in seeds)
        tag = f"composition_{name}"
        register(ctx, tag, res[0], res[1:])
        save_arrays(ctx, STAGE, tag, {"ref": res[0], "reps": res[1:]})
        st = variant_stats(y, {"ref": res[0], "reps": res[1:]}, cfg["item1"]["bootstrap_B"], cfg["seeds"]["bootstrap"])
        out[tag] = st
        ctx.log.info("[%s] %d features | seed42 pooled %.4f, per-fold mean %.4f | repeated CV mean-fold AUC %.4f (CI-A %.4f-%.4f)",
                     tag, len(cols), st["seed42"]["pooled_oof_auc"], st["seed42"]["per_fold_mean"],
                     st["repeated_cv"]["mean_fold_auc"], st["repeated_cv"]["ci_A_subject_bootstrap"]["lo"],
                     st["repeated_cv"]["ci_A_subject_bootstrap"]["hi"])
    return out


def _panel_vs_composition(ctx, dec):
    ds, y = ctx.ds, ctx.ds.y
    genes = list(dict.fromkeys(ctx.cfg["baseline"]["panel"] + ctx.cache.get("panel", {}).get("symbols", [])
                               + (ctx.cache["panel"]["order"].symbol.head(5).tolist() if "panel" in ctx.cache else [])))
    rows = []
    for g in genes:
        j = np.flatnonzero(ds.symbols == g)
        if len(j) != 1:
            continue
        x = ds.LC[:, j[0]]
        for c in dec.columns:
            v = dec[c].to_numpy(float)
            r, p = stats.spearmanr(x, v)
            within = [stats.spearmanr(x[y == k], v[y == k]).statistic for k in (0, 1)]
            rows.append({"gene": g, "in_final_panel": g in ctx.cache.get("panel", {}).get("symbols", []),
                         "in_baseline_panel": g in ctx.cfg["baseline"]["panel"], "score": c, "spearman_rho": r,
                         "p": p, "within_group_mean_rho": float(np.mean(within))})
    df = pd.DataFrame(rows)
    df["q_BH"] = multipletests(df.p, method="fdr_bh")[1]
    df.sort_values("p").to_csv(ctx.out(STAGE, "panel_genes_vs_composition.csv"), index=False)
    return df


def _p1(ctx):
    y, cfg = ctx.ds.y, ctx.cfg
    B, rs = cfg["item1"]["bootstrap_B"], cfg["seeds"]["bootstrap"]
    prim = get_variant(ctx, "primary", ModelCfg.from_config(cfg), True)
    oa, fa = repeat_arrays(prim)
    out = {}
    for other in ("PC_only", "S7_PC_only_to_20", "composition_all"):
        run_ = ctx.cache.get("runs", {}).get(other)
        if run_ is None or not run_["reps"]:
            ctx.log.warning("P1: %s not available", other)
            continue
        ob, fb = repeat_arrays(run_)
        d = np.array([a["mean_fold_auc"] - b["mean_fold_auc"] for a, b in zip(prim["reps"], run_["reps"])])
        out[other] = {
            "comparison": f"primary minus {other} (mean-fold AUC; same repeat seeds and outer splits)",
            "CI_A_paired_subject_bootstrap": paired_bootstrap_diff(y, oa, fa, ob, fb, B=B, seed=rs),
            "partition_variability_of_difference": repeat_summary(d),
            "frac_repeats_primary_better": float((d > 0).mean()),
            "seed42_pooled_difference_DeLong_paired": delong_paired(y, prim["ref"]["oof"], run_["ref"]["oof"]),
            "seed42_per_fold_mean_difference": prim["ref"]["mean_fold_auc"] - run_["ref"]["mean_fold_auc"]}
        o = out[other]
        ctx.log.info("P1 primary - %s: mean-fold AUC diff %.4f | CI-A paired bootstrap %.4f to %.4f | partition variability "
                     "%.4f to %.4f | primary better in %.0f%% of repeats | seed 42 pooled diff %.4f (DeLong %.4f to %.4f, p=%.3f)",
                     other, o["CI_A_paired_subject_bootstrap"]["estimate"], o["CI_A_paired_subject_bootstrap"]["boot_lo"],
                     o["CI_A_paired_subject_bootstrap"]["boot_hi"], o["partition_variability_of_difference"]["p2.5"],
                     o["partition_variability_of_difference"]["p97.5"], 100 * o["frac_repeats_primary_better"],
                     o["seed42_pooled_difference_DeLong_paired"]["diff"], o["seed42_pooled_difference_DeLong_paired"]["lo"],
                     o["seed42_pooled_difference_DeLong_paired"]["hi"], o["seed42_pooled_difference_DeLong_paired"]["p"])
    ctx.out(STAGE, "P1_paired_comparisons.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run(ctx):
    log = ctx.log
    if not (ctx.run_dir / "item6" / "vst_blindTRUE.tsv").exists():
        raise RuntimeError("item6b needs stage item6 in the same run")
    dec = _deconv(ctx)
    if dec is None:
        log.warning("no deconvolution scores available: composition parts are skipped")
    cov, grp = _covariates(ctx, dec)
    cors = _pc_correlations(ctx, cov, grp)
    _heatmap(ctx, cors["blindTRUE"])
    c = cors["blindTRUE"]
    for k in range(1, 6):
        s = c[(c.PC == k)].sort_values("spearman_p").head(4)
        log.info("PC%d (%.1f%% variance), strongest Spearman associations (BH q): %s", k, s.pct_variance.iloc[0],
                 "; ".join(f"{r.covariate} rho={r.spearman_rho:+.2f} (q={r.q_BH_spearman:.3g})" for r in s.itertuples()))
    if dec is not None:
        _composition_baselines(ctx, dec)
        pg = _panel_vs_composition(ctx, dec)
        log.info("panel genes vs composition: %d of %d correlations with BH q < 0.05; strongest:\n%s", int((pg.q_BH < 0.05).sum()),
                 len(pg), pg.sort_values("p").head(8)[["gene", "score", "spearman_rho", "within_group_mean_rho", "q_BH"]].round(3).to_string(index=False))
    _p1(ctx)
