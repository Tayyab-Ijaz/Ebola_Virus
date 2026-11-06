"""Item 6: PCA variance explained + PC-only baseline classifier."""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from evd import plotting
from evd.pcbase import pc_baseline_cv
from evd.runs import register, save_arrays, variant_stats
from evd.stats import delong_ci, repeat_summary, repeated_cv_bootstrap

STAGE = "item6"


def _pca(path, sample_order):
    v = pd.read_csv(path, sep="\t").set_index("GeneID")
    v = v[sample_order]
    X = v.T.to_numpy(float)
    X = X - X.mean(0)
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum() * 100
    return ev, u * s, v.shape[0]


def run(ctx):
    cfg, ds, log = ctx.cfg, ctx.ds, ctx.log
    d = ctx.stage_dir(STAGE)
    inp = ctx.inputs_dir()
    ctx.run_r("vst_pca.R", inp, d)
    names = list(ds.sample_ids)
    y = ds.y

    # ---- scree + single-PC AUCs ---------------------------------------------------------------
    rows, scores = [], {}
    sources = {"vst_blindFALSE (as legacy)": d / "vst_blindFALSE.tsv", "vst_blindTRUE": d / "vst_blindTRUE.tsv",
               "stored legacy vst_matrix.tsv": cfg.path("stored_vst")}
    for label, p in sources.items():
        ev, sc, ng = _pca(p, names)
        scores[label] = sc
        for k in range(10):
            auc = roc_auc_score(y, sc[:, k])
            rows.append({"matrix": label, "n_genes": ng, "PC": k + 1, "pct_variance": round(ev[k], 3),
                         "cum_pct": round(ev[:k + 1].sum(), 3), "auc_raw": round(auc, 4),
                         "auc_orientation_free": round(max(auc, 1 - auc), 4)})
    scree = pd.DataFrame(rows)
    scree.to_csv(ctx.out(STAGE, "pca_scree_and_pc_auc.csv"), index=False)
    log.info("PC1-5 %% variance (blind=FALSE): %s",
             scree[scree.matrix.str.startswith("vst_blindFALSE")].pct_variance.head(5).tolist())
    log.info("PC1-5 %% variance (blind=TRUE):  %s",
             scree[scree.matrix == "vst_blindTRUE"].pct_variance.head(5).tolist())
    a = scores["vst_blindFALSE (as legacy)"]
    b = scores["stored legacy vst_matrix.tsv"]
    log.info("agreement of re-generated and stored blind=FALSE PC1/PC2 scores (|corr|): %.6f / %.6f",
             abs(np.corrcoef(a[:, 0], b[:, 0])[0, 1]), abs(np.corrcoef(a[:, 1], b[:, 1])[0, 1]))

    # ---- figures ------------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    plotting.style()
    sex = pd.read_csv(inp / "samples.tsv", sep="\t")["sex"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, label in zip(axes[:2], ["vst_blindFALSE (as legacy)", "vst_blindTRUE"]):
        sc = scores[label]
        ev = scree[scree.matrix.str.startswith(label[:14])].pct_variance.to_numpy()
        for lab, col, nm in ((0, plotting.HEALTHY, "Healthy"), (1, plotting.EVD, "EVD-S")):
            for sx, mk in (("male", "o"), ("female", "^")):
                m = (y == lab) & (sex == sx)
                ax.scatter(sc[m, 0], sc[m, 1], c=col, marker=mk, s=28, edgecolor="white", linewidth=0.6,
                           label=f"{nm}, {sx}")
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
        ax.set_title(label.replace("vst_", "VST ").replace("blindFALSE", "blind=FALSE").replace("blindTRUE", "blind=TRUE"))
    axes[0].legend(fontsize=7, loc="best")
    ax = axes[2]
    for label, ls in (("vst_blindFALSE (as legacy)", "-"), ("vst_blindTRUE", "--")):
        s = scree[scree.matrix.str.startswith(label[:14])]
        ax.plot(s.PC, s.pct_variance, marker="o", ls=ls, color=plotting.INK if ls == "-" else plotting.MUTED,
                label="blind=FALSE" if ls == "-" else "blind=TRUE")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Scree")
    ax.legend()
    fig.tight_layout()
    plotting.save(fig, ctx.out(STAGE, "pca_scatter_scree.png"))

    # ---- PC-only baseline classifier ----------------------------------------------------------
    seeds = [cfg["seeds"]["reference"]] + cfg.repeat_seeds
    log.info("PC-only baseline: %d nested CVs (seed %d + %d repeats), n_jobs=%d", len(seeds),
             seeds[0], len(seeds) - 1, ctx.n_jobs)
    res = Parallel(n_jobs=ctx.n_jobs)(delayed(pc_baseline_cv)(ds, y, s, cfg) for s in seeds)
    ref, reps = res[0], res[1:]
    register(ctx, "PC_only", ref, reps)
    save_arrays(ctx, STAGE, "PC_only", {"ref": ref, "reps": reps})
    pd.DataFrame([f for r in res for f in r["folds"]]).to_csv(ctx.out(STAGE, "pc_baseline_folds.csv"), index=False)
    pd.DataFrame({"sample": ds.sample_ids, "y": y, "fold_seed42": ref["fold_id"], "oof_seed42": ref["oof"]}
                 ).to_csv(ctx.out(STAGE, "pc_baseline_oof_seed42.csv"), index=False)
    per_repeat = [r["mean_fold_auc"] for r in reps]
    boot = repeated_cv_bootstrap(y, np.vstack([r["oof"] for r in reps]), np.vstack([r["fold_id"] for r in reps]),
                                 B=cfg["item6"]["bootstrap_B"], seed=cfg["seeds"]["bootstrap"])
    pooled_boot = repeated_cv_bootstrap(y, ref["oof"], np.zeros(len(y), int), B=cfg["item6"]["bootstrap_B"],
                                        seed=cfg["seeds"]["bootstrap"])
    fa = [f["auc"] for f in ref["folds"]]
    summary = {
        "classifier": "PC-only (top-2000 variance genes -> PCA -> L2 logistic), tuned n_pc and C inside CV",
        "seed42": {"pooled_oof_auc": ref["pooled_auc"], "pooled_oof_ap": ref["pooled_ap"],
                   "pooled_delong": delong_ci(y, ref["oof"]), "pooled_bootstrap": pooled_boot,
                   "per_fold_auc": fa, "per_fold_mean": float(np.mean(fa)), "per_fold_sd_ddof1": float(np.std(fa, ddof=1)),
                   "chosen_n_pc": [f["n_pc"] for f in ref["folds"]]},
        "repeated_cv": {"statistic": "mean over repeats of mean-of-fold AUC", "per_repeat_summary": repeat_summary(per_repeat),
                        "subject_bootstrap": boot,
                        "n_pc_chosen_counts": pd.Series([f["n_pc"] for r in reps for f in r["folds"]]).value_counts().sort_index().to_dict()},
    }
    (ctx.out(STAGE, "pc_baseline_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("PC baseline seed42: pooled OOF AUC %.4f, per-fold mean %.4f", ref["pooled_auc"], np.mean(fa))
    log.info("PC baseline repeated CV: mean %.4f, bootstrap 95%% CI %.4f-%.4f, repeat 2.5-97.5%% %.4f-%.4f",
             boot["estimate"], boot["boot_lo"], boot["boot_hi"],
             summary["repeated_cv"]["per_repeat_summary"]["p2.5"], summary["repeated_cv"]["per_repeat_summary"]["p97.5"])

    # ---- S7: grid-boundary sensitivity, up to 20 PCs ------------------------------------------
    grid7 = cfg["sensitivity"]["pc_grid_s7"]
    res7 = Parallel(n_jobs=ctx.n_jobs)(delayed(pc_baseline_cv)(ds, y, s, cfg, grid7) for s in seeds)
    register(ctx, "S7_PC_only_to_20", res7[0], res7[1:])
    save_arrays(ctx, STAGE, "S7_PC_only_to_20", {"ref": res7[0], "reps": res7[1:]})
    st7 = variant_stats(y, {"ref": res7[0], "reps": res7[1:]}, cfg["item6"]["bootstrap_B"], cfg["seeds"]["bootstrap"])
    st7["n_pc_chosen_counts"] = pd.Series([f["n_pc"] for r in res7[1:] for f in r["folds"]]).value_counts().sort_index().to_dict()
    ctx.out(STAGE, "pc_baseline_S7_summary.json").write_text(json.dumps(st7, indent=2), encoding="utf-8")
    log.info("S7 PC-only (PCs up to 20): seed42 pooled %.4f, per-fold mean %.4f | repeated CV mean-fold AUC %.4f "
             "(CI-A %.4f-%.4f; partition variability %.4f-%.4f) | n_pc chosen: %s", res7[0]["pooled_auc"],
             res7[0]["mean_fold_auc"], st7["repeated_cv"]["mean_fold_auc"], st7["repeated_cv"]["ci_A_subject_bootstrap"]["lo"],
             st7["repeated_cv"]["ci_A_subject_bootstrap"]["hi"], st7["repeated_cv"]["partition_variability"]["p2.5"],
             st7["repeated_cv"]["partition_variability"]["p97.5"], st7["n_pc_chosen_counts"])