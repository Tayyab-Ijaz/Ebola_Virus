"""Item 9 / P2: exploratory in-silico cell-type deconvolution (EPIC blood reference, quanTIseq, MCP-counter via
immunedeconv) from TPM = counts / gene length. No CIBERSORT, no validation possible (no measured blood counts).
Outputs the per-sample scores and the group comparison (Welch t, Mann-Whitney U, BH over all tests)."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

STAGE = "item9"


def _tpm(ctx):
    ds = ctx.ds
    annot = pd.read_csv(ctx.cfg.path("annotation"), sep="\t", dtype={"GeneID": str}, low_memory=False)
    length = annot.drop_duplicates("GeneID").set_index("GeneID")["Length"].reindex(ds.gene_ids).to_numpy(float)
    ok = np.isfinite(length) & (length > 0)
    rate = ds.counts[:, ok].astype(float) / (length[ok] / 1000.0)
    tpm = rate / rate.sum(1, keepdims=True) * 1e6
    df = pd.DataFrame(tpm.T, index=ds.symbols[ok], columns=ds.sample_ids)
    df = df[df.index != ""]
    df = df.loc[df.mean(1).sort_values(ascending=False).index]
    df = df[~df.index.duplicated(keep="first")]                       # duplicated symbols: keep the most expressed gene
    return df.rename_axis("gene_symbol"), int((~ok).sum())


def run(ctx):
    cfg, ds, log, y = ctx.cfg, ctx.ds, ctx.log, ctx.ds.y
    d = ctx.stage_dir(STAGE)
    tpm, n_nolen = _tpm(ctx)
    log.info("TPM matrix: %d symbols x %d samples (%d genes without a usable length dropped)", *tpm.shape, n_nolen)
    tpm.to_csv(d / "tpm_symbols_input.tsv", sep="\t")
    try:
        ctx.run_r("deconv.R", d / "tpm_symbols_input.tsv", d, ",".join(cfg["item9"]["methods"]))
    except RuntimeError as exc:
        (d / "SKIPPED.txt").write_text(f"deconvolution not available: {exc}\n", encoding="utf-8")
        ctx.cache["deconv"] = None
        log.error("deconvolution failed; dependent analyses will be skipped: %s", exc)
        return
    feats, rows = {}, []
    for m in cfg["item9"]["methods"]:
        p = d / f"deconv_{m}.tsv"
        if not p.exists():
            log.error("method %s produced no output", m)
            continue
        df = pd.read_csv(p, sep="\t").set_index("cell_type")
        df = df[ds.sample_ids]
        for ct, v in df.iterrows():
            v = v.to_numpy(float)
            if not np.isfinite(v).all() or np.nanstd(v) == 0:
                continue
            feats[f"{m}:{ct}"] = v
            a, b = v[y == 1], v[y == 0]
            t, pw = stats.ttest_ind(a, b, equal_var=False)
            u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"method": m, "cell_type": ct, "mean_EVD_S": a.mean(), "mean_Healthy": b.mean(),
                         "diff": a.mean() - b.mean(), "welch_t": t, "p_welch": pw, "p_mannwhitney": pu})
    F = pd.DataFrame(feats, index=ds.sample_ids)
    F.insert(0, "condition", np.where(y == 1, "EVD-S", "Healthy"))
    F.to_csv(ctx.out(STAGE, "deconvolution_scores_per_sample.csv"))
    comp = pd.DataFrame(rows)
    comp["q_welch_BH"] = multipletests(comp.p_welch, method="fdr_bh")[1]
    comp["q_mannwhitney_BH"] = multipletests(comp.p_mannwhitney, method="fdr_bh")[1]
    comp.sort_values("p_mannwhitney").to_csv(ctx.out(STAGE, "group_comparison.csv"), index=False)
    ctx.cache["deconv"] = F.drop(columns="condition")
    log.info("deconvolution: %d score columns from %s; %d with BH q < 0.05 (Mann-Whitney) between EVD-S and Healthy",
             F.shape[1] - 1, cfg["item9"]["methods"], int((comp.q_mannwhitney_BH < 0.05).sum()))
    log.info("top group differences:\n%s", comp.sort_values("p_mannwhitney").head(8)[
        ["method", "cell_type", "mean_EVD_S", "mean_Healthy", "q_mannwhitney_BH"]].round(4).to_string(index=False))
