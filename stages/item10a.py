"""Item 10(a): inferred sex, DESeq2 with and without sex as covariate, sex-linked-gene flags (D5)."""
import json

import numpy as np
import pandas as pd
from scipy import stats

from evd import plotting
from evd.splits import outer_splits

STAGE = "item10a"


def _read(d, name):
    return pd.read_csv(d / name, sep="\t", dtype={"GeneID": str})


def run(ctx):
    cfg, ds, log = ctx.cfg, ctx.ds, ctx.log
    d = ctx.stage_dir(STAGE)
    inp = ctx.inputs_dir()
    sex = pd.read_csv(inp / "samples.tsv", sep="\t")
    sex.to_csv(ctx.out(STAGE, "inferred_sex_per_sample.csv"), index=False)

    # ---- sex vs group, concordance, fold composition ------------------------------------------
    tab = pd.crosstab(sex.condition, sex.sex)
    fisher = stats.fisher_exact(tab.loc[["Healthy", "EVD-S"], ["female", "male"]].to_numpy())
    log.info("inferred sex by group:\n%s\nFisher exact p = %.4f", tab.to_string(), fisher[1])
    disc = sex[~sex.concordant_with_xist]
    log.info("Y-gene vs XIST discordant samples: %d %s", len(disc), disc["sample"].tolist())
    y = ds.y
    comp = []
    for f, (tr, te) in enumerate(outer_splits(y, cfg["seeds"]["reference"], cfg["model"]["outer_splits"]), 1):
        for part, idx in (("train", tr), ("test", te)):
            s = sex.iloc[idx]
            comp.append({"fold": f, "part": part, "n": len(idx),
                         "HD_female": int(((s.label == 0) & (s.sex == "female")).sum()),
                         "HD_male": int(((s.label == 0) & (s.sex == "male")).sum()),
                         "EVD_female": int(((s.label == 1) & (s.sex == "female")).sum()),
                         "EVD_male": int(((s.label == 1) & (s.sex == "male")).sum())})
    pd.DataFrame(comp).to_csv(ctx.out(STAGE, "sex_by_group_and_fold_seed42.csv"), index=False)
    (ctx.out(STAGE, "sex_group_table.json")).write_text(json.dumps({
        "table": tab.to_dict(), "fisher_p": float(fisher[1]), "fisher_odds_ratio": float(fisher[0]),
        "n_discordant_Y_vs_XIST": int(len(disc))}, indent=2), encoding="utf-8")

    # ---- DESeq2 -------------------------------------------------------------------------------
    ctx.run_r("sex_deseq.R", inp, d)
    c0 = _read(d, "deseq_condition_only.tsv")
    c1 = _read(d, "deseq_condition_adj_sex.tsv")
    sx = _read(d, "deseq_sex_effect.tsv")
    m = c0.merge(c1, on="GeneID", suffixes=("_unadj", "_adj")).merge(
        sx[["GeneID", "log2FC", "stat", "padj"]].rename(columns={"log2FC": "log2FC_sex", "stat": "stat_sex",
                                                                "padj": "padj_sex"}), on="GeneID")
    sym = pd.DataFrame({"GeneID": ds.gene_ids, "symbol": ds.symbols, "on_sex_chrom": ds.sex_chrom})
    m = m.merge(sym, on="GeneID", how="left")
    m["sex_DE"] = m.padj_sex < 0.05
    m["sex_linked_flag"] = m.on_sex_chrom | m.sex_DE
    m.to_csv(ctx.out(STAGE, "deseq_unadjusted_vs_sex_adjusted_all_genes.csv"), index=False)

    deg0 = (m.padj_unadj < .05) & (m.log2FC_unadj.abs() >= 1)
    deg1 = (m.padj_adj < .05) & (m.log2FC_adj.abs() >= 1)
    orig = pd.read_csv(cfg.path("stored_deseq2_degs"), sep="\t", dtype={"GeneID": str})
    in_orig = m.GeneID.isin(orig.GeneID)
    summ = {
        "n_genes_tested": int(len(m)),
        "n_DEG_unadjusted": int(deg0.sum()), "n_DEG_sex_adjusted": int(deg1.sum()), "n_DEG_both": int((deg0 & deg1).sum()),
        "pearson_log2FC_unadj_vs_adj": float(m.log2FC_unadj.corr(m.log2FC_adj)),
        "n_original85_still_DEG_after_sex_adjustment": int((in_orig & deg1).sum()), "n_original": int(in_orig.sum()),
        "n_genes_sex_DE_padj_lt_0.05": int(m.sex_DE.sum()),
        "n_sex_DE_on_autosomes": int((m.sex_DE & ~m.on_sex_chrom).sum()),
        "n_sex_chrom_genes_tested": int(m.on_sex_chrom.sum()),
    }
    log.info("DEGs: unadjusted %d, sex-adjusted %d, shared %d; original-85 still DEG after adjustment: %d/%d",
             summ["n_DEG_unadjusted"], summ["n_DEG_sex_adjusted"], summ["n_DEG_both"],
             summ["n_original85_still_DEG_after_sex_adjustment"], summ["n_original"])
    log.info("sex-DE genes (padj<0.05): %d (%d autosomal)", summ["n_genes_sex_DE_padj_lt_0.05"], summ["n_sex_DE_on_autosomes"])

    # ---- panel / stability genes (D5 flags) ---------------------------------------------------
    genes = cfg["baseline"]["panel"] + [g for g in cfg["baseline"]["stability_genes"] if g not in cfg["baseline"]["panel"]]
    pg = m[m.symbol.isin(genes)].drop_duplicates("symbol").set_index("symbol").reindex(genes)   # one symbol is shared by 3 unrelated genes; none of `genes`
    pg.insert(0, "in_baseline_panel", [g in cfg["baseline"]["panel"] for g in genes])
    cols = ["in_baseline_panel", "on_sex_chrom", "log2FC_unadj", "padj_unadj", "log2FC_adj", "padj_adj",
            "log2FC_sex", "padj_sex", "sex_DE", "sex_linked_flag"]
    pg[cols].round(5).reset_index().to_csv(ctx.out(STAGE, "panel_and_stability_genes_sex_check.csv"), index=False)
    for g in genes:
        r = pg.loc[g]
        log.info("  %-8s %s LFC %.2f -> %.2f (adj) | sex effect LFC %.2f padj %.2g | sex-chrom=%s sex_linked_flag=%s",
                 g, "PANEL" if r.in_baseline_panel else "     ", r.log2FC_unadj, r.log2FC_adj, r.log2FC_sex,
                 r.padj_sex, r.on_sex_chrom, r.sex_linked_flag)
    summ["baseline_panel_sex_linked"] = [g for g in cfg["baseline"]["panel"] if bool(pg.loc[g].sex_linked_flag)]
    summ["stability_genes_sex_linked"] = [g for g in genes if bool(pg.loc[g].sex_linked_flag)]
    reprod = c0.merge(pd.read_csv(cfg.path("stored_deseq2_all"), sep="\t", dtype={"GeneID": str}), on="GeneID",
                      suffixes=("_new", "_stored"))
    summ["unadjusted_rerun_vs_stored_max_abs_diff_log2FC"] = float((reprod.log2FC - reprod.log2FoldChange).abs().max())
    (ctx.out(STAGE, "summary.json")).write_text(json.dumps(summ, indent=2), encoding="utf-8")

    # ---- figure -------------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    plotting.style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    ax.scatter(m.log2FC_unadj, m.log2FC_adj, s=3, color="#9a9a9a", alpha=0.35, linewidths=0, rasterized=True)
    ax.plot([-4, 6], [-4, 6], color=plotting.MUTED, lw=0.8)
    for g in cfg["baseline"]["panel"]:
        r = pg.loc[g]
        ax.scatter(r.log2FC_unadj, r.log2FC_adj, s=22, color=plotting.EVD, zorder=3)
        ax.annotate(g, (r.log2FC_unadj, r.log2FC_adj), xytext=(4, -9), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Condition log2FC, ~ Condition")
    ax.set_ylabel("Condition log2FC, ~ sex + Condition")
    ax.set_title("Effect of adjusting for inferred sex")
    ax = axes[1]
    ax.scatter(m.log2FC_sex, -np.log10(m.padj_sex.clip(lower=1e-300)), s=3, color="#9a9a9a", alpha=0.35, linewidths=0,
               rasterized=True)
    ax.axhline(-np.log10(0.05), color=plotting.MUTED, lw=0.8, ls="--")
    for g in cfg["baseline"]["panel"]:
        r = pg.loc[g]
        ax.scatter(r.log2FC_sex, -np.log10(max(r.padj_sex, 1e-300)), s=22, color=plotting.EVD, zorder=3)
        ax.annotate(g, (r.log2FC_sex, -np.log10(max(r.padj_sex, 1e-300))), xytext=(4, 2), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Sex effect log2FC (male vs female)")
    ax.set_ylabel("-log10 adjusted p")
    ax.set_title("Sex effect (dashed: FDR 0.05)")
    fig.tight_layout()
    plotting.save(fig, ctx.out(STAGE, "sex_adjustment.png"))
