"""Item 7: Welch (log2 CPM+1) vs DESeq2 vs limma-voom rankings, full data and on the seed-42 splits."""
import json

import numpy as np
import pandas as pd

from evd import plotting
from evd.selection import welch_rank
from evd.splits import splits_json
from evd.stats import spearman_abs, topn_overlap

STAGE = "item7"


def _read(d, method, key):
    return pd.read_csv(d / f"rank_{method}_{key}.tsv", sep="\t", dtype={"GeneID": str})


def _welch_table(ds, idx, cfg):
    rk = welch_rank(ds, idx, ds.y[idx], None, cfg["selection"]["gate_fdr"], cfg["selection"]["gate_lfc"])
    cols = rk.order
    return pd.DataFrame({"GeneID": ds.gene_ids[cols], "stat": rk.t, "log2FC": rk.lfc, "padj": rk.padj})


def _gate(df, lfc_thr, fdr):
    return set(df.loc[(df.padj < fdr) & (df.log2FC.abs() >= lfc_thr), "GeneID"])


def _compare(a, b, name_a, name_b, key, top_n, cfg):
    m = a.merge(b, on="GeneID", suffixes=("_a", "_b")).dropna(subset=["stat_a", "stat_b"])
    row = {"split": key, "pair": f"{name_a} vs {name_b}", "n_genes_a": len(a), "n_genes_b": len(b), "n_common": len(m),
           "spearman_abs_stat": spearman_abs(m.stat_a.to_numpy(), m.stat_b.to_numpy()),
           "spearman_log2FC": float(pd.Series(m.log2FC_a).corr(pd.Series(m.log2FC_b), method="spearman"))}
    for n in top_n:
        row[f"top{n}_overlap"] = topn_overlap(m.stat_a.to_numpy(), m.stat_b.to_numpy(), n)
    ga = _gate(a, cfg["selection"]["gate_lfc"], cfg["selection"]["gate_fdr"])
    gb = _gate(b, cfg["selection"]["gate_lfc"], cfg["selection"]["gate_fdr"])
    row.update({"gate_n_a": len(ga), "gate_n_b": len(gb), "gate_overlap": len(ga & gb)})
    return row


def run(ctx):
    cfg, ds, log = ctx.cfg, ctx.ds, ctx.log
    d = ctx.stage_dir(STAGE)
    inp = ctx.inputs_dir()
    seed = cfg["seeds"]["reference"]
    sj = splits_json(ds.y, seed, cfg["model"]["outer_splits"], cfg["model"]["inner_splits"])
    ctx.out(STAGE, "splits_seed42.json").write_text(json.dumps(sj), encoding="utf-8")
    rows = []
    for o in sj["outer"]:
        rows += [(f"outer{o['fold']}_train", i + 1) for i in o["train"]]
        for s in o["inner"]:
            rows += [(f"outer{o['fold']}_inner{s['j']}_train", i + 1) for i in s["train"]]
    spl = ctx.out(STAGE, "splits_seed42.tsv")
    pd.DataFrame(rows, columns=["key", "idx"]).to_csv(spl, sep="\t", index=False)
    ctx.run_r("rankings.R", inp, spl, d)

    top_n = cfg["item7"]["top_n"]
    all_idx = np.arange(ds.n)
    keys = {"full": all_idx}
    for o in sj["outer"]:
        keys[f"outer{o['fold']}"] = np.array(o["train"])
        for s in o["inner"]:
            keys[f"outer{o['fold']}_inner{s['j']}"] = np.array(s["train"])

    out_rows, tables = [], {}
    for key, idx in keys.items():
        w = _welch_table(ds, idx, cfg)
        lim = _read(d, "limma", key)
        tables[key] = (w, lim)
        out_rows.append(_compare(w, lim, "Welch", "limma-voom", key, top_n, cfg))
        if "inner" not in key:
            de = _read(d, "deseq2", key)
            tables[key] = (w, lim, de)
            out_rows.append(_compare(w, de, "Welch", "DESeq2", key, top_n, cfg))
            out_rows.append(_compare(lim, de, "limma-voom", "DESeq2", key, top_n, cfg))
    cmp_df = pd.DataFrame(out_rows)
    cmp_df.to_csv(ctx.out(STAGE, "ranking_agreement.csv"), index=False)

    full = cmp_df[cmp_df.split == "full"].set_index("pair")
    log.info("FULL DATA agreement (Spearman of |stat| / top-10 / top-50 overlap / gate sizes):")
    for pair, r in full.iterrows():
        log.info("  %-24s rho=%.3f  top10=%d top50=%d  gate %d vs %d (shared %d)", pair, r.spearman_abs_stat,
                 r.top10_overlap, r.top50_overlap, r.gate_n_a, r.gate_n_b, r.gate_overlap)
    of = cmp_df[cmp_df.split.str.match(r"^outer\d$")]
    for pair, g in of.groupby("pair"):
        log.info("  OUTER FOLDS %-22s rho %.3f-%.3f  top10 %d-%d  top50 %d-%d", pair, g.spearman_abs_stat.min(),
                 g.spearman_abs_stat.max(), g.top10_overlap.min(), g.top10_overlap.max(), g.top50_overlap.min(),
                 g.top50_overlap.max())
    inn = cmp_df[cmp_df.split.str.contains("inner")]
    log.info("  INNER splits Welch vs limma-voom: rho %.3f-%.3f, top10 %d-%d", inn.spearman_abs_stat.min(),
             inn.spearman_abs_stat.max(), inn.top10_overlap.min(), inn.top10_overlap.max())

    # ---- reproducibility of the stored full-data DESeq2 result ---------------------------------
    stored = pd.read_csv(cfg.path("stored_deseq2_all"), sep="\t", dtype={"GeneID": str})
    new = _read(d, "deseq2", "full").merge(stored, on="GeneID", suffixes=("_new", "_stored"))
    rep = {"n_compared": int(len(new)),
           "max_abs_diff_log2FC": float((new.log2FC - new.log2FoldChange).abs().max()),
           "max_abs_diff_stat": float((new.stat_new - new.stat_stored).abs().max()),
           "n_new_DEGs": int(((new.padj_new < .05) & (new.log2FC.abs() >= 1)).sum()),
           "n_stored_DEGs": int(((new.padj_stored < .05) & (new.log2FoldChange.abs() >= 1)).sum())}
    (ctx.out(STAGE, "deseq2_reproducibility_vs_stored.json")).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log.info("DESeq2 full-data rerun vs stored file: %s", rep)

    # ---- where do the baseline panel genes rank? ----------------------------------------------
    w, lim, de = tables["full"]
    sym = dict(zip(ds.gene_ids, ds.symbols))
    pr = []
    for g in cfg["baseline"]["panel"] + [x for x in cfg["baseline"]["stability_genes"] if x not in cfg["baseline"]["panel"]]:
        gid = ds.gene_ids[np.flatnonzero(ds.symbols == g)[0]]
        r = {"symbol": g}
        for name, t in (("welch", w), ("limma_voom", lim), ("deseq2", de)):
            t = t.assign(rank=t.stat.abs().rank(ascending=False, method="first"))
            row = t[t.GeneID == gid]
            r[f"{name}_rank_by_abs_stat"] = int(row["rank"].iloc[0]) if len(row) else None
            r[f"{name}_log2FC"] = float(row.log2FC.iloc[0]) if len(row) else None
            r[f"{name}_padj"] = float(row.padj.iloc[0]) if len(row) else None
        pr.append(r)
    pd.DataFrame(pr).to_csv(ctx.out(STAGE, "panel_gene_ranks_full_data.csv"), index=False)

    # ---- figure -------------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    plotting.style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    m = w.merge(de, on="GeneID", suffixes=("_w", "_d"))
    ax = axes[0]
    ax.scatter(m.stat_d.abs(), m.stat_w.abs(), s=3, color="#9a9a9a", alpha=0.35, linewidths=0, rasterized=True)
    for g in cfg["baseline"]["panel"]:
        gid = ds.gene_ids[np.flatnonzero(ds.symbols == g)[0]]
        r = m[m.GeneID == gid]
        if len(r):
            ax.scatter(r.stat_d.abs(), r.stat_w.abs(), s=22, color=plotting.EVD, zorder=3)
            ax.annotate(g, (r.stat_d.abs().iloc[0], r.stat_w.abs().iloc[0]), xytext=(4, 3), textcoords="offset points",
                        fontsize=7, color=plotting.INK)
    ax.set_xlabel("|DESeq2 Wald statistic|")
    ax.set_ylabel("|Welch t|, log2(CPM+1)")
    ax.set_title(f"Full data, {len(m):,} genes (rho = {full.loc['Welch vs DESeq2'].spearman_abs_stat:.2f})")
    for ax, pair, col, title in ((axes[1], "Welch vs DESeq2", plotting.EVD, "Welch vs DESeq2"),
                                 (axes[2], "Welch vs limma-voom", plotting.HEALTHY, "Welch vs limma-voom")):
        sub = cmp_df[(cmp_df.pair == pair) & cmp_df.split.str.match(r"^outer\d$")]
        for _, r in sub.iterrows():
            ax.plot(top_n, [r[f"top{n}_overlap"] / n for n in top_n], color=col, alpha=0.35, lw=1, marker=None)
        f = cmp_df[(cmp_df.pair == pair) & (cmp_df.split == "full")].iloc[0]
        ax.plot(top_n, [f[f"top{n}_overlap"] / n for n in top_n], color=col, lw=2, marker="o", label="full data")
        ax.plot([], [], color=col, alpha=0.35, lw=1, label="5 outer training folds")
        ax.set_xscale("log")
        ax.set_xticks(top_n)
        ax.set_xticklabels(top_n)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Top-N genes by |statistic|")
        ax.set_ylabel("Fraction shared")
        ax.set_title(title)
        ax.legend(loc="lower right")
    fig.tight_layout()
    plotting.save(fig, ctx.out(STAGE, "ranking_agreement.png"))
