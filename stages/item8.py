"""Item 8 (cheap version): concordance of selected genes between the NCBI GRCh38.p13 gene counts used here and the
authors' own quantification (STAR -> hg19, GENCODE v19, Partek; GEO supplementary CPM+0.0001 table).

WHAT THIS CAN AND CANNOT SHOW: both tables come from the same sequencing reads. Agreement shows that the signal is
not an artefact of one annotation/aligner/counting pipeline; it does NOT exclude ambiguous read assignment
(multimapping) that both pipelines could share. No reads or BAM files are available (no SRA download), so the
mapping analysis requested by Reviewer 2 is not done and stays a limitation."""
import gzip
import re

import numpy as np
import pandas as pd
from scipy import stats

STAGE = "item8"


def _series_map(path):
    """description (authors' column name) -> GSM"""
    gsm = desc = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("!Sample_geo_accession"):
                gsm = [x.strip('"') for x in ln.rstrip("\n").split("\t")[1:]]
            elif ln.startswith("!Sample_description"):
                desc = [x.strip('"') for x in ln.rstrip("\n").split("\t")[1:]]
            if gsm and desc:
                break
    return dict(zip(desc, gsm))


def _rankrows(a):
    return np.apply_along_axis(stats.rankdata, 1, a)


def run(ctx):
    cfg, ds, log, y = ctx.cfg, ctx.ds, ctx.log, ctx.ds.y
    with gzip.open(cfg.path("authors_cpm"), "rt", encoding="utf-8", errors="replace") as fh:
        au = pd.read_csv(fh, sep="\t", low_memory=False)
    m = _series_map(cfg.path("series_matrix"))
    scols = [c for c in au.columns if c in m]
    assert len(scols) == 59, f"expected 59 sample columns in the authors' table, found {len(scols)}"
    au = au.rename(columns=m)
    au["Gene Symbol"] = au["Gene Symbol"].astype(str)
    log.info("authors' table: %d rows, %d samples; chromosomes: %s", len(au), 59, sorted(au.Chromosome.astype(str).unique())[:6])
    A = au[ds.sample_ids]                                                  # aligned to our sample order
    sym_n = au["Gene Symbol"].value_counts()

    # ---- genome-wide concordance (symbols unique in both tables) ------------------------------
    ours_idx = {s: i for i, s in enumerate(ds.symbols) if (ds.symbols == s).sum() == 1}
    uniq = [s for s in au["Gene Symbol"].unique() if sym_n[s] == 1 and s in ours_idx]
    Aa = np.log2(au.set_index("Gene Symbol").loc[uniq, list(ds.sample_ids)].to_numpy(float) + 1.0)
    Oo = ds.LC[:, [ours_idx[s] for s in uniq]].T
    expressed = (ds.GE.sum(0)[[ours_idx[s] for s in uniq]] >= 26) & (Aa.mean(1) > 1.0)
    ra, ro = _rankrows(Aa), _rankrows(Oo)
    ra -= ra.mean(1, keepdims=True)
    ro -= ro.mean(1, keepdims=True)
    rho_all = (ra * ro).sum(1) / np.sqrt((ra ** 2).sum(1) * (ro ** 2).sum(1))
    gw = pd.Series(rho_all, index=uniq)
    log.info("genome-wide per-gene Spearman (authors vs NCBI log-CPM across 59 samples), %d shared unique symbols, "
             "%d expressed: median %.3f, 5th percentile %.3f", len(uniq), int(expressed.sum()),
             np.nanmedian(rho_all[expressed]), np.nanpercentile(rho_all[expressed], 5))

    # ---- the genes of interest ----------------------------------------------------------------
    genes = list(dict.fromkeys(cfg["item8"]["genes"] + ctx.cache.get("panel", {}).get("symbols", [])[:15]))
    rows = []
    for g in genes:
        in_au = au[au["Gene Symbol"] == g]
        j = np.flatnonzero(ds.symbols == g)
        r = {"gene": g, "in_final_panel": g in ctx.cache.get("panel", {}).get("symbols", []),
             "in_authors_table_rows": len(in_au), "in_our_table_rows": len(j)}
        if len(in_au) == 0 or len(j) != 1:
            r["note"] = "locus not present in the authors' hg19/GENCODE v19 table" if len(in_au) == 0 else "ambiguous in ours"
            rows.append(r)
            continue
        row = in_au.iloc[in_au[ds.sample_ids].mean(1).argmax()]            # most expressed row if duplicated
        a = np.log2(row[ds.sample_ids].to_numpy(float) + 1.0)
        o = ds.LC[:, j[0]]
        rho = stats.spearmanr(a, o)
        r.update({"authors_chr": row["Chromosome"], "authors_start": row["Start"], "authors_stop": row["Stop"],
                  "authors_gene_status": row["gene_status"], "spearman_rho": rho.statistic, "spearman_p": rho.pvalue,
                  "pearson_r": float(np.corrcoef(a, o)[0, 1]),
                  "genomewide_percentile_of_rho": float((gw < rho.statistic).mean() * 100),
                  "frac_samples_at_authors_floor_0.0001": float((row[ds.sample_ids].to_numpy(float) <= 0.00011).mean()),
                  "frac_samples_count_ge10_ours": float(ds.GE[:, j[0]].mean()),
                  "log2FC_ours_EVD_minus_HD": float(o[y == 1].mean() - o[y == 0].mean()),
                  "log2FC_authors_EVD_minus_HD": float(a[y == 1].mean() - a[y == 0].mean()),
                  "welch_t_ours": float(stats.ttest_ind(o[y == 1], o[y == 0], equal_var=False).statistic),
                  "welch_t_authors": float(stats.ttest_ind(a[y == 1], a[y == 0], equal_var=False).statistic)})
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(ctx.out(STAGE, "gene_concordance_authors_vs_ncbi.csv"), index=False)
    gw.rename("spearman_rho").to_csv(ctx.out(STAGE, "genomewide_concordance_per_gene.csv"), header=True, index_label="symbol")
    show = [c for c in ("gene", "in_final_panel", "in_authors_table_rows", "authors_chr", "spearman_rho", "genomewide_percentile_of_rho",
                        "log2FC_ours_EVD_minus_HD", "log2FC_authors_EVD_minus_HD", "welch_t_ours", "welch_t_authors", "note") if c in df.columns]
    log.info("gene concordance (authors' STAR/hg19/GENCODE v19 vs NCBI GRCh38.p13):\n%s", df[show].round(3).to_string(index=False))
    log.info("NOTE: same reads, two quantifications; concordance does not exclude shared multimapping (no reads/BAMs available)")
