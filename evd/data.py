"""Data loading, label-free per-sample transforms, inferred sex.

Everything computed here is either sample-wise (log-CPM of a sample uses only that
sample's counts) or label-free, so it can be done once for all samples without
leaking label information into cross-validation.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEX_CHROM_ACC = ("NC_000023.11", "NC_000024.10")


@dataclass
class Dataset:
    counts: np.ndarray        # n x G integer counts
    LC: np.ndarray            # n x G log2(CPM+1), library size = sum over ALL genes of the sample
    GE: np.ndarray            # n x G bool, count >= min_counts
    gene_ids: np.ndarray      # G (str)
    symbols: np.ndarray       # G (str)
    sex_chrom: np.ndarray     # G bool
    sample_ids: np.ndarray    # n (str)
    y: np.ndarray             # n int (0 Healthy, 1 EVD-S)
    min_counts: int = 10

    @property
    def n(self):
        return self.counts.shape[0]

    @property
    def G(self):
        return self.counts.shape[1]

    @classmethod
    def from_arrays(cls, counts, y, gene_ids=None, symbols=None, sex_chrom=None,
                    sample_ids=None, min_counts=10):
        counts = np.asarray(counts)
        n, G = counts.shape
        lib = counts.sum(axis=1, keepdims=True).astype(float)
        LC = np.log2(counts / lib * 1e6 + 1.0)
        GE = counts >= min_counts
        gene_ids = np.asarray(gene_ids if gene_ids is not None else [f"g{i}" for i in range(G)], dtype=str)
        symbols = np.asarray(symbols if symbols is not None else gene_ids, dtype=str)
        sex_chrom = np.zeros(G, bool) if sex_chrom is None else np.asarray(sex_chrom, bool)
        sample_ids = np.asarray(sample_ids if sample_ids is not None else [f"s{i}" for i in range(n)], dtype=str)
        return cls(counts, LC, GE, gene_ids, symbols, sex_chrom, sample_ids,
                   np.asarray(y, dtype=int), min_counts)

    def subset_genes(self, mask):
        """Return a copy restricted to genes in mask (library sizes are NOT recomputed)."""
        mask = np.asarray(mask, bool)
        return Dataset(self.counts[:, mask], self.LC[:, mask], self.GE[:, mask], self.gene_ids[mask],
                       self.symbols[mask], self.sex_chrom[mask], self.sample_ids, self.y, self.min_counts)


def load_dataset(cfg):
    counts = pd.read_csv(cfg.path("counts"), sep="\t")
    annot = pd.read_csv(cfg.path("annotation"), sep="\t")
    meta = pd.read_csv(cfg.path("metadata"))

    annot["GeneID"] = annot["GeneID"].astype(str)
    annot["Symbol"] = annot["Symbol"].fillna("").astype(str)
    annot["GeneType"] = annot["GeneType"].fillna("").astype(str)
    annot["ChrAcc"] = annot["ChrAcc"].fillna("").astype(str)
    rm = annot["GeneType"].str.contains("pseudo", case=False, na=False) | annot["Symbol"].str.startswith("LOC")
    keep_ids = set(annot.loc[~rm, "GeneID"])

    counts["GeneID"] = counts["GeneID"].astype(str)
    counts = counts.set_index("GeneID")
    n_raw_genes = counts.shape[0]
    counts = counts.loc[counts.index.isin(keep_ids)]

    meta["ID_REF"] = meta["ID_REF"].astype(str)
    meta["Condition"] = meta["Condition"].str.strip()
    meta["Label"] = meta["Condition"].map({"Healthy": 0, "EVD-S": 1})
    assert meta["Label"].notna().all(), "unexpected Condition value"
    samples = [s for s in meta["ID_REF"] if s in counts.columns]
    assert len(samples) == len(meta) == 59, f"expected 59 samples, got {len(samples)}"
    counts = counts[samples]
    meta = meta.set_index("ID_REF").loc[samples]

    ann = annot.drop_duplicates("GeneID").set_index("GeneID")
    gene_ids = counts.index.to_numpy(str)
    symbols = ann.reindex(gene_ids)["Symbol"].fillna("").to_numpy(str)
    chr_acc = ann.reindex(gene_ids)["ChrAcc"].fillna("").to_numpy(str)
    sex_chrom = np.array([any(a in c for a in SEX_CHROM_ACC) for c in chr_acc])

    ds = Dataset.from_arrays(counts.to_numpy().T.astype(np.int64), meta["Label"].to_numpy(int),
                             gene_ids=gene_ids, symbols=symbols, sex_chrom=sex_chrom,
                             sample_ids=np.array(samples), min_counts=cfg["selection"]["min_counts"])
    ds.meta_condition = meta["Condition"].to_numpy(str)
    ds.n_raw_genes = n_raw_genes
    for s in ("XIST", "KDM5D", "RPS4Y1", "DDX3Y", "UTY"):
        j = np.flatnonzero(symbols == s)
        assert len(j) == 1 and sex_chrom[j[0]], f"{s} must be unique and flagged as sex-chromosome"
    return ds


def gene_index(ds, symbol):
    j = np.flatnonzero(ds.symbols == symbol)
    if len(j) != 1:
        raise KeyError(f"{symbol}: {len(j)} matches")
    return int(j[0])


def infer_sex(ds, cfg):
    sc = cfg["sex"]
    yj = [gene_index(ds, g) for g in sc["y_genes"]]
    ylev = ds.LC[:, yj].mean(1)   # mean of log2(CPM+1) over the Y genes
    xist = ds.LC[:, gene_index(ds, sc["xist"])]
    male = ylev > sc["threshold"]
    xist_female = xist > sc["threshold"]
    return pd.DataFrame({
        "sample": ds.sample_ids,
        "label": ds.y,
        "condition": np.where(ds.y == 1, "EVD-S", "Healthy"),
        "ygene_mean_log2cpm": ylev,
        "xist_log2cpm": xist,
        "sex": np.where(male, "male", "female"),
        "concordant_with_xist": male != xist_female,
    })
