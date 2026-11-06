# Shared helpers for the revision_v2 R scripts (DESeq2 / limma-voom).
options(warn = 1)    # print every warning immediately, with its call, so it ends up in the run log
suppressPackageStartupMessages({
  library(DESeq2)
  library(edgeR)
  library(limma)
})

script_dir <- function() {
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  dirname(normalizePath(f))
}

load_inputs <- function(inp) {
  cnt <- read.delim(file.path(inp, "counts_gene_filtered.tsv"), check.names = FALSE, row.names = 1)
  smp <- read.delim(file.path(inp, "samples.tsv"), stringsAsFactors = FALSE)
  stopifnot(identical(colnames(cnt), smp$sample))
  stopifnot(all(abs(as.matrix(cnt) - round(as.matrix(cnt))) == 0))   # counts are integers
  smp$Condition <- factor(smp$condition, levels = c("Healthy", "EVD-S"))
  smp$sex <- factor(smp$sex, levels = c("female", "male"))
  list(counts = as.matrix(cnt), samples = smp)
}

# low-expression filter: >= min_counts in >= (size of the smaller class) samples
expr_keep <- function(counts, cond, min_counts = 10) {
  rowSums(counts >= min_counts) >= min(table(cond))
}

fit_deseq <- function(counts, coldata, design) {
  dds <- DESeqDataSetFromMatrix(countData = round(counts), colData = coldata, design = design)
  for (ft in c("parametric", "local", "mean")) {
    out <- tryCatch(DESeq(dds, quiet = TRUE, fitType = ft), error = function(e) NULL)
    if (!is.null(out)) { attr(out, "fitType_used") <- ft; return(out) }
  }
  stop("DESeq failed for all fit types")
}

tidy_deseq <- function(res) {
  d <- as.data.frame(res)
  data.frame(GeneID = rownames(d), baseMean = d$baseMean, log2FC = d$log2FoldChange,
             stat = d$stat, pvalue = d$pvalue, padj = d$padj)
}

deseq_condition <- function(counts, smp) {
  keep <- expr_keep(counts, smp$Condition)
  dds <- fit_deseq(counts[keep, , drop = FALSE], data.frame(Condition = smp$Condition, row.names = smp$sample),
                   ~ Condition)
  tidy_deseq(results(dds, contrast = c("Condition", "EVD-S", "Healthy")))
}

limma_condition <- function(counts, smp) {
  keep <- expr_keep(counts, smp$Condition)
  dge <- calcNormFactors(DGEList(counts = counts[keep, , drop = FALSE]))
  design <- model.matrix(~ Condition, data = smp)
  v <- voom(dge, design, plot = FALSE)
  fit <- eBayes(lmFit(v, design))
  tt <- topTable(fit, coef = "ConditionEVD-S", number = Inf, sort.by = "none")
  data.frame(GeneID = rownames(tt), log2FC = tt$logFC, stat = tt$t, pvalue = tt$P.Value, padj = tt$adj.P.Val)
}

write_tsv <- function(df, path) write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
