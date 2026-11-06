# Item 6: VST (blind = FALSE as in the legacy script, and blind = TRUE) on all samples.
# usage: Rscript vst_pca.R <inputs_dir> <out_dir>
args <- commandArgs(trailingOnly = TRUE)
source(file.path(script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))), "common.R"))
inp <- args[1]; out <- args[2]
d <- load_inputs(inp); counts <- d$counts; smp <- d$samples

keep <- expr_keep(counts, smp$Condition)
cat("genes after global low-expression filter:", sum(keep), "\n")
dds <- fit_deseq(counts[keep, , drop = FALSE], data.frame(Condition = smp$Condition, row.names = smp$sample), ~ Condition)
cat("fitType used:", attr(dds, "fitType_used"), "\n")

for (blind in c(FALSE, TRUE)) {
  v <- assay(vst(dds, blind = blind))
  df <- data.frame(GeneID = rownames(v), v, check.names = FALSE)
  write_tsv(df, file.path(out, sprintf("vst_blind%s.tsv", toupper(as.character(blind)))))
}
cat("done\n")
