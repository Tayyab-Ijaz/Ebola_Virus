# Item 9 / P2: cell-type deconvolution of whole-blood RNA-seq with immunedeconv (EPIC with the blood reference,
# quanTIseq, MCP-counter). Each method scores every sample on its own (no label information is used).
# usage: Rscript deconv.R <tpm_symbols.tsv> <out_dir> <method1,method2,...>
args <- commandArgs(trailingOnly = TRUE)
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))), "common.R"))
suppressPackageStartupMessages(library(immunedeconv))
tpm <- read.delim(args[1], check.names = FALSE, row.names = 1)
m <- as.matrix(tpm)
cat("TPM matrix:", nrow(m), "genes x", ncol(m), "samples\n")
for (method in strsplit(args[3], ",")[[1]]) {
  res <- tryCatch(deconvolute(m, method, tumor = FALSE),
                  error = function(e) { cat("FAILED", method, ":", conditionMessage(e), "\n"); NULL })
  if (!is.null(res)) {
    write_tsv(as.data.frame(res), file.path(args[2], paste0("deconv_", method, ".tsv")))
    cat(method, ": ", nrow(res), "cell types\n")
  }
}
cat("immunedeconv", as.character(packageVersion("immunedeconv")), "| EPIC", as.character(packageVersion("EPIC")),
    "| MCPcounter", as.character(packageVersion("MCPcounter")), "\n")
cat("done\n")
