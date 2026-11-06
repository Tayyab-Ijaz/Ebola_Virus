# Item 10(a): DESeq2 with and without inferred sex on the full data.
# usage: Rscript sex_deseq.R <inputs_dir> <out_dir>
args <- commandArgs(trailingOnly = TRUE)
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))), "common.R"))
inp <- args[1]; out <- args[2]
d <- load_inputs(inp); counts <- d$counts; smp <- d$samples

keep <- expr_keep(counts, smp$Condition)
cat("genes after global low-expression filter:", sum(keep), "\n")
cn <- counts[keep, , drop = FALSE]
cd <- data.frame(Condition = smp$Condition, sex = smp$sex, row.names = smp$sample)
print(table(cd$Condition, cd$sex))

d0 <- fit_deseq(cn, cd, ~ Condition)
write_tsv(tidy_deseq(results(d0, contrast = c("Condition", "EVD-S", "Healthy"))), file.path(out, "deseq_condition_only.tsv"))

d1 <- fit_deseq(cn, cd, ~ sex + Condition)
cat("resultsNames (sex + Condition):", paste(resultsNames(d1), collapse = ", "), "\n")
write_tsv(tidy_deseq(results(d1, contrast = c("Condition", "EVD-S", "Healthy"))), file.path(out, "deseq_condition_adj_sex.tsv"))
write_tsv(tidy_deseq(results(d1, contrast = c("sex", "male", "female"))), file.path(out, "deseq_sex_effect.tsv"))
cat("done\n")
