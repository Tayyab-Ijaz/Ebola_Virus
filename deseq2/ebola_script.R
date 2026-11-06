# 01_deseq2_pipeline.R  (start-to-end, single block)
# - Removes pseudogenes AND "LOC*" genes from the very start
# - QC low-expression filtering
# - DESeq2 DEGs + exports
# - VST matrix export for ML
# - Publication-ready figures (PNG, 300 DPI) with ascending F01..F10 filenames

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tibble)
  library(tidyr)
  library(DESeq2)
  library(ggplot2)
  library(ggrepel)
  library(pheatmap)
})

# -------------------------
# Paths
# -------------------------
counts_path <- "GSE143549_raw_counts_GRCh38.p13_NCBI.tsv"
annot_path  <- "Human.GRCh38.p13.annot.tsv"
meta_path   <- "meta_data.csv"

# -------------------------
# Load
# -------------------------
counts_df <- fread(counts_path) %>% as.data.frame()
annot_df  <- fread(annot_path)  %>% as.data.frame()
meta_df   <- fread(meta_path)   %>% as.data.frame()

stopifnot("GeneID" %in% colnames(counts_df))
stopifnot(all(c("GeneID", "Symbol", "EnsemblGeneID", "GeneType") %in% colnames(annot_df)))
stopifnot(all(c("ID_REF", "Condition") %in% colnames(meta_df)))
stopifnot(all(meta_df$ID_REF %in% colnames(counts_df)))

# Make counts matrix: rows=GeneID, cols=samples (ordered like metadata)
rownames(counts_df) <- as.character(counts_df$GeneID)
counts_df$GeneID <- NULL
counts_df <- counts_df[, meta_df$ID_REF, drop = FALSE]

# -------------------------
# Remove pseudogenes + LOC* from start
# -------------------------
annot_df <- annot_df %>%
  mutate(
    GeneID = as.integer(GeneID),
    Symbol = ifelse(is.na(Symbol), "", as.character(Symbol)),
    GeneType = ifelse(is.na(GeneType), "", as.character(GeneType)),
    EnsemblGeneID = ifelse(is.na(EnsemblGeneID), "", as.character(EnsemblGeneID))
  )

rm_mask <- grepl("pseudo", annot_df$GeneType, ignore.case = TRUE) |
  grepl("^LOC", annot_df$Symbol, ignore.case = FALSE)

keep_geneids <- annot_df$GeneID[!rm_mask]
keep_geneids <- keep_geneids[!is.na(keep_geneids)]

# Keep only genes that remain after removal AND exist in annotation
counts_df <- counts_df[rownames(counts_df) %in% as.character(keep_geneids), , drop = FALSE]
annot_df  <- annot_df[annot_df$GeneID %in% keep_geneids, , drop = FALSE]

# -------------------------
# Low expression filter (QC)
# Keep genes with >=10 counts in at least min_group samples
# -------------------------
meta_df$Condition <- factor(meta_df$Condition, levels = c("Healthy", "EVD-S"))
min_group <- min(table(meta_df$Condition))

keep_low <- rowSums(as.matrix(counts_df) >= 10) >= min_group
counts_df <- counts_df[keep_low, , drop = FALSE]

stopifnot(nrow(counts_df) > 0)

# -------------------------
# DESeq2
# -------------------------
coldata <- data.frame(
  row.names = meta_df$ID_REF,
  Condition = meta_df$Condition
)

dds <- DESeqDataSetFromMatrix(
  countData = round(as.matrix(counts_df)),
  colData   = coldata,
  design    = ~ Condition
)

dds <- DESeq(dds)
res <- results(dds, contrast = c("Condition", "EVD-S", "Healthy"))

res_df <- as.data.frame(res) %>%
  rownames_to_column("GeneID") %>%
  mutate(GeneID = as.integer(GeneID)) %>%
  left_join(
    annot_df %>% select(GeneID, Symbol, EnsemblGeneID, GeneType),
    by = "GeneID"
  ) %>%
  arrange(padj)

deg_df <- res_df %>%
  filter(!is.na(padj), padj < 0.05, abs(log2FoldChange) >= 1) %>%
  arrange(padj)

write.table(res_df, file = "deseq2_all_results.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
write.table(deg_df, file = "deseq2_DEGs.tsv",         sep = "\t", quote = FALSE, row.names = FALSE)

# -------------------------
# Export matrix for ML (VST)
# -------------------------
vsd <- vst(dds, blind = FALSE)
vst_mat <- assay(vsd)
vst_out <- data.frame(GeneID = rownames(vst_mat), vst_mat, check.names = FALSE)
write.table(vst_out, file = "vst_matrix.tsv", sep = "\t", quote = FALSE, row.names = FALSE)

# -------------------------
# Figures (Publication-ready: PNG, 300 DPI, vibrant colors, explanatory annotations)
# -------------------------
fig_dir <- "figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# Vibrant palette (2-class)
cond_cols <- c("Healthy" = "#00BFC4", "EVD-S" = "#F8766D")

# Heatmap palettes (vibrant, high-contrast)
hm_colors  <- colorRampPalette(c("#313695", "#FFFFFF", "#A50026"))(120)
cor_colors <- colorRampPalette(c("#132B43", "#FFFFFF", "#A50026"))(120)

save_figure <- function(p, fname, w = 7, h = 5) {
  ggsave(
    filename = file.path(fig_dir, fname),
    plot = p,
    width = w, height = h, units = "in",
    dpi = 300, bg = "white",
    device = "png"
  )
}

# Prepare long VST for density plots
vst_long <- as.data.frame(vst_mat) %>%
  rownames_to_column("GeneID") %>%
  pivot_longer(-GeneID, names_to = "Sample", values_to = "VST") %>%
  left_join(
    data.frame(Sample = rownames(coldata), Condition = coldata$Condition),
    by = "Sample"
  )

# F02: VST density
p2 <- ggplot(vst_long, aes(x = VST, group = Sample, color = Condition)) +
  geom_density(alpha = 0.15, linewidth = 0.4) +
  scale_color_manual(values = cond_cols) +
  labs(
    title = "Distribution of VST expression values",
    subtitle = "Comparable distributions indicate successful normalization",
    x = "VST expression", y = "Density"
  ) +
  theme_classic(base_size = 12) +
  theme(legend.position = "top")
save_figure(p2, "F02_vst_density.png", w = 7, h = 5)

# F03: Sample correlation heatmap (VST)
sample_cor <- cor(vst_mat, method = "pearson")
ann_col <- data.frame(Condition = coldata$Condition)
rownames(ann_col) <- rownames(coldata)

png(file.path(fig_dir, "F03_sample_correlation_heatmap.png"),
    width = 12, height = 12, units = "in", res = 300)
pheatmap(
  sample_cor,
  color = cor_colors,
  annotation_col = ann_col,
  annotation_row = ann_col,
  main = "Sample–sample correlation (VST)",
  fontsize = 10,
  border_color = NA
)
dev.off()

# F04: PCA (VST)
pca <- prcomp(t(vst_mat), scale. = FALSE)
pca_df <- data.frame(
  Sample = rownames(coldata),
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  Condition = coldata$Condition
)
pct <- (pca$sdev^2) / sum(pca$sdev^2) * 100

p4 <- ggplot(pca_df, aes(PC1, PC2, color = Condition)) +
  geom_point(size = 3, alpha = 0.9) +
  scale_color_manual(values = cond_cols) +
  labs(
    title = "PCA of samples (VST)",
    subtitle = "Axes show variance explained; separation suggests condition signal",
    x = sprintf("PC1 (%.1f%%)", pct[1]),
    y = sprintf("PC2 (%.1f%%)", pct[2])
  ) +
  theme_classic(base_size = 12) +
  theme(legend.position = "top")
save_figure(p4, "F04_PCA.png", w = 7, h = 5)

# F05: Hierarchical clustering dendrogram (VST)
d <- dist(t(vst_mat))
hc <- hclust(d, method = "complete")

png(file.path(fig_dir, "F05_dendrogram.png"),
    width = 8, height = 5, units = "in", res = 300)
plot(hc, main = "Hierarchical clustering (VST)", xlab = "", sub = "", cex = 0.7)
dev.off()

# F06: Dispersion plot
png(file.path(fig_dir, "F06_dispersion_plot.png"),
    width = 6, height = 5, units = "in", res = 300)
plotDispEsts(dds, main = "DESeq2 dispersion estimates")
dev.off()

# F07: MA plot
png(file.path(fig_dir, "F07_MA_plot.png"),
    width = 6, height = 5, units = "in", res = 300)
plotMA(res, main = "MA plot: EVD-S vs Healthy", ylim = c(-6, 6))
abline(h = c(-1, 1), lty = 2)
dev.off()

# Label more genes: strong signals (tune thresholds)
res_df2 <- res_df %>%
  mutate(
    neglog10padj = -log10(padj),
    sig = ifelse(!is.na(padj) & padj < 0.05 & abs(log2FoldChange) >= 1, "DEG", "Not DEG"),
    Symbol = ifelse(is.na(Symbol) | Symbol == "", paste0("GeneID_", GeneID), Symbol),
    label = ifelse(!is.na(padj) & padj < 1e-4 & abs(log2FoldChange) >= 1.5, Symbol, NA)
  )

p8 <- ggplot(res_df2, aes(x = log2FoldChange, y = neglog10padj, color = sig)) +
  geom_point(alpha = 0.75, size = 1.6) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
  scale_color_manual(values = c("DEG" = "#D81B60", "Not DEG" = "grey70")) +
  ggrepel::geom_text_repel(
    aes(label = label),
    max.overlaps = Inf,
    max.time = 2,
    max.iter = 5000,
    size = 3
  ) +
  labs(
    title = "Volcano plot (EVD-S vs Healthy)",
    subtitle = "Dashed lines: |log2FC| ≥ 1 and FDR < 0.05; labeled padj < 1e-4 and |log2FC| ≥ 1.5",
    x = "log2 fold change",
    y = expression(-log[10]("adjusted p-value"))
  ) +
  theme_classic(base_size = 12) +
  theme(legend.position = "top")

save_figure(p8, "F08_volcano.png", w = 7, h = 5)


# F09: Heatmap of top 50 DEGs (padj ascending) - show SYMBOLS (not GeneIDs)

topN <- 50

top_deg <- deg_df %>%
  arrange(padj) %>%
  slice_head(n = min(topN, nrow(.))) %>%
  mutate(
    GeneID_chr = as.character(GeneID),
    Symbol = ifelse(is.na(Symbol) | Symbol == "", paste0("GeneID_", GeneID_chr), Symbol)
  )

top_deg_geneids <- top_deg$GeneID_chr

if (length(top_deg_geneids) > 1) {
  
  # Subset matrix using GeneIDs (rownames of vst_mat are GeneIDs)
  hm_mat <- vst_mat[rownames(vst_mat) %in% top_deg_geneids, , drop = FALSE]
  
  # Reorder rows to match padj-ascending order in top_deg
  hm_mat <- hm_mat[top_deg_geneids, , drop = FALSE]
  
  # Replace rownames (GeneIDs) with Symbols (make unique to avoid duplicates)
  rownames(hm_mat) <- make.unique(top_deg$Symbol)
  
  # Row Z-score
  hm_mat_z <- t(scale(t(hm_mat)))
  
  png(file.path(fig_dir, "F09_top50_DEG_heatmap.png"),
      width = 8, height = 7, units = "in", res = 300)
  
  pheatmap(
    hm_mat_z,
    color = hm_colors,
    annotation_col = ann_col,
    main = "Top DEGs heatmap (VST; row Z-score)",
    fontsize = 9,
    border_color = NA,
    show_colnames = FALSE,
    show_rownames = TRUE
  )
  
  dev.off()
}

# -------------------------
# Export DEGs-only expression matrix (VST) to CSV (SYMBOLS, not IDs)
# -------------------------

# Ensure DEG list is ordered (padj ascending) and has usable symbols
deg_order <- deg_df %>%
  arrange(padj) %>%
  mutate(
    GeneID_chr = as.character(GeneID),
    Symbol = ifelse(is.na(Symbol) | Symbol == "", paste0("GeneID_", GeneID_chr), Symbol)
  )

# Keep only DEG gene IDs that exist in vst_mat rownames
deg_ids_in <- intersect(deg_order$GeneID_chr, rownames(vst_mat))
deg_order  <- deg_order %>% filter(GeneID_chr %in% deg_ids_in)

# Subset + keep the same DEG ordering
deg_vst <- vst_mat[deg_order$GeneID_chr, , drop = FALSE]

# Replace rownames with gene symbols (make unique to avoid duplicates)
deg_symbols_unique <- make.unique(deg_order$Symbol)
rownames(deg_vst) <- deg_symbols_unique

# Build CSV (genes x samples). Includes GeneID + Symbol for traceability.
deg_expr_df <- data.frame(
  GeneID = deg_order$GeneID_chr,
  Symbol = deg_symbols_unique,
  deg_vst,
  check.names = FALSE
)

write.csv(deg_expr_df, file = "DEG_expression_matrix_VST.csv", row.names = FALSE)


# F10: Box/violin plots for top 6 DEGs (normalized counts; log10 scale)
norm_counts <- counts(dds, normalized = TRUE)

top6 <- deg_df %>%
  arrange(padj) %>%
  slice_head(n = min(12, nrow(.))) %>%
  mutate(GeneID = as.character(GeneID))   # <-- make join key character

if (nrow(top6) > 0) {
  
  bc_long <- as.data.frame(norm_counts[top6$GeneID, , drop = FALSE]) %>%  # <-- GeneID already character
    rownames_to_column("GeneID") %>%
    mutate(GeneID = as.character(GeneID)) %>%                             # <-- enforce character
    pivot_longer(-GeneID, names_to = "Sample", values_to = "NormCount") %>%
    left_join(top6 %>% select(GeneID, Symbol), by = "GeneID") %>%
    left_join(
      data.frame(Sample = rownames(coldata), Condition = coldata$Condition),
      by = "Sample"
    ) %>%
    mutate(Symbol = ifelse(is.na(Symbol) | Symbol == "", paste0("GeneID_", GeneID), Symbol))
  
  p10 <- ggplot(bc_long, aes(x = Condition, y = NormCount, fill = Condition)) +
    geom_violin(trim = TRUE, alpha = 0.6) +
    geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.9) +
    geom_jitter(width = 0.12, size = 1.2, alpha = 0.7) +
    scale_fill_manual(values = cond_cols) +
    scale_y_continuous(trans = "log10") +
    facet_wrap(~ Symbol, scales = "free_y", ncol = 3) +
    labs(
      title = "Top DEGs: normalized expression by condition",
      subtitle = "Log10 scale; each point is a sample; violin shows distribution",
      x = NULL, y = "Normalized counts (log10)"
    ) +
    theme_classic(base_size = 12) +
    theme(legend.position = "none")
  
  save_figure(p10, "F10_topDEG_boxplots.png", w = 9, h = 6)
}
