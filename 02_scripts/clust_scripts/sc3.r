### Load Data

# Try first with existing dataset (Zeisel)

library(scRNAseq)
library(SC3)
library(scater)
library(SingleCellExperiment)


# sce <- fetchDataset("zeisel-brain-2015", "2023-12-14")

# B <- read.table("./03_clustering/S_GSE111672_PDAC-A-indrop-filtered-expMat_select.csv", header=TRUE, sep=",")
B <- read.table("./01_real_data/S_OB_6_runs_processed_seurat.dge_filtered_select.csv", header=TRUE, sep=",")

# Remove the first column
# B <- B[, -ncol(B)]
B <- B[,-1]

sce <- SingleCellExperiment(B)

# meta <- fetchMetadata("zeisel-brain-2015", "2023-12-14")

rowData(sce)$feature_symbol <- rownames(sce)
sce <- sce[!duplicated(rowData(sce)$feature_symbol), ]

# print("Data:")
# print(sce)

# print("Data Assay:")
# print(assay(sce))

# print("Dimensions:")
# str(dim(assay(sce)))

# print("Column Names:")
# str(colnames(colData(sce)))

# print("Row names:")
# str(colnames(rowData(sce)))

# print("Meta Data:")
# str(meta)

# Gets the Matrix
counts <- assay(sce)
# Sums each column
libsizes <- colSums(counts)
# Normalizes to mean=1
size.factors <- libsizes/mean(libsizes)
# idk
logcounts(sce) <- log2(as.matrix(t(t(counts)/size.factors)+1))
assayNames(sce)

# print(plotColData(sce, x = "sum", y="detected", colour_by="tissue"))

print("Starting SC3")
counts <- as.matrix(assay(sce))

# sce <- runPCA(sce)
# reducedDim(sce, "PCA")

sce <- sc3(object=sce, ks = 5, gene_filter=FALSE, svm_num_cells=50)

col_data <- colData(sce)
head(col_data[ , grep("sc3_", colnames(col_data))])

sce <- sc3_run_svm(sce, ks = 5)
col_data <- colData(sce)
head(col_data[ , grep("sc3_", colnames(col_data))])

metadata(sce)$sc3$svm_train_inds <- NULL
sce <- sc3_calc_biology(sce, ks = 5)

col_data <- colData(sce)
head(col_data[ , grep("sc3_", colnames(col_data))])


fileName="sc3_MOB_5C.csv"
write.csv(col_data, fileName)

# no_svm_labels <- colData(sce)$sc3_3_clusters


