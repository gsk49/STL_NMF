# library(Seurat)
# library(dplyr)
# library(Matrix)
# # Load the PBMC dataset

# # pbmc.data <- read.csv("data_sc3_seurat/A_100.csv", row.names = 1);
# B <- read.table("./01_real_data/pdac_b_ref.tsv", header=TRUE, sep="\t")
# B <- B[,-1]

# pbmc.data <- as.matrix(B)
# pbmc.data <- as(pbmc.data, "dgCMatrix")


# start_time <- Sys.time()

# # Examine the memory savings between regular and sparse matrices
# dense.size <- object.size(x = as.matrix(x = pbmc.data))
# sparse.size <- object.size(x = pbmc.data)

# mincell=0 # Keep all genes expressed in >= mincell cells (parameter)
# mingene=0 # Keep all cells with at least mingene detected genes (parameter)

# # pbmc <- CreateSeuratObject(raw.data = pbmc.data, min.cells = 0, min.genes = 0, project = "10X_PBMC")
# pbmc <- CreateSeuratObject(counts = pbmc.data, min.cells = 0, project = "10X_PBMC")
# pbmc <- NormalizeData(object = pbmc, normalization.method = "LogNormalize", scale.factor = 100)


# pbmc <- FindVariableFeatures(object = pbmc)
# pbmc <- ScaleData(object = pbmc)
# pbmc <- RunPCA(object = pbmc)
# pbmc <- FindNeighbors(object = pbmc)

# j=1.30; # a tunable parameter
# results <- FindClusters(object = pbmc, resolution = j)

# end_time <- Sys.time()
# t = end_time - start_time
# print(t)

# name = "seurat_output.csv"
# # write.csv(results@ident, name);
# write.csv(pbmc@meta.data$seurat_clusters, name);

library(Seurat)
library(dplyr)
library(Matrix)

# Load the PBMC dataset

B <- read.table("03_clustering/S_OB_6_runs_processed_seurat.dge_filtered_select.csv", header=TRUE, sep=",")
# B <- read.table("03_clustering/S_GSE111672_PDAC-A-indrop-filtered-expMat_select.csv", header=TRUE, sep=",")
# B <- read.table("./Desktop/deepNMF/01_real_data/S_OB_6_runs_processed_seurat.dge_filtered_select.csv", header=TRUE, sep=",")
B <- B[,-1]

pbmc.data <- as.matrix(B)
pbmc.data <- as(pbmc.data, "dgCMatrix")

start_time <- Sys.time()

# Examine the memory savings between regular and sparse matrices
dense.size <- object.size(as.matrix(pbmc.data))
sparse.size <- object.size(pbmc.data)

mincell = 0 # Keep all genes expressed in >= mincell cells (parameter)
mingene = 0 # Keep all cells with at least mingene detected genes (parameter)

# Create Seurat object
pbmc <- CreateSeuratObject(counts = pbmc.data, min.cells = mincell, project = "10X_PBMC")
# Filter cells and genes based on the specified thresholds
pbmc <- subset(pbmc, subset = nFeature_RNA >= mingene)

# Normalize the data
pbmc <- NormalizeData(object = pbmc, normalization.method = "LogNormalize", scale.factor = 100)

# Find variable features
pbmc <- FindVariableFeatures(object = pbmc)

# Scale the data
pbmc <- ScaleData(object = pbmc)

# Perform linear dimensional reduction (PCA)
pbmc <- RunPCA(object = pbmc)

# Find neighbors
pbmc <- FindNeighbors(object = pbmc)

# Find clusters
j = .0495 # a tunable parameter
pbmc <- FindClusters(object = pbmc, resolution = j, algorithm=4)

end_time <- Sys.time()
t = end_time - start_time
print(t)

# Check if clusters were added to metadata
print(head(pbmc@meta.data))

# Save results
name = "./01_real_data/SEURAT_MOB_leiden.csv"
write.csv(pbmc@meta.data$seurat_clusters, name)

