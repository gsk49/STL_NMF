library(MuSiC)
library(ggplot2)
library(SingleCellExperiment)
library(SpatialExperiment)
library(scater)
library(scran)
library(Biobase)

# EMTAB.sce = readRDS('https://xuranw.github.io/MuSiC/data/XinT2Dsce.rds')
# print(EMTAB.sce)
sc_counts_matrix <- as.matrix(read.csv("00_synthetic/ruitao/square_2x2/S.tsv", row.names = 1, sep = "\t"))
cell_meta <- read.csv("00_synthetic/ruitao/square_2x2/meta.tsv", sep = "\t")
sp_counts_matrix <- as.matrix(read.csv("00_synthetic/ruitao/square_2x2/X.tsv", row.names = 1, sep="\t"))

print(dim(sp_counts_matrix))
print(dim(sc_counts_matrix))

colnames(sc_counts_matrix) <- cell_meta$CellID
rownames(cell_meta) <- cell_meta$CellID
rownames(sc_counts_matrix) <- rownames(sp_counts_matrix)  # Align genes

rownames(sp_counts_matrix) = as.character(c(1:nrow(sp_counts_matrix)))

## ----Data Preparation---------------------------------------------------------
# Single Cell Data Prepare
# colnames(sc_counts_matrix) = cell_meta$Cell_ID
# rownames(sc_counts_matrix) = rownames(B)
print("Sample")
cell_meta$SampleID= 1
print("sampled")

# print(dim(sp_counts_matrix))       # Should match the number of genes
# print(dim(sc_counts_matrix))      # Should match the number of genes and cells
# print(colnames(sc_counts_matrix)) # Should match cell_meta$Cell_ID
# print(rownames(sc_counts_matrix)) # Should match rownames(sp_counts_matrix)

# print("Next")
# print(colnames(cell_meta))    # Should include Cell_ID, bio_celltype, SampleID
# print(nrow(cell_meta))        # Should match ncol(sc_counts_matrix)

# print("again")

# print(head(sp_counts_matrix))
# sc_counts_matrix <- sc_counts_matrix[rowSums(sc_counts_matrix) > 0, ]
# print(dim(sc_counts_matrix))





# cell_meta <- cell_meta[,-2]


sce <- SingleCellExperiment(assays = list(counts = sc_counts_matrix),
                            colData = as.matrix(cell_meta), mainExpName = "main")
# print(sce)
# print("colN")
colnames(sce) <- cell_meta$CellID
# print("colNd")
# Spatial Transcriptomics Data Prepare
rownames(sp_counts_matrix) = as.character(c(1:nrow(sp_counts_matrix)))
# print("rowN")
rownames(sp_counts_matrix) <- make.unique(paste0("Gene", seq_len(nrow(sp_counts_matrix))))
sp_counts_matrix <- as.matrix(sp_counts_matrix)

# print(dim(sce))
print(head(sce))


Est.prop = music_prop(bulk.mtx = sp_counts_matrix, sc.sce = sce,
                      clusters = "CellType", samples = "SampleID")
save(Est.prop,file = "result-MuSiC.Rdata")
norm_weights = Est.prop$Est.prop.weighted
write.csv(norm_weights,file = "result-MuSiC.csv")