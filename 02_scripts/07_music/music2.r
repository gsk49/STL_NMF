library(MuSiC)
library(ggplot2)
library(SingleCellExperiment)
library(SpatialExperiment)
library(scater)
library(scran)
library(Biobase)


sc_counts_matrix <- as.matrix(read.csv("00_synthetic/ruitao/square_2x2/S.tsv", row.names = 1, sep = "\t"))
cell_meta <- read.csv("00_synthetic/ruitao/square_2x2/meta2.tsv", sep = "\t")
sp_counts_matrix <- as.matrix(read.csv("00_synthetic/ruitao/square_2x2/X.tsv", row.names = 1, sep="\t"))

print(dim(sp_counts_matrix))
print(dim(sc_counts_matrix))

rownames(sp_counts_matrix) <- rownames(sc_counts_matrix)
colnames(sc_counts_matrix) <- cell_meta$CellID
rownames(sc_counts_matrix) <- rownames(sp_counts_matrix)  # Align genes

# Single Cell Data Prepare
colnames(sc_counts_matrix) = cell_meta$CellID
rownames(sc_counts_matrix) = rownames(sp_counts_matrix)
rownames(cell_meta) <- cell_meta$CellID
cell_meta$SampleID= 1

sce <- SingleCellExperiment(assays = list(counts = sc_counts_matrix),
                            colData = cell_meta)

colnames(sce) <- cell_meta$CellID

print(head(sce))

# Spatial Transcriptomics Data Prepare
rownames(sp_counts_matrix) = as.character(c(1:nrow(sp_counts_matrix)))
Est.prop = music_prop(bulk.mtx = as.matrix(sp_counts_matrix), sc.sce = sce,
                      markers=rownames(sce),clusters = "CellType", samples = "SampleID")
save(Est.prop,file = "result-MuSiC.Rdata")

norm_weights = Est.prop$Est.prop.weighted
write.csv(norm_weights,file = "result-MuSiC.csv")