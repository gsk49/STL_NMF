install.packages('devtools')
install.packages("BiocManager")
BiocManager::install("Biobase", force=TRUE)
BiocManager::install("SingleCellExperiment", force=TRUE)
BiocManager::install("SummarizedExperiment", force=TRUE)
BiocManager::install("TOAST", force=TRUE)
devtools::install_github("YingMa0107/CARD", force=TRUE)
devtools::install_github("xuranw/MuSiC")

library(devtools)
library(CARD)
library(MuSiC)
library(Matrix)


X_rowNames <- read.csv("../deepNMF/X_rowNames.csv", sep = "\n", header=FALSE)$V1
X_colNames <- read.csv("../deepNMF/01_real_data/locs.csv", sep = "\n", header=FALSE)$V1

loc_cols <- c("x", "y")
spatial_location <- read.csv("../deepNMF/01_real_data/locs.csv", sep=",", header=FALSE)
rownames(spatial_location) <- X_colNames
colnames(spatial_location) <- loc_cols

spatial_location[1:4,]




sc_count <- read.csv("../deepNMF/01_real_data/S_GSE111672_PDAC-A-indrop-filtered-expMat_select.csv", sep=',', header=TRUE)
sc_count <- sc_count[, -ncol(sc_count)]
rownames(sc_count) <- X_rowNames

sc_count <- as(as(as.matrix(sc_count), "sparseMatrix"), "dgCMatrix")
sc_count[1:3,1:3]



sc_meta <- read.csv("../deepNMF/01_real_data/new_cluster_number.csv", sep=',', header=TRUE)
sc_meta <- sc_meta[, -ncol(sc_meta)]
sc_meta <- sc_meta[, -1]
meta_row <- sc_meta[, 1]
print(meta_row)
rownames(sc_meta) <- meta_row
samples <- rep(1, 1926)
sc_meta$sample <- samples

sc_meta[1:20,]

x_names <- read.csv("../deepNMF/x_names.csv", header=FALSE, sep="\n")$V1
x <- 0
for(name in x_names){
  print(x)
  x_name <- paste("../deepNMF/00_synthetic/00_PDAC_A/01_X_Noise/05_clust/01_x/",name, sep="")
  print(x_name)
  spatial_count <- read.csv(x_name, sep=",", header=FALSE)
  # spatial_count <- spatial_count[, -1]
  names <- read.csv("../deepNMF/Genes.csv", sep=",", header=FALSE)$V1
  rownames(spatial_count) <- names
  colnames(spatial_count) <- X_colNames
  
  spatial_count <- as(as(as.matrix(spatial_count), "sparseMatrix"), "dgCMatrix")
  spatial_count[1:4,1:4]
  
  CARD_obj = createCARDObject(
    sc_count=sc_count,
    sc_meta=sc_meta,
    spatial_count = spatial_count,
    spatial_location = spatial_location,
    ct.varname = "CellType",
    # ct.select = unique(sc_meta$cellType),
    ct.select = c("Acinar cells", "Cancer clone A", "Cancer clone B"),
    # ct.select = c("Acinar cells", "Cancer clone A", "Endocrine cells", "Fibroblasts", "Ductal - terminal ductal like"),
    sample.varname = "sample",
    # minCountGene = 100,
    # minCountSpot = 5
    minCountGene = 1,
    minCountSpot = 1
    )
  
  CARD_obj = CARD_deconvolution(CARD_object = CARD_obj)
  
  v = as.matrix(CARD_obj@Proportion_CARD)
  write.csv(v, paste("./noisy_3_v/card_",name, sep=""), row.names=FALSE)
  
  # print(CARD_obj@Proportion_CARD[1:3,])
  x <- x+1
}
