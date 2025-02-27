library(ggplot2)
library(SPOTlight)
library(SingleCellExperiment)
library(SpatialExperiment)
library(scater)
library(scran)



## Input data directory
o_data_dir = "./00_synthetic/00_PDAC_A/00_No_X_Noise/03_clust/03_x_tsv/"

## SC data
# B = read.csv(paste0(o_data_dir, "S.tsv", sep = "\t")
# B_cell = read.csv(paste0(o_data_dir, "S.csv"))
B = read.csv("S.tsv", sep = "\t")
B_cell = read.csv("S.tsv", sep = "\t")
sc_counts_matrix <- as.matrix(B_cell[, -which(colnames(B_cell) == "Genes")])
cell_meta = read.csv("01_real_data/new_cluster_number.csv")
cell_meta <- cell_meta[,-ncol(cell_meta)]
cell_meta <- cell_meta[,-1]

location_meta = read.csv("real_loc.csv")

## ----Data Preparation---------------------------------------------------------
# Single Cell Data Prepare
colnames(sc_counts_matrix) = cell_meta$CellID
rownames(sc_counts_matrix) = rownames(B)
rownames(cell_meta) <- cell_meta$CellID


sce <- SingleCellExperiment(assays = list(counts = sc_counts_matrix),
                            colData = cell_meta)
colLabels(sce) <- colData(sce)$CellType
colnames(sce) <- cell_meta$CellID



x_names <- read.csv("./zzz_outputs/xfiles.csv", header=FALSE, sep="\n")$V1
x = 0
for(name in x_names){

  ## SP data
  X = read.csv(paste0(o_data_dir, name), sep = "\t")
  sp_counts_matrix= as.matrix(X[, -which(colnames(X) == "Genes")])



# Spatial Transciptomics Data Prepare
    spe <- SpatialExperiment(assays = list(counts = sp_counts_matrix),
                            colData = DataFrame(coord_x = location_meta$x, coord_y = location_meta$y))
    rownames(spe) = rownames(X)

    ## ----lognorm------------------------------------------------------------------
    sce <- logNormCounts(sce)

    ## ----mgs----------------------------------------------------------------------
    mgs <- scoreMarkers(sce, subset.row = rownames(sce))
    mgs_fil <- lapply(names(mgs), function(i) {
      x <- mgs[[i]]
      x$gene <- rownames(x)
      x$cluster <- i
      data.frame(x)
    })
    mgs_df <- do.call(rbind, mgs_fil)

    ## ----downsample---------------------------------------------------------------
    # split cell indices by identity
    idx <- split(seq(ncol(sce)), sce$CellType)
    # downsample to at most 20 per identity & subset
    # set to 75-100 for your real life analysis
    n_cells <- 75
    cs_keep <- lapply(idx, function(i) {
      n <- length(i)
      if (n < n_cells)
        n_cells <- n
      sample(i, n_cells)
    })
    sce <- sce[, unlist(cs_keep)]

    ## ----SPOTlight----------------------------------------------------------------

    # res <- SPOTlight(
    #     x = sce,
    #     y = spe,
    #     groups = as.character(sce$CellType),
    #     mgs = mgs_df,
    #     weight_id = "mean.AUC",
    #     group_id = "cluster",
    #     gene_id = "gene")
    if (x == 0){
    mod_ls <- trainNMF(
    x = sce,
    y = spe,
    groups = as.character(sce$CellType),
    mgs = mgs_df,
    weight_id = "mean.AUC",
    group_id = "cluster",
    gene_id = "gene")
    }
    res <- runDeconvolution(
    x = spe,
    mod = mod_ls[["mod"]],
    ref = mod_ls[["topic"]])

    write.table(res$mat, file = paste0("spotlight_", name), sep = "\t", row.names = TRUE, col.names = TRUE)
    print(x)
    x = x+1
}