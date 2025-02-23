#!/usr/local/bin/Rscript

options(repos = c(CRAN = "https://cloud.r-project.org"))
update.packages(ask = FALSE, checkBuilt = TRUE)


if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}
# check if necessary packages are installed
pkg.list <- installed.packages()[,"Package"] 
BiocManager::install("GenomicRanges")
BiocManager::install("S4Vectors")
BiocManager::install("IRanges")
install.packages(c("matrixStats", "GenomicRanges", "S4Vectors", "IRanges", "argparse"))

# install.packages("argparse", repos = "https://cloud.r-project.org")



# BiocManager::install(c("edgeR", "scran"))

# BiocManager::install(version = "3.20")  # Replace with your R version-compatible Bioconductor version.


if (!("deconvSeq" %in% pkg.list)) {
    if (!("devtools") %in% pkg.list) {
        install.packages("devtools")
    }
    devtools::install_github("rosedu1/deconvSeq")
}

if (!("SingleCellExperiment") %in% pkg.list) {
    if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager")

    BiocManager::install("SingleCellExperiment")
}

library(deconvSeq)
library(SingleCellExperiment)
library(argparse)

parser <- ArgumentParser()

parser$add_argument('-sc','--sc_data',
                    type = 'character',
                    help = ''
                    )

parser$add_argument('-st','--st_data',
                    type = 'character',
                    help = ''
                    )

parser$add_argument('-mt','--meta_data',
                    type = 'character',
                    help = ''
                    )

parser$add_argument('-o','--out_dir',
                    type = 'character',
                    default = NULL,
                    help =''
                    )

args <- parser$parse_args()


sc_cnt_pth <- args$sc_data
sc_mta_pth <- args$meta_data
st_cnt_pth <- args$st_data 

print(sc_cnt_pth)
print(sc_mta_pth)
print(args$out_dir)



dir.create(args$out_dir,
           showWarnings = F)

lbl_vec <- read.table(sc_mta_pth,
                      row.names = 1,
                      sep = '\t',
                      header = T,
                      stringsAsFactors = F)

lbl_vec <- lbl_vec[,"bio_celltype"]

print("read sc count data")

cnts.sc <- read.table(sc_cnt_pth,
                      row.names = 1,
                      sep = '\t',
                      header = T,
                      stringsAsFactors = F)

cnts.sc <- t(cnts.sc)

print("read st count data")

cnts.st <- read.table(st_cnt_pth,
                      row.names = 1,
                      header = T,
                      sep = '\t',
                      stringsAsFactors = F)

cnts.st <- t(cnts.st)

keep.genes <- rowMeans(cnts.sc) > 0.05
cnts.sc <- cnts.sc[keep.genes,]
keep.cells <- colSums(cnts.sc) > 300
cnts.sc <- cnts.sc[,keep.cells]


lbl_vec <- lbl_vec[keep.cells]

ori_lnames <- levels(as.factor(lbl_vec))
lbl_vec <- gsub(",","",lbl_vec)
inter <- intersect(rownames(cnts.st),rownames(cnts.sc))
cnts.st <- as.matrix(cnts.st[inter,])
cnts.sc <- as.matrix(cnts.sc[inter,])
names(lbl_vec) <- colnames(cnts.sc)

cnts.sc <- prep_scrnaseq(cnts.sc,
                         genenametype = "hgnc_symbol",
                         cellcycle = NULL,
                         count.threshold = 0.05)

lbl_vec <-lbl_vec[colnames(cnts.sc)]
lbl_vec <- as.factor(lbl_vec)
lbl_vec <- gsub(" ", "_", lbl_vec)  # Replace spaces with underscores
design.sc = model.matrix(~-1+lbl_vec)
colnames(design.sc) <- levels(lbl_vec)
rownames(design.sc) <- colnames(cnts.sc)


cnts.st <- prep_scrnaseq(cnts.st,
                         genenametype = "hgnc_symbol",
                         cellcycle =NULL,
                         count.threshold = 0.05)


set.seed(1337)
print("filter st data")
cnts.st <-cnts.st[intersect(rownames(cnts.st),rownames(cnts.sc)),]
cnts.st <- cnts.st[,colSums(cnts.st) > 0]

print("Get DGE from sc data")
dge.sc = getdge(cnts.sc,
                design.sc,
                ncpm.min=.01,
                nsamp.min=1,
                method="bin.loess")

# summary(dge.sc$counts)
# summary(dge.sc$common.dispersion)
#summary(dge.sc$tagwise.dispersion)


# print(dim(dge.sc$counts))   # Dimensions of the single-cell data after DGE
# print(dim(design.sc))       # Dimensions of the design matrix

# print(summary(as.vector(dge.sc$counts)))

# print(head(design.sc))
# print(colSums(design.sc))  # Ensure all cell types have non-zero counts

# print(dim(cnts.sc))  # Dimensions after filtering
# # print(head(cnts.sc)) # First few rows/columns of the filtered data

# print(length(inter))  # Number of overlapping genes

# print(summary(dge.sc$dispersion))
print(dim(dge.sc$counts))  # Check dimensions of count data
print(dim(design.sc))      # Check dimensions of the design matrix

dge.sc$counts <- na.omit(dge.sc$counts)  # Remove rows with NA values
design.sc <- na.omit(design.sc)          # Remove rows with NA values in the design matrix


print("fit b0")
b0.sc = getb0.rnaseq(dge.sc,
                     design.sc,
                     ncpm.min=.01,
                     nsamp.min=2, sigg=NULL)



print("Get DGE from ST")
dge.st = getdge(cnts.st,
                NULL,
                ncpm.min=.01,
                nsamp.min=4,
                method="bin.loess")

print("DGE from ST:")

res = getx1.rnaseq(NB0=10,
                   b0.sc,
                   dge.st)

colnames(res$x1) <- ori_lnames
save(res,file = file.path(args$out_dir,"results.R"))

write.table(res$x1,
            file = file.path(args$out_dir,
                             "deconvSeq-proportions.tsv"),
            sep = '\t',
            quote = F,
            col.names = T,
            row.names = T
            )

