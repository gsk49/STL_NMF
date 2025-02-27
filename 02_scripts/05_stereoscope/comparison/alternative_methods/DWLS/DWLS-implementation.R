#!/usr/local/bin/Rscript
chooseCRANmirror(graphics=FALSE, ind=1)  # Automatically choose the first CRAN mirror

pkg.list <- installed.packages()[,"Package"] 

# if (!("argparse") %in% pkg.list) {
#     install.packages("argparse")
# }
# if (!("zeallot") %in% pkg.list) {
#     install.packages("zeallot")
# }

# install.packages("tidyverse")
# install.packages("varhandle")
# install.packages("dmm")


# # Install Bioconductor packages using the correct method
# if (!requireNamespace("BiocManager", quietly = TRUE)) {
#   install.packages("BiocManager")
# }

# install.packages("igraph", force = TRUE)
# BiocManager::install(c("reshape", "varhandle", "MAST"), force = TRUE)


# taken from:
# https://stackoverflow.com/questions/47044068/get-the-path-of-current-scriptk
getCurrentFileLocation <-  function()
{
    this_file <- commandArgs() %>% 
    tibble::enframe(name = NULL) %>%
    tidyr::separate(col=value, into=c("key", "value"), sep="=", fill='right') %>%
    dplyr::filter(key == "--file") %>%
    dplyr::pull(value)
    if (length(this_file)==0)
    {
      this_file <- rstudioapi::getSourceEditorContext()$path
    }
    return(dirname(this_file))
}

library(zeallot)
library(argparse)
library(tidyverse)
library(dmm)
library(MAST)
library(varhandle)

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

parser$add_argument('-wd','--workdir',
                    type = 'character',
                    default = NULL,
                    help =''
                    )

args <- parser$parse_args()

workdir <-  ifelse(is.null(args$workdir),
                 'project',
                 args$workdir)

sc_cnt_pth <- args$sc_data
sc_mta_pth <- args$meta_data
st_cnt_pth <- args$st_data 



dir.create(file.path(workdir,"results"), showWarnings = F)

script.dir <- getCurrentFileLocation()

source(file.path(script.dir,"Modded_Deconvolution_functions.R"))
print("Loading SC count data")
dataSC <-  read.table(sc_cnt_pth,
                      sep = '\t',
                      header = T,
                      row.names = 1,
                      stringsAsFactors = F
                      )


print(dataSC[1:10,1:10])
print("Loading ST count data")

dataST <- read.table(st_cnt_pth,
                     sep = '\t',
                     header = T,
                     row.names = 1,
                     stringsAsFactors = F
                     )
print(dataST[1:10,1:10])
print("Loading SC labels")

labels <- read.table(sc_mta_pth,
                     sep = '\t',
                     header = T,
                     row.names = 1,
                     stringsAsFactors = F
                     )
oldest_labels <- labels
labels <- labels['bio_celltype']



print(labels[1:10,])
print(workdir)
setwd(workdir)
print("Prepare Data for analysis")
intermeta <- intersect(rownames(dataSC),rownames(labels))
dataSC <- dataSC[intermeta,]
labels <- labels[intermeta,]
labels <- gsub(',| |\\.|-','_',labels, perl = T)
types <- unique(labels)
n_types <- length(types)

interst <- intersect(colnames(dataSC),colnames(dataST))
dataSC <- dataSC[,interst]
dataST <- dataST[,interst]

dataSC <- t(dataSC) # transposition converts to matrix
# labels <- as.vector(unlist(labels))
# if (is.data.frame(oldest_labels)) {
#     oldest_labels <- oldest_labels$bio_celltype  # Extract bio_celltype column
# } else {
#     stop("oldest_labels is not a data frame")
# }
# oldest_labels <- as.character(oldest_labels)  # Convert to character if it was a factor or data frame

# # Convert to factor if needed (check the requirements of buildSignatureMatrixMAST)
# oldest_labels <- factor(oldest_labels)
oldest_labels <- oldest_labels$bio_celltype


print(length(oldest_labels))       # Length should match nrow(dataSC)
print(class(oldest_labels))        # Should be a character or factor vector
print(unique(oldest_labels))  



dim(dataSC)
print(class(dataSC))        # Should be a character or factor vector
print("Building Signatures")

set.seed(1337)
Signatures <- buildSignatureMatrixMAST(scdata=dataSC,
                                       id=oldest_labels,
                                       path="results",
                                       # fitDEA = T,
                                       diff.cutoff=0.5,
                                       pval.cutoff=0.01
                                       ) 
print("Signatures built")
c(n_spots,n_genes) %<-% dim(dataST)

prop_mat <- as.data.frame(matrix(0,
                                 nrow = n_spots,
                                 ncol = n_types
                                )
                         )
rownames(prop_mat) <- rownames(dataST)
colnames(prop_mat) <- unique(old_labels)

print("Estimate propotions in each spot")
dataST <- as.matrix(dataST)
for (s in 1:n_spots) {

    print(sprintf("Estimating proportion for spot : %d / %d",
                  s,n_spots)
          )

    spot <- dataST[s,]
    tr <- trimData(Signatures,spot)

    tr$sig <- tr$sig[,colSums(tr$sig) > 0]
    is_pd <- eigen(t(tr$sig)%*%tr$sig)$values
    is_pd <- all(is_pd > 10e-6)

    if (!(is_pd)) { 
        next
    }
    
    try(solDWLS <- solveDampenedWLS(tr$sig,tr$bulk),next)
    print("Proportions >> ")
    prop_mat[s,names(solDWLS)] <- solDWLS
}

write.table(prop_mat,
            file = 'results/DWLS-proportions.tsv',
            sep = '\t',
            quote = F,
            col.names = T,
            row.names = T
            )