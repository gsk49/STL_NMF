library("pcaReduce")
library("pcaMethods")
library(Matrix)
library(dplyr)


B <- read.table("./01_real_data/pdac_b_ref.tsv", header=TRUE, sep="\t")
B <- B[,-1]


tags <- as.matrix(B)
# tags <- as(tags, "dgCMatrix")
# print(tags)

# tags <- read.table('data_pcareduce/A_100.txt')
# tags <- as.matrix(tags)

D <- log2(tags + 1)
Input <- t(D)
start_time <- Sys.time()
print(start_time)
Output_S <- PCAreduce(Input, nbt=2, q=9, method='S')
end_time <- Sys.time()

print(end_time - start_time)

write.csv(Output_S[1], "results_pcareduce/pcaReduce_output.txt");